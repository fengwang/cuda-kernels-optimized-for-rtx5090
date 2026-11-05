// Row-wise softmax over the last dim, fp32 and bf16 rows (dtype 0 and 1), every value accumulated in fp32.
// Paths, chosen by row length:
//  - warp per row (row <= 32 * 32 elements): the row lives in registers, one DRAM read and one write;
//  - block per row (row <= 1024 * 32 elements): same, with block reductions;
//  - cluster per row, register-cached (longer 16-byte vectorisable rows): the row is split over a thread block
//    cluster, each block holds its chunk in registers as raw 16-byte vectors, the blocks exchange the max and
//    then the sum through distributed shared memory, and each block normalises from registers: one read, one write;
//  - cluster per row, re-reading (rows that are not 16-byte vectorisable): each block runs an online max/sum over
//    its chunk, the blocks exchange (max, sum), then each block re-reads its chunk (L2-resident) and normalises it.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <math_constants.h>
#include <stdint.h>

namespace cg = cooperative_groups;

template <typename T> struct Vec;
template <> struct Vec<float> { static constexpr int N = 4; };
template <> struct Vec<__nv_bfloat16> { static constexpr int N = 8; };

// Raw 16-byte vector type of a dtype, kept packed in registers.
template <typename T> struct Raw;
template <> struct Raw<float> { using type = float4; };
template <> struct Raw<__nv_bfloat16> { using type = uint4; };

__device__ __forceinline__ float ld(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void st(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void st(__nv_bfloat16* p, long long i, float v) { p[i] = __float2bfloat16_rn(v); }

// 16-byte vector load / evict-first store of vector index v.
__device__ __forceinline__ void ldv(const float* p, long long v, float* o) {
  float4 q = __ldg(reinterpret_cast<const float4*>(p) + v);
  o[0] = q.x; o[1] = q.y; o[2] = q.z; o[3] = q.w;
}

__device__ __forceinline__ void bf2(unsigned w, float* o) {
  o[0] = __uint_as_float(w << 16);
  o[1] = __uint_as_float(w & 0xffff0000u);
}

__device__ __forceinline__ void ldv(const __nv_bfloat16* p, long long v, float* o) {
  uint4 q = __ldg(reinterpret_cast<const uint4*>(p) + v);
  bf2(q.x, o); bf2(q.y, o + 2); bf2(q.z, o + 4); bf2(q.w, o + 6);
}

// Raw 16-byte vector load of vector index v, and its expansion to fp32.
__device__ __forceinline__ float4 ldraw(const float* p, long long v) {
  return __ldg(reinterpret_cast<const float4*>(p) + v);
}

__device__ __forceinline__ uint4 ldraw(const __nv_bfloat16* p, long long v) {
  return __ldg(reinterpret_cast<const uint4*>(p) + v);
}

__device__ __forceinline__ void unpack(const float4& q, float* o) {
  o[0] = q.x; o[1] = q.y; o[2] = q.z; o[3] = q.w;
}

__device__ __forceinline__ void unpack(const uint4& q, float* o) {
  bf2(q.x, o); bf2(q.y, o + 2); bf2(q.z, o + 4); bf2(q.w, o + 6);
}

__device__ __forceinline__ void stv(float* p, long long v, const float* i) {
  __stcs(reinterpret_cast<float4*>(p) + v, make_float4(i[0], i[1], i[2], i[3]));
}

__device__ __forceinline__ unsigned pack(float a, float b) {
  return (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(a)) |
         ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(b)) << 16);
}

__device__ __forceinline__ void stv(__nv_bfloat16* p, long long v, const float* i) {
  uint4 q;
  q.x = pack(i[0], i[1]); q.y = pack(i[2], i[3]); q.z = pack(i[4], i[5]); q.w = pack(i[6], i[7]);
  __stcs(reinterpret_cast<uint4*>(p) + v, q);
}

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Block reductions through one shared slot per warp; every thread returns the block's result.
__device__ float block_max(float v, float* red) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
  v = warp_max(v);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = warp_max(lane < warps ? red[lane] : -CUDART_INF_F);
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  v = red[0];
  __syncthreads();
  return v;
}

__device__ float block_sum(float v, float* red) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = warp_sum(lane < warps ? red[lane] : 0.0f);
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  v = red[0];
  __syncthreads();
  return v;
}

// Merges the online-softmax state (om, os) into (m, s): s is the sum of exp(x - m).
__device__ __forceinline__ void combine(float& m, float& s, float om, float os) {
  float M = fmaxf(m, om);
  if (M == -CUDART_INF_F) return;
  s = s * expf(m - M) + os * expf(om - M);
  m = M;
}

__device__ __forceinline__ void warp_combine(float& m, float& s) {
  for (int o = 16; o > 0; o >>= 1) {
    float om = __shfl_xor_sync(0xffffffffu, m, o);
    float os = __shfl_xor_sync(0xffffffffu, s, o);
    combine(m, s, om, os);
  }
}

__device__ void block_combine(float& m, float& s, float* rm, float* rs) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
  warp_combine(m, s);
  if (lane == 0) { rm[warp] = m; rs[warp] = s; }
  __syncthreads();
  if (warp == 0) {
    m = lane < warps ? rm[lane] : -CUDART_INF_F;
    s = lane < warps ? rs[lane] : 0.0f;
    warp_combine(m, s);
    if (lane == 0) { rm[0] = m; rs[0] = s; }
  }
  __syncthreads();
  m = rm[0];
  s = rs[0];
  __syncthreads();
}

// One warp per row, NV vectors per lane held in registers. Needs cols % V == 0 and 16-byte aligned x, y.
template <typename T, int NV>
__global__ void __launch_bounds__(128) softmax_warp_rows(const T* __restrict__ x, T* __restrict__ y, long long rows,
                                                         long long cols) {
  constexpr int V = Vec<T>::N;
  long long row = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (row >= rows) return;
  long long nvec = cols / V;
  const T* xr = x + row * cols;
  T* yr = y + row * cols;
  float v[NV][V];
  float m = -CUDART_INF_F;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * 32 + lane;
    if (idx < nvec) {
      ldv(xr, idx, v[k]);
#pragma unroll
      for (int i = 0; i < V; ++i) m = fmaxf(m, v[k][i]);
    }
  }
  m = warp_max(m);
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * 32 + lane;
    if (idx < nvec) {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        v[k][i] = expf(v[k][i] - m);
        s += v[k][i];
      }
    }
  }
  s = warp_sum(s);
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * 32 + lane;
    if (idx < nvec) {
#pragma unroll
      for (int i = 0; i < V; ++i) v[k][i] = v[k][i] / s;
      stv(yr, idx, v[k]);
    }
  }
}

// One block per row, NV vectors per thread held in registers. Needs cols % V == 0 and 16-byte aligned x, y.
template <typename T, int NV>
__global__ void __launch_bounds__(1024) softmax_block_rows(const T* __restrict__ x, T* __restrict__ y,
                                                           long long cols) {
  constexpr int V = Vec<T>::N;
  __shared__ float red[32];
  long long nvec = cols / V;
  const T* xr = x + (long long)blockIdx.x * cols;
  T* yr = y + (long long)blockIdx.x * cols;
  float v[NV][V];
  float m = -CUDART_INF_F;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * blockDim.x + threadIdx.x;
    if (idx < nvec) {
      ldv(xr, idx, v[k]);
#pragma unroll
      for (int i = 0; i < V; ++i) m = fmaxf(m, v[k][i]);
    }
  }
  m = block_max(m, red);
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * blockDim.x + threadIdx.x;
    if (idx < nvec) {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        v[k][i] = expf(v[k][i] - m);
        s += v[k][i];
      }
    }
  }
  s = block_sum(s, red);
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * blockDim.x + threadIdx.x;
    if (idx < nvec) {
#pragma unroll
      for (int i = 0; i < V; ++i) v[k][i] = v[k][i] / s;
      stv(yr, idx, v[k]);
    }
  }
}

// One cluster per row (grid (cluster, rows)); block `rank` owns vectors [rank * chunk / V, (rank + 1) * chunk / V)
// and holds them in registers as raw 16-byte vectors, NV per thread. Needs cols % V == 0, chunk % V == 0,
// blockDim.x * NV >= chunk / V and 16-byte aligned x, y.
template <typename T, int NV>
__global__ void __launch_bounds__(1024) softmax_cluster_reg(const T* __restrict__ x, T* __restrict__ y,
                                                            long long cols, long long chunk) {
  constexpr int V = Vec<T>::N;
  using R = typename Raw<T>::type;
  __shared__ float red[32];
  __shared__ float part[2];
  __shared__ float stat;
  cg::cluster_group cluster = cg::this_cluster();
  unsigned rank = cluster.block_rank();
  unsigned csize = cluster.num_blocks();
  const T* xr = x + (long long)blockIdx.y * cols;
  T* yr = y + (long long)blockIdx.y * cols;
  long long nvec = cols / V;
  long long beg = (long long)rank * (chunk / V);
  long long end = beg + chunk / V < nvec ? beg + chunk / V : nvec;
  long long cnt = end - beg;  // <= 0 for a trailing block past the row's end

  R q[NV];
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * blockDim.x + threadIdx.x;
    if (idx < cnt) q[k] = ldraw(xr, beg + idx);
  }
  float m = -CUDART_INF_F;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * blockDim.x + threadIdx.x;
    if (idx < cnt) {
      float a[V];
      unpack(q[k], a);
#pragma unroll
      for (int i = 0; i < V; ++i) m = fmaxf(m, a[i]);
    }
  }
  m = block_max(m, red);
  if (threadIdx.x == 0) part[0] = m;
  cluster.sync();
  if (threadIdx.x < 32) {
    float pm = threadIdx.x < csize ? cluster.map_shared_rank(part, threadIdx.x)[0] : -CUDART_INF_F;
    pm = warp_max(pm);
    if (threadIdx.x == 0) stat = pm;
  }
  __syncthreads();
  m = stat;

  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * blockDim.x + threadIdx.x;
    if (idx < cnt) {
      float a[V];
      unpack(q[k], a);
#pragma unroll
      for (int i = 0; i < V; ++i) s += expf(a[i] - m);
    }
  }
  s = block_sum(s, red);  // its barriers also order every read of `stat` above before the write below
  if (threadIdx.x == 0) part[1] = s;
  cluster.sync();
  if (threadIdx.x < 32) {
    float ps = threadIdx.x < csize ? cluster.map_shared_rank(part, threadIdx.x)[1] : 0.0f;
    ps = warp_sum(ps);
    if (threadIdx.x == 0) stat = ps;
  }
  cluster.sync();  // peers are done reading `part`, and `stat` is visible block-wide
  s = stat;

#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long idx = (long long)k * blockDim.x + threadIdx.x;
    if (idx < cnt) {
      float a[V];
      unpack(q[k], a);
#pragma unroll
      for (int i = 0; i < V; ++i) a[i] = expf(a[i] - m) / s;
      stv(yr, beg + idx, a);
    }
  }
}

// One cluster per row (grid (cluster, rows)); block `rank` owns elements [rank * chunk, (rank + 1) * chunk).
// VECT needs cols % V == 0, chunk % V == 0 and 16-byte aligned x, y.
template <typename T, bool VECT>
__global__ void __launch_bounds__(1024) softmax_cluster(const T* __restrict__ x, T* __restrict__ y, long long cols,
                                                        long long chunk) {
  constexpr int V = Vec<T>::N;
  __shared__ float rm[32], rs[32];
  __shared__ float part[2];
  cg::cluster_group cluster = cg::this_cluster();
  unsigned rank = cluster.block_rank();
  unsigned csize = cluster.num_blocks();
  const T* xr = x + (long long)blockIdx.y * cols;
  T* yr = y + (long long)blockIdx.y * cols;
  long long beg = (long long)rank * chunk;
  long long end = beg + chunk < cols ? beg + chunk : cols;

  float m = -CUDART_INF_F, s = 0.0f;
  if constexpr (VECT) {
    for (long long i = beg / V + threadIdx.x; i < end / V; i += blockDim.x) {
      float a[V];
      ldv(xr, i, a);
      float lm = a[0];
#pragma unroll
      for (int j = 1; j < V; ++j) lm = fmaxf(lm, a[j]);
      if (lm > m) {
        s *= expf(m - lm);
        m = lm;
      }
#pragma unroll
      for (int j = 0; j < V; ++j) s += expf(a[j] - m);
    }
  } else {
    for (long long i = beg + threadIdx.x; i < end; i += blockDim.x) {
      float a = ld(xr, i);
      if (a > m) {
        s *= expf(m - a);
        m = a;
      }
      s += expf(a - m);
    }
  }
  block_combine(m, s, rm, rs);
  if (threadIdx.x == 0) { part[0] = m; part[1] = s; }
  cluster.sync();
  if (threadIdx.x < 32) {
    float pm = -CUDART_INF_F, ps = 0.0f;
    if (threadIdx.x < csize) {
      float* rp = cluster.map_shared_rank(part, threadIdx.x);
      pm = rp[0];
      ps = rp[1];
    }
    warp_combine(pm, ps);
    if (threadIdx.x == 0) { rm[0] = pm; rs[0] = ps; }
  }
  cluster.sync();  // peers are done reading `part`, and rm[0], rs[0] are visible block-wide
  m = rm[0];
  s = rs[0];

  if constexpr (VECT) {
    for (long long i = beg / V + threadIdx.x; i < end / V; i += blockDim.x) {
      float a[V];
      ldv(xr, i, a);
#pragma unroll
      for (int j = 0; j < V; ++j) a[j] = expf(a[j] - m) / s;
      stv(yr, i, a);
    }
  } else {
    for (long long i = beg + threadIdx.x; i < end; i += blockDim.x) st(yr, i, expf(ld(xr, i) - m) / s);
  }
}

// Cluster size for a long row: up to 8 blocks, each owning at least 4096 elements.
static int cluster_size(long long cols) {
  int cl = 1;
  while (cl < 8 && cols >= (long long)cl * 2 * 4096) cl *= 2;
  return cl;
}

static cudaLaunchConfig_t cluster_config(int cl, long long rows, int t, cudaStream_t stream,
                                         cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)cl, (unsigned)rows, 1);
  cfg.blockDim = dim3((unsigned)t, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (unsigned)cl;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cfg;
}

template <typename T, bool VECT>
static int launch_cluster(const T* x, T* y, long long rows, long long cols, int threads, cudaStream_t stream) {
  constexpr int V = Vec<T>::N;
  int cl = cluster_size(cols);
  long long chunk = (cols + cl - 1) / cl;
  if (VECT) chunk = (chunk + V - 1) / V * V;
  long long work = VECT ? chunk / V : chunk;
  int t = threads;
  long long need = (work + 31) / 32 * 32;
  if (need < t) t = (int)need;
  if (t < 32) t = 32;
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = cluster_config(cl, rows, t, stream, attr);
  cudaError_t err = cudaLaunchKernelEx(&cfg, softmax_cluster<T, VECT>, x, y, cols, chunk);
  if (err != cudaSuccess) return (int)err;
  return (int)cudaGetLastError();
}

template <typename T, int NV>
static int launch_cluster_reg(const T* x, T* y, long long rows, long long cols, long long chunk, int cl, int t,
                              cudaStream_t stream) {
  cudaLaunchAttribute attr[1];
  cudaLaunchConfig_t cfg = cluster_config(cl, rows, t, stream, attr);
  cudaError_t err = cudaLaunchKernelEx(&cfg, softmax_cluster_reg<T, NV>, x, y, cols, chunk);
  if (err != cudaSuccess) return (int)err;
  return (int)cudaGetLastError();
}

// Register-cached cluster path for a vectorisable row: the smallest NV whose block fits in `cap` threads, else NV 8
// with up to 1024 threads. Returns -1 when the chunk does not fit in 8 vectors per thread.
template <typename T>
static int cluster_reg(const T* x, T* y, long long rows, long long cols, int cap, cudaStream_t stream) {
  constexpr int V = Vec<T>::N;
  int cl = cluster_size(cols);
  long long chunk = ((cols + cl - 1) / cl + V - 1) / V * V;
  long long nvc = chunk / V;
  auto fit = [nvc](long long nv) {
    long long t = ((nvc + nv - 1) / nv + 31) / 32 * 32;
    return t < 32 ? 32LL : t;
  };
  if (fit(2) <= cap) return launch_cluster_reg<T, 2>(x, y, rows, cols, chunk, cl, (int)fit(2), stream);
  if (fit(4) <= cap) return launch_cluster_reg<T, 4>(x, y, rows, cols, chunk, cl, (int)fit(4), stream);
  if (fit(8) <= 1024) return launch_cluster_reg<T, 8>(x, y, rows, cols, chunk, cl, (int)fit(8), stream);
  return -1;
}

template <typename T>
static int launch_t(const T* x, T* y, long long rows, long long cols, int threads, int big, cudaStream_t stream) {
  constexpr int V = Vec<T>::N;
  constexpr int MAXNV = 32 / V;  // at most 32 cached fp32 values per thread
  bool vec = (cols % V == 0) && ((uintptr_t)x % 16 == 0) && ((uintptr_t)y % 16 == 0);
  if (vec) {
    long long nvec = cols / V;
    if (nvec <= 32LL * MAXNV) {
      unsigned grid = (unsigned)((rows + 3) / 4);
      if (nvec <= 32) {
        softmax_warp_rows<T, 1><<<grid, 128, 0, stream>>>(x, y, rows, cols);
      } else if (nvec <= 64) {
        softmax_warp_rows<T, 2><<<grid, 128, 0, stream>>>(x, y, rows, cols);
      } else if (nvec <= 128 || MAXNV < 8) {
        softmax_warp_rows<T, 4><<<grid, 128, 0, stream>>>(x, y, rows, cols);
      } else {
        if constexpr (MAXNV >= 8) softmax_warp_rows<T, 8><<<grid, 128, 0, stream>>>(x, y, rows, cols);
      }
      return (int)cudaGetLastError();
    }
    int bt = threads / 32 * 32;
    if (bt < 32) bt = 32;
    if (bt > 1024) bt = 1024;
    while (bt < 1024 && (long long)bt * MAXNV < nvec) bt = bt * 2 > 1024 ? 1024 : bt * 2;
    if ((long long)bt * MAXNV >= nvec) {
      long long nv = (nvec + bt - 1) / bt;
      unsigned grid = (unsigned)rows;
      if (nv <= 1) {
        softmax_block_rows<T, 1><<<grid, bt, 0, stream>>>(x, y, cols);
      } else if (nv <= 2) {
        softmax_block_rows<T, 2><<<grid, bt, 0, stream>>>(x, y, cols);
      } else if (nv <= 4 || MAXNV < 8) {
        softmax_block_rows<T, 4><<<grid, bt, 0, stream>>>(x, y, cols);
      } else {
        if constexpr (MAXNV >= 8) softmax_block_rows<T, 8><<<grid, bt, 0, stream>>>(x, y, cols);
      }
      return (int)cudaGetLastError();
    }
  }
  int t = big / 32 * 32;
  if (t < 32) t = 32;
  if (t > 1024) t = 1024;
  if (vec) {
    int err = cluster_reg<T>(x, y, rows, cols, t, stream);
    if (err != -1) return err;
    return launch_cluster<T, true>(x, y, rows, cols, t, stream);
  }
  return launch_cluster<T, false>(x, y, rows, cols, t, stream);
}

extern "C" int launch(const void* x, void* y, long long rows, long long cols, int dtype, int threads, int aux,
                      cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return 0;
  if (dtype == 0) return launch_t<float>((const float*)x, (float*)y, rows, cols, threads, aux, stream);
  if (dtype == 1)
    return launch_t<__nv_bfloat16>((const __nv_bfloat16*)x, (__nv_bfloat16*)y, rows, cols, threads, aux, stream);
  return -1;
}
