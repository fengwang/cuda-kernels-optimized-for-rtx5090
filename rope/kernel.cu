// Vectorized rotary position embedding of q and k. Block (head_dim/16) x ROWS per token (grid.x), grid.y splits the
// heads: thread (c, y) owns the 8 consecutive pairs 8c..8c+7 of head_dim (element i with i + head_dim/2, half-split
// rotate_half), reads their cos/sin values from the token's cache row once with 16-byte loads and rotates those pairs
// in heads y + ROWS * blockIdx.y, ... with 16-byte loads and stores of q/k:
//   out[i]        = x[i] * cos[i]        - x[i + h] * sin[i]
//   out[i + h]    = x[i + h] * cos[i + h] + x[i] * sin[i + h]          (h = head_dim / 2)
// fp32 math with the products and the sum each rounded (__fmul_rn, __fadd_rn: no contraction), as the reference's
// x * cos + rotate_half(x) * sin, then one round-to-nearest-even conversion to the output dtype. q/k dtypes: bf16 (1)
// and fp16 (2); cache dtypes: fp32 (0), bf16 (1) and fp16 (2), each read as fp32. Row-major contiguous tensors;
// position_ids are int64 in [0, max_position). head_dim % 16 != 0 or a pointer not 16-byte aligned falls back to the
// scalar seed kernel (one pair per thread).
// The vectorized kernel is software-pipelined: the first head's q/k loads are issued before the position id and the
// dependent cos/sin loads, and each later head's q/k are loaded before the previous head is rotated and stored.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float ld(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float ld(const __half* p, long long i) { return __half2float(p[i]); }

__device__ __forceinline__ void st(__nv_bfloat16* p, long long i, float v) { p[i] = __float2bfloat16_rn(v); }
__device__ __forceinline__ void st(__half* p, long long i, float v) { p[i] = __float2half_rn(v); }

// 8 packed 16-bit values <-> 8 floats.
__device__ __forceinline__ void unpack8(uint4 r, float (&v)[8], const __nv_bfloat16*) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(h[j]);
    v[2 * j] = f.x;
    v[2 * j + 1] = f.y;
  }
}
__device__ __forceinline__ void unpack8(uint4 r, float (&v)[8], const __half*) {
  const __half2* h = reinterpret_cast<const __half2*>(&r);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __half22float2(h[j]);
    v[2 * j] = f.x;
    v[2 * j + 1] = f.y;
  }
}
__device__ __forceinline__ uint4 pack8(const float (&v)[8], const __nv_bfloat16*) {
  uint4 r;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
  for (int j = 0; j < 4; ++j) h[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
  return r;
}
__device__ __forceinline__ uint4 pack8(const float (&v)[8], const __half*) {
  uint4 r;
  __half2* h = reinterpret_cast<__half2*>(&r);
#pragma unroll
  for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
  return r;
}

// 8 consecutive cache values (16- or 32-byte aligned) as floats, through the read-only path.
__device__ __forceinline__ void ld8(const float* p, long long i, float (&v)[8]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p + i)), b = __ldg(reinterpret_cast<const float4*>(p + i + 4));
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
  v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void ld8(const __nv_bfloat16* p, long long i, float (&v)[8]) {
  unpack8(__ldg(reinterpret_cast<const uint4*>(p + i)), v, p);
}
__device__ __forceinline__ void ld8(const __half* p, long long i, float (&v)[8]) {
  unpack8(__ldg(reinterpret_cast<const uint4*>(p + i)), v, p);
}

template <typename T>
__device__ __forceinline__ void rotate_pair(const T* x, T* out, long long base, int i, int half, float c1, float s1,
                                            float c2, float s2) {
  float a = ld(x, base + i), b = ld(x, base + i + half);
  st(out, base + i, __fadd_rn(__fmul_rn(a, c1), -__fmul_rn(b, s1)));
  st(out, base + i + half, __fadd_rn(__fmul_rn(b, c2), __fmul_rn(a, s2)));
}

// Element offset of head h (q heads first, then k heads) of a token in its tensor.
__device__ __forceinline__ long long head_base(long long token, int h, int q_heads, int kv_heads, int head_dim) {
  return h < q_heads ? (token * q_heads + h) * head_dim : (token * kv_heads + (h - q_heads)) * head_dim;
}

template <typename T, typename C>
__global__ void rope_vec(const T* __restrict__ q, const T* __restrict__ k, const C* __restrict__ cos_rows,
                         const C* __restrict__ sin_rows, const long long* __restrict__ ids, T* __restrict__ q_out,
                         T* __restrict__ k_out, int q_heads, int kv_heads, int head_dim) {
  const long long token = blockIdx.x;
  const int half = head_dim >> 1;
  const int c = threadIdx.x * 8;
  const int heads = q_heads + kv_heads;
  const int step = blockDim.y * gridDim.y;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (h >= heads) return;
  // First head's q/k in flight before the position id -> cos/sin chain.
  long long base = head_base(token, h, q_heads, kv_heads, head_dim) + c;
  const T* x = h < q_heads ? q : k;
  uint4 ra = __ldcs(reinterpret_cast<const uint4*>(x + base));
  uint4 rb = __ldcs(reinterpret_cast<const uint4*>(x + base + half));
  const long long row = __ldg(ids + token) * head_dim;
  float c1[8], s1[8], c2[8], s2[8];
  ld8(cos_rows, row + c, c1);
  ld8(sin_rows, row + c, s1);
  ld8(cos_rows, row + c + half, c2);
  ld8(sin_rows, row + c + half, s2);
  while (true) {
    const int hn = h + step;
    const bool more = hn < heads;
    long long next = 0;
    uint4 na = ra, nb = rb;
    if (more) {
      next = head_base(token, hn, q_heads, kv_heads, head_dim) + c;
      const T* xn = hn < q_heads ? q : k;
      na = __ldcs(reinterpret_cast<const uint4*>(xn + next));
      nb = __ldcs(reinterpret_cast<const uint4*>(xn + next + half));
    }
    float a[8], b[8], oa[8], ob[8];
    unpack8(ra, a, q);
    unpack8(rb, b, q);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      oa[j] = __fadd_rn(__fmul_rn(a[j], c1[j]), -__fmul_rn(b[j], s1[j]));
      ob[j] = __fadd_rn(__fmul_rn(b[j], c2[j]), __fmul_rn(a[j], s2[j]));
    }
    T* out = h < q_heads ? q_out : k_out;
    __stcs(reinterpret_cast<uint4*>(out + base), pack8(oa, out));
    __stcs(reinterpret_cast<uint4*>(out + base + half), pack8(ob, out));
    if (!more) break;
    h = hn;
    base = next;
    ra = na;
    rb = nb;
  }
}

template <typename T, typename C>
__global__ void rope_seed(const T* __restrict__ q, const T* __restrict__ k, const C* __restrict__ cos_rows,
                          const C* __restrict__ sin_rows, const long long* __restrict__ ids, T* __restrict__ q_out,
                          T* __restrict__ k_out, int q_heads, int kv_heads, int head_dim) {
  const long long token = blockIdx.x;
  const int half = head_dim >> 1;
  const long long row = ids[token] * head_dim;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float c1 = ld(cos_rows, row + i), s1 = ld(sin_rows, row + i);
    const float c2 = ld(cos_rows, row + i + half), s2 = ld(sin_rows, row + i + half);
    for (int h = threadIdx.y; h < q_heads + kv_heads; h += blockDim.y) {
      if (h < q_heads) {
        rotate_pair(q, q_out, (token * q_heads + h) * head_dim, i, half, c1, s1, c2, s2);
      } else {
        rotate_pair(k, k_out, (token * kv_heads + (h - q_heads)) * head_dim, i, half, c1, s1, c2, s2);
      }
    }
  }
}

static bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

template <typename T, typename C>
static int launch_typed(const void* q, const void* k, const void* cos_rows, const void* sin_rows, const void* ids, void* q_out,
                        void* k_out, long long tokens, int q_heads, int kv_heads, int head_dim, int rows, int splits,
                        cudaStream_t stream) {
  const int heads = q_heads + kv_heads;
  const int lanes = head_dim / 16;  // threads per head: 8 pairs each
  const bool vec = head_dim % 16 == 0 && lanes <= 1024 && aligned16(q) && aligned16(k) && aligned16(q_out) &&
                   aligned16(k_out) && aligned16(cos_rows) && aligned16(sin_rows);
  if (vec) {
    int r = rows < heads ? rows : heads;
    if (lanes * r > 1024) r = 1024 / lanes;
    const int groups = (heads + r - 1) / r;
    const int gy = splits < groups ? splits : groups;
    dim3 block(lanes, r), grid((unsigned)tokens, gy);
    rope_vec<T, C><<<grid, block, 0, stream>>>((const T*)q, (const T*)k, (const C*)cos_rows, (const C*)sin_rows,
                                               (const long long*)ids, (T*)q_out, (T*)k_out, q_heads, kv_heads, head_dim);
  } else {
    dim3 block(64, rows < 16 ? rows : 16);
    rope_seed<T, C><<<(unsigned)tokens, block, 0, stream>>>((const T*)q, (const T*)k, (const C*)cos_rows,
                                                            (const C*)sin_rows, (const long long*)ids, (T*)q_out,
                                                            (T*)k_out, q_heads, kv_heads, head_dim);
  }
  return (int)cudaGetLastError();
}

template <typename T>
static int launch_cache(int cache_dtype, const void* q, const void* k, const void* cos_rows, const void* sin_rows,
                        const void* ids, void* q_out, void* k_out, long long tokens, int q_heads, int kv_heads,
                        int head_dim, int rows, int splits, cudaStream_t stream) {
  if (cache_dtype == 0)
    return launch_typed<T, float>(q, k, cos_rows, sin_rows, ids, q_out, k_out, tokens, q_heads, kv_heads, head_dim, rows,
                                  splits, stream);
  if (cache_dtype == 1)
    return launch_typed<T, __nv_bfloat16>(q, k, cos_rows, sin_rows, ids, q_out, k_out, tokens, q_heads, kv_heads, head_dim,
                                          rows, splits, stream);
  if (cache_dtype == 2)
    return launch_typed<T, __half>(q, k, cos_rows, sin_rows, ids, q_out, k_out, tokens, q_heads, kv_heads, head_dim, rows,
                                   splits, stream);
  return -1;
}

extern "C" int launch(const void* q, const void* k, const void* cos_rows, const void* sin_rows, const void* ids, void* q_out,
                      void* k_out, long long tokens, int q_heads, int kv_heads, int head_dim, int qk_dtype,
                      int cache_dtype, int rows, int splits, cudaStream_t stream) {
  if (tokens <= 0) return 0;
  if (tokens > 2147483647LL || head_dim <= 0 || head_dim % 2 || q_heads <= 0 || kv_heads <= 0 || rows < 1 ||
      splits < 1 || splits > 65535)
    return -1;
  if (qk_dtype == 1)
    return launch_cache<__nv_bfloat16>(cache_dtype, q, k, cos_rows, sin_rows, ids, q_out, k_out, tokens, q_heads, kv_heads,
                                       head_dim, rows, splits, stream);
  if (qk_dtype == 2)
    return launch_cache<__half>(cache_dtype, q, k, cos_rows, sin_rows, ids, q_out, k_out, tokens, q_heads, kv_heads, head_dim,
                                rows, splits, stream);
  return -1;
}
