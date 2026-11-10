// RMSNorm of x[rows, cols] over cols, no weight, one block per row, register-resident and 16-byte vectorized. Each
// thread loads VPT uint4 vectors (8 elements each) of its row into registers once, sums their squares in fp32, and the
// block reduces the partial sums (warp shuffles, then one shared slot per warp); inv = rsqrtf(sum / cols + eps) is
// rounded to x's dtype, and each thread writes y = x * inv rounded to x's dtype from its registers (round to nearest
// even both times), as the reference's x * rsqrt(mean + eps).to(x.dtype). A NaN in a row makes the whole row NaN, and
// an overflowing sum makes inv 0, as in the reference. VPT = ceil(cols / 8 / threads) is a template (1..8); when cols
// is not a multiple of 8, a pointer is not 16-byte aligned or VPT would exceed 8, the scalar seed kernel runs instead.
// dtypes: bf16 (1) and fp16 (2); row-major contiguous tensors; the block size is a multiple of 32 from 32 to 1024.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }

constexpr int kWarp = 32;
constexpr int kMaxWarps = 1024 / kWarp;
constexpr int kVecElems = 8;  // elements of a 2-byte dtype in one uint4
constexpr int kMaxVpt = 8;

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

// The block's total of sum, in every thread; inv rounded to T as the reference does.
template <typename T>
__device__ __forceinline__ float block_inv(float sum, int cols, float eps) {
  __shared__ float partial[kMaxWarps];
  __shared__ float row_inv;
  sum = warp_sum(sum);
  const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < (int)(blockDim.x / kWarp) ? partial[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) row_inv = to_float(from_float<T>(rsqrtf(total / (float)cols + eps)));
  }
  __syncthreads();
  return row_inv;
}

template <typename T>
__global__ void rmsnorm_seed(const T* __restrict__ x, T* __restrict__ y, int cols, float eps) {
  const long long base = (long long)blockIdx.x * cols;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = to_float(x[base + i]);
    sum += v * v;                      // exact square: a bf16/fp16 value squared fits fp32's mantissa
  }
  const float inv = block_inv<T>(sum, cols, eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) y[base + i] = from_float<T>(to_float(x[base + i]) * inv);
}

template <typename T, int VPT>
__global__ void rmsnorm_vec(const T* __restrict__ x, T* __restrict__ y, int cols, int nvec, float eps) {
  const long long base = (long long)blockIdx.x * cols;
  const uint4* __restrict__ xr = reinterpret_cast<const uint4*>(x + base);
  uint4* __restrict__ yr = reinterpret_cast<uint4*>(y + base);
  uint4 v[VPT];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i < nvec) {
      v[k] = __ldg(xr + i);
      const T* e = reinterpret_cast<const T*>(&v[k]);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) {
        const float f = to_float(e[j]);
        sum += f * f;                  // exact square: a bf16/fp16 value squared fits fp32's mantissa
      }
    }
  }
  const float inv = block_inv<T>(sum, cols, eps);
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i = threadIdx.x + k * blockDim.x;
    if (i < nvec) {
      const T* e = reinterpret_cast<const T*>(&v[k]);
      uint4 o;
      T* oe = reinterpret_cast<T*>(&o);
#pragma unroll
      for (int j = 0; j < kVecElems; ++j) oe[j] = from_float<T>(to_float(e[j]) * inv);
      yr[i] = o;
    }
  }
}

template <typename T, int VPT>
static void launch_vec(const void* x, void* y, long long rows, int cols, int nvec, float eps, int threads,
                       cudaStream_t stream) {
  rmsnorm_vec<T, VPT><<<(unsigned)rows, threads, 0, stream>>>((const T*)x, (T*)y, cols, nvec, eps);
}

template <typename T>
static int launch_typed(const void* x, void* y, long long rows, int cols, float eps, int threads, cudaStream_t stream) {
  const bool aligned = ((uintptr_t)x % 16 == 0) && ((uintptr_t)y % 16 == 0);
  if (cols % kVecElems == 0 && aligned) {
    const int nvec = cols / kVecElems;
    int t = threads;
    const int needed = (nvec + kWarp - 1) / kWarp * kWarp;  // no more threads than vectors, in whole warps
    if (t > needed) t = needed;
    const int vpt = (nvec + t - 1) / t;
    if (vpt <= kMaxVpt) {
      switch (vpt) {
        case 1: launch_vec<T, 1>(x, y, rows, cols, nvec, eps, t, stream); break;
        case 2: launch_vec<T, 2>(x, y, rows, cols, nvec, eps, t, stream); break;
        case 3: launch_vec<T, 3>(x, y, rows, cols, nvec, eps, t, stream); break;
        case 4: launch_vec<T, 4>(x, y, rows, cols, nvec, eps, t, stream); break;
        case 5: launch_vec<T, 5>(x, y, rows, cols, nvec, eps, t, stream); break;
        case 6: launch_vec<T, 6>(x, y, rows, cols, nvec, eps, t, stream); break;
        case 7: launch_vec<T, 7>(x, y, rows, cols, nvec, eps, t, stream); break;
        default: launch_vec<T, 8>(x, y, rows, cols, nvec, eps, t, stream); break;
      }
      return (int)cudaGetLastError();
    }
  }
  rmsnorm_seed<T><<<(unsigned)rows, threads, 0, stream>>>((const T*)x, (T*)y, cols, eps);
  return (int)cudaGetLastError();
}

extern "C" int launch(const void* x, void* y, long long rows, int cols, int dtype, float eps, int threads,
                      cudaStream_t stream) {
  if (rows <= 0) return 0;
  if (rows > 2147483647LL || cols <= 0 || threads < kWarp || threads > 1024 || threads % kWarp) return -1;
  if (dtype == 1) return launch_typed<__nv_bfloat16>(x, y, rows, cols, eps, threads, stream);
  if (dtype == 2) return launch_typed<__half>(x, y, rows, cols, eps, threads, stream);
  return -1;
}
