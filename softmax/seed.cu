// Seed (search root): row-wise softmax over the last dim, one block of 256 threads per row. Three passes over the
// row: a block max, a block sum of exp(x - max) in fp32, then a separate normalisation pass. fp32 and bf16 rows
// (dtype 0 and 1); every value is accumulated in fp32.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

__device__ __forceinline__ float ld(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void st(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void st(__nv_bfloat16* p, long long i, float v) { p[i] = __float2bfloat16_rn(v); }

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

template <typename T>
__global__ void softmax_seed(const T* __restrict__ x, T* __restrict__ y, long long cols) {
  __shared__ float red[32];
  const T* xr = x + blockIdx.x * cols;
  T* yr = y + blockIdx.x * cols;
  float m = -CUDART_INF_F;
  for (long long j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, ld(xr, j));
  m = block_max(m, red);
  float s = 0.0f;
  for (long long j = threadIdx.x; j < cols; j += blockDim.x) s += expf(ld(xr, j) - m);
  s = block_sum(s, red);
  for (long long j = threadIdx.x; j < cols; j += blockDim.x) st(yr, j, expf(ld(xr, j) - m) / s);
}

extern "C" int launch(const void* x, void* y, long long rows, long long cols, int dtype, int threads, int aux,
                      cudaStream_t stream) {
  (void)threads;
  (void)aux;
  if (rows <= 0 || cols <= 0) return 0;
  if (dtype == 0) {
    softmax_seed<float><<<(unsigned)rows, 256, 0, stream>>>((const float*)x, (float*)y, cols);
  } else if (dtype == 1) {
    softmax_seed<__nv_bfloat16><<<(unsigned)rows, 256, 0, stream>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)y, cols);
  } else {
    return -1;
  }
  return (int)cudaGetLastError();
}
