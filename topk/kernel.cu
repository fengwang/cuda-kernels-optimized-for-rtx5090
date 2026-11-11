// Row-wise top-k (largest, sorted) as a one-pass threshold filter. Each element becomes a 64-bit composite
// (order-preserving float key << 32 | ~index), so composites are distinct and a larger one comes earlier in the output
// (value descending, index ascending). A row is split into chunks, one block each: the block seeds its threshold from
// the first tile (a lower bound with at least k composites strictly above), streams its chunk with DEPTH tiles of raw
// loads in flight (a register ring: a slot is refilled with the tile DEPTH ahead as soon as it is decoded) and appends
// composites above its running threshold to a shared buffer, a whole tile per warp with one shared atomic. Warps with
// no composite above the threshold skip the append, and a tile ends with a single barrier that also tells the block
// whether the buffer passed PRUNE_AT. Once it holds more than PRUNE_AT, the block prunes it: a seed taken from the
// buffer the same way raises the threshold and the composites at or below it are dropped (a bitonic sort keeps the top
// k should more than COMPACT_AT remain). At the end a last prune and a rank count (or a sort) leave the top k sorted.
// Each chunk's top k goes to a workspace; the last block of a row to finish (per-row counter, reset by that block) takes
// a floor from the heads of the row's sorted chunk lists and runs the same filter over the chunk results from that
// floor, then writes the values (decoded bit-exactly from the composites, so x is not read again) and int64 indices.
// plan() sizes the chunks from the kernel's occupancy so the whole grid is resident in one wave. Composite 0 means
// "no element": every finite input has a key above 0. fp32 and bf16 rows are dtype 0 and 1.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <climits>
#include <cstdint>

typedef unsigned long long u64;

constexpr int COMPACT_AT = 512;  // sort once a prune leaves more; k <= COMPACT_AT
constexpr int PRUNE_AT = 256;    // prune once a tile leaves the buffer holding more; PRUNE_AT <= COMPACT_AT
constexpr int PRUNE_R = 4;       // buffered composites a thread moves per prune round
constexpr int MIN_TILE = 1024;   // a block filters at least this many elements per step
constexpr int MERGE_DEPTH = 4;   // most workspace tiles in flight in the row merge

constexpr int pow2_at_least(int v) {
  int n = 1;
  while (n < v) n <<= 1;
  return n;
}

// Per block size: a tile is U vectors of 4 per thread. Between tiles the buffer holds at most COMPACT_AT, so after one
// it holds at most COMPACT_AT + TILE; BUF rounds that up to a power of two, the largest sort compact() can run.
template <int THREADS>
struct Cfg {
  static constexpr int U = THREADS * 4 >= MIN_TILE ? 1 : MIN_TILE / (THREADS * 4);
  static constexpr int TILE = U * THREADS * 4;
  static constexpr int BUF = pow2_at_least(COMPACT_AT + TILE);
};

template <int BUF>
struct Shared {
  u64 buf[BUF];
  u64 thr;   // admit composites strictly above
  u64 seed;  // prune's minimum over the warps; the merge's head floor
  int count;
  int last;
};

template <int THREADS>
using SharedT = Shared<Cfg<THREADS>::BUF>;

__device__ __forceinline__ u64 pack(float f, int i) {
  unsigned u = __float_as_uint(f);
  unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((u64)key << 32) | (u64)(~(unsigned)i);
}
__device__ __forceinline__ int unpack_index(u64 c) { return (int)(~(unsigned)c); }

// The float bits a composite's key came from: pack() set the top bit of a non-negative float and inverted a negative.
__device__ __forceinline__ unsigned unpack_bits(u64 c) {
  const unsigned key = (unsigned)(c >> 32);
  return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
}

// The element a composite encodes, bit-exact: a bf16 element is the high half of the float it was widened to.
template <typename T>
__device__ __forceinline__ T unpack_value(u64 c);

template <>
__device__ __forceinline__ float unpack_value<float>(u64 c) {
  return __uint_as_float(unpack_bits(c));
}

template <>
__device__ __forceinline__ __nv_bfloat16 unpack_value<__nv_bfloat16>(u64 c) {
  return __ushort_as_bfloat16((unsigned short)(unpack_bits(c) >> 16));
}

// Sources of composites, 4 at a time: fetch(e0, r) issues the loads of elements [e0, e0 + 4) into raw registers,
// decode(r, e0, c) turns them into composites (0 past the end). Keeping the ring raw keeps it small and lets the loads
// stay in flight until their tile is filtered.
template <typename T, bool VEC>
struct XSource;

template <bool VEC>
struct XSource<float, VEC> {
  typedef float4 Raw;
  const float* xr;
  int end;
  __device__ __forceinline__ void fetch(int e0, Raw& r) const {
    if (VEC) {  // chunk ends and cols are multiples of 4: a vector is wholly in or out
      r = e0 < end ? __ldg(reinterpret_cast<const float4*>(xr + e0)) : make_float4(0.f, 0.f, 0.f, 0.f);
    } else {
      r.x = e0 < end ? __ldg(xr + e0) : 0.f;
      r.y = e0 + 1 < end ? __ldg(xr + e0 + 1) : 0.f;
      r.z = e0 + 2 < end ? __ldg(xr + e0 + 2) : 0.f;
      r.w = e0 + 3 < end ? __ldg(xr + e0 + 3) : 0.f;
    }
  }
  __device__ __forceinline__ void decode(const Raw& r, int e0, u64* c) const {
    const float f[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) c[e] = e0 + e < end ? pack(f[e], e0 + e) : 0ull;
  }
};

// Element e sits in the 16 bits at (e & 1) * 16 of word e >> 1.
template <bool VEC>
struct XSource<__nv_bfloat16, VEC> {
  typedef uint2 Raw;
  const __nv_bfloat16* xr;
  int end;
  __device__ __forceinline__ void fetch(int e0, Raw& r) const {
    if (VEC) {
      r = e0 < end ? __ldg(reinterpret_cast<const uint2*>(xr + e0)) : make_uint2(0u, 0u);
    } else {
      const unsigned short* p = reinterpret_cast<const unsigned short*>(xr);
      unsigned a[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) a[e] = e0 + e < end ? (unsigned)__ldg(p + e0 + e) : 0u;
      r.x = a[0] | (a[1] << 16);
      r.y = a[2] | (a[3] << 16);
    }
  }
  __device__ __forceinline__ void decode(const Raw& r, int e0, u64* c) const {
    const float f[4] = {__uint_as_float(r.x << 16), __uint_as_float(r.x & 0xffff0000u), __uint_as_float(r.y << 16),
                        __uint_as_float(r.y & 0xffff0000u)};
#pragma unroll
    for (int e = 0; e < 4; ++e) c[e] = e0 + e < end ? pack(f[e], e0 + e) : 0ull;
  }
};

// The row's chunk results in the workspace (already composites, read through L2).
struct WSource {
  struct Raw {
    u64 v[4];
  };
  const u64* wr;
  int total;
  __device__ __forceinline__ void fetch(int e0, Raw& r) const {
#pragma unroll
    for (int e = 0; e < 4; ++e) r.v[e] = e0 + e < total ? __ldcg(wr + e0 + e) : 0ull;
  }
  __device__ __forceinline__ void decode(const Raw& r, int, u64* c) const {
#pragma unroll
    for (int e = 0; e < 4; ++e) c[e] = r.v[e];
  }
};

// Warp-aggregated append of the N composites c[0, N) of each lane that are above thr; every lane of the warp calls it.
// One shared atomic reserves the warp's total; composite j of a lane lands after the earlier ballots' composites and
// the lower lanes' of ballot j. Returns the buffer fill right after the warp's append (0 if the warp appended nothing):
// the largest such fill over a block's appends is the final count.
template <int N, typename S>
__device__ __forceinline__ int push_n(S& s, const u64* c, u64 thr) {
  unsigned m[N];
  int total = 0;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    m[j] = __ballot_sync(0xffffffffu, c[j] > thr);
    total += __popc(m[j]);
  }
  if (total == 0) return 0;
  const int lane = threadIdx.x & 31;
  int base = 0;
  if (lane == 0) base = atomicAdd(&s.count, total);
  base = __shfl_sync(0xffffffffu, base, 0);
  const unsigned below = (1u << lane) - 1u;
  int off = base;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    if (c[j] > thr) s.buf[off + __popc(m[j] & below)] = c[j];
    off += __popc(m[j]);
  }
  return base + total;
}

// A bound below the lanes' composites m: the per-th largest lane key (ties counted, per <= 32, warp-uniform) shifted
// up. If that key t is above 0, every counted lane holds a real composite whose low word ~index is nonzero, so at least
// per lanes' composites lie strictly above t << 32; if t is 0 the bound is 0 and admits every real composite.
__device__ __forceinline__ u64 warp_kth_key(u64 m, int per) {
  const unsigned key = (unsigned)(m >> 32);
  bool active = true;
  unsigned t = 0u;
  int left = per;
  for (int it = 0; it < 32; ++it) {
    const unsigned mx = __reduce_max_sync(0xffffffffu, active ? key : 0u);
    left -= __popc(__ballot_sync(0xffffffffu, active && key == mx));
    t = mx;
    if (left <= 0) break;
    active = active && key < mx;
  }
  return (u64)t << 32;
}

// Descending bitonic sort of buf[0, n), n a power of two >= 64.
template <int THREADS>
__device__ void sort_desc(u64* buf, int n) {
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < (n >> 1); i += THREADS) {
        int lo = ((i & ~(j - 1)) << 1) | (i & (j - 1)), hi = lo + j;
        u64 a = buf[lo], b = buf[hi];
        bool desc = (lo & k) == 0;
        if (desc ? a < b : a > b) {
          buf[lo] = b;
          buf[hi] = a;
        }
      }
      __syncthreads();
    }
  }
}

// Sorts the cnt buffered composites (zero padded), keeps the top k and raises the threshold to the k-th.
// cnt <= BUF and k <= COMPACT_AT < BUF, so the power-of-two length stays within the buffer.
template <int THREADS>
__device__ void compact(SharedT<THREADS>& s, int cnt, int k) {
  int n = 64;
  while (n < cnt || n < k) n <<= 1;
  for (int i = cnt + threadIdx.x; i < n; i += THREADS) s.buf[i] = 0ull;
  __syncthreads();
  sort_desc<THREADS>(s.buf, n);
  if (threadIdx.x == 0 && cnt >= k) {
    s.count = k;
    s.thr = s.buf[k - 1];
  }
  __syncthreads();
}

// Raises the threshold from the cnt buffered composites and drops those at or below it. Slot w + WARPS * lane of warp w
// takes buf[slot], buf[slot + THREADS], ...: the slots are disjoint and spread every warp over the buffer, so the seed
// argument of scan() applies to the buffer and at least k buffered composites stay strictly above the new threshold;
// the top k seen so far all survive. cnt is uniform and the block is synchronized on entry and on return.
template <int THREADS>
__device__ void prune(SharedT<THREADS>& s, int cnt, int k) {
  constexpr int WARPS = THREADS / 32;
  const int per = (k + WARPS) / WARPS;
  if (per > 32) return;
  const int slot = (int)(threadIdx.x >> 5) + WARPS * (int)(threadIdx.x & 31);
  u64 m = 0ull;
  for (int i = slot; i < cnt; i += THREADS) {
    const u64 v = s.buf[i];
    m = v > m ? v : m;
  }
  m = warp_kth_key(m, per);
  if (threadIdx.x == 0) s.seed = ~0ull;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) atomicMin(&s.seed, m);
  __syncthreads();
  const u64 thr = s.seed > s.thr ? s.seed : s.thr;
  __syncthreads();
  if (threadIdx.x == 0) {
    s.thr = thr;
    s.count = 0;
  }
  // In place: a round's appends land below the composites it read, and the next round reads above them.
  for (int r = 0; r < cnt; r += THREADS * PRUNE_R) {
    u64 v[PRUNE_R];
#pragma unroll
    for (int j = 0; j < PRUNE_R; ++j) {
      const int i = r + j * THREADS + (int)threadIdx.x;
      v[j] = i < cnt ? s.buf[i] : 0ull;
    }
    __syncthreads();
    push_n<PRUNE_R>(s, v, thr);
  }
  __syncthreads();
}

// Leaves the top k of the cnt buffered composites sorted in buf[0, k), zero padded. With G consecutive lanes per
// composite (G the largest power of two <= 32 with cnt * G <= THREADS) each group counts the composites above its own
// (they are distinct, so ranks are unique), combined by xor shuffles; with more than THREADS the buffer is sorted.
template <int THREADS>
__device__ void finish(SharedT<THREADS>& s, int cnt, int k) {
  if (cnt > THREADS) {
    compact<THREADS>(s, cnt, k);
    return;
  }
  int g = 32;
  while (g > 1 && cnt * g > THREADS) g >>= 1;
  const int item = (int)threadIdx.x / g, sub = (int)threadIdx.x & (g - 1);
  u64 v = 0ull;
  int rank = 0;
  if (item < cnt) {
    v = s.buf[item];
#pragma unroll 4
    for (int j = sub; j < cnt; j += g) rank += s.buf[j] > v;
  }
  for (int o = g >> 1; o > 0; o >>= 1) rank += __shfl_xor_sync(0xffffffffu, rank, o);
  __syncthreads();
  if (item < cnt && sub == 0 && rank < k) s.buf[rank] = v;
  for (int j = cnt + (int)threadIdx.x; j < k; j += THREADS) s.buf[j] = 0ull;
  __syncthreads();
}

// Filters composites [begin, end) of src; leaves the top k sorted in buf[0, k).
// With floor 0 the threshold starts from the first tile: each lane's max is a distinct composite and warp_kth_key
// leaves at least per lane maxes of each warp strictly above its bound, so the minimum over the warps has
// WARPS * per >= k + 1 composites strictly above it. Padding (composite 0) can only lower that seed, to at worst 0.
// A nonzero floor (the caller guarantees at least k of src's composites strictly above it) replaces the seed.
// Tiles go through a ring of DEPTH raw register slots: tiles 0 .. DEPTH-1 are fetched up front, and step d of each
// group decodes slot d and at once refetches it with the tile DEPTH ahead, so DEPTH tiles of loads stay in flight
// across the filter and barriers. Slot indices are compile-time, so no register copy waits on a pending load.
// A tile ends with one barrier: the largest fill any warp's append returned is the tile's final count, so
// __syncthreads_or(fill > PRUNE_AT) is the uniform "prune now". Without appends the count stays where the last prune or
// compact left it (<= COMPACT_AT), so between tiles the buffer never holds more than COMPACT_AT.
template <int THREADS, int DEPTH, typename Src>
__device__ __forceinline__ void scan(SharedT<THREADS>& s, int begin, int end, int k, const Src& src, u64 floor) {
  constexpr int U = Cfg<THREADS>::U;
  constexpr int TILE = Cfg<THREADS>::TILE;
  constexpr int WARPS = THREADS / 32;
  static_assert(COMPACT_AT + TILE <= Cfg<THREADS>::BUF && PRUNE_AT <= COMPACT_AT, "buffer too small for a tile");
  const int per = (k + WARPS) / WARPS;
  const bool seed = per <= 32 && floor == 0ull;
  const int own = (int)threadIdx.x * 4;  // the thread's first element within a tile
  if (threadIdx.x == 0) {
    s.count = 0;
    s.thr = seed ? ~0ull : floor;
  }
  typename Src::Raw r[DEPTH][U];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d)
    if (begin + d * TILE < end) {
#pragma unroll
      for (int u = 0; u < U; ++u) src.fetch(begin + d * TILE + u * THREADS * 4 + own, r[d][u]);
    }
  __syncthreads();
  if (seed) {
    u64 m = 0ull;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      u64 c[4];
      src.decode(r[0][u], begin + u * THREADS * 4 + own, c);
#pragma unroll
      for (int e = 0; e < 4; ++e) m = c[e] > m ? c[e] : m;
    }
    m = warp_kth_key(m, per);
    if ((threadIdx.x & 31) == 0) atomicMin(&s.thr, m);
  }
  __syncthreads();
  for (int base = begin; base < end; base += DEPTH * TILE) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
      const int tb = base + d * TILE;
      if (tb >= end) break;
      u64 c[U * 4];
#pragma unroll
      for (int u = 0; u < U; ++u) src.decode(r[d][u], tb + u * THREADS * 4 + own, c + u * 4);
      const int nb = tb + DEPTH * TILE;
      if (nb < end) {
#pragma unroll
        for (int u = 0; u < U; ++u) src.fetch(nb + u * THREADS * 4 + own, r[d][u]);
      }
      const u64 thr = s.thr;
      bool any = false;
#pragma unroll
      for (int e = 0; e < U * 4; ++e) any |= c[e] > thr;
      int fill = 0;
      if (__any_sync(0xffffffffu, any)) fill = push_n<U * 4>(s, c, thr);
      if (__syncthreads_or(fill > PRUNE_AT)) {
        const int cnt = s.count;  // every append is done and nothing writes the count before prune's barriers
        prune<THREADS>(s, cnt, k);
        const int left = s.count;
        __syncthreads();
        if (left > COMPACT_AT) compact<THREADS>(s, left, k);
      }
    }
  }
  int cnt = s.count;
  __syncthreads();
  if (cnt > 2 * k) {
    prune<THREADS>(s, cnt, k);
    cnt = s.count;
    __syncthreads();
  }
  finish<THREADS>(s, cnt, k);
}

// A floor for the row merge from the heads of the row's nchunks sorted chunk lists (k composites each, descending):
// the first r = ceil(k / nchunks) of each list, nh = nchunks * r >= k of them. If their k-th largest L is a real
// composite, the k heads at or above it are k distinct elements of the row, so the row's top k all lie at or above L
// and the merge may admit only composites strictly above L - 1. Real heads are distinct, so exactly one has k - 1
// heads above it; if fewer than k heads are real nothing is found and the floor is 0 (the seed path). The heads are
// staged in buf, which holds nothing live on entry; scan() synchronizes before it writes buf or the seed again.
template <int THREADS>
__device__ u64 head_floor(SharedT<THREADS>& s, const u64* wr, int nchunks, int k) {
  const int r = (k + nchunks - 1) / nchunks;
  const int nh = nchunks * r;
  if (nh > Cfg<THREADS>::BUF) return 0ull;
  if (threadIdx.x == 0) s.seed = 0ull;
  for (int i = threadIdx.x; i < nh; i += THREADS) s.buf[i] = __ldcg(wr + (i / r) * k + i % r);
  __syncthreads();
  for (int i = threadIdx.x; i < nh; i += THREADS) {
    const u64 v = s.buf[i];
    if (v == 0ull) continue;
    int rank = 0;
#pragma unroll 4
    for (int j = 0; j < nh; ++j) rank += s.buf[j] > v;
    if (rank == k - 1) s.seed = v;
  }
  __syncthreads();
  const u64 l = s.seed;
  return l ? l - 1ull : 0ull;
}

template <typename T, int THREADS, bool VEC, int DEPTH>
__global__ void __launch_bounds__(THREADS)
    topk_filter(const T* __restrict__ x, T* __restrict__ values, long long* __restrict__ indices, u64* __restrict__ work,
                unsigned* __restrict__ counters, int cols, int chunk, int k) {
  __shared__ SharedT<THREADS> s;
  const int row = blockIdx.y, nchunks = gridDim.x;
  const T* xr = x + (long long)row * cols;
  const int begin = blockIdx.x * chunk, end = min(begin + chunk, cols);
  const XSource<T, VEC> src{xr, end};
  scan<THREADS, DEPTH>(s, begin, end, k, src, 0ull);
  if (nchunks > 1) {
    u64* wr = work + (long long)row * nchunks * k;
    for (int j = threadIdx.x; j < k; j += THREADS) wr[(long long)blockIdx.x * k + j] = s.buf[j];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s.last = atomicAdd(&counters[row], 1u) == (unsigned)nchunks - 1u;
    __syncthreads();
    if (!s.last) return;
    __threadfence();
    const int total = nchunks * k;
    const u64 floor = head_floor<THREADS>(s, wr, nchunks, k);
    const WSource ws{wr, total};
    scan<THREADS, (DEPTH < MERGE_DEPTH ? DEPTH : MERGE_DEPTH)>(s, 0, total, k, ws, floor);
    if (threadIdx.x == 0) counters[row] = 0u;
  }
  // k <= cols, so all k composites are real elements and decode to exactly the stored values.
  for (int j = threadIdx.x; j < k; j += THREADS) {
    const u64 c = s.buf[j];
    values[(long long)row * k + j] = unpack_value<T>(c);
    indices[(long long)row * k + j] = unpack_index(c);
  }
}

template <typename T, int THREADS, int DEPTH>
static void start(bool vec, dim3 grid, cudaStream_t stream, const void* x, void* values, void* indices, void* work,
                  void* counters, int cols, int chunk, int k) {
  if (vec) {
    topk_filter<T, THREADS, true, DEPTH><<<grid, THREADS, 0, stream>>>(
        (const T*)x, (T*)values, (long long*)indices, (u64*)work, (unsigned*)counters, cols, chunk, k);
  } else {
    topk_filter<T, THREADS, false, DEPTH><<<grid, THREADS, 0, stream>>>(
        (const T*)x, (T*)values, (long long*)indices, (u64*)work, (unsigned*)counters, cols, chunk, k);
  }
}

template <typename T, int THREADS>
static int start_depth(int depth, bool vec, dim3 grid, cudaStream_t stream, const void* x, void* values, void* indices,
                       void* work, void* counters, int cols, int chunk, int k) {
  switch (depth) {
    case 2: start<T, THREADS, 2>(vec, grid, stream, x, values, indices, work, counters, cols, chunk, k); return 0;
    case 4: start<T, THREADS, 4>(vec, grid, stream, x, values, indices, work, counters, cols, chunk, k); return 0;
    case 8: start<T, THREADS, 8>(vec, grid, stream, x, values, indices, work, counters, cols, chunk, k); return 0;
    default: return -1;
  }
}

template <typename T>
static int start_threads(int threads, int depth, bool vec, dim3 grid, cudaStream_t stream, const void* x, void* values,
                         void* indices, void* work, void* counters, int cols, int chunk, int k) {
  switch (threads) {
    case 128: return start_depth<T, 128>(depth, vec, grid, stream, x, values, indices, work, counters, cols, chunk, k);
    case 256: return start_depth<T, 256>(depth, vec, grid, stream, x, values, indices, work, counters, cols, chunk, k);
    case 512: return start_depth<T, 512>(depth, vec, grid, stream, x, values, indices, work, counters, cols, chunk, k);
    default: return -1;
  }
}

// Resident blocks per SM of one instantiation, with the carveout set to the most shared memory.
template <typename T, int THREADS, bool VEC, int DEPTH>
static int resident_one() {
  auto f = topk_filter<T, THREADS, VEC, DEPTH>;
  cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);
  int n = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, THREADS, 0) != cudaSuccess) n = 0;
  return n;
}

template <typename T, int THREADS, int DEPTH>
static int resident_pair() {
  int a = resident_one<T, THREADS, true, DEPTH>(), b = resident_one<T, THREADS, false, DEPTH>();
  return a < b ? a : b;
}

template <typename T, int THREADS>
static int resident_depth(int depth) {
  switch (depth) {
    case 2: return resident_pair<T, THREADS, 2>();
    case 4: return resident_pair<T, THREADS, 4>();
    case 8: return resident_pair<T, THREADS, 8>();
    default: return 0;
  }
}

template <typename T>
static int resident(int threads, int depth) {
  int n = 0;
  switch (threads) {
    case 128: n = resident_depth<T, 128>(depth); break;
    case 256: n = resident_depth<T, 256>(depth); break;
    case 512: n = resident_depth<T, 512>(depth); break;
    default: break;
  }
  cudaGetLastError();  // a failed query must not surface as the next launch's error
  return n > 0 ? n : 1;
}

// Chunk length (columns per block) for a rows x cols launch: as many chunks per row as keep the whole grid resident in
// one wave (SMs times the kernel's resident blocks per SM), chunks of at least min_chunk columns, a multiple of 4 when
// a row is split. The row's chunk count is ceil(cols / chunk).
extern "C" long long plan(long long rows, long long cols, int dtype, int threads, int depth, int min_chunk) {
  static int sms = 0;
  static int occ[2][3][3] = {};
  if (sms <= 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) sms = 1;
    cudaGetLastError();
  }
  if (rows <= 0 || cols <= 0) return cols > 0 ? cols : 1;
  const int d = dtype == 1 ? 1 : 0;
  const int t = threads == 128 ? 0 : threads == 256 ? 1 : threads == 512 ? 2 : -1;
  const int p = depth == 2 ? 0 : depth == 4 ? 1 : depth == 8 ? 2 : -1;
  int per_sm = 1;
  if (t >= 0 && p >= 0) {
    if (occ[d][t][p] <= 0)
      occ[d][t][p] = d ? resident<__nv_bfloat16>(threads, depth) : resident<float>(threads, depth);
    per_sm = occ[d][t][p];
  }
  if (min_chunk < MIN_TILE) min_chunk = MIN_TILE;
  long long slots = (long long)sms * per_sm;
  long long per_row = slots / rows;
  long long most = (cols + min_chunk - 1) / min_chunk;
  if (per_row > most) per_row = most;
  if (per_row <= 1) return cols;
  long long chunk = (cols + per_row - 1) / per_row;
  return (chunk + 3) / 4 * 4;
}

extern "C" int launch(const void* x, void* values, void* indices, void* work, void* counters, long long rows,
                      long long cols, int k, int dtype, int threads, int depth, long long chunk, cudaStream_t stream) {
  if (rows <= 0 || k <= 0) return 0;
  if (k > cols || k > COMPACT_AT || cols > INT_MAX / 2 || rows > 65535 || chunk <= 0) return -1;
  long long nchunks = (cols + chunk - 1) / chunk;
  if (nchunks > 1 && (chunk % 4 || nchunks * k > INT_MAX / 2)) return -1;
  dim3 grid((unsigned)nchunks, (unsigned)rows);
  int err;
  if (dtype == 0) {
    bool vec = cols % 4 == 0 && (uintptr_t)x % 16 == 0;
    err = start_threads<float>(threads, depth, vec, grid, stream, x, values, indices, work, counters, (int)cols,
                               (int)chunk, k);
  } else if (dtype == 1) {
    bool vec = cols % 4 == 0 && (uintptr_t)x % 8 == 0;
    err = start_threads<__nv_bfloat16>(threads, depth, vec, grid, stream, x, values, indices, work, counters,
                                       (int)cols, (int)chunk, k);
  } else {
    return -1;
  }
  if (err) return err;
  return (int)cudaGetLastError();
}
