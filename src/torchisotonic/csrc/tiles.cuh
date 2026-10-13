// The CUDA pipeline behind torchisotonic::isotonic_tiles (registered in ops_tiles.cu). It is a
// header so that the variants in experiments/ can reuse the parts they do not change.
//
// The fit is the slope of the lower convex hull of the cumulative sum diagram (hull.h). Every
// element is read once, by one thread; after that only hull vertices travel:
//
// 1. chunk_kernel: one block per chunk of up to 4096 neighbouring elements of a problem
//    (fewer when a larger one's shared memory would not fit), 32 per lane. Each lane builds
//    the hull of its 32 points with the monotone chain, its stack kept as a bit mask over its
//    points (their cumulative sums are in shared memory). Then the lane hulls merge pairwise,
//    in rounds - within warps first, then across them: a merge walks to its bridge and clears
//    the mask bits under it, nothing is copied. A problem that fits in one chunk is finished
//    right there: the block writes its fit. Otherwise it writes the chunk's hull vertices and
//    sums.
// 2. chunk_starts_kernel: prefix sums over the chunks' sums place every chunk in its problem.
// 3. merge_kernel: neighbouring chunk hulls merge pairwise, log2(chunks) rounds, one launch
//    per round: a thread per merge while there are many (it walks to the bridge, and searches
//    for it if the walk is long), a warp per merge in the last rounds. A hull is a list of
//    segments - ranges of vertices of the chunks' hulls - so a merge shortens two segments and
//    moves the list's tail: it costs segments, not vertices.
// 4. ends_kernel / fit_kernel: every chunk learns which of its vertices survive and the slopes
//    of the bridges into and out of it; then every element takes the slope of its edge.
//
// Problems of up to 512 elements are packed several per warp instead (packed_kernel) and
// finished in one kernel.
//
// Coordinates are relative, so that the difference of two nearby points keeps the precision
// of the difference rather than that of the sums: a lane's points are relative to the lane's
// start, lane starts to the chunk's start, chunk starts to the problem's start. Lane starts
// and chunk starts are hi + lo pairs (of the working type and of double); in float32 the sums
// within a lane are pairs too.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace torchisotonic {

namespace {

constexpr unsigned kFull = 0xffffffffu;
constexpr int kLanes = 32;
constexpr int kThreads = 256;
constexpr int64_t kThreadMerges = 256;  // rounds with this many merges: a thread per merge

// Built with -DTI_PHASES (bench/phases.py), chunk_kernel adds up its warps' clock cycles per
// phase: 0 load, 1 lane hulls, 2 lane starts, 3 merging lane hulls, 4 writing.
#ifdef TI_PHASES
__device__ unsigned long long phase_cycles[8];
// Merge statistics: per round, nodes / segment descriptors / vertices alive after it; and
// over all merges, how many there were and how many walks gave up (kSegWalk) for a search.
constexpr int kStatRounds = 32;
__device__ unsigned long long merge_level_stats[kStatRounds][3];
__device__ unsigned long long merge_counts[2];
#define TI_MARK(i)                                                          \
  do {                                                                      \
    if (lane == 0) {                                                        \
      const long long now = clock64();                                      \
      atomicAdd(&phase_cycles[i], static_cast<unsigned long long>(now - mark)); \
      mark = now;                                                           \
    }                                                                       \
  } while (0)
// ...and with phase_stop = i, returns right after phase i: timing the kernel cut at every
// phase gives each phase's real share (the write that depends on `keep` never happens, it
// only stops the compiler from dropping or delaying the work before the cut).
__device__ int phase_stop = 99;
#define TI_STOP(i, keep)                      \
  do {                                        \
    if (phase_stop == (i)) {                  \
      if ((keep) == 0x7654321) {              \
        out[start] = scalar_t(0);             \
      }                                       \
      return;                                 \
    }                                         \
  } while (0)
#else
#define TI_MARK(i) \
  do {             \
  } while (0)
#define TI_STOP(i, keep) \
  do {                   \
  } while (0)
#endif

// Lane l's elements live at l * 33 + k: a warp reading one element per lane hits 32 banks.
__device__ __forceinline__ int pad(int e) {
  return e + (e >> 5);
}

__device__ __forceinline__ int imin(int a, int b) {
  return a < b ? a : b;
}

__device__ __forceinline__ int imax(int a, int b) {
  return a > b ? a : b;
}

// ---------------------------------------------------------------------------------------------
// Chunk level: one block of WARPS warps; 32 elements per lane, 1024 per warp.

// EXACT (the default; see isotonic_tiles_cuda): the float32 sums within a lane are pairs and
// the chunk level's turns are filtered. Without it, float32 runs about twice as fast, with
// errors up to about 1e-3 of the data's scale on hard, weighted data. Float64 ignores it.
template <typename acc_t, bool WEIGHTED, int WARPS, bool EXACT = true>
struct Chunk {
  using acc = acc_t;
  static constexpr bool kW = WEIGHTED;
  static constexpr int kL = WARPS * kLanes;  // lanes
  static constexpr int kN = kL * kLanes;     // elements
  // In float32 a lane's sums are pairs hi + lo: with weighted data the sums of 32 terms can
  // be far larger than a single term, and the float32 rounding of the sum would decide which
  // points are vertices. The lo part is stored as a 16-bit code (pack_lo), half the shared
  // memory of a float. In float64 the hi part alone is accurate enough.
  static constexpr bool kExact = EXACT || sizeof(acc_t) == 8;
  static constexpr bool kPairs = EXACT && sizeof(acc_t) == 4;
  acc_t sy[kN + kL];                         // lane-local cumulative sums of w * y, by element
  acc_t sx[WEIGHTED ? kN + kL : 1];          // ... and of w
  int16_t syl[kPairs ? kN + kL : 1];         // their lo parts, packed (float32 only)
  int16_t sxl[kPairs && WEIGHTED ? kN + kL : 1];
  acc_t by[2][kL + 1];                       // lane starts, hi and lo; [kL] is the chunk end
  acc_t bx[2][WEIGHTED ? kL + 1 : 1];
  uint32_t mask[kL + 1];                     // each lane's points that are still vertices
  int off[kL + 1];                           // rank of each lane's first vertex
  acc_t wy[2][WARPS], wx[2][WARPS];          // per warp: its lanes' total (hi, lo)
  int wi[WARPS];                             // per warp: for block-wide scans
  acc_t wf[WARPS];
  int bridge[2][WARPS];                      // merges across warps: their bridges
};

// The largest block (in warps, at most 4) whose shared memory fits the 48 KB of a launch.
template <typename T, bool W, bool EXACT = true>
constexpr int chunk_warps() {
  return sizeof(Chunk<T, W, 4, EXACT>) <= 48 * 1024   ? 4
         : sizeof(Chunk<T, W, 2, EXACT>) <= 48 * 1024 ? 2
                                                      : 1;
}

// v (0 <= v < 2^23) as a float without I2F, which issues at a quarter of the rate of a float
// add: the bits of 2^23 + v, minus 2^23.
__device__ __forceinline__ float small_float(int v, float) {
  return __int_as_float(v | 0x4B000000) - 8388608.0f;
}
__device__ __forceinline__ double small_float(int v, double) {
  return __longlong_as_double(static_cast<long long>(v) | 0x4330000000000000LL) -
         4503599627370496.0;
}
template <typename acc_t>
__device__ __forceinline__ acc_t to_acc(int v) {
  return small_float(v, acc_t(0));
}

// s + e == a + b exactly (Knuth's two-sum).
template <typename T>
__device__ __forceinline__ void two_sum(T a, T b, T& s, T& e) {
  s = a + b;
  const T bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// (h, l) += (bh, bl), as unevaluated sums hi + lo.
template <typename T>
__device__ __forceinline__ void add2(T& h, T& l, T bh, T bl) {
  T s, e;
  two_sum(h, bh, s, e);
  e += l + bl;
  h = s + e;
  l = e - (h - s);
}

// The lo part of a float pair as a 16-bit code: lo / 2^E in units of 2^-38, E the exponent
// of hi (|lo| <= ulp(hi) / 2 = 2^(E - 24) after a renormalized two-sum, so the code is within
// +-2^14). Sums below 2^-89 in magnitude keep no lo part: it would be below 2^-113.
__device__ __forceinline__ int16_t pack_lo(float hi, float lo) {
  const int e = (__float_as_int(hi) >> 23) & 0xff;  // biased exponent of hi
  if (e <= 38) {
    return 0;
  }
  const float scale = __int_as_float((292 - e) << 23);  // 2^(38 - E)
  return static_cast<int16_t>(__float2int_rn(lo * scale));
}
// Float64 sums keep no lo parts (Chunk::kPairs): these only make the templates compile.
__device__ __forceinline__ int16_t pack_lo(double, double) {
  return 0;
}

__device__ __forceinline__ float unpack_lo(float hi, int16_t code) {
  const int e = (__float_as_int(hi) >> 23) & 0xff;
  return e <= 38 ? 0.0f : static_cast<float>(code) * __int_as_float((e - 38) << 23);
}
__device__ __forceinline__ double unpack_lo(double, int16_t) {
  return 0.0;
}

// An unevaluated sum h + l. The chunk level's turn predicates work on the hi parts with a
// bound on their error, and fall back to the pairs, with error-free transformations
// (two-diff, two-prod), when the bound is not met.
template <typename T>
struct Pair {
  T h, l;
};

// The slope dy / dx of pairs, rounded once: a faithful float32 quotient of the hi parts,
// corrected by its remainder, which fma gives exactly; the result is the rounding of a value
// within about u^2 of the exact slope, so it is monotone in the slope but for near-ties of
// that size. No division: a reciprocal and multiplications.
template <typename acc_t>
__device__ __forceinline__ acc_t rounded_slope(const Pair<acc_t>& dy, const Pair<acc_t>& dx) {
  if constexpr (std::is_same_v<acc_t, float>) {
    const float r = __frcp_rn(dx.h);
    const float q = dy.h * r;
    const float e = (fmaf(-q, dx.h, dy.h) + dy.l) - q * dx.l;
    return fmaf(e, r, q);
  } else {
    return (dy.h + dy.l) / (dx.h + dx.l);
  }
}

// (ah + al) - (bh + bl) as a pair.
template <typename T>
__device__ __forceinline__ Pair<T> diff2(T ah, T al, T bh, T bl) {
  const T s = ah - bh;
  const T bb = s - ah;
  const T e = ((ah - (s - bb)) - (bh + bb)) + (al - bl);
  const T h = s + e;
  return Pair<T>{h, e - (h - s)};
}

// x1 * y2 - y1 * x2 from pairs, evaluated with two-prod (fma) and a two-diff.
template <typename T>
__device__ __forceinline__ T cross2(Pair<T> x1, Pair<T> y1, Pair<T> x2, Pair<T> y2) {
  const T p1 = x1.h * y2.h;
  const T p1l = fma(x1.h, y2.h, -p1) + x1.h * y2.l + x1.l * y2.h;
  const T p2 = y1.h * x2.h;
  const T p2l = fma(y1.h, x2.h, -p2) + y1.h * x2.l + y1.l * x2.h;
  const T s = p1 - p2;
  const T bb = s - p1;
  const T e = ((p1 - (s - bb)) - (p2 + bb)) + (p1l - p2l);
  return s + e;
}

// Error bounds for the fast path of a turn predicate on the hi parts of pair sums. The hi
// part of a renormalized pair is the exact sum rounded once (the lo part holds the rest), so
// a coordinate is within u M of the truth for u the unit roundoff and M the magnitude; a
// difference of two is within 2 u M + u |diff|, a product of two differences within
// 2 u M (|dx| + |dy|) + 3 u |p|, and the cross product within 2 u (M_y (|dx1| + |dx2|) +
// M_x (|dy1| + |dy2|)) + 4 u (|p1| + |p2|) + u |c|. The coefficients below double that.
template <typename T>
__device__ __forceinline__ constexpr T bound_m() {  // on the magnitude terms
  return sizeof(T) == 4 ? T(4 * 5.9604644775390625e-08) : T(4 * 1.1102230246251565e-16);
}
template <typename T>
__device__ __forceinline__ constexpr T bound_p() {  // on the product terms
  return sizeof(T) == 4 ? T(8 * 5.9604644775390625e-08) : T(8 * 1.1102230246251565e-16);
}

template <typename acc_t>
struct Local {  // a point of the chunk: its index, and coordinates from its lane's start
  int p;
  acc_t x, y;    // x: weighted only (otherwise the point index is the coordinate)
  acc_t xl, yl;  // lo parts (float only; otherwise 0)
};

// Point p of the chunk (p = after its first p elements; the chunk's end has p = its length).
template <typename C>
__device__ __forceinline__ Local<typename C::acc> local(const C& c, int p) {
  Local<typename C::acc> q{p, 0, 0, 0, 0};
  if (p & (kLanes - 1)) {
    q.y = c.sy[pad(p - 1)];
    if (C::kPairs) {
      q.yl = unpack_lo(q.y, c.syl[pad(p - 1)]);
    }
    if (C::kW) {
      q.x = c.sx[pad(p - 1)];
      if (C::kPairs) {
        q.xl = unpack_lo(q.x, c.sxl[pad(p - 1)]);
      }
    }
  }
  return q;
}

// b - a in chunk coordinates (b right of a), as a pair: the hi parts of two lane starts
// differ at the ulp of the chunk's sums, so a slope of the output, formed from this once per
// vertex, needs the lo parts.
template <typename C, typename acc_t>
__device__ __forceinline__ void delta(
    const C& c, const Local<acc_t>& a, const Local<acc_t>& b, Pair<acc_t>& dx, Pair<acc_t>& dy) {
  const int la = a.p / kLanes, lb = b.p / kLanes;
  const Pair<acc_t> ly = diff2(b.y, b.yl, a.y, a.yl);
  dy = diff2(c.by[0][lb], c.by[1][lb], c.by[0][la], c.by[1][la]);
  add2(dy.h, dy.l, ly.h, ly.l);
  if (C::kW) {
    const Pair<acc_t> lx = diff2(b.x, b.xl, a.x, a.xl);
    dx = diff2(c.bx[0][lb], c.bx[1][lb], c.bx[0][la], c.bx[1][la]);
    add2(dx.h, dx.l, lx.h, lx.l);
  } else {
    dx = Pair<acc_t>{to_acc<acc_t>(b.p - a.p), acc_t(0)};
  }
}

template <typename T>
__device__ __forceinline__ T warp_inclusive_sum(T v, int lane) {
  for (int d = 1; d < kLanes; d <<= 1) {
    const T o = __shfl_up_sync(kFull, v, d);
    if (lane >= d) {
      v += o;
    }
  }
  return v;
}

// (h, l) += v: a two-sum into the pair when PAIRS, a plain addition otherwise.
template <bool PAIRS, typename T>
__device__ __forceinline__ void scan_step(T& h, T& l, T v) {
  if constexpr (PAIRS) {
    T s, e;
    two_sum(h, v, s, e);
    e += l;
    h = s + e;  // renormalized: |l| stays below half an ulp of h (pack_lo needs that)
    l = e - (h - s);
  } else {
    h += v;
  }
}

// Inclusive prefix sums over the warp of v (+ vl), as hi + lo.
template <typename T>
__device__ __forceinline__ void warp_inclusive_sum2(T v, T vl, int lane, T& hi, T& lo) {
  hi = v;
  lo = vl;
  for (int d = 1; d < kLanes; d <<= 1) {
    const T oh = __shfl_up_sync(kFull, hi, d), ol = __shfl_up_sync(kFull, lo, d);
    if (lane >= d) {
      add2(hi, lo, oh, ol);
    }
  }
}

// Every lane's start (exclusive prefix sum of the lanes' totals over the chunk) into
// starts[0 / 1][lane], the chunk's total into [.][kL]; returns that total in double (in the
// chunk's last thread). warp_totals: per-warp scratch. Ends with a barrier.
template <typename T, int WARPS>
__device__ double lane_starts(
    T total, T total_lo, int lane, int warp, T (*starts)[WARPS * kLanes + 1],
    T (*warp_totals)[WARPS]) {
  T hi, lo;
  warp_inclusive_sum2(total, total_lo, lane, hi, lo);
  T eh = __shfl_up_sync(kFull, hi, 1), el = __shfl_up_sync(kFull, lo, 1);
  if (lane == 0) {
    eh = el = T(0);
  }
  if (lane == kLanes - 1) {
    warp_totals[0][warp] = hi;
    warp_totals[1][warp] = lo;
  }
  __syncthreads();
  T oh = 0, ol = 0;  // the warps before this one
  for (int v = 0; v < warp; ++v) {
    add2(oh, ol, warp_totals[0][v], warp_totals[1][v]);
  }
  add2(eh, el, oh, ol);
  const int t = warp * kLanes + lane;
  starts[0][t] = eh;
  starts[1][t] = el;
  double whole = 0.0;
  if (t == WARPS * kLanes - 1) {
    add2(oh, ol, hi, lo);
    starts[0][t + 1] = oh;
    starts[1][t + 1] = ol;
    whole = static_cast<double>(oh) + static_cast<double>(ol);
  }
  __syncthreads();
  return whole;
}

// The vertex before point p, by the lanes' masks in shared memory; there must be one.
template <typename C>
__device__ __forceinline__ int prev_vertex(const C& c, int p) {
  int t = p / kLanes;
  uint32_t m = c.mask[t] & ((1u << (p % kLanes)) - 1u);
  while (!m) {
    m = c.mask[--t];
  }
  return t * kLanes + kLanes - 1 - __clz(m);
}

// The vertex after point p among lanes before t1, or `end` if there is none.
template <typename C>
__device__ __forceinline__ int next_vertex(const C& c, int p, int t1, int end) {
  int t = p / kLanes;
  uint32_t m = c.mask[t] & ~((2u << (p % kLanes)) - 1u);
  while (!m) {
    if (++t >= t1) {
      return end;
    }
    m = c.mask[t];
  }
  return t * kLanes + __ffs(m) - 1;
}

// A point in chunk coordinates, each as an unevaluated sum hi + lo: loaded once, then every
// turn it takes part in needs no shared memory.
template <typename acc_t>
struct Point2 {
  int p;
  acc_t yh, yl, xh, xl;  // x: weighted only
};

template <typename C>
__device__ __forceinline__ Point2<typename C::acc> chunk_point(const C& c, int p) {
  using acc_t = typename C::acc;
  const int t = p / kLanes;
  const bool inside = p & (kLanes - 1);
  Point2<acc_t> q;
  q.p = p;
  acc_t s, e;
  const acc_t ly = inside ? c.sy[pad(p - 1)] : acc_t(0);
  two_sum(c.by[0][t], ly, s, e);
  q.yh = s;
  q.yl = e + c.by[1][t] + (C::kPairs && inside ? unpack_lo(ly, c.syl[pad(p - 1)]) : acc_t(0));
  if (C::kW) {
    const acc_t lx = inside ? c.sx[pad(p - 1)] : acc_t(0);
    two_sum(c.bx[0][t], lx, s, e);
    q.xh = s;
    q.xl = e + c.bx[1][t] + (C::kPairs && inside ? unpack_lo(lx, c.sxl[pad(p - 1)]) : acc_t(0));
  }
  return q;
}

// turn2 from the pairs, when the fast path cannot decide. Rarely taken, so kept out of line.
template <bool W, typename acc_t>
__device__ __noinline__ acc_t turn2_slow(
    const Point2<acc_t>& a, const Point2<acc_t>& b, const Point2<acc_t>& d, acc_t x1,
    acc_t x2) {
  const Pair<acc_t> py1 = diff2(b.yh, b.yl, a.yh, a.yl), py2 = diff2(d.yh, d.yl, a.yh, a.yl);
  Pair<acc_t> px1, px2;
  if (W) {
    px1 = diff2(b.xh, b.xl, a.xh, a.xl);
    px2 = diff2(d.xh, d.xl, a.xh, a.xl);
  } else {
    px1 = Pair<acc_t>{x1, acc_t(0)};
    px2 = Pair<acc_t>{x2, acc_t(0)};
  }
  return cross2(px1, py1, px2, py2);
}

// > 0 when a, b, d turn left (counter-clockwise).
// FILTER: decide from the hi parts when that is safe, from the pairs otherwise; without it,
// from the hi and lo differences added (as EXACT = false asks).
template <bool W, bool FILTER = true, typename acc_t>
__device__ __forceinline__ acc_t turn2(
    const Point2<acc_t>& a, const Point2<acc_t>& b, const Point2<acc_t>& d) {
  if constexpr (!FILTER) {
    const acc_t y1 = (b.yh - a.yh) + (b.yl - a.yl), y2 = (d.yh - a.yh) + (d.yl - a.yl);
    acc_t x1, x2;
    if (W) {
      x1 = (b.xh - a.xh) + (b.xl - a.xl);
      x2 = (d.xh - a.xh) + (d.xl - a.xl);
    } else {
      x1 = to_acc<acc_t>(b.p - a.p);
      x2 = to_acc<acc_t>(d.p - a.p);
    }
    return x1 * y2 - y1 * x2;
  }
  const acc_t y1 = b.yh - a.yh, y2 = d.yh - a.yh;
  acc_t x1, x2;
  if (W) {
    x1 = b.xh - a.xh;
    x2 = d.xh - a.xh;
  } else {
    x1 = to_acc<acc_t>(b.p - a.p);
    x2 = to_acc<acc_t>(d.p - a.p);
  }
  const acc_t p1 = x1 * y2, p2 = y1 * x2, c = p1 - p2;
  // A coordinate here is a lane start's hi plus a lane sum's hi, rounded: within 3 u M rather
  // than u M, so the magnitude terms take three times lane_chain's coefficient.
  const acc_t my = fmax(fmax(fabs(a.yh), fabs(b.yh)), fabs(d.yh));
  acc_t bound = fma(3 * bound_m<acc_t>() * my, fabs(x1) + fabs(x2),
                    bound_p<acc_t>() * (fabs(p1) + fabs(p2)));
  if (W) {
    const acc_t mx = fmax(fmax(fabs(a.xh), fabs(b.xh)), fabs(d.xh));
    bound = fma(3 * bound_m<acc_t>() * mx, fabs(y1) + fabs(y2), bound);
  }
  if (fabs(c) > bound) {
    return c;
  }
  return turn2_slow<W>(a, b, d, x1, x2);
}

// The bridge between the node of points [start, shared] and the one of [shared, end] (lanes
// before t1), if the shared point needs one: *a = -1 if not. Found by walking from the shared
// point: the left end moves left and the right end right until the line between them supports
// both hulls. Every step but a last one per pass passes over a vertex that the merge removes,
// and a vertex is removed once, so all the walks of a chunk take O(its size) steps whatever
// the data. One thread; returns the number of steps (for bench/phases.py).
template <typename C, typename Load>
__device__ int walk_bridge(
    const C& c, Load load, int start, int shared, int t1, int end, int* a_out, int* b_out) {
  using acc_t = typename C::acc;
  int a = prev_vertex(c, shared), b = next_vertex(c, shared, t1, end);
  Point2<acc_t> pa = load(a), pb = load(b);
  if (turn2<C::kW, C::kExact>(pa, load(shared), pb) > 0) {
    *a_out = -1;  // convex at the shared point
    return 0;
  }
  int steps = 0;
  bool moved = true;
  while (moved) {
    moved = false;
    while (b != end) {
      const int nb = next_vertex(c, b, t1, end);
      const Point2<acc_t> q = load(nb);
      ++steps;
      if (turn2<C::kW, C::kExact>(pa, pb, q) > 0) {
        break;
      }
      b = nb;
      pb = q;
      moved = true;
    }
    while (a != start) {
      const int na = prev_vertex(c, a);
      const Point2<acc_t> q = load(na);
      ++steps;
      if (turn2<C::kW, C::kExact>(q, pa, pb) > 0) {
        break;
      }
      a = na;
      pa = q;
      moved = true;
    }
  }
  *a_out = a;
  *b_out = b;
  return steps;
}

// The bits of lane t's points strictly between points a and b.
__device__ __forceinline__ uint32_t between(int a, int b, int t) {
  const int lo = imax(a + 1 - t * kLanes, 0);
  const int hi = imin(b - 1 - t * kLanes, kLanes - 1);
  if (lo > hi) {
    return 0;
  }
  const uint32_t upto = hi == kLanes - 1 ? kFull : (1u << (hi + 1)) - 1u;
  return upto & ~((1u << lo) - 1u);
}

// lane_chain's turn from the pairs, when the fast path cannot decide: the lo parts are read
// by the points' indices (point 0 is 0). Rarely taken, so kept out of line.
template <bool W, typename acc_t>
__device__ __noinline__ acc_t lane_turn_slow(
    const int16_t* syl, const int16_t* sxl, int ka, int kb, int k, acc_t ax, acc_t ay, acc_t bx,
    acc_t by, acc_t qx, acc_t qy, acc_t dx1, acc_t dx2) {
  const acc_t al = ka ? unpack_lo(ay, syl[ka - 1]) : acc_t(0);
  const acc_t bl = kb ? unpack_lo(by, syl[kb - 1]) : acc_t(0);
  const acc_t ql = unpack_lo(qy, syl[k - 1]);
  const Pair<acc_t> py1 = diff2(by, bl, ay, al), py2 = diff2(qy, ql, ay, al);
  Pair<acc_t> px1{dx1, acc_t(0)}, px2{dx2, acc_t(0)};
  if (W) {
    const acc_t axl = ka ? unpack_lo(ax, sxl[ka - 1]) : acc_t(0);
    const acc_t bxl = kb ? unpack_lo(bx, sxl[kb - 1]) : acc_t(0);
    px1 = diff2(bx, bxl, ax, axl);
    px2 = diff2(qx, unpack_lo(qx, sxl[k - 1]), ax, axl);
  }
  return cross2(px1, py1, px2, py2);
}

// The monotone chain over one lane's points 0..count, point k's cumulative sums being sy[k - 1]
// (and sx[k - 1]; point 0 is the lane's start): the hull's points as the bits of the result
// (point count, the lane's end, belongs to the next lane and has none). The stack is the set
// bits plus its top b; a is the entry under b. Every iteration either pops b or pushes the
// next point q, chosen by selects, and both loads it might need are made: no branch in the
// loop but the rare slow path, so lanes stay together and the body stays short.
//
// PAIRS: the sums are pairs hi + lo (lo parts packed in syl, sxl). The turn is computed from
// the hi parts and accepted when it exceeds its error bound: slack_y (slack_x) is bound_m
// times the lane's largest sum of w * y (of w).
template <bool W, bool PAIRS, typename acc_t>
__device__ __forceinline__ uint32_t lane_chain(
    const acc_t* sy, const int16_t* syl, const acc_t* sx, const int16_t* sxl, int count,
    acc_t slack_y, acc_t slack_x) {
  uint32_t mask = count > 0 ? 1u : 0u;
  int ka = 0, kb = 0, k = 1;  // the stack has two entries or more exactly when b is not point 0
  acc_t ax = 0, ay = 0, bx = 0, by = 0;
  acc_t qx = W ? sx[0] : acc_t(1), qy = sy[0];
  while (k <= count) {
    const acc_t dx1 = bx - ax, dy1 = by - ay, dx2 = qx - ax, dy2 = qy - ay;
    const acc_t p1 = dx1 * dy2, p2 = dy1 * dx2;
    acc_t c = p1 - p2;
    if constexpr (PAIRS) {
      acc_t bound =
          fma(slack_y, fabs(dx1) + fabs(dx2), bound_p<acc_t>() * (fabs(p1) + fabs(p2)));
      if (W) {
        bound = fma(slack_x, fabs(dy1) + fabs(dy2), bound);
      }
      if (!(fabs(c) > bound)) {
        c = lane_turn_slow<W>(syl, sxl, ka, kb, k, ax, ay, bx, by, qx, qy, dx1, dx2);
      }
    }
    const bool pop = kb != 0 && c <= 0;
    const uint32_t popped = mask & ~(1u << kb);
    const uint32_t pushed = mask | ((k < count ? 1u : 0u) << (k & (kLanes - 1)));
    const int under = kLanes - 1 - __clz(popped & ((1u << ka) - 1u));  // -1: a is point 0
    const int ui = under > 0 ? under - 1 : 0;
    const acc_t uy = sy[ui], ux = W ? sx[ui] : to_acc<acc_t>(ui + 1);
    const int ni = k < kLanes ? k : kLanes - 1;
    const acc_t ny = sy[ni], nx = W ? sx[ni] : qx + acc_t(1);
    const bool deep = under > 0;
    mask = pop ? popped : pushed;
    const int nka = pop ? (deep ? under : 0) : kb;
    const acc_t nax = pop ? (deep ? ux : acc_t(0)) : bx, nay = pop ? (deep ? uy : acc_t(0)) : by;
    kb = pop ? ka : k;
    bx = pop ? ax : qx;
    by = pop ? ay : qy;
    ka = nka;
    ax = nax;
    ay = nay;
    qx = pop ? qx : nx;
    qy = pop ? qy : ny;
    k += pop ? 0 : 1;
  }
  return mask;
}

// A lane's hull: its elements' values (at base.., padded) become its cumulative sums, in
// place, and lane_chain finds the hull of points 0..count. Returns the hull's mask; (ty, tyl)
// and (tx, txl) are the lane's totals of w * y and of w (for unweighted data, tx = count).
template <typename C>
__device__ __forceinline__ uint32_t lane_hull(
    C& c, int base, int count, typename C::acc& ty, typename C::acc& tyl, typename C::acc& tx,
    typename C::acc& txl) {
  using acc_t = typename C::acc;
  constexpr bool W = C::kW;
  acc_t my = 0, mx = 0;  // the largest sums, for lane_chain's error bound
  acc_t py[kLanes], px[W ? kLanes : 1];
#pragma unroll
  for (int k = 0; k < kLanes; ++k) {
    py[k] = c.sy[base + k];
    if (W) {
      px[k] = c.sx[base + k];
    }
  }
#pragma unroll
  for (int k = 0; k < kLanes; ++k) {  // py[k]: point k + 1
    scan_step<C::kPairs>(ty, tyl, k < count ? py[k] : acc_t(0));
    c.sy[base + k] = ty;
    my = fmax(my, fabs(ty));
    if (C::kPairs) {
      c.syl[base + k] = pack_lo(ty, tyl);
    }
    if (W) {
      scan_step<C::kPairs>(tx, txl, k < count ? px[k] : acc_t(0));
      c.sx[base + k] = tx;
      mx = fmax(mx, fabs(tx));
      if (C::kPairs) {
        c.sxl[base + k] = pack_lo(tx, txl);
      }
    }
  }
  if (!W) {
    tx = acc_t(count);
  }
  return lane_chain<W, C::kPairs>(c.sy + base, c.syl + (C::kPairs ? base : 0),
                                  c.sx + (W ? base : 0), c.sxl + (W && C::kPairs ? base : 0),
                                  count, bound_m<acc_t>() * my, bound_m<acc_t>() * mx);
}

struct Seg {  // vertices lo..hi of one chunk's hull; voff: where that hull starts
  int64_t voff;
  int32_t chunk, lo, hi, unused;
};

struct Hulls {  // the chunks' hulls (vertices in their chunk's coordinates) and sums
  int32_t* p;   // point index in the problem
  double* y;
  double* x;    // weighted only; otherwise x is the point index
  Seg* seg;     // each node's segment list, from its first chunk on
  int32_t* nseg;
  double* total_y;
  double* total_x;
  int32_t* row_nan;
  // Chunk starts, [batch, per_row + 1], as unevaluated sums hi + lo (the last column: the
  // row's total). A slope between vertices of different chunks is formed from differences of
  // these, never from absolute coordinates: at 1e8 elements the absolute sums reach 1e10 and
  // a difference of two of them loses six digits.
  const double* cy_lo;
  const double* cx_lo;
};

template <typename scalar_t, bool W, int WARPS, bool EXACT = true>
__global__ void __launch_bounds__(WARPS * kLanes) chunk_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ w,
    int64_t n,
    int64_t per_row,
    scalar_t sign,
    scalar_t* __restrict__ out,
    Hulls h) {
  using acc_t = scalar_t;
  using C = Chunk<acc_t, W, WARPS, EXACT>;
  constexpr int kL = C::kL, kN = C::kN;
  __shared__ C c;
  const int t = threadIdx.x, lane = t % kLanes, warp = t / kLanes;  // t: this thread's lane
#ifdef TI_PHASES
  long long mark = clock64();
#endif
  const int64_t g = blockIdx.x, r = g / per_row, j = g % per_row, start = r * n + j * kN;
  const int len = static_cast<int>(n - j * kN < kN ? n - j * kN : kN);

  // Coalesced, and all 32 loads of a thread issued before any is used.
  acc_t ry[kLanes], rw[W ? kLanes : 1];
#pragma unroll
  for (int i = 0; i < kLanes; ++i) {
    const int e = i * kL + t;
    ry[i] = e < len ? static_cast<acc_t>(y[start + e]) : acc_t(0);
    if (W) {
      rw[i] = e < len ? static_cast<acc_t>(w[start + e]) : acc_t(0);
    }
  }
  bool nan = false;
#pragma unroll
  for (int i = 0; i < kLanes; ++i) {
    const int e = i * kL + t;
    const acc_t v = sign * ry[i];
    nan |= v != v;
    if (W) {
      c.sy[pad(e)] = rw[i] * v;
      c.sx[pad(e)] = rw[i];
    } else {
      c.sy[pad(e)] = v;
    }
  }
  nan = __syncthreads_or(nan);
  TI_MARK(0);
  TI_STOP(0, __float_as_int(static_cast<float>(c.sy[pad(t)])) + int(nan));

  // Each lane: the hull of its points 0..count (point count is the next lane's 0).
  const int count = imax(0, imin(kLanes, len - t * kLanes));
  acc_t tx = 0, ty = 0, txl = 0, tyl = 0;
  uint32_t mask = lane_hull(c, t * (kLanes + 1), count, ty, tyl, tx, txl);

  TI_MARK(1);
  TI_STOP(1, static_cast<int>(mask) ^ __float_as_int(static_cast<float>(ty)));
  const double iy = lane_starts<acc_t, WARPS>(ty, tyl, lane, warp, c.by, c.wy);
  double ix = 0.0;
  if constexpr (W) {
    ix = lane_starts<acc_t, WARPS>(tx, txl, lane, warp, c.bx, c.wx);
  }

  TI_MARK(2);
  TI_STOP(2, static_cast<int>(mask) ^ __float_as_int(static_cast<float>(c.by[0][t])));
  // Merge the lane hulls: round `level` merges pairs of 2^level lanes. Merge q joins lanes
  // [t0, t0 + half) and [t0 + half, t0 + 2 half); its first lane walks to the bridge, and its
  // lanes clear the mask bits under it. Up to round 4 a merge is within a warp.
  const int active = (len + kLanes - 1) / kLanes;
  for (int level = 0; (1 << level) < active; ++level) {
    const bool in_warp = level < 5;
    const int half = 1 << level, G = 2 * half;
    const int t0 = t / G * G, tm = t0 + half;
    const bool valid = tm < active;
    c.mask[t] = mask;
    if (in_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }
    int a = -1, b = -1, steps = 0;
    if (valid && t == t0) {
      const int t1 = imin(t0 + G, active);
      steps = walk_bridge(c, [&](int p) { return chunk_point(c, p); }, t0 * kLanes, tm * kLanes,
                          t1, t1 < active ? t1 * kLanes : len, &a, &b);
    }
    if (in_warp) {
      a = __shfl_sync(kFull, a, t0 % kLanes);
      b = __shfl_sync(kFull, b, t0 % kLanes);
    } else {
      if (valid && t == t0) {
        c.bridge[0][t0 / G] = a;
        c.bridge[1][t0 / G] = b;
      }
      __syncthreads();
      if (valid) {
        a = c.bridge[0][t0 / G];
        b = c.bridge[1][t0 / G];
      }
    }
#ifdef TI_PHASES
    {  // walk steps: in all, and the longest of the round (what the warp waits for)
      int most = steps, all = steps;
      for (int d = 16; d > 0; d >>= 1) {
        most = imax(most, __shfl_xor_sync(kFull, most, d));
        all += __shfl_xor_sync(kFull, all, d);
      }
      if (lane == 0) {
        atomicAdd(&phase_cycles[5], static_cast<unsigned long long>(all));
        atomicAdd(&phase_cycles[6], static_cast<unsigned long long>(most));
      }
    }
#else
    (void)steps;
#endif
    if (valid && a >= 0) {
      mask &= ~between(a, b, t);
    }
    if (in_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  }

  // Rank the vertices: lane by lane, then the chunk's end point (point len) at rank top.
  c.mask[t] = mask;
  const int count_v = __popc(mask);
  const int incl = warp_inclusive_sum(count_v, lane);
  if (lane == kLanes - 1) {
    c.wi[warp] = incl;
  }
  __syncthreads();
  int first = incl - count_v;
  for (int v = 0; v < warp; ++v) {
    first += c.wi[v];
  }
  c.off[t] = first;
  if (t == kL - 1) {
    c.off[kL] = first + count_v;
  }
  __syncthreads();
  TI_MARK(3);
  TI_STOP(3, static_cast<int>(mask) ^ c.off[kL]);
  const int top = c.off[kL];

  if (per_row == 1) {  // the chunk is the whole problem: write its fit
    // Each lane its own 32 elements: backwards, the slope of the edge out of each of its
    // vertices (to the next vertex: its own, or the first one of a later lane, or the end);
    // forwards, every element takes the slope of the last vertex at or before it (for the
    // elements before its first vertex, the last slope of an earlier lane). The slopes are
    // staged where the sums were, then written coalesced.
    const int p0 = t * kLanes;
    int after = mask ? p0 + __ffs(mask) - 1 : len;  // suffix minimum of first vertices
    for (int d = 1; d < kLanes; d <<= 1) {
      const int o = __shfl_down_sync(kFull, after, d);
      if (lane + d < kLanes && o < after) {
        after = o;
      }
    }
    if (lane == 0) {
      c.wi[warp] = after;  // the warp's first vertex
    }
    int next_first = __shfl_down_sync(kFull, after, 1);
    __syncthreads();
    if (lane == kLanes - 1) {
      next_first = len;
    }
    for (int v = warp + 1; v < WARPS; ++v) {
      next_first = imin(next_first, c.wi[v]);
    }
    next_first = imin(next_first, len);
    acc_t slope[kLanes];
    acc_t last = 0;  // the slope out of this lane's last vertex
    {
      bool found = false;
      int nv = next_first;  // the vertex after the one being looked at
      Local<acc_t> pn = local(c, nv);
#pragma unroll
      for (int k = kLanes - 1; k >= 0; --k) {
        if ((mask >> k) & 1u) {
          const Local<acc_t> pa = local(c, p0 + k);
          Pair<acc_t> dx, dy;
          if (nv / kLanes == t) {  // both ends in this lane: no lane starts involved
            dy = diff2(pn.y, pn.yl, pa.y, pa.yl);
            dx = W ? diff2(pn.x, pn.xl, pa.x, pa.xl)
                   : Pair<acc_t>{to_acc<acc_t>(nv - p0 - k), acc_t(0)};
          } else {
            delta(c, pa, pn, dx, dy);
          }
          slope[k] = sign * rounded_slope(dy, dx);
          if (!found) {
            last = slope[k];
            found = true;
          }
          nv = p0 + k;
          pn = pa;
        }
      }
    }
    // The slope coming into this lane: the last slope of the nearest earlier lane with
    // vertices (lane 0 always has one, its point 0).
    bool has = mask != 0;
    acc_t into = last;
    for (int d = 1; d < kLanes; d <<= 1) {
      const acc_t o = __shfl_up_sync(kFull, into, d);
      const bool oh = __shfl_up_sync(kFull, has, d);
      if (lane >= d && !has) {
        into = o;
        has = oh;
      }
    }
    acc_t prev = __shfl_up_sync(kFull, into, 1);
    bool prev_has = __shfl_up_sync(kFull, has, 1) && lane > 0;
    __syncthreads();  // everyone has read c.wi
    if (lane == kLanes - 1) {
      c.wi[warp] = has;
      c.wf[warp] = into;
    }
    __syncthreads();
    for (int v = warp - 1; v >= 0 && !prev_has; --v) {
      if (c.wi[v]) {
        prev = c.wf[v];
        prev_has = true;
      }
    }
    {
      acc_t cur = prev;
#pragma unroll
      for (int k = 0; k < kLanes; ++k) {
        if ((mask >> k) & 1u) {
          cur = slope[k];
        }
        c.sy[p0 + t + k] = cur;  // = pad(p0 + k)
      }
    }
    __syncthreads();
    for (int e = t; e < len; e += kL) {
      out[start + e] = nan ? std::numeric_limits<scalar_t>::quiet_NaN() : c.sy[pad(e)];
    }
    TI_MARK(4);
    return;
  }

  // Otherwise write the chunk's hull: every lane its vertices, the last lane the end point.
  const int64_t voff = g * (kN + 1);
  auto put = [&](int i, int p) {
    const Local<acc_t> q = local(c, p);
    const int u = p / kLanes;
    h.p[voff + i] = static_cast<int32_t>(j * kN + p);
    h.y[voff + i] = static_cast<double>(c.by[0][u]) + static_cast<double>(c.by[1][u]) +
                    static_cast<double>(q.y) + static_cast<double>(q.yl);
    if (W) {
      h.x[voff + i] = static_cast<double>(c.bx[0][u]) + static_cast<double>(c.bx[1][u]) +
                      static_cast<double>(q.x) + static_cast<double>(q.xl);
    }
  };
  {
    int i = first;
    for (uint32_t m = mask; m; m &= m - 1) {
      put(i++, t * kLanes + __ffs(m) - 1);
    }
  }
  if (t == kL - 1) {
    put(top, len);
    h.seg[g] = Seg{voff, static_cast<int32_t>(j), 0, top, 0};
    h.nseg[g] = 1;
    h.total_y[g] = iy;
    if (W) {
      h.total_x[g] = ix;
    }
    if (nan) {
      h.row_nan[r] = 1;
    }
  }
  TI_MARK(4);
}

// Problems of up to 512 elements, several per warp: problem q of a warp takes its lanes
// [q L, q L + L), L the smallest power of two with 32 L >= n, so that no lane idles for want
// of a problem; everything else is chunk_kernel's, done per problem. A point other than its
// problem's start is taken as the one after the element before it: so the end of a problem
// is the end of its last element, though it may share its index with the next one's start.
template <typename scalar_t, bool W, int WARPS, bool EXACT = true>
__global__ void __launch_bounds__(WARPS * kLanes) packed_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ w,
    int64_t batch,
    int n,
    int L,
    unsigned long long inv_n,  // ceil(2^32 / n): e / n = (e * inv_n) >> 32 for these e
    scalar_t sign,
    scalar_t* __restrict__ out) {
  using acc_t = scalar_t;
  using C = Chunk<acc_t, W, 1, EXACT>;
  __shared__ C shared[WARPS];
  __shared__ int bad[WARPS][kLanes];  // per problem of the warp: whether it holds a NaN
  const int lane = threadIdx.x % kLanes, warp = threadIdx.x / kLanes;
  C& c = shared[warp];
  const int per_warp = kLanes / L, span = L * kLanes;  // problems per warp; points per problem
  const int64_t r0 = (static_cast<int64_t>(blockIdx.x) * WARPS + warp) * per_warp;
  if (r0 >= batch) {
    return;  // the whole warp
  }
  const int rows = static_cast<int>(batch - r0 < per_warp ? batch - r0 : per_warp);
  const int m = rows * n;
  const int64_t start = r0 * n;
  auto at = [&](int e) {  // where element e of the warp's problems lives in shared memory
    const int q = static_cast<int>((static_cast<unsigned long long>(e) * inv_n) >> 32);
    return q * span + (e - q * n);
  };
  bad[warp][lane] = 0;
  acc_t ry[kLanes], rw[W ? kLanes : 1];
#pragma unroll
  for (int i = 0; i < kLanes; ++i) {
    const int e = i * kLanes + lane;
    ry[i] = e < m ? static_cast<acc_t>(y[start + e]) : acc_t(0);
    if (W) {
      rw[i] = e < m ? static_cast<acc_t>(w[start + e]) : acc_t(0);
    }
  }
  __syncwarp();
#pragma unroll
  for (int i = 0; i < kLanes; ++i) {
    const int e = i * kLanes + lane;
    if (e < m) {
      const int pos = at(e);
      const acc_t v = sign * ry[i];
      if (v != v) {
        bad[warp][pos / span] = 1;
      }
      if (W) {
        c.sy[pad(pos)] = rw[i] * v;
        c.sx[pad(pos)] = rw[i];
      } else {
        c.sy[pad(pos)] = v;
      }
    }
  }
  __syncwarp();

  // The lane's hull, as in chunk_kernel.
  const int q = lane / L, u = lane % L;  // this lane's problem, and its place in it
  const int count = q < rows ? imax(0, imin(kLanes, n - u * kLanes)) : 0;
  acc_t tx = 0, ty = 0, txl = 0, tyl = 0;
  uint32_t mask = lane_hull(c, lane * (kLanes + 1), count, ty, tyl, tx, txl);

  // Lane starts, from each problem's start: prefix sums restarted at every problem.
  auto starts = [&](acc_t total, acc_t total_lo, acc_t (*b)[kLanes + 1]) {
    acc_t hi = total, lo = total_lo;
    for (int d = 1; d < L; d <<= 1) {
      const acc_t oh = __shfl_up_sync(kFull, hi, d), ol = __shfl_up_sync(kFull, lo, d);
      if (u >= d) {
        add2(hi, lo, oh, ol);
      }
    }
    const acc_t eh = __shfl_up_sync(kFull, hi, 1), el = __shfl_up_sync(kFull, lo, 1);
    b[0][lane] = u ? eh : acc_t(0);
    b[1][lane] = u ? el : acc_t(0);
  };
  starts(ty, tyl, c.by);
  if constexpr (W) {
    starts(tx, txl, c.bx);
  }
  __syncwarp();
  const int first_point = q * span, end_point = q * span + n;
  auto load = [&](int p) {  // point p of this lane's problem, in the problem's coordinates
    Point2<acc_t> pt;
    pt.p = p;
    pt.yh = pt.yl = pt.xh = pt.xl = acc_t(0);
    if (p != first_point) {
      const int e = p - 1, t = e / kLanes;
      acc_t s, err;
      two_sum(c.by[0][t], c.sy[pad(e)], s, err);
      pt.yh = s;
      pt.yl = err + c.by[1][t] + (C::kPairs ? unpack_lo(c.sy[pad(e)], c.syl[pad(e)]) : acc_t(0));
      if (W) {
        two_sum(c.bx[0][t], c.sx[pad(e)], s, err);
        pt.xh = s;
        pt.xl = err + c.bx[1][t] + (C::kPairs ? unpack_lo(c.sx[pad(e)], c.sxl[pad(e)]) : acc_t(0));
      }
    }
    return pt;
  };

  // Merge the lane hulls of each problem.
  const int active = (n + kLanes - 1) / kLanes;  // lanes of a problem holding elements
  for (int level = 0; (1 << level) < active; ++level) {
    const int half = 1 << level, G = 2 * half;
    const int t0 = lane / G * G, u0 = t0 % L;
    const bool valid = q < rows && u0 + half < active;
    c.mask[lane] = mask;
    __syncwarp();
    int a = -1, b = -1;
    if (valid && lane == t0) {
      const int u1 = imin(u0 + G, active);
      walk_bridge(c, load, t0 * kLanes, (t0 + half) * kLanes, t0 - u0 + u1,
                  u1 < active ? (t0 - u0 + u1) * kLanes : end_point, &a, &b);
    }
    a = __shfl_sync(kFull, a, t0);
    b = __shfl_sync(kFull, b, t0);
    if (valid && a >= 0) {
      mask &= ~between(a, b, lane);
    }
    __syncwarp();
  }
  c.mask[lane] = mask;
  __syncwarp();

  // The fit, as chunk_kernel's, per problem.
  const int p0 = lane * kLanes;
  int after = mask ? p0 + __ffs(mask) - 1 : end_point;  // suffix minimum within the problem
  for (int d = 1; d < L; d <<= 1) {
    const int o = __shfl_down_sync(kFull, after, d);
    if (u + d < L && o < after) {
      after = o;
    }
  }
  int next_first = __shfl_down_sync(kFull, after, 1);
  if (u == L - 1 || next_first > end_point) {
    next_first = end_point;
  }
  acc_t slope[kLanes];
  acc_t last = 0;
  {
    bool found = false;
    int nv = next_first;
    Point2<acc_t> pn = load(nv);
#pragma unroll
    for (int k = kLanes - 1; k >= 0; --k) {
      if ((mask >> k) & 1u) {
        const Point2<acc_t> pa = load(p0 + k);
        const Pair<acc_t> dy = diff2(pn.yh, pn.yl, pa.yh, pa.yl);
        const Pair<acc_t> dx = W ? diff2(pn.xh, pn.xl, pa.xh, pa.xl)
                                 : Pair<acc_t>{to_acc<acc_t>(nv - p0 - k), acc_t(0)};
        slope[k] = sign * rounded_slope(dy, dx);
        if (!found) {
          last = slope[k];
          found = true;
        }
        nv = p0 + k;
        pn = pa;
      }
    }
  }
  bool has = mask != 0;
  acc_t into = last;
  for (int d = 1; d < L; d <<= 1) {
    const acc_t o = __shfl_up_sync(kFull, into, d);
    const bool oh = __shfl_up_sync(kFull, has, d);
    if (u >= d && !has) {
      into = o;
      has = oh;
    }
  }
  acc_t cur = __shfl_up_sync(kFull, into, 1);  // lane 0 of a problem always has a vertex
  __syncwarp();
#pragma unroll
  for (int k = 0; k < kLanes; ++k) {
    if ((mask >> k) & 1u) {
      cur = slope[k];
    }
    c.sy[p0 + lane + k] = cur;  // = pad(p0 + k)
  }
  __syncwarp();
  for (int e = lane; e < m; e += kLanes) {
    const int pos = at(e);
    out[start + e] =
        bad[warp][pos / span] ? std::numeric_limits<scalar_t>::quiet_NaN() : c.sy[pad(pos)];
  }
}

// ---------------------------------------------------------------------------------------------
// Problem level: hulls as segment lists.

struct Pt {
  double x, y;
};

// The plain cross product (> 0: a left turn), without Row::turn's filter.
__device__ __forceinline__ double turn(const Pt& a, const Pt& b, const Pt& d) {
  return (b.x - a.x) * (d.y - a.y) - (b.y - a.y) * (d.x - a.x);
}

// The biased exponent of |v| (0 for zero and subnormals), read from the high word.
__device__ __forceinline__ int bexp(double v) {
  return static_cast<int>((static_cast<unsigned>(__double2hiint(v)) << 1) >> 21);
}

template <bool W>
struct Row {  // one problem's vertices, in its coordinates
  Hulls h;
  const double* cy;  // chunk starts, hi parts
  const double* cx;
  const double* cyl = nullptr;  // their lo parts (needed by slope and the local turn only)
  const double* cxl = nullptr;

  // Absolute coordinates, one addition each: cheap, and exact enough for most predicates.
  __device__ __forceinline__ Pt at(const Seg& s, int pos) const {
    const int64_t i = s.voff + pos;
    return Pt{W ? cx[s.chunk] + h.x[i] : static_cast<double>(h.p[i]), cy[s.chunk] + h.y[i]};
  }

  // The turn of a, b, d (> 0: a left turn), with a floating-point filter. The absolute
  // coordinates are sums of up to 1e10 rounded to double, so the cross product computed from
  // them carries an error of a few ulp of that magnitude times the edge lengths; a result
  // larger than that bound has the right sign. Otherwise - nearly collinear points, or far
  // from the origin - the turn is recomputed from coordinates relative to a, formed from
  // differences of the chunk starts' hi and lo parts and of the chunks' own coordinates. The
  // points are passed again as (list, segment, position) for that. Without lo parts the
  // filtered result is returned as it is.
  //
  // The bound costs no floating-point operation (the merges are bound by double
  // throughput): it is compared by exponents. With u = 2^-53 the error is below
  // 8u (M (|bx| + |by| + |dx| + |dy|) + |p1| + |p2|) <= 16u M (|bx| + |by| + |dx| + |dy|) for
  // M the largest coordinate, as |p1| + |p2| <= M (|bx| + |by|). Bounding each factor by a
  // power of two from its exponent - M <= 2^(E_m + 1) with E_m <= max(E_a, E_e) + 1 over a's
  // coordinates and the four differences, the sum of the differences by 2^(E_e + 3) - gives
  // a bound of at most 2^(E_m + E_e - 45), while |c| >= 2^E_c for a normal c. The derivation
  // assumes normal numbers, so a zero or subnormal c or difference always takes the slow
  // path. That path is double arithmetic too, not exact: its rounding is at the scale of the
  // chunks' sums rather than the problem's.
  __device__ __forceinline__ double turn(
      const Pt& a, const Pt& b, const Pt& d, const Seg* A, int as, int ap, const Seg* B, int bs,
      int bp, const Seg* D, int ds, int dp) const {
    const double bx = b.x - a.x, by = b.y - a.y, dx = d.x - a.x, dy = d.y - a.y;
    const double p1 = bx * dy, p2 = by * dx, c = p1 - p2;
    const int ea = imax(bexp(a.x), bexp(a.y));
    const int ee = imax(imax(bexp(bx), bexp(by)), imax(bexp(dx), bexp(dy)));
    const int em = imax(ea, ee) + 1;
    const int ec = bexp(c);
    // Biased exponents: E = e - 1023, so the condition E_c >= E_m + E_e - 44 reads
    // e_c >= e_m + e_e - 1067.
    if ((ec > 0 && ee > 0 && ec >= em + ee - 1067) || cyl == nullptr) {
      return c;
    }
    double rbx, rby, rdx, rdy;
    rel(A[as], ap, B[bs], bp, rbx, rby);
    rel(A[as], ap, D[ds], dp, rdx, rdy);
    return rbx * rdy - rby * rdx;
  }

  // Vertex (sv, pv) minus vertex (sa, pa), from differences: of the chunk starts' hi parts
  // (exact when the chunks are close: Sterbenz), of their lo parts, and of the chunks' own
  // coordinates.
  __device__ __forceinline__ void rel(
      const Seg& sa, int pa, const Seg& sv, int pv, double& x, double& y) const {
    const int64_t ia = sa.voff + pa, iv = sv.voff + pv;
    const int ja = sa.chunk, jv = sv.chunk;
    y = (cy[jv] - cy[ja]) + (cyl[jv] - cyl[ja]) + (h.y[iv] - h.y[ia]);
    if (W) {
      x = (cx[jv] - cx[ja]) + (cxl[jv] - cxl[ja]) + (h.x[iv] - h.x[ia]);
    } else {
      x = static_cast<double>(h.p[iv] - h.p[ia]);
    }
  }

  // The slope of the edge from vertex (sa, pa) to (sb, pb), from the same differences.
  __device__ __forceinline__ double slope(const Seg& sa, int pa, const Seg& sb, int pb) const {
    const int64_t ia = sa.voff + pa, ib = sb.voff + pb;
    const int ja = sa.chunk, jb = sb.chunk;
    const double dy = (cy[jb] - cy[ja]) + (cyl[jb] - cyl[ja]) + (h.y[ib] - h.y[ia]);
    double dx;
    if (W) {
      dx = (cx[jb] - cx[ja]) + (cxl[jb] - cxl[ja]) + (h.x[ib] - h.x[ia]);
    } else {
      dx = static_cast<double>(h.p[ib] - h.p[ia]);
    }
    return dy / dx;
  }
};

// The first index in [0, count) where a predicate that is true, then false, is false (count
// if none), searched by the whole warp, 32 candidates per round.
template <typename Pred>
__device__ int warp_first_false(int count, int lane, Pred pred) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int step = (hi - lo + kLanes - 1) / kLanes;
    const int i = lo + lane * step;
    const unsigned truths = __ballot_sync(kFull, i < hi && pred(i));
    const int f = __ffs(~truths) - 1;  // the first false sample (rounding may break monotony)
    if (f < 0) {
      lo += (kLanes - 1) * step + 1;
      continue;
    }
    const int next_hi = lo + f * step;
    lo = f > 0 ? lo + (f - 1) * step + 1 : lo;
    hi = next_hi < hi ? next_hi : hi;
  }
  return lo;
}

// The same, by the whole warp (WARP) or by one thread.
template <bool WARP, typename Pred>
__device__ int first_false(int count, int lane, Pred pred) {
  if constexpr (WARP) {
    return warp_first_false(count, lane, pred);
  } else {
    int lo = 0, hi = count;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (pred(mid)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }
}

// Where the lower tangent from p = vertex (ps, pp) of list P, left of the right hull R,
// touches it (not at its first vertex, the one shared with the left hull): segment *ts,
// position *tp. One thread.
template <bool W>
__device__ void tangent(
    const Row<W>& row, const Seg* R, int nr, const Pt& p, const Seg* P, int ps, int pp, int* ts,
    int* tp) {
  // Slopes from p fall, then rise, along R. First narrow it down by segments' first vertices
  // to (first of segment s - 1, first of segment s + 1], then search that window.
  int sa = 0, pa = R[0].lo + 1, na = R[0].hi - R[0].lo;  // window part A: after a first vertex
  int sb = 0, pb = 0, nb = 0;                            // part B: a whole segment
  int sc = 0, pc = 0, nc = 0;                            // part C: the next first vertex
  if (nr > 1) {
    int lo = 1, hi = nr - 1;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      const Seg u = R[mid], v = R[mid + 1];
      if (row.turn(p, row.at(u, u.lo), row.at(v, v.lo), P, ps, pp, R, mid, u.lo, R, mid + 1,
                   v.lo) < 0) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const Seg a = R[lo - 1], b = R[lo];
    sa = lo - 1;
    pa = a.lo + 1;
    na = a.hi - a.lo;
    sb = lo;
    pb = b.lo;
    nb = b.hi - b.lo + 1;
    if (lo + 1 < nr) {
      sc = lo + 1;
      pc = R[lo + 1].lo;
      nc = 1;
    }
  }
  auto vertex = [&](int i, int* s, int* pos) {
    if (i < na) {
      *s = sa;
      *pos = pa + i;
    } else if (i - na < nb) {
      *s = sb;
      *pos = pb + i - na;
    } else {
      *s = sc;
      *pos = pc;
    }
  };
  int lo = 0, hi = na + nb + nc - 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    int s0, p0, s1, p1;
    vertex(mid, &s0, &p0);
    vertex(mid + 1, &s1, &p1);
    if (row.turn(p, row.at(R[s0], p0), row.at(R[s1], p1), P, ps, pp, R, s0, p0, R, s1, p1) <
        0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  vertex(lo, ts, tp);
}

// The left end of the bridge between L and R (sharing L's last vertex and R's first), when
// there is one: the first vertex of L whose successor is not below its tangent to R.
template <bool W, bool WARP>
__device__ void bridge_start(
    const Row<W>& row, const Seg* L, int nl, const Seg* R, int nr, int lane, int* ss, int* sp) {
  auto further = [&](int s, int pos) {  // the bridge starts right of vertex (s, pos)
    const Seg cs = L[s];
    const Pt p = row.at(cs, pos);
    const int ns = pos < cs.hi ? s : s + 1, np = pos < cs.hi ? pos + 1 : L[s + 1].lo;
    const Pt next = row.at(L[ns], np);
    int ts, tp;
    tangent(row, R, nr, p, L, s, pos, &ts, &tp);
    return row.turn(p, row.at(R[ts], tp), next, L, s, pos, R, ts, tp, L, ns, np) < 0;
  };
  // Candidates: every vertex but the shared one. By segments' first vertices first...
  const int firsts = L[nl - 1].lo == L[nl - 1].hi ? nl - 1 : nl;
  const int s = first_false<WARP>(firsts, lane, [&](int i) { return further(i, L[i].lo); });
  if (s == 0) {
    *ss = 0;
    *sp = L[0].lo;
    return;
  }
  // ...then within (first of segment s - 1, first of segment s].
  const Seg a = L[s - 1];
  const int inside = (s - 1 == nl - 1 ? a.hi - 1 : a.hi) - a.lo;
  const int width = inside + (s < firsts ? 1 : 0);
  if (width == 0) {
    *ss = s - 1;
    *sp = a.lo;
    return;
  }
  auto vertex = [&](int i, int* vs, int* vp) {
    if (i < inside) {
      *vs = s - 1;
      *vp = a.lo + 1 + i;
    } else {
      *vs = s;
      *vp = L[s].lo;
    }
  };
  const int i = first_false<WARP>(width, lane, [&](int i) {
    int vs, vp;
    vertex(i, &vs, &vp);
    return further(vs, vp);
  });
  vertex(i < width ? i : width - 1, ss, sp);
}

constexpr int kSegWalk = 64;  // steps of walk_segments before it gives up

// The bridge between L and R (sharing L's last vertex and R's first), found by walking from
// the shared point as in the chunks (walk_bridge), over the segment lists: few steps on the
// short hulls of the many-merge rounds. False (nothing written) after kSegWalk steps, for a
// search instead. One thread. (ls, lp): L's vertex before the shared one, lp = -1 meaning
// the last vertex of the segment before ls.
template <bool W>
__device__ bool walk_segments(
    const Row<W>& row, const Seg* L, const Seg* R, int nr, int ls, int lp, int* sa, int* pa,
    int* sb, int* pb) {
  if (lp < 0) {
    --ls;
    lp = L[ls].hi;
  }
  int rs = 0, rp = R[0].lo + 1;  // R's vertex after the shared one
  if (rp > R[0].hi) {
    rs = 1;
    rp = R[1].lo;
  }
  Pt a = row.at(L[ls], lp), b = row.at(R[rs], rp);
  int steps = 0;
  bool moved = true;
  while (moved) {
    moved = false;
    while (rs < nr - 1 || rp < R[rs].hi) {  // b is not R's last vertex
      int ns = rs, np = rp + 1;
      if (np > R[rs].hi) {
        ++ns;
        np = R[ns].lo;
      }
      const Pt q = row.at(R[ns], np);
      if (row.turn(a, b, q, L, ls, lp, R, rs, rp, R, ns, np) > 0) {
        break;
      }
      rs = ns;
      rp = np;
      b = q;
      moved = true;
      if (++steps > kSegWalk) {
        return false;
      }
    }
    while (ls > 0 || lp > L[0].lo) {  // a is not L's first vertex
      int ns = ls, np = lp - 1;
      if (np < L[ls].lo) {
        --ns;
        np = L[ns].hi;
      }
      const Pt q = row.at(L[ns], np);
      if (row.turn(q, a, b, L, ns, np, L, ls, lp, R, rs, rp) > 0) {
        break;
      }
      ls = ns;
      lp = np;
      a = q;
      moved = true;
      if (++steps > kSegWalk) {
        return false;
      }
    }
  }
  *sa = ls;
  *pa = lp;
  *sb = rs;
  *pb = rp;
  return true;
}

// Pairs of neighbouring nodes (half chunks wide) of one problem merge, one warp per pair
// (WARP: few pairs, long lists) or one thread per pair (many pairs, short lists: the warp's
// 32 parallel candidates would be mostly wasted float64 work). A node's segment list starts
// at its first chunk's slot; the merged list goes where the left one was.
template <bool W, bool WARP>
__device__ void merge_pair(Hulls h, const Row<W>& row, int64_t first, int64_t half, int lane) {
  Seg* L = h.seg + first;
  const Seg* R = L + half;
  const int nl = h.nseg[first], nr = h.nseg[first + half];

  const Seg last = L[nl - 1], head = R[0];
  const Pt shared = row.at(last, last.hi);
  // The vertices next to the shared one: before it in L, after it in R (segment, position).
  const int before_s = last.hi > last.lo ? nl - 1 : nl - 2;
  const int before_p = last.hi > last.lo ? last.hi - 1 : L[nl - 2].hi;
  const int after_s = head.hi > head.lo ? 0 : 1;
  const int after_p = head.hi > head.lo ? head.lo + 1 : R[1].lo;
  const Pt before = row.at(L[before_s], before_p), after = row.at(R[after_s], after_p);
  // The merged list: L's segments [0, keep), the last cut at hi_l, then R's from `from` on,
  // the first cut at lo_r.
  int keep, hi_l, from, lo_r;
  if (row.turn(before, shared, after, L, before_s, before_p, L, nl - 1, last.hi, R, after_s,
               after_p) > 0) {  // convex at the shared point: concatenate
    keep = nl;
    hi_l = last.hi;
    from = 0;
    lo_r = head.lo + 1;  // the shared point once
    if (lo_r > head.hi) {
      from = 1;
      lo_r = R[1].lo;
    }
  } else {
    int sa, pa, sb, pb;
    const bool walked =
        !WARP && walk_segments(row, L, R, nr, nl - 1, last.hi - 1 >= last.lo ? last.hi - 1 : -1,
                               &sa, &pa, &sb, &pb);
    if (!walked) {
      bridge_start<W, WARP>(row, L, nl, R, nr, lane, &sa, &pa);
      tangent(row, R, nr, row.at(L[sa], pa), L, sa, pa, &sb, &pb);
    }
#ifdef TI_PHASES
    if (lane == 0) {
      atomicAdd(&merge_counts[0], 1ull);
      if (!walked) {
        atomicAdd(&merge_counts[1], 1ull);
      }
    }
#endif
    keep = sa + 1;
    hi_l = pa;
    from = sb;
    lo_r = pb;
  }
  // Move R's tail down to follow L's. Destinations never pass their sources, so reading a
  // round of 32 before writing it is enough.
  const int moved = nr - from;
  if constexpr (WARP) {
    __syncwarp();
    if (lane == 0) {
      L[keep - 1].hi = hi_l;
    }
    for (int base = 0; base < moved; base += kLanes) {
      const int k = base + lane;
      Seg s{};
      if (k < moved) {
        s = R[from + k];
        if (k == 0) {
          s.lo = lo_r;
        }
      }
      __syncwarp();
      if (k < moved) {
        L[keep + k] = s;
      }
      __syncwarp();
    }
  } else {
    L[keep - 1].hi = hi_l;
    for (int k = 0; k < moved; ++k) {
      Seg s = R[from + k];
      if (k == 0) {
        s.lo = lo_r;
      }
      L[keep + k] = s;
    }
  }
  if (lane == 0) {
    h.nseg[first] = keep + moved;
  }
}

template <bool W, bool WARP>
__global__ void merge_kernel(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row, int64_t half) {
  const int64_t id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
                     (WARP ? kLanes : 1);
  const int lane = WARP ? threadIdx.x % kLanes : 0;
  const int64_t pairs = (per_row + half - 1) / (2 * half);
  if (id >= batch * pairs) {
    return;
  }
  const int64_t r = id / pairs, ro = r * (per_row + 1);
  const Row<W> row{h, cy + ro, W ? cx + ro : nullptr, h.cy_lo + ro, W ? h.cx_lo + ro : nullptr};
  merge_pair<W, WARP>(h, row, r * per_row + id % pairs * 2 * half, half, lane);
}

// Per chunk: its surviving vertices lo..hi (none: lo > hi) and the slopes of the edges into
// its first one and out of its last one (for a chunk without vertices, of the edge over it).
template <bool W>
__global__ void ends_kernel(
    Hulls h, const double* cy, const double* cx, int64_t chunks, int64_t per_row,
    int2* __restrict__ range, double2* __restrict__ ends) {
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= chunks) {
    return;
  }
  const int64_t r = g / per_row;
  const int32_t j = static_cast<int32_t>(g % per_row);
  const Seg* L = h.seg + r * per_row;
  const int nl = h.nseg[r * per_row];
  const int64_t ro = r * (per_row + 1);
  const Row<W> row{h, cy + ro, W ? cx + ro : nullptr, h.cy_lo + ro, W ? h.cx_lo + ro : nullptr};
  int lo = 0, hi = nl - 1;  // the last segment of a chunk <= j
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (L[mid].chunk <= j) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  auto slope = [&](int s) {  // of the edge from segment s's last vertex to s + 1's first
    return row.slope(L[s], L[s].hi, L[s + 1], L[s + 1].lo);
  };
  const Seg s = L[lo];
  if (s.chunk == j) {
    range[g] = make_int2(s.lo, s.hi);
    ends[g] = make_double2(lo > 0 ? slope(lo - 1) : 0.0, lo + 1 < nl ? slope(lo) : 0.0);
  } else {
    const double v = slope(lo);
    range[g] = make_int2(1, 0);
    ends[g] = make_double2(v, v);
  }
}

constexpr int kFitSpan = 1024;  // elements per block of fit_kernel

// One block per 1024 elements of a chunk: the slopes of the chunk's surviving edges over
// them, then every element's.
template <typename scalar_t, bool W>
__global__ void __launch_bounds__(kThreads) fit_kernel(
    Hulls h, const int2* __restrict__ range, const double2* __restrict__ ends, int64_t n,
    int64_t per_row, int chunk, scalar_t sign, scalar_t* __restrict__ out) {
  using acc_t = scalar_t;
  __shared__ int32_t at[kFitSpan + 2];
  __shared__ acc_t slope[kFitSpan + 2];
  const int subs = chunk / kFitSpan;
  const int64_t g = blockIdx.x / subs, r = g / per_row, j = g % per_row;
  const int64_t c0 = j * chunk + static_cast<int64_t>(blockIdx.x % subs) * kFitSpan;
  if (c0 >= n) {
    return;
  }
  const int len = static_cast<int>(n - c0 < kFitSpan ? n - c0 : kFitSpan);
  const int64_t start = r * n + c0;
  if (h.row_nan[r]) {
    for (int e = threadIdx.x; e < len; e += blockDim.x) {
      out[start + e] = std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return;
  }
  const int2 rg = range[g];
  const double2 en = ends[g];
  const acc_t in = static_cast<acc_t>(en.x);
  const int64_t vb = g * (chunk + 1);
  const int32_t p0 = static_cast<int32_t>(c0), p1 = p0 + len;
  // The chunk's surviving vertices that matter here: from the last at or before p0 (if any)
  // to the last before p1.
  int lo = rg.x, hi = rg.y + 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (h.p[vb + mid] <= p0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int first = lo > rg.x ? lo - 1 : rg.x;
  hi = rg.y + 1;
  lo = first;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (h.p[vb + mid] < p1) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int count = lo - first;  // vertices first .. lo - 1 (none if the chunk has none)
  for (int u = threadIdx.x; u < count; u += blockDim.x) {
    const int64_t i = vb + first + u;
    at[u] = h.p[i];
    if (first + u < rg.y) {
      // In double, rounded once, as the slopes across chunks (ends_kernel): neighbouring
      // slopes then compare as the exact ones do, but for ties far below float32's
      // resolution, and the float32 fit does not step down.
      const double dy = h.y[i + 1] - h.y[i];
      const double dx = W ? h.x[i + 1] - h.x[i] : static_cast<double>(h.p[i + 1] - h.p[i]);
      slope[u] = static_cast<acc_t>(dy / dx);
    } else {
      slope[u] = static_cast<acc_t>(en.y);
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < len; e += blockDim.x) {
    const int32_t p = p0 + e;
    int a = 0, b = count;  // vertices at or before element e's start
    while (a < b) {
      const int mid = (a + b) / 2;
      if (at[mid] <= p) {
        a = mid + 1;
      } else {
        b = mid;
      }
    }
    out[start + e] = sign * (a ? slope[a - 1] : in);
  }
}

int64_t blocks_for_warps(int64_t warps) {
  return (warps * kLanes + kThreads - 1) / kThreads;
}

#ifdef TI_PHASES
// After a round: the nodes' descriptor counts and vertices alive, summed into round `r`.
__global__ void merge_stats_kernel(
    const Seg* seg, const int32_t* nseg, int64_t batch, int64_t per_row, int64_t span, int r) {
  const int64_t id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nodes = (per_row + span - 1) / span;
  if (id >= batch * nodes) {
    return;
  }
  const int64_t first = (id / nodes) * per_row + (id % nodes) * span;
  const int n = nseg[first];
  unsigned long long vertices = 0;
  for (int i = 0; i < n; ++i) {
    vertices += seg[first + i].hi - seg[first + i].lo + 1;
  }
  atomicAdd(&merge_level_stats[r][0], 1ull);
  atomicAdd(&merge_level_stats[r][1], static_cast<unsigned long long>(n));
  atomicAdd(&merge_level_stats[r][2], vertices);
}
#endif

// Chunk starts: per row, the exclusive prefix sums of its chunks' totals as hi + lo pairs,
// plus the row's total in column per_row. One block per row; a thread sums a slice of
// chunks sequentially (two_sum), the block scans the slices, the thread writes its slice.
constexpr int kStartThreads = 256;  // more threads spill the double-double scan's registers

// THREADS: 256 for long rows, 32 (one warp, no block scan) for rows of up to 32 chunks.
template <int THREADS>
__global__ void __launch_bounds__(THREADS) chunk_starts_kernel(
    const double* __restrict__ totals, int64_t per_row, double* __restrict__ hi,
    double* __restrict__ lo) {
  __shared__ double wh[THREADS / kLanes], wl[THREADS / kLanes];
  const int64_t r = blockIdx.x, t = threadIdx.x, lane = t % kLanes, warp = t / kLanes;
  const int64_t per = (per_row + THREADS - 1) / THREADS;
  const int64_t j0 = t * per, j1 = j0 + per < per_row ? j0 + per : per_row;
  double sh = 0, sl = 0;  // this thread's slice total
  for (int64_t j = j0; j < j1; ++j) {
    add2(sh, sl, totals[r * per_row + j], 0.0);
  }
  // Exclusive prefix over the threads: warp scans, then the warps' totals.
  double ph = sh, pl = sl;
  for (int d = 1; d < kLanes; d <<= 1) {
    const double oh = __shfl_up_sync(kFull, ph, d), ol = __shfl_up_sync(kFull, pl, d);
    if (lane >= d) {
      add2(ph, pl, oh, ol);
    }
  }
  if (lane == kLanes - 1) {
    wh[warp] = ph;
    wl[warp] = pl;
  }
  __syncthreads();
  double oh = 0, ol = 0;
  for (int v = 0; v < warp; ++v) {
    add2(oh, ol, wh[v], wl[v]);
  }
  double eh = __shfl_up_sync(kFull, ph, 1), el = __shfl_up_sync(kFull, pl, 1);
  if (lane == 0) {
    eh = el = 0;
  }
  add2(eh, el, oh, ol);  // the exclusive prefix at this thread's first chunk
  double* oh_row = hi + r * (per_row + 1);
  double* ol_row = lo + r * (per_row + 1);
  for (int64_t j = j0; j < j1; ++j) {
    oh_row[j] = eh;
    ol_row[j] = el;
    add2(eh, el, totals[r * per_row + j], 0.0);
  }
  if (t == THREADS - 1) {
    oh_row[per_row] = eh;
    ol_row[per_row] = el;
  }
}

// The chunks' hulls of every problem merged into one, round after round of pairs.
template <bool W>
void merge_rounds(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row,
    cudaStream_t stream) {
  for (int64_t half = 1; half < per_row; half *= 2) {
    const int64_t merges = batch * ((per_row + half - 1) / (2 * half));
    if (merges >= kThreadMerges) {
      merge_kernel<W, false><<<(merges + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half);
    } else {
      merge_kernel<W, true><<<blocks_for_warps(merges), kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#ifdef TI_PHASES
    {
      static int round = 0;  // reset by tiles_phases
      const int64_t nodes = batch * ((per_row + 2 * half - 1) / (2 * half));
      if (round < kStatRounds) {
        merge_stats_kernel<<<(nodes + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            h.seg, h.nseg, batch, per_row, 2 * half, round);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      round = half * 2 < per_row ? round + 1 : 0;
    }
#endif
  }
}

// The whole pipeline for chunks of `chunk` elements, whose hulls (or, for one chunk per
// problem, fits) launch(chunks, n, per_row, sign, out, hulls, stream) computes, which
// merge(hulls, cy, cx, batch, per_row, stream) merges, and from which finish(hulls, cy, cx,
// batch, n, per_row, sign, out, stream) writes the fit: chunk_kernel, merge_rounds and
// fit_segments below, or an experiment's (experiments/).
template <typename scalar_t, bool W, int chunk, typename Launch, typename Merge, typename Finish>
void run_chunks(
    const at::Tensor& y, const at::Tensor* w, bool increasing, at::Tensor& out, Launch launch,
    Merge merge, Finish finish) {
  constexpr int64_t slots = chunk + 1;  // vertices of a chunk's hull, at most
  const int64_t batch = y.size(0), n = y.size(1);
  const int64_t per_row = (n + chunk - 1) / chunk, chunks = batch * per_row;
  const scalar_t sign = increasing ? scalar_t(1) : scalar_t(-1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto ints = y.options().dtype(at::kInt);
  const auto doubles = y.options().dtype(at::kDouble);

  at::Tensor vp, vy, vx, seg, nseg, ty, tx, row_nan;
  Hulls h{};
  if (per_row > 1) {
    vp = at::empty({chunks * slots}, ints);
    vy = at::empty({chunks * slots}, doubles);
    seg = at::empty({chunks * static_cast<int64_t>(sizeof(Seg) / sizeof(int64_t))},
                    y.options().dtype(at::kLong));
    nseg = at::empty({chunks}, ints);
    ty = at::empty({chunks}, doubles);
    row_nan = at::zeros({batch}, ints);
    if (W) {
      vx = at::empty({chunks * slots}, doubles);
      tx = at::empty({chunks}, doubles);
    }
    h = Hulls{vp.data_ptr<int32_t>(), vy.data_ptr<double>(), W ? vx.data_ptr<double>() : nullptr,
              reinterpret_cast<Seg*>(seg.data_ptr<int64_t>()), nseg.data_ptr<int32_t>(),
              ty.data_ptr<double>(), W ? tx.data_ptr<double>() : nullptr,
              row_nan.data_ptr<int32_t>()};
  }

  launch(chunks, n, per_row, sign, out.data_ptr<scalar_t>(), h, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (per_row == 1) {
    return;
  }

  // Chunk starts: [batch, per_row + 1] as hi + lo, the last column being each problem's total.
  auto starts = [&](const at::Tensor& totals, at::Tensor& lo) {
    at::Tensor hi = at::empty({batch, per_row + 1}, doubles);
    lo = at::empty({batch, per_row + 1}, doubles);
    if (per_row <= kLanes) {
      chunk_starts_kernel<kLanes><<<batch, kLanes, 0, stream>>>(
          totals.data_ptr<double>(), per_row, hi.data_ptr<double>(), lo.data_ptr<double>());
    } else {
      chunk_starts_kernel<kStartThreads><<<batch, kStartThreads, 0, stream>>>(
          totals.data_ptr<double>(), per_row, hi.data_ptr<double>(), lo.data_ptr<double>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return hi;
  };
  at::Tensor cy_lo, cx_lo;
  const at::Tensor cy = starts(ty, cy_lo);
  const at::Tensor cx = W ? starts(tx, cx_lo) : at::Tensor();
  const double* cyp = cy.data_ptr<double>();
  const double* cxp = W ? cx.data_ptr<double>() : nullptr;
  h.cy_lo = cy_lo.data_ptr<double>();
  h.cx_lo = W ? cx_lo.data_ptr<double>() : nullptr;

  merge(h, cyp, cxp, batch, per_row, stream);
  finish(h, cyp, cxp, batch, n, per_row, sign, out.data_ptr<scalar_t>(), stream);
}

// The fit from the merged segment lists: ends_kernel, then fit_kernel.
template <typename scalar_t, bool W, int chunk>
void fit_segments(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t n, int64_t per_row,
    scalar_t sign, scalar_t* out, cudaStream_t stream) {
  const int64_t chunks = batch * per_row;
  const auto opts = at::TensorOptions().device(at::kCUDA, at::cuda::current_device());
  at::Tensor range = at::empty({chunks * 2}, opts.dtype(at::kInt));
  at::Tensor ends = at::empty({chunks * 2}, opts.dtype(at::kDouble));
  auto* rp = reinterpret_cast<int2*>(range.data_ptr<int32_t>());
  auto* ep = reinterpret_cast<double2*>(ends.data_ptr<double>());
  ends_kernel<W><<<(chunks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      h, cy, cx, chunks, per_row, rp, ep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  fit_kernel<scalar_t, W><<<chunks * (chunk / kFitSpan), kThreads, 0, stream>>>(
      h, rp, ep, n, per_row, chunk, sign, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool W, int chunk, typename Launch, typename Merge>
void run_chunks(
    const at::Tensor& y, const at::Tensor* w, bool increasing, at::Tensor& out, Launch launch,
    Merge merge) {
  run_chunks<scalar_t, W, chunk>(
      y, w, increasing, out, launch, merge, fit_segments<scalar_t, W, chunk>);
}

template <typename scalar_t, bool W, int chunk, typename Launch>
void run_chunks(
    const at::Tensor& y, const at::Tensor* w, bool increasing, at::Tensor& out, Launch launch) {
  run_chunks<scalar_t, W, chunk>(y, w, increasing, out, launch, merge_rounds<W>);
}

template <typename scalar_t, bool W, int warps, bool EXACT>
void run(const at::Tensor& y, const at::Tensor* w, bool increasing, at::Tensor& out) {
  const scalar_t* yp = y.data_ptr<scalar_t>();
  const scalar_t* wp = W ? w->data_ptr<scalar_t>() : nullptr;
  run_chunks<scalar_t, W, Chunk<scalar_t, W, warps, EXACT>::kN>(
      y, w, increasing, out,
      [&](int64_t chunks, int64_t n, int64_t per_row, scalar_t sign, scalar_t* o, Hulls h,
          cudaStream_t stream) {
        // The largest shared memory carveout: shared memory limits the blocks per SM.
        static const bool carved = [] {
          C10_CUDA_CHECK(cudaFuncSetAttribute(chunk_kernel<scalar_t, W, warps, EXACT>,
                                              cudaFuncAttributePreferredSharedMemoryCarveout,
                                              cudaSharedmemCarveoutMaxShared));
          return true;
        }();
        (void)carved;
        chunk_kernel<scalar_t, W, warps, EXACT><<<chunks, warps * kLanes, 0, stream>>>(
            yp, wp, n, per_row, sign, o, h);
      });
}

// Problems of up to 512 elements, several per warp (packed_kernel).
template <typename scalar_t, bool W, bool EXACT>
void run_packed(const at::Tensor& y, const at::Tensor* w, bool increasing, at::Tensor& out) {
  // Warps per block, each on its own problems: up to 4, as shared memory allows.
  constexpr int per_warp_shared = static_cast<int>(sizeof(Chunk<scalar_t, W, 1, EXACT>)) +
                                  kLanes * static_cast<int>(sizeof(int));
  constexpr int warps = 48 * 1024 / per_warp_shared >= 4   ? 4
                        : 48 * 1024 / per_warp_shared >= 2 ? 2
                                                           : 1;
  const int64_t batch = y.size(0);
  const int n = static_cast<int>(y.size(1));
  int L = 1;
  while (L * kLanes < n) {
    L *= 2;
  }
  const int64_t per_warp = kLanes / L, groups = (batch + per_warp - 1) / per_warp;
  const unsigned long long inv_n = ((1ULL << 32) + n - 1) / n;
  packed_kernel<scalar_t, W, warps, EXACT>
      <<<(groups + warps - 1) / warps, warps * kLanes, 0, at::cuda::getCurrentCUDAStream()>>>(
          y.data_ptr<scalar_t>(), W ? w->data_ptr<scalar_t>() : nullptr, batch, n, L, inv_n,
          increasing ? scalar_t(1) : scalar_t(-1), out.data_ptr<scalar_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The chunk size, tuned on a T4: a problem of up to 4096 elements is one chunk of as few warps
// as cover it (finished in one kernel, no warp idle); longer ones are split into one-warp
// chunks, whose hulls cost least to build. TI_CHUNK_WARPS (1, 2 or 4) overrides it.
template <typename scalar_t, bool W, bool EXACT>
void run_sized(const at::Tensor& y, const at::Tensor* w, bool increasing, at::Tensor& out) {
  constexpr int most = chunk_warps<scalar_t, W, EXACT>();
  const int64_t n = y.size(1);
  if (n <= 512 && !std::getenv("TI_CHUNK_WARPS")) {  // several problems per warp
    run_packed<scalar_t, W, EXACT>(y, w, increasing, out);
    return;
  }
  int warps = n <= 1024 ? 1 : n <= 2048 ? 2 : n <= 4096 ? 4 : 1;
  if (const char* forced = std::getenv("TI_CHUNK_WARPS")) {
    warps = std::atoi(forced);
  }
  if (warps >= 4 && most >= 4) {
    run<scalar_t, W, (most >= 4 ? 4 : 1), EXACT>(y, w, increasing, out);
  } else if (warps >= 2 && most >= 2) {
    run<scalar_t, W, (most >= 2 ? 2 : 1), EXACT>(y, w, increasing, out);
  } else {
    run<scalar_t, W, 1, EXACT>(y, w, increasing, out);
  }
}

// exact = false: float32 without the pairs within lanes (see Chunk); float64 is unaffected.
at::Tensor isotonic_tiles_cuda(
    const at::Tensor& y, const c10::optional<at::Tensor>& w, bool increasing, bool exact = true) {
  TORCH_CHECK(y.dim() == 2 && y.size(1) >= 1, "y must be [batch, n] with n >= 1");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
  const bool weighted = w.has_value();
  if (weighted) {
    TORCH_CHECK(w->sizes() == y.sizes() && w->scalar_type() == y.scalar_type(), "bad w");
    TORCH_CHECK(w->is_contiguous(), "w must be contiguous");
  }
  TORCH_CHECK(y.size(1) < (int64_t{1} << 31) - 1,
              "problems must have fewer than 2^31 - 1 elements");
  const c10::cuda::CUDAGuard guard(y.device());
  at::Tensor out = at::empty_like(y);
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_tiles_cuda", [&] {
    // Only float32 has a separate fast path, so only it is compiled twice.
    if constexpr (sizeof(scalar_t) == 4) {
      if (!exact) {
        if (weighted) {
          run_sized<scalar_t, true, false>(y, &*w, increasing, out);
        } else {
          run_sized<scalar_t, false, false>(y, nullptr, increasing, out);
        }
        return;
      }
    }
    if (weighted) {
      run_sized<scalar_t, true, true>(y, &*w, increasing, out);
    } else {
      run_sized<scalar_t, false, true>(y, nullptr, increasing, out);
    }
  });
  return out;
}

}  // namespace

}  // namespace torchisotonic
