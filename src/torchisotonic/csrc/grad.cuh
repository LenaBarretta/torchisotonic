// The gradient of the fit on CUDA: torchisotonic::isotonic_grad.
//
// The fit is constant on blocks - runs of equal fitted values in a row (equal up to rounding:
// same_block) - each the weighted mean of y over its block, so with g the gradient with
// respect to the fit, and G and W the sums of g and w over an element's block:
//
//   d/dy_i = w_i G / W,   d/dw_i = (y_i - fit_i) G / W.
//
// Block sums without atomics (a block can be millions of elements long). A warp takes a tile
// of 32 E elements of a row, E per lane, E chosen so that a row of up to 1024 elements is one
// tile with every lane busy. A row that is one tile is finished by grad_fill_kernel alone. A
// longer row (E = 32) takes three steps:
//
// 1. grad_ends_kernel: per tile, where blocks start, and the tile's sums before its first
//    start and from its last start on, and whether it has a start at all.
// 2. On the tiles only (a thousandth of the data), with prefix sums: the sums of the blocks
//    that cross tiles.
// 3. grad_fill_kernel: the tiles again; within a tile, block sums by scans over each lane's
//    elements and over the lanes, the blocks crossing its edges from step 2; then the
//    gradients, written coalesced.
#pragma once

#include <vector>

#include "tiles.cuh"

namespace torchisotonic {

namespace {

// Whether neighbouring fitted values a and b belong to one block: equal up to 16 units in the
// last place, as the fit's own rounding cannot tell them apart (blocks whose means are equal
// - a point where the fit is not differentiable - are then one, on any device).
template <typename T>
__device__ __forceinline__ bool same_block(T a, T b) {
  const T scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
  return fabs(a - b) <= T(16) * std::numeric_limits<T>::epsilon() * scale;
}

// Lane l's elements at l * (E + 1) + k: reading one element per lane hits E + 1 banks apart.
template <int E>
__device__ __forceinline__ int gpad(int e) {
  return e + e / E;
}

template <typename T, bool W, bool Y, int E>
struct GradShared {
  static constexpr int kN = kLanes * E + kLanes;
  T out[kN];
  T g[kN];
  T w[W ? kN : 1];
  T y[Y ? kN : 1];
};

// Warps per block for a tile type: as many as keep the block's shared memory under 40 KB.
template <typename Shared>
constexpr int grad_warps() {
  constexpr int fit = (40 * 1024) / static_cast<int>(sizeof(Shared));
  return fit >= 8 ? 8 : fit >= 4 ? 4 : fit >= 2 ? 2 : 1;
}

// One tile, as each lane sees its E elements: where blocks start, and sums of g and w before
// its first start (F), from its last start on (L); without a start, F = L = the lane's sums.
template <typename T, bool W, bool Y, int E>
struct LaneRuns {
  uint32_t starts = 0;  // bit k: a block starts at the lane's element k
  int count = 0;
  T fg = 0, fw = 0, lg = 0, lw = 0;
  T pg[E], pw[E];  // sums within the block, up to each element

  __device__ void scan(
      const GradShared<T, W, Y, E>& s, int lane, int len, bool row_start, T before) {
    const int e0 = lane * E, base = lane * (E + 1);
    count = imax(0, imin(E, len - e0));
    T prev = lane == 0 ? before : s.out[gpad<E>(e0 - 1)];
    T cg = 0, cw = 0;
#pragma unroll
    for (int k = 0; k < E; ++k) {
      if (k < count) {
        const T o = s.out[base + k];
        const bool start = (row_start && lane == 0 && k == 0) || !same_block(o, prev);
        prev = o;
        if (start) {
          if (!starts) {
            fg = cg;
            fw = cw;
          }
          starts |= 1u << k;
          cg = cw = 0;
        }
        cg += s.g[base + k];
        cw += W ? s.w[base + k] : T(1);
        pg[k] = cg;
        pw[k] = cw;
      }
    }
    if (starts) {
      lg = cg;
      lw = cw;
    } else {
      fg = lg = cg;
      fw = lw = cw;
    }
  }
};

template <typename T, bool W, bool Y, int E>
__device__ void load_tile(
    GradShared<T, W, Y, E>& s, const T* out, const T* g, const T* w, const T* y, int64_t start,
    int len, int lane) {
  for (int e = lane; e < len; e += kLanes) {
    s.out[gpad<E>(e)] = out[start + e];
    s.g[gpad<E>(e)] = g[start + e];
    if (W) {
      s.w[gpad<E>(e)] = w[start + e];
    }
    if (Y) {
      s.y[gpad<E>(e)] = y[start + e];
    }
  }
  __syncwarp();
}

// Sums as pairs hi + lo of the working type (Pair, from tiles.cuh): the precision of the
// forward pass's sums without float64 arithmetic, which most GPUs run far slower.
template <typename T>
__device__ __forceinline__ Pair<T> p2(T v) {
  return {v, T(0)};
}

template <typename T>
__device__ __forceinline__ Pair<T> p2(double v) {  // exact split of a double
  const T h = static_cast<T>(v);
  return {h, static_cast<T>(v - static_cast<double>(h))};
}

template <typename T>
__device__ __forceinline__ Pair<T> padd(Pair<T> a, Pair<T> b) {
  add2(a.h, a.l, b.h, b.l);
  return a;
}

template <typename T>
__device__ __forceinline__ Pair<T> psub(Pair<T> a, Pair<T> b) {
  return padd(a, Pair<T>{-b.h, -b.l});
}

template <typename T>
__device__ __forceinline__ Pair<T> pshfl(Pair<T> v, int src) {
  return {__shfl_sync(kFull, v.h, src), __shfl_sync(kFull, v.l, src)};
}

template <typename T>
__device__ __forceinline__ Pair<T> pscan(Pair<T> v, int lane) {  // inclusive, over the warp
  for (int d = 1; d < kLanes; d <<= 1) {
    const Pair<T> o{__shfl_up_sync(kFull, v.h, d), __shfl_up_sync(kFull, v.l, d)};
    if (lane >= d) {
      v = padd(v, o);
    }
  }
  return v;
}

__device__ __forceinline__ double warp_sum(double v) {
  for (int d = 16; d > 0; d >>= 1) {
    v += __shfl_xor_sync(kFull, v, d);
  }
  return v;
}

constexpr int kGradTile = kLanes * kLanes;  // elements per tile of a long row (E = 32)

// Step 1, long rows: per tile, F (g, w), L (g, w) and whether a block starts in it.
template <typename T, bool W, int WARPS>
__global__ void __launch_bounds__(WARPS * kLanes) grad_ends_kernel(
    const T* __restrict__ out, const T* __restrict__ g, const T* __restrict__ w, int64_t tiles,
    int64_t n, int64_t per_row, double4* __restrict__ ends, int32_t* __restrict__ has) {
  constexpr int E = kLanes;
  __shared__ GradShared<T, W, false, E> shared[WARPS];
  const int lane = threadIdx.x % kLanes, warp = threadIdx.x / kLanes;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
  if (t >= tiles) {
    return;  // the whole warp
  }
  GradShared<T, W, false, E>& s = shared[warp];
  const int64_t j = t % per_row, start = (t / per_row) * n + j * kGradTile;
  const int len = static_cast<int>(n - j * kGradTile < kGradTile ? n - j * kGradTile : kGradTile);
  load_tile<T, W, false, E>(s, out, g, w, nullptr, start, len, lane);
  LaneRuns<T, W, false, E> lr;
  lr.scan(s, lane, len, j == 0, j > 0 ? out[start - 1] : T(0));
  const uint32_t heads = __ballot_sync(kFull, lr.starts != 0);
  const int first = heads ? __ffs(heads) - 1 : kLanes, last = heads ? 31 - __clz(heads) : -1;
  // Before the first start: whole lanes before the first lane with one, then its F; from the
  // last start on: its L, then whole lanes after it.
  const double fg = warp_sum(lane <= first ? double(lr.fg) : 0.0);
  const double fw = warp_sum(lane <= first ? double(lr.fw) : 0.0);
  const double lg = warp_sum(lane >= last ? double(lr.lg) : 0.0);
  const double lw = warp_sum(lane >= last ? double(lr.lw) : 0.0);
  if (lane == 0) {
    ends[t] = make_double4(fg, fw, lg, lw);
    has[t] = heads != 0;
  }
}

// Step 3 (or the whole job, SINGLE: the row is one tile): the gradients. For a long row,
// edges[t] holds the sums (g, w) of the block holding the tile's first element and of the
// one holding its last.
template <typename T, bool W, bool Y, int E, bool SINGLE, int WARPS>
__global__ void __launch_bounds__(WARPS * kLanes) grad_fill_kernel(
    const T* __restrict__ out, const T* __restrict__ g, const T* __restrict__ w,
    const T* __restrict__ y, int64_t tiles, int64_t n, int64_t per_row,
    const double4* __restrict__ edges, T* __restrict__ grad_y, T* __restrict__ grad_w) {
  constexpr int kTile = kLanes * E;
  __shared__ GradShared<T, W, Y, E> shared[WARPS];
  const int lane = threadIdx.x % kLanes, warp = threadIdx.x / kLanes;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
  if (t >= tiles) {
    return;  // the whole warp
  }
  GradShared<T, W, Y, E>& s = shared[warp];
  const int64_t j = SINGLE ? 0 : t % per_row;
  const int64_t start = SINGLE ? t * n : (t / per_row) * n + j * kTile;
  const int len = SINGLE ? static_cast<int>(n)
                         : static_cast<int>(n - j * kTile < kTile ? n - j * kTile : kTile);
  load_tile<T, W, Y, E>(s, out, g, w, y, start, len, lane);
  LaneRuns<T, W, Y, E> lr;
  lr.scan(s, lane, len, j == 0, j > 0 ? out[start - 1] : T(0));
  // A lane's last block (from its last start) runs on to the next lane with a start, the
  // lanes in between whole: L + (Q[next] - Q[lane]), Q the prefix sums of F over the lanes.
  // Without a start further in the tile, it is the tile's last block.
  const uint32_t heads = __ballot_sync(kFull, lr.starts != 0);
  const Pair<T> qg = pscan(p2(lr.fg), lane), qw = pscan(p2(lr.fw), lane);
  Pair<T> first_g, first_w, last_g, last_w;  // the tile's first and last blocks
  if constexpr (SINGLE) {
    // The row starts a block at its element 0 (lane 0), so the first block is never needed
    // from outside; the last block is the last lane-with-start's L plus the lanes after it.
    const int last = heads ? 31 - __clz(heads) : 0;
    last_g = padd(pshfl(p2(lr.lg), last), psub(pshfl(qg, kLanes - 1), pshfl(qg, last)));
    last_w = padd(pshfl(p2(lr.lw), last), psub(pshfl(qw, kLanes - 1), pshfl(qw, last)));
    first_g = first_w = p2(T(0));
  } else {
    const double4 edge = edges[t];
    first_g = p2<T>(edge.x);
    first_w = p2<T>(edge.y);
    last_g = p2<T>(edge.z);
    last_w = p2<T>(edge.w);
  }
  const uint32_t later = heads & ~((2u << lane) - 1u);
  const int next = later ? __ffs(later) - 1 : 0;
  const Pair<T> ng = pshfl(qg, next), nw = pshfl(qw, next);
  Pair<T> tg = last_g, tw = last_w;  // this lane's last block
  if (later) {
    tg = padd(p2(lr.lg), psub(ng, qg));
    tw = padd(p2(lr.lw), psub(nw, qw));
  }
  // A lane's first block (before its first start) is the last block of the previous lane
  // with a start, or the tile's first block.
  const uint32_t earlier = heads & ((1u << lane) - 1u);
  const int prev = earlier ? 31 - __clz(earlier) : 0;
  Pair<T> hg = pshfl(tg, prev), hw = pshfl(tw, prev);
  if (!earlier) {
    hg = first_g;
    hw = first_w;
  }
  // Backwards over the lane's elements, the sums of the block each is in; the gradients go
  // where g and y were (the lane's own slots, which no other lane reads any more).
  {
    const int base = lane * (E + 1);
    const int first = lr.starts ? __ffs(lr.starts) - 1 : E;
    T bg = lr.starts ? tg.h + tg.l : hg.h + hg.l, bw = lr.starts ? tw.h + tw.l : hw.h + hw.l;
#pragma unroll
    for (int k = E - 1; k >= 0; --k) {
      if (k < lr.count) {
        const T mean_grad = bg / bw;
        s.g[base + k] = (W ? s.w[base + k] : T(1)) * mean_grad;
        if (Y) {
          s.y[base + k] = (s.y[base + k] - s.out[base + k]) * mean_grad;
        }
        if ((lr.starts >> k) & 1u) {  // the elements before k are in the block before
          if (k > first) {
            bg = lr.pg[k - 1];
            bw = lr.pw[k - 1];
          } else {
            bg = hg.h + hg.l;
            bw = hw.h + hw.l;
          }
        }
      }
    }
  }
  __syncwarp();
  for (int e = lane; e < len; e += kLanes) {
    grad_y[start + e] = s.g[gpad<E>(e)];
    if (Y) {
      grad_w[start + e] = s.y[gpad<E>(e)];
    }
  }
}

// Rows of up to 32 E elements, one tile each.
template <typename T, bool W, bool Y, int E>
void run_grad_single(
    const T* out, const T* g, const T* w, const T* y, int64_t batch, int64_t n, T* grad_y,
    T* grad_w, cudaStream_t stream) {
  constexpr int warps = grad_warps<GradShared<T, W, Y, E>>();
  grad_fill_kernel<T, W, Y, E, true, warps>
      <<<(batch + warps - 1) / warps, warps * kLanes, 0, stream>>>(
          out, g, w, y, batch, n, 1, nullptr, grad_y, grad_w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, bool W, bool Y>
void run_grad(
    const at::Tensor& out, const at::Tensor& g, const at::Tensor* w, const at::Tensor* y,
    at::Tensor& grad_y, at::Tensor& grad_w) {
  const int64_t batch = out.size(0), n = out.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const T* op = out.data_ptr<T>();
  const T* gp = g.data_ptr<T>();
  const T* wp = W ? w->data_ptr<T>() : nullptr;
  const T* yp = Y ? y->data_ptr<T>() : nullptr;
  T* gyp = grad_y.data_ptr<T>();
  T* gwp = Y ? grad_w.data_ptr<T>() : nullptr;
  if (n <= kGradTile) {  // one tile per row: E elements per lane, the fewest that cover it
    if (n <= 32) {
      run_grad_single<T, W, Y, 1>(op, gp, wp, yp, batch, n, gyp, gwp, stream);
    } else if (n <= 64) {
      run_grad_single<T, W, Y, 2>(op, gp, wp, yp, batch, n, gyp, gwp, stream);
    } else if (n <= 128) {
      run_grad_single<T, W, Y, 4>(op, gp, wp, yp, batch, n, gyp, gwp, stream);
    } else if (n <= 256) {
      run_grad_single<T, W, Y, 8>(op, gp, wp, yp, batch, n, gyp, gwp, stream);
    } else if (n <= 512) {
      run_grad_single<T, W, Y, 16>(op, gp, wp, yp, batch, n, gyp, gwp, stream);
    } else {
      run_grad_single<T, W, Y, 32>(op, gp, wp, yp, batch, n, gyp, gwp, stream);
    }
    return;
  }

  const int64_t per_row = (n + kGradTile - 1) / kGradTile, tiles = batch * per_row;
  const auto doubles = out.options().dtype(at::kDouble);
  at::Tensor ends = at::empty({tiles, 4}, doubles);
  at::Tensor has = at::empty({tiles}, out.options().dtype(at::kInt));
  {
    constexpr int warps = grad_warps<GradShared<T, W, false, kLanes>>();
    grad_ends_kernel<T, W, warps><<<(tiles + warps - 1) / warps, warps * kLanes, 0, stream>>>(
        op, gp, wp, tiles, n, per_row, reinterpret_cast<double4*>(ends.data_ptr<double>()),
        has.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Step 2, per row over its tiles: a block that starts in tile s (its last block) and next
  // starts in tile e sums L[s] + F[s+1] + ... + F[e] (or to the row's end): with P the prefix
  // sums of F, L[s] + P[e] - P[s]. A tile's first block is the last block of the latest tile
  // before it with a start (the first tile of a row starts one at its first element).
  const at::Tensor e4 = ends.view({batch, per_row, 4});
  const at::Tensor f = e4.narrow(2, 0, 2), l = e4.narrow(2, 2, 2);
  const at::Tensor p = at::cumsum(f, 1);
  const at::Tensor starts = has.view({batch, per_row}).to(at::kBool);
  const at::Tensor index =
      at::arange(per_row, out.options().dtype(at::kLong)).expand({batch, per_row});
  at::Tensor next = at::where(starts, index, at::full_like(index, per_row));
  next = std::get<0>(at::cummin(next.flip(1), 1)).flip(1);  // first start tile >= t
  next = at::cat({next.narrow(1, 1, per_row - 1), at::full_like(next.narrow(1, 0, 1), per_row)}, 1)
             .clamp_max(per_row - 1);  // first start tile > t, or the row's last tile
  const at::Tensor at_next = at::gather(p, 1, next.unsqueeze(2).expand({batch, per_row, 2}));
  const at::Tensor last_block = l + at_next - p;  // for tiles with a start
  at::Tensor prev = at::where(starts, index, at::full_like(index, -1));
  prev = std::get<0>(at::cummax(prev, 1));
  prev = at::cat({at::zeros_like(prev.narrow(1, 0, 1)), prev.narrow(1, 0, per_row - 1)}, 1)
             .clamp_min(0);  // latest start tile < t
  const at::Tensor first_block =
      at::gather(last_block, 1, prev.unsqueeze(2).expand({batch, per_row, 2}));
  // For a tile without a start, its last block is its first.
  const at::Tensor last_or_first = at::where(starts.unsqueeze(2), last_block, first_block);
  const at::Tensor edges = at::cat({first_block, last_or_first}, 2).contiguous();

  {
    constexpr int warps = grad_warps<GradShared<T, W, Y, kLanes>>();
    grad_fill_kernel<T, W, Y, kLanes, false, warps>
        <<<(tiles + warps - 1) / warps, warps * kLanes, 0, stream>>>(
            op, gp, wp, yp, tiles, n, per_row,
            reinterpret_cast<const double4*>(edges.data_ptr<double>()), gyp, gwp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// out: the fit ([batch, n]); g: the gradient with respect to it; w: the weights, if any; y: the
// data, if the gradient with respect to the weights is wanted (with w). Returns the gradient
// with respect to y, and the one with respect to w (empty without y).
std::vector<at::Tensor> isotonic_grad_cuda(
    const at::Tensor& out, const at::Tensor& g, const c10::optional<at::Tensor>& w,
    const c10::optional<at::Tensor>& y) {
  TORCH_CHECK(out.dim() == 2 && out.is_contiguous(), "out must be a contiguous [batch, n]");
  TORCH_CHECK(g.sizes() == out.sizes() && g.scalar_type() == out.scalar_type(), "bad g");
  TORCH_CHECK(!y.has_value() || w.has_value(), "y is only needed for the weights' gradient");
  const c10::cuda::CUDAGuard guard(out.device());
  const at::Tensor gc = g.contiguous();
  const at::Tensor wc = w.has_value() ? w->contiguous() : at::Tensor();
  const at::Tensor yc = y.has_value() ? y->contiguous() : at::Tensor();
  at::Tensor grad_y = at::empty_like(out);
  at::Tensor grad_w = y.has_value() ? at::empty_like(out) : at::empty({0}, out.options());
  if (out.numel() > 0) {
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "isotonic_grad_cuda", [&] {
      if (y.has_value()) {
        run_grad<scalar_t, true, true>(out, gc, &wc, &yc, grad_y, grad_w);
      } else if (w.has_value()) {
        run_grad<scalar_t, true, false>(out, gc, &wc, nullptr, grad_y, grad_w);
      } else {
        run_grad<scalar_t, false, false>(out, gc, nullptr, nullptr, grad_y, grad_w);
      }
    });
  }
  return {grad_y, grad_w};
}

}  // namespace

}  // namespace torchisotonic
