// CUDA implementation of torchisotonic::isotonic: one thread per problem.
//
// The batch is transposed to [n, batch] first, so that at every step of PAVA the threads of
// a warp, each at the same position of its own problem, read and write neighbouring
// addresses. The block stack of each problem lives in global memory with the same layout.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <limits>

#include "hull.h"
#include "pava.h"

namespace torchisotonic {

namespace {

constexpr int kThreads = 128;

template <typename scalar_t>
__global__ void isotonic_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t batch,
    int32_t* __restrict__ start,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ total) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= batch) {
    return;
  }
  pava<scalar_t, scalar_t, int32_t>(
      y + b, w + b, out + b, n, batch, start + b, weight + b, total + b);
}

at::Tensor isotonic_cuda(const at::Tensor& y, const at::Tensor& w) {
  TORCH_CHECK(y.dim() == 2 && y.sizes() == w.sizes(), "y and w must be [batch, n] and alike");
  TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y and w must have the same dtype");
  TORCH_CHECK(y.size(1) < (int64_t{1} << 31), "problems must have fewer than 2^31 elements");
  const c10::cuda::CUDAGuard guard(y.device());
  const int64_t batch = y.size(0), n = y.size(1);
  const at::Tensor yt = y.t().contiguous(), wt = w.t().contiguous();  // [n, batch]
  at::Tensor out = at::empty_like(yt);
  at::Tensor start = at::empty({n, batch}, yt.options().dtype(at::kInt));
  at::Tensor weight = at::empty_like(yt), total = at::empty_like(yt);
  const int blocks = static_cast<int>((batch + kThreads - 1) / kThreads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_cuda", [&] {
    isotonic_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        yt.data_ptr<scalar_t>(),
        wt.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        n,
        batch,
        start.data_ptr<int32_t>(),
        weight.data_ptr<scalar_t>(),
        total.data_ptr<scalar_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.t().contiguous();  // back to [batch, n]
}

// The hull algorithm for few, long problems. Rows are blockIdx.y. A hull is three arrays
// (x, y, point index) in a fixed-size slot of a vertex buffer: slot k of a level starts at
// k * cap. Two sets of buffers take turns from one level of the merge tree to the next.

struct Vertices {
  double* x;
  double* y;
  int32_t* k;
};

constexpr int kChunkThreads = 128;  // one chunk per thread
constexpr int kRound = 8;           // points per chunk loaded per round

// Every thread builds the lower hull of its chunk in local coordinates (cumulative sums
// from the chunk's start), with the stack in per-thread arrays. The block loads the chunks'
// points together, kRound per chunk per round, so that the loads are coalesced: the kRound
// threads that load one chunk's points read neighbouring addresses.
//
// Inside a chunk the numbers are small, so the hull is computed in the input's precision:
// float32 input costs float32 arithmetic, which a T4 does 32 times faster than float64. The
// vertices and the chunk totals are then kept in float64 for the merges.
template <typename scalar_t, int MAX>
__global__ void chunk_hull_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ w,
    const double* __restrict__ centre,
    int64_t n,
    int64_t chunk,
    int64_t nchunks,
    Vertices out,
    int32_t* __restrict__ cnt,
    double* __restrict__ total_x,
    double* __restrict__ total_y,
    int64_t cap,
    int64_t row_vert) {
  __shared__ scalar_t tile_y[kChunkThreads * kRound], tile_w[kChunkThreads * kRound];
  const int tid = threadIdx.x;
  const int64_t r = blockIdx.y;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * kChunkThreads;  // block's 1st chunk
  const int64_t c = first + tid;
  const int64_t lo = c * chunk;
  const int64_t len = c < nchunks ? (lo + chunk < n ? chunk : n - lo) : 0;
  const scalar_t cr = static_cast<scalar_t>(centre[r]);
  const scalar_t* yr = y + r * n;
  const scalar_t* wr = w + r * n;
  scalar_t sx[MAX + 1], sy[MAX + 1];
  int32_t si[MAX + 1];
  int32_t h = 0;
  scalar_t x = 0, s = 0;
  if (len > 0) {
    hull_push(sx, sy, si, h, scalar_t(0), scalar_t(0), static_cast<int32_t>(lo));
  }
  for (int64_t k0 = 0; k0 < chunk; k0 += kRound) {
    __syncthreads();
    for (int e = tid; e < kChunkThreads * kRound; e += kChunkThreads) {
      const int64_t ce = first + e / kRound, ke = k0 + e % kRound;
      const int64_t pos = ce * chunk + ke;
      const bool ok = ce < nchunks && ke < chunk && pos < n;
      tile_y[e] = ok ? yr[pos] : scalar_t(0);
      tile_w[e] = ok ? wr[pos] : scalar_t(0);
    }
    __syncthreads();
    for (int k = 0; k < kRound && k0 + k < len; ++k) {
      const scalar_t wi = tile_w[tid * kRound + k];
      x += wi;
      s += wi * (tile_y[tid * kRound + k] - cr);
      hull_push(sx, sy, si, h, x, s, static_cast<int32_t>(lo + k0 + k + 1));
    }
  }
  if (len == 0) {
    return;
  }
  const int64_t slot = r * row_vert + c * cap;
  for (int32_t v = 0; v < h; ++v) {
    out.x[slot + v] = static_cast<double>(sx[v]);
    out.y[slot + v] = static_cast<double>(sy[v]);
    out.k[slot + v] = si[v];
  }
  cnt[r * nchunks + c] = h;
  total_x[r * nchunks + c] = x;
  total_y[r * nchunks + c] = s;
}

// Moves every chunk's vertices from local to global coordinates (base: where it starts).
__global__ void shift_kernel(
    Vertices v,
    const int32_t* __restrict__ cnt,
    const double* __restrict__ base_x,
    const double* __restrict__ base_y,
    int64_t nchunks,
    int64_t cap,
    int64_t row_vert) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = blockIdx.y;
  if (c >= nchunks) {
    return;
  }
  const int64_t slot = r * row_vert + c * cap;
  const double bx = base_x[r * nchunks + c], by = base_y[r * nchunks + c];
  for (int32_t i = 0; i < cnt[r * nchunks + c]; ++i) {
    v.x[slot + i] += bx;
    v.y[slot + i] += by;
  }
}

constexpr int kMergeThreads = 256;

// One block per pair of neighbouring hulls: every thread tries some left vertices as the
// bridge, then the block takes the one with the smallest violation (ties: leftmost) and
// writes the merged hull.
__global__ void merge_kernel(
    Vertices in,
    const int32_t* __restrict__ in_cnt,
    int64_t cap_in,
    int64_t nodes_in,
    Vertices out,
    int32_t* __restrict__ out_cnt,
    int64_t cap_out,
    int64_t row_vert,
    int64_t row_cnt) {
  __shared__ double best_v[kMergeThreads];
  __shared__ int32_t best_a[kMergeThreads], best_b[kMergeThreads];
  const int64_t j = blockIdx.x, r = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t ls = r * row_vert + 2 * j * cap_in, os = r * row_vert + j * cap_out;
  const int32_t hl = in_cnt[r * row_cnt + 2 * j];
  if (2 * j + 1 >= nodes_in) {  // no right neighbour: carry the hull over
    for (int32_t k = tid; k < hl; k += kMergeThreads) {
      out.x[os + k] = in.x[ls + k];
      out.y[os + k] = in.y[ls + k];
      out.k[os + k] = in.k[ls + k];
    }
    if (tid == 0) {
      out_cnt[r * row_cnt + j] = hl;
    }
    return;
  }
  const int64_t rs = ls + cap_in;
  const int32_t hr = in_cnt[r * row_cnt + 2 * j + 1];
  const double *lx = in.x + ls, *ly = in.y + ls, *rx = in.x + rs, *ry = in.y + rs;
  int32_t a = hl - 1, b = 0;
  if (needs_bridge(lx, ly, hl, rx, ry, hr)) {
    double v_best = std::numeric_limits<double>::infinity();
    int32_t a_best = 0, b_best = 1;
    for (int32_t k = tid; k + 1 < hl; k += kMergeThreads) {
      int32_t bk;
      const double v = violation(lx, ly, k, rx, ry, hr, &bk);
      if (v < v_best) {
        v_best = v;
        a_best = k;
        b_best = bk;
      }
    }
    best_v[tid] = v_best;
    best_a[tid] = a_best;
    best_b[tid] = b_best;
    __syncthreads();
    for (int stride = kMergeThreads / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        const int o = tid + stride;
        if (best_v[o] < best_v[tid] || (best_v[o] == best_v[tid] && best_a[o] < best_a[tid])) {
          best_v[tid] = best_v[o];
          best_a[tid] = best_a[o];
          best_b[tid] = best_b[o];
        }
      }
      __syncthreads();
    }
    a = best_a[0];
    b = best_b[0];
  }
  const int32_t size = merged_size(a, b, hr);
  for (int32_t k = tid; k < size; k += kMergeThreads) {
    int32_t pos;
    const int64_t src = (merged_source(a, b, k, &pos) ? rs : ls) + pos;
    out.x[os + k] = in.x[src];
    out.y[os + k] = in.y[src];
    out.k[os + k] = in.k[src];
  }
  if (tid == 0) {
    out_cnt[r * row_cnt + j] = size;
  }
}

__global__ void fit_kernel(
    Vertices v,
    const int32_t* __restrict__ cnt,
    const double* __restrict__ centre,
    int64_t n,
    int64_t row_vert,
    int64_t row_cnt,
    double* __restrict__ out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = blockIdx.y;
  if (i >= n) {
    return;
  }
  const int64_t s = r * row_vert;
  out[r * n + i] = fit(v.x + s, v.y + s, v.k + s, cnt[r * row_cnt], i) + centre[r];
}

at::Tensor isotonic_hull_cuda(
    const at::Tensor& y, const at::Tensor& w, const at::Tensor& centre, int64_t chunk) {
  TORCH_CHECK(y.dim() == 2 && y.sizes() == w.sizes(), "y and w must be [batch, n] and alike");
  TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y and w must have the same dtype");
  TORCH_CHECK(y.is_contiguous() && w.is_contiguous(), "y and w must be contiguous");
  TORCH_CHECK(centre.scalar_type() == at::kDouble && centre.numel() == y.size(0), "bad centre");
  TORCH_CHECK(chunk >= 1 && chunk <= 512 && y.size(1) >= 1, "chunk must be in [1, 512], n >= 1");
  TORCH_CHECK(y.size(0) < 65536, "at most 65535 problems at a time");
  TORCH_CHECK(y.size(1) < (int64_t{1} << 31) - 1, "problems must have fewer than 2^31 - 1 elements");
  const c10::cuda::CUDAGuard guard(y.device());
  const int64_t batch = y.size(0), n = y.size(1);
  const int64_t nchunks = (n + chunk - 1) / chunk;
  // The largest level of the merge tree sets the buffer size.
  int64_t row_vert = 0;
  for (int64_t nodes = nchunks, cap = chunk + 1;; nodes = (nodes + 1) / 2, cap = 2 * cap - 1) {
    row_vert = nodes * cap > row_vert ? nodes * cap : row_vert;
    if (nodes == 1) {
      break;
    }
  }
  const auto doubles = y.options().dtype(at::kDouble);
  const auto ints = y.options().dtype(at::kInt);
  at::Tensor vx[2], vy[2], vk[2], cnt[2];
  for (int t = 0; t < 2; ++t) {
    vx[t] = at::empty({batch, row_vert}, doubles);
    vy[t] = at::empty({batch, row_vert}, doubles);
    vk[t] = at::empty({batch, row_vert}, ints);
    cnt[t] = at::empty({batch, nchunks}, ints);
  }
  auto verts = [&](int t) {
    return Vertices{vx[t].data_ptr<double>(), vy[t].data_ptr<double>(), vk[t].data_ptr<int32_t>()};
  };
  at::Tensor total_x = at::empty({batch, nchunks}, doubles);
  at::Tensor total_y = at::empty({batch, nchunks}, doubles);
  const at::Tensor c = centre.contiguous();
  const auto stream = at::cuda::getCurrentCUDAStream();

  const dim3 grid((nchunks + kChunkThreads - 1) / kChunkThreads, batch);
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_hull_cuda", [&] {
    auto launch = [&](auto kernel) {
      kernel<<<grid, kChunkThreads, 0, stream>>>(
          y.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), c.data_ptr<double>(), n, chunk,
          nchunks, verts(0), cnt[0].data_ptr<int32_t>(), total_x.data_ptr<double>(),
          total_y.data_ptr<double>(), chunk + 1, row_vert);
    };
    if (chunk <= 32) {
      launch(chunk_hull_kernel<scalar_t, 32>);
    } else if (chunk <= 128) {
      launch(chunk_hull_kernel<scalar_t, 128>);
    } else {
      launch(chunk_hull_kernel<scalar_t, 512>);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Where each chunk starts: exclusive prefix sums of the chunks' totals (only nchunks of them).
  const at::Tensor base_x = at::cumsum(total_x, 1) - total_x;
  const at::Tensor base_y = at::cumsum(total_y, 1) - total_y;
  shift_kernel<<<dim3((nchunks + 255) / 256, batch), 256, 0, stream>>>(
      verts(0), cnt[0].data_ptr<int32_t>(), base_x.data_ptr<double>(), base_y.data_ptr<double>(),
      nchunks, chunk + 1, row_vert);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int cur = 0;
  for (int64_t nodes = nchunks, cap = chunk + 1; nodes > 1; cur ^= 1) {
    const int64_t out_nodes = (nodes + 1) / 2, out_cap = 2 * cap - 1;
    merge_kernel<<<dim3(out_nodes, batch), kMergeThreads, 0, stream>>>(
        verts(cur), cnt[cur].data_ptr<int32_t>(), cap, nodes, verts(cur ^ 1),
        cnt[cur ^ 1].data_ptr<int32_t>(), out_cap, row_vert, nchunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nodes = out_nodes;
    cap = out_cap;
  }
  at::Tensor out = at::empty({batch, n}, doubles);
  fit_kernel<<<dim3((n + 255) / 256, batch), 256, 0, stream>>>(
      verts(cur), cnt[cur].data_ptr<int32_t>(), c.data_ptr<double>(), n, row_vert, nchunks,
      out.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(torchisotonic, CUDA, m) {
  m.impl("isotonic", &isotonic_cuda);
  m.impl("isotonic_hull", &isotonic_hull_cuda);
}

}  // namespace torchisotonic
