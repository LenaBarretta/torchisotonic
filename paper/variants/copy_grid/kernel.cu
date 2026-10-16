// The control `copy_per_merge` with its copies made by the whole GPU. A round of merges is two
// kernels: bridge_merge_kernel finds every merge's bridge (the same walk and search, by the
// same thread-or-warp rule) and writes how much of each side survives; copy_merge_kernel then
// writes every merged hull into a second buffer, L's vertices up to the bridge and R's from
// it, with all of the grid's blocks: many merges per block in the first rounds, many blocks per
// merge in the last ones, each thread a vertex at a time, coalesced. The buffers swap after
// every round, so a round moves every surviving vertex once, at the memory's bandwidth; the
// segment lists move none. The rest is `copy_per_merge`: the package's chunk_kernel, the
// coordinates of a vertex's own chunk with the chunk's start added at every access, and the
// fit from one vertex array per problem (flat_fit_kernel).
#include "tiles.cuh"

// A named namespace: see the README.
namespace tiexp_copy_grid {

using namespace torchisotonic;

struct Flat {  // a problem's buffer of vertices (slots of chunk + 1 per chunk)
  const double* x;
  const double* y;
  const int32_t* p;
  const int32_t* c;  // the vertex's chunk
  const double* cy;  // the problem's chunk starts
  const double* cx;
};

struct Buffer {  // one of the two vertex buffers
  double* x;
  double* y;
  int32_t* p;
  int32_t* c;
};

template <bool W>
__device__ __forceinline__ Pt at(const Flat& f, int64_t i) {
  return Pt{W ? f.cx[f.c[i]] + f.x[i] : static_cast<double>(f.p[i]), f.cy[f.c[i]] + f.y[i]};
}

// Where the lower tangent from p touches R's vertices 1..nr-1.
template <bool W>
__device__ int tangent_flat(const Flat& R, int nr, const Pt& p) {
  int lo = 1, hi = nr - 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (turn(p, at<W>(R, mid), at<W>(R, mid + 1)) < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// The bridge by walking from the shared point (L[nl-1] == R[0]); false after kSegWalk steps.
template <bool W>
__device__ bool walk_flat(const Flat& L, int nl, const Flat& R, int nr, int* a, int* b) {
  int ia = nl - 2, ib = 1;
  Pt pa = at<W>(L, ia), pb = at<W>(R, ib);
  int steps = 0;
  bool moved = true;
  while (moved) {
    moved = false;
    while (ib < nr - 1) {
      const Pt q = at<W>(R, ib + 1);
      if (turn(pa, pb, q) > 0) {
        break;
      }
      ++ib;
      pb = q;
      moved = true;
      if (++steps > kSegWalk) {
        return false;
      }
    }
    while (ia > 0) {
      const Pt q = at<W>(L, ia - 1);
      if (turn(q, pa, pb) > 0) {
        break;
      }
      --ia;
      pa = q;
      moved = true;
      if (++steps > kSegWalk) {
        return false;
      }
    }
  }
  *a = ia;
  *b = ib;
  return true;
}

// The bridge by search: the first L vertex whose successor is not below its tangent to R.
template <bool W, bool WARP>
__device__ void search_flat(
    const Flat& L, int nl, const Flat& R, int nr, int lane, int* a, int* b) {
  const int i = first_false<WARP>(nl - 1, lane, [&](int i) {
    const Pt p = at<W>(L, i);
    return turn(p, at<W>(R, tangent_flat<W>(R, nr, p)), at<W>(L, i + 1)) < 0;
  });
  *a = i < nl - 1 ? i : nl - 2;
  *b = tangent_flat<W>(R, nr, at<W>(L, *a));
}

// The bridge of the merge of the node of chunks [first, first + half) with the one of
// [first + half, first + 2 half): L keeps its vertices 0..a, R its b..nr-1. Writes a + 1 to
// kept[first], b to from[first] and the merged count to cnt[first].
template <bool W, bool WARP>
__global__ void bridge_merge_kernel(
    Buffer v, const double* cy, const double* cx, int32_t* cnt, int32_t* kept, int32_t* from,
    int64_t slots, int64_t batch, int64_t per_row, int64_t half) {
  const int64_t id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
                     (WARP ? kLanes : 1);
  const int lane = WARP ? threadIdx.x % kLanes : 0;
  const int64_t pairs = (per_row + half - 1) / (2 * half);
  if (id >= batch * pairs) {
    return;
  }
  const int64_t r = id / pairs, first = r * per_row + id % pairs * 2 * half;
  const double* rcy = cy + r * (per_row + 1);
  const double* rcx = W ? cx + r * (per_row + 1) : nullptr;
  const int64_t ol = first * slots, orr = (first + half) * slots;
  const Flat L{W ? v.x + ol : nullptr, v.y + ol, v.p + ol, v.c + ol, rcy, rcx};
  const Flat R{W ? v.x + orr : nullptr, v.y + orr, v.p + orr, v.c + orr, rcy, rcx};
  const int nl = cnt[first], nr = cnt[first + half];
  int a = nl - 1, b = 1;  // convex at the shared point: concatenate, the shared point once
  if (turn(at<W>(L, nl - 2), at<W>(L, nl - 1), at<W>(R, 1)) <= 0) {
    bool walked = false;
    if (lane == 0) {
      walked = walk_flat<W>(L, nl, R, nr, &a, &b);
    }
    if (WARP) {
      walked = __shfl_sync(kFull, walked, 0);
      a = __shfl_sync(kFull, a, 0);
      b = __shfl_sync(kFull, b, 0);
    }
    if (!walked) {
      search_flat<W, WARP>(L, nl, R, nr, lane, &a, &b);
    }
  }
  if (lane == 0) {
    kept[first] = a + 1;
    from[first] = b;
    cnt[first] = a + 1 + nr - b;
  }
}

// Every node of the round, merged or left alone (the last of a row without a partner),
// written from src into dst at its own place. Block j of `grid` takes the nodes j, j + grid,
// ... when there are at least `grid` nodes, else one share of one node.
template <bool W>
__global__ void __launch_bounds__(kThreads) copy_merge_kernel(
    Buffer src, Buffer dst, const int32_t* cnt, const int32_t* kept, const int32_t* from,
    int64_t slots, int64_t batch, int64_t per_row, int64_t half) {
  const int64_t per = (per_row + 2 * half - 1) / (2 * half), nodes = batch * per;
  const int64_t grid = gridDim.x;
  const int64_t shares = nodes >= grid ? 1 : grid / nodes;
  const int64_t share = nodes >= grid ? 0 : blockIdx.x % shares;
  for (int64_t node = nodes >= grid ? blockIdx.x : blockIdx.x / shares; node < nodes;
       node += nodes >= grid ? grid : nodes) {
    const int64_t r = node / per, j = node % per * 2 * half, first = r * per_row + j;
    const bool merged = j + half < per_row;
    const int total = cnt[first];
    const int left = merged ? kept[first] : total;
    const int64_t right = merged ? (first + half) * slots + from[first] - left : 0;
    const int64_t size = (total + shares - 1) / shares;
    const int64_t k0 = share * size, k1 = k0 + size < total ? k0 + size : total;
    for (int64_t k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
      const int64_t s = k < left ? first * slots + k : right + k, d = first * slots + k;
      dst.y[d] = src.y[s];
      dst.p[d] = src.p[s];
      dst.c[d] = src.c[s];
      if (W) {
        dst.x[d] = src.x[s];
      }
    }
    if (nodes < grid) {
      break;
    }
  }
}

// After chunk_kernel: every slot's chunk id, and each chunk's vertex count into cnt (= h.nseg,
// which chunk_kernel set to 1). Coordinates stay the chunks' own.
__global__ void flat_init_kernel(Hulls h, int32_t* vc, int64_t slots, int64_t chunks, int64_t per_row) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= chunks * slots) {
    return;
  }
  const int64_t g = i / slots, k = i % slots;
  vc[i] = static_cast<int32_t>(g % per_row);
  if (k == 0) {
    h.nseg[g] = h.seg[g].hi + 1;
  }
}

// The fit from one vertex array per problem: a block per kFitSpan elements.
template <typename scalar_t, bool W>
__global__ void __launch_bounds__(kThreads) flat_fit_kernel(
    Buffer v, const int32_t* cnt, const int32_t* row_nan, const double* cy, const double* cx,
    int64_t slots, int64_t n, int64_t per_row, scalar_t sign, scalar_t* __restrict__ out) {
  using acc_t = scalar_t;
  __shared__ int32_t at_[kFitSpan + 2];
  __shared__ acc_t slope[kFitSpan + 2];
  const int64_t spans = (n + kFitSpan - 1) / kFitSpan;
  const int64_t r = blockIdx.x / spans, c0 = (blockIdx.x % spans) * kFitSpan;
  const int len = static_cast<int>(n - c0 < kFitSpan ? n - c0 : kFitSpan);
  const int64_t start = r * n + c0;
  if (row_nan[r]) {
    for (int e = threadIdx.x; e < len; e += blockDim.x) {
      out[start + e] = std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return;
  }
  const int64_t vb = r * per_row * slots;
  const Flat f{W ? v.x + vb : nullptr, v.y + vb, v.p + vb, v.c + vb, cy + r * (per_row + 1),
               W ? cx + r * (per_row + 1) : nullptr};
  const int count = cnt[r * per_row];
  const int32_t p0 = static_cast<int32_t>(c0), p1 = p0 + len;
  int lo = 0, hi = count;  // the first vertex after p0
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (v.p[vb + mid] <= p0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int first = lo - 1;  // the vertex at or before p0 (vertex 0 is point 0)
  hi = count;
  while (lo < hi) {  // the first vertex at or past p1 (the end point is one)
    const int mid = (lo + hi) / 2;
    if (v.p[vb + mid] < p1) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int m = lo - first + 1;  // vertices first .. lo
  for (int u = threadIdx.x; u < m; u += blockDim.x) {
    const int64_t i = first + u;
    at_[u] = v.p[vb + i];
    if (u + 1 < m) {
      const Pt a = at<W>(f, i), b = at<W>(f, i + 1);
      slope[u] = static_cast<acc_t>((b.y - a.y) / (b.x - a.x));
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < len; e += blockDim.x) {
    const int32_t p = p0 + e;
    int a = 0, b = m - 1;  // the last vertex at or before p
    while (a < b) {
      const int mid = (a + b + 1) / 2;
      if (at_[mid] <= p) {
        a = mid;
      } else {
        b = mid - 1;
      }
    }
    out[start + e] = sign * slope[a];
  }
}

// As the package's run(): as much shared memory as the SM allows, for the same occupancy.
template <typename scalar_t, bool W>
void carve() {
  static const bool carved = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(chunk_kernel<scalar_t, W, 1>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    return true;
  }();
  (void)carved;
}

at::Tensor isotonic(const at::Tensor& y, const c10::optional<at::Tensor>& w, bool increasing) {
  if (y.size(1) <= 4096) {
    return isotonic_tiles_cuda(y, w, increasing, true, false)[0];  // one chunk per problem: nothing to merge
  }
  const c10::cuda::CUDAGuard guard(y.device());
  at::Tensor out = at::empty_like(y);
  // Enough blocks of kThreads to fill every SM.
  const int64_t grid = static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) *
                       (2048 / kThreads);
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "copy_grid", [&] {
    auto go = [&](auto weighted) {
      constexpr bool W = decltype(weighted)::value;
      constexpr int chunk = Chunk<scalar_t, W, 1>::kN;
      constexpr int64_t slots = chunk + 1;
      const scalar_t* yp = y.data_ptr<scalar_t>();
      const scalar_t* wp = W ? w->data_ptr<scalar_t>() : nullptr;
      const int64_t chunks = y.size(0) * ((y.size(1) + chunk - 1) / chunk), all = chunks * slots;
      const auto ints = y.options().dtype(at::kInt), doubles = y.options().dtype(at::kDouble);
      at::Tensor c0 = at::empty({all}, ints), c1 = at::empty({all}, ints);
      at::Tensor p1 = at::empty({all}, ints), y1 = at::empty({all}, doubles);
      at::Tensor x1 = W ? at::empty({all}, doubles) : at::Tensor();
      at::Tensor kept = at::empty({chunks}, ints), from = at::empty({chunks}, ints);
      Buffer cur{}, other{nullptr, y1.data_ptr<double>(), p1.data_ptr<int32_t>(),
                          c1.data_ptr<int32_t>()};
      if (W) {
        other.x = x1.data_ptr<double>();
      }
      run_chunks<scalar_t, W, chunk>(
          y, W ? &*w : nullptr, increasing, out, nullptr,
          [&](int64_t chunks, int64_t n, int64_t per_row, scalar_t sign, scalar_t* o, Hulls h,
              cudaStream_t stream) {
            carve<scalar_t, W>();
            chunk_kernel<scalar_t, W, 1><<<chunks, kLanes, 0, stream>>>(
                yp, wp, n, per_row, sign, o, h);
          },
          [&](Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row,
              cudaStream_t stream) {
            cur = Buffer{h.x, h.y, h.p, c0.data_ptr<int32_t>()};
            flat_init_kernel<<<(chunks * slots + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                h, cur.c, slots, chunks, per_row);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            int32_t* kp = kept.data_ptr<int32_t>();
            int32_t* fp = from.data_ptr<int32_t>();
            for (int64_t half = 1; half < per_row; half *= 2) {
              const int64_t merges = batch * ((per_row + half - 1) / (2 * half));
              if (merges >= kThreadMerges) {
                bridge_merge_kernel<W, false><<<(merges + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    cur, cy, cx, h.nseg, kp, fp, slots, batch, per_row, half);
              } else {
                bridge_merge_kernel<W, true><<<blocks_for_warps(merges), kThreads, 0, stream>>>(
                    cur, cy, cx, h.nseg, kp, fp, slots, batch, per_row, half);
              }
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              copy_merge_kernel<W><<<grid, kThreads, 0, stream>>>(
                  cur, other, h.nseg, kp, fp, slots, batch, per_row, half);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              std::swap(cur, other);
            }
          },
          [&](Hulls h, const double* cy, const double* cx, int64_t batch, int64_t n,
              int64_t per_row, scalar_t sign, scalar_t* o, cudaStream_t stream) {
            const int64_t spans = (n + kFitSpan - 1) / kFitSpan;
            flat_fit_kernel<scalar_t, W><<<batch * spans, kThreads, 0, stream>>>(
                cur, h.nseg, h.row_nan, cy, cx, slots, n, per_row, sign, o);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    };
    if (w.has_value()) {
      go(std::true_type{});
    } else {
      go(std::false_type{});
    }
  });
  return out;
}

}  // namespace tiexp_copy_grid

TORCH_LIBRARY(tiexp_copy_grid, m) {
  m.def("isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiexp_copy_grid, CUDA, m) {
  m.impl("isotonic", &tiexp_copy_grid::isotonic);
}
