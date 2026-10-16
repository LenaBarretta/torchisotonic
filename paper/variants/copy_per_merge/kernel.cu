// The control for the segment lists: after every merge the merged hull is written out as one
// array of vertices (L's up to the bridge, then R's from it: the right part moves down), so a
// merge costs vertices, not segments. Everything else is the package's: chunk_kernel for the
// chunks' hulls, the same bridge search (a walk from the shared point, a search when it takes
// over kSegWalk steps), the same rounds and the same thread-or-warp-per-merge rule, and the
// same coordinates: a vertex keeps its chunk's coordinates and its chunk id, and the chunk's
// start is added at every access, exactly as the package's Row::at does. The one other
// difference is the fit at the end: from one array per problem (flat_fit_kernel) instead of
// per chunk from its slots and segment; it is timed as its own stage.
#include "tiles.cuh"

// A named namespace: see the README.
namespace tiexp_copy_per_merge {

using namespace torchisotonic;

struct Flat {  // a problem's buffer of vertices (slots of chunk + 1 per chunk)
  const double* x;
  const double* y;
  const int32_t* p;
  const int32_t* c;  // the vertex's chunk
  const double* cy;  // the problem's chunk starts
  const double* cx;
};

template <bool W>
__device__ __forceinline__ Pt at(const Flat& f, int64_t i) {
  return Pt{W ? f.cx[f.c[i]] + f.x[i] : static_cast<double>(f.p[i]), f.cy[f.c[i]] + f.y[i]};
}

// Where the lower tangent from p touches R's vertices 1..nr-1: as the package's tangent, on
// an array.
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

// Merges the node of chunks [first, first + half) with the one of [first + half, first + 2
// half) of a problem; vertex counts in cnt, by first chunk. The merged array goes where the
// left one was (the right part moves down, over the left's tail and its own gap: every
// destination is before its source).
template <bool W, bool WARP>
__device__ void merge_flat(
    double* vx, double* vy, int32_t* vp, int32_t* vc, const double* cy, const double* cx,
    int32_t* cnt, int64_t slots, int64_t first, int64_t half, int lane) {
  const int64_t ol = first * slots, orr = (first + half) * slots;
  const Flat L{W ? vx + ol : nullptr, vy + ol, vp + ol, vc + ol, cy, cx};
  const Flat R{W ? vx + orr : nullptr, vy + orr, vp + orr, vc + orr, cy, cx};
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
  const int moved = nr - b;
  if constexpr (WARP) {
    __syncwarp();
    for (int base = 0; base < moved; base += kLanes) {
      const int k = base + lane;
      double x = 0, y = 0;
      int32_t p = 0, c = 0;
      if (k < moved) {
        y = vy[orr + b + k];
        p = vp[orr + b + k];
        c = vc[orr + b + k];
        if (W) {
          x = vx[orr + b + k];
        }
      }
      __syncwarp();
      if (k < moved) {
        vy[ol + a + 1 + k] = y;
        vp[ol + a + 1 + k] = p;
        vc[ol + a + 1 + k] = c;
        if (W) {
          vx[ol + a + 1 + k] = x;
        }
      }
      __syncwarp();
    }
  } else {
    for (int k = 0; k < moved; ++k) {
      vy[ol + a + 1 + k] = vy[orr + b + k];
      vp[ol + a + 1 + k] = vp[orr + b + k];
      vc[ol + a + 1 + k] = vc[orr + b + k];
      if (W) {
        vx[ol + a + 1 + k] = vx[orr + b + k];
      }
    }
  }
  if (lane == 0) {
    cnt[first] = a + 1 + moved;
  }
}

template <bool W, bool WARP>
__global__ void flat_merge_kernel(
    double* vx, double* vy, int32_t* vp, int32_t* vc, const double* cy, const double* cx,
    int32_t* cnt, int64_t slots, int64_t batch, int64_t per_row, int64_t half) {
  const int64_t id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
                     (WARP ? kLanes : 1);
  const int lane = WARP ? threadIdx.x % kLanes : 0;
  const int64_t pairs = (per_row + half - 1) / (2 * half);
  if (id >= batch * pairs) {
    return;
  }
  const int64_t r = id / pairs;
  merge_flat<W, WARP>(vx, vy, vp, vc, cy + r * (per_row + 1), W ? cx + r * (per_row + 1) : nullptr,
                      cnt, slots, r * per_row + id % pairs * 2 * half, half, lane);
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
    Hulls h, const int32_t* vc, const double* cy, const double* cx, int64_t slots, int64_t n,
    int64_t per_row, scalar_t sign, scalar_t* __restrict__ out) {
  using acc_t = scalar_t;
  __shared__ int32_t at_[kFitSpan + 2];
  __shared__ acc_t slope[kFitSpan + 2];
  const int64_t spans = (n + kFitSpan - 1) / kFitSpan;
  const int64_t r = blockIdx.x / spans, c0 = (blockIdx.x % spans) * kFitSpan;
  const int len = static_cast<int>(n - c0 < kFitSpan ? n - c0 : kFitSpan);
  const int64_t start = r * n + c0;
  if (h.row_nan[r]) {
    for (int e = threadIdx.x; e < len; e += blockDim.x) {
      out[start + e] = std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return;
  }
  const int64_t vb = r * per_row * slots;
  const Flat f{W ? h.x + vb : nullptr, h.y + vb, h.p + vb, vc + vb, cy + r * (per_row + 1),
               W ? cx + r * (per_row + 1) : nullptr};
  const int count = h.nseg[r * per_row];
  const int32_t p0 = static_cast<int32_t>(c0), p1 = p0 + len;
  int lo = 0, hi = count;  // the first vertex after p0
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (h.p[vb + mid] <= p0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int first = lo - 1;  // the vertex at or before p0 (vertex 0 is point 0)
  hi = count;
  while (lo < hi) {  // the first vertex at or past p1 (the end point is one)
    const int mid = (lo + hi) / 2;
    if (h.p[vb + mid] < p1) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int m = lo - first + 1;  // vertices first .. lo
  for (int u = threadIdx.x; u < m; u += blockDim.x) {
    const int64_t i = first + u;
    at_[u] = h.p[vb + i];
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
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "copy_per_merge", [&] {
    auto go = [&](auto weighted) {
      constexpr bool W = decltype(weighted)::value;
      constexpr int chunk = Chunk<scalar_t, W, 1>::kN;
      constexpr int64_t slots = chunk + 1;
      const scalar_t* yp = y.data_ptr<scalar_t>();
      const scalar_t* wp = W ? w->data_ptr<scalar_t>() : nullptr;
      const int64_t all = y.size(0) * ((y.size(1) + chunk - 1) / chunk) * slots;
      at::Tensor chunk_ids = at::empty({all}, y.options().dtype(at::kInt));
      int32_t* vc = chunk_ids.data_ptr<int32_t>();
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
            const int64_t chunks = batch * per_row;
            flat_init_kernel<<<(chunks * slots + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                h, vc, slots, chunks, per_row);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            for (int64_t half = 1; half < per_row; half *= 2) {
              const int64_t merges = batch * ((per_row + half - 1) / (2 * half));
              if (merges >= kThreadMerges) {
                flat_merge_kernel<W, false><<<(merges + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    h.x, h.y, h.p, vc, cy, cx, h.nseg, slots, batch, per_row, half);
              } else {
                flat_merge_kernel<W, true><<<blocks_for_warps(merges), kThreads, 0, stream>>>(
                    h.x, h.y, h.p, vc, cy, cx, h.nseg, slots, batch, per_row, half);
              }
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
          },
          [&](Hulls h, const double* cy, const double* cx, int64_t batch, int64_t n,
              int64_t per_row, scalar_t sign, scalar_t* o, cudaStream_t stream) {
            const int64_t spans = (n + kFitSpan - 1) / kFitSpan;
            flat_fit_kernel<scalar_t, W><<<batch * spans, kThreads, 0, stream>>>(
                h, vc, cy, cx, slots, n, per_row, sign, o);
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

}  // namespace tiexp_copy_per_merge

TORCH_LIBRARY(tiexp_copy_per_merge, m) {
  m.def("isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiexp_copy_per_merge, CUDA, m) {
  m.impl("isotonic", &tiexp_copy_per_merge::isotonic);
}
