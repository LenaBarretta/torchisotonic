// The package's rounds, with the merges that a thread cannot finish cheaply handed to warps.
// In a round with a thread per merge, a merge whose walk reaches its cap does not search for
// the bridge alone: it leaves its lists untouched and adds itself to a list, and a second
// kernel merges the listed pairs with a warp each, by the package's warp-parallel search. Rounds
// with few merges take a warp per merge, as in the package. Only rows longer than one chunk
// have merges; everything else is the package's.
#include "tiles.cuh"

// A named namespace: see the README.
namespace tiexp_warps_for_capped {

using namespace torchisotonic;

// The package's merge_pair for a thread, but without its search: false (and nothing
// changed) if the bridge needs one.
template <bool W>
__device__ bool merge_pair_walk(Hulls h, const Row<W>& row, int64_t first, int64_t half) {
  Seg* L = h.seg + first;
  const Seg* R = L + half;
  const int nl = h.nseg[first], nr = h.nseg[first + half];

  const Seg last = L[nl - 1], head = R[0];
  const Pt shared = row.at(last, last.hi);
  const int before_s = last.hi > last.lo ? nl - 1 : nl - 2;
  const int before_p = last.hi > last.lo ? last.hi - 1 : L[nl - 2].hi;
  const int after_s = head.hi > head.lo ? 0 : 1;
  const int after_p = head.hi > head.lo ? head.lo + 1 : R[1].lo;
  const Pt before = row.at(L[before_s], before_p), after = row.at(R[after_s], after_p);
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
    if (!walk_segments(row, L, R, nr, nl - 1, last.hi - 1 >= last.lo ? last.hi - 1 : -1, &sa,
                       &pa, &sb, &pb)) {
      return false;
    }
    keep = sa + 1;
    hi_l = pa;
    from = sb;
    lo_r = pb;
  }
  const int moved = nr - from;
  L[keep - 1].hi = hi_l;
  for (int k = 0; k < moved; ++k) {
    Seg s = R[from + k];
    if (k == 0) {
      s.lo = lo_r;
    }
    L[keep + k] = s;
  }
  h.nseg[first] = keep + moved;
  return true;
}

template <bool W>
__device__ __forceinline__ Row<W> row_of(Hulls h, const double* cy, const double* cx, int64_t r,
                                         int64_t per_row) {
  const int64_t ro = r * (per_row + 1);
  return Row<W>{h, cy + ro, W ? cx + ro : nullptr, h.cy_lo + ro, W ? h.cx_lo + ro : nullptr};
}

// A thread per merge; the merges it cannot walk go to `deferred`.
template <bool W>
__global__ void walk_merge_kernel(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row, int64_t half,
    int64_t* __restrict__ deferred, int* __restrict__ count) {
  const int64_t id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs = (per_row + half - 1) / (2 * half);
  if (id >= batch * pairs) {
    return;
  }
  const int64_t r = id / pairs, first = r * per_row + id % pairs * 2 * half;
  if (!merge_pair_walk<W>(h, row_of<W>(h, cy, cx, r, per_row), first, half)) {
    deferred[atomicAdd(count, 1)] = first;
  }
}

// A warp per deferred merge, over the list.
template <bool W>
__global__ void deferred_merge_kernel(
    Hulls h, const double* cy, const double* cx, int64_t per_row, int64_t half,
    const int64_t* __restrict__ deferred, const int* __restrict__ count) {
  const int64_t warps = static_cast<int64_t>(gridDim.x) * blockDim.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kLanes;
       i < *count; i += warps) {
    const int64_t first = deferred[i];
    merge_pair<W, true>(h, row_of<W>(h, cy, cx, first / per_row, per_row), first, half, lane);
  }
}

template <bool W>
void merge_deferred(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row,
    cudaStream_t stream) {
  const auto options = at::TensorOptions().device(at::kCUDA, at::cuda::current_device());
  at::Tensor deferred = at::empty({batch * per_row}, options.dtype(at::kLong));
  at::Tensor count = at::empty({1}, options.dtype(at::kInt));
  int64_t* dp = deferred.data_ptr<int64_t>();
  int* cp = count.data_ptr<int>();
  // Enough warps for every SM, for the deferred merges.
  const int64_t sm_warps =
      static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * 64;
  for (int64_t half = 1; half < per_row; half *= 2) {
    const int64_t merges = batch * ((per_row + half - 1) / (2 * half));
    if (merges >= kThreadMerges) {
      C10_CUDA_CHECK(cudaMemsetAsync(cp, 0, sizeof(int), stream));
      walk_merge_kernel<W><<<(merges + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half, dp, cp);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      deferred_merge_kernel<W><<<blocks_for_warps(std::min(merges, sm_warps)), kThreads, 0,
                                 stream>>>(h, cy, cx, per_row, half, dp, cp);
    } else {
      merge_kernel<W, true><<<blocks_for_warps(merges), kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// As the package's run(): as much shared memory as the SM allows, for the same occupancy.
template <typename scalar_t, bool W, typename Kernel>
void carve(Kernel kernel) {
  static const bool carved = [&] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    return true;
  }();
  (void)carved;
}

at::Tensor isotonic(const at::Tensor& y, const c10::optional<at::Tensor>& w, bool increasing) {
  if (y.size(1) <= 4096) {
    return isotonic_tiles_cuda(y, w, increasing, true, false)[0];  // one chunk per row: nothing to merge
  }
  const c10::cuda::CUDAGuard guard(y.device());
  at::Tensor out = at::empty_like(y);
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "warps_for_capped", [&] {
    auto go = [&](auto weighted) {
      constexpr bool W = decltype(weighted)::value;
      const scalar_t* yp = y.data_ptr<scalar_t>();
      const scalar_t* wp = W ? w->data_ptr<scalar_t>() : nullptr;
      run_chunks<scalar_t, W, Chunk<scalar_t, W, 1>::kN>(
          y, W ? &*w : nullptr, increasing, out, nullptr,
          [&](int64_t chunks, int64_t n, int64_t per_row, scalar_t sign, scalar_t* o, Hulls h,
              cudaStream_t stream) {
            carve<scalar_t, W>(chunk_kernel<scalar_t, W, 1>);
            chunk_kernel<scalar_t, W, 1><<<chunks, kLanes, 0, stream>>>(
                yp, wp, n, per_row, sign, o, h);
          },
          merge_deferred<W>);
    };
    if (w.has_value()) {
      go(std::true_type{});
    } else {
      go(std::false_type{});
    }
  });
  return out;
}

}  // namespace tiexp_warps_for_capped

TORCH_LIBRARY(tiexp_warps_for_capped, m) {
  m.def("isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiexp_warps_for_capped, CUDA, m) {
  m.impl("isotonic", &tiexp_warps_for_capped::isotonic);
}
