// Ablation: a warp for every merge, in every round (the package: a thread per merge in
// rounds with at least kThreadMerges merges, a warp per merge after that).
// Only rows longer than one chunk have merges; everything else is the package's.
#include "tiles.cuh"

// A named namespace: see the README.
namespace tiexp_warp_per_merge {

using namespace torchisotonic;

template <bool W>
void merge_all(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row,
    cudaStream_t stream) {
  for (int64_t half = 1; half < per_row; half *= 2) {
    const int64_t merges = batch * ((per_row + half - 1) / (2 * half));
    if constexpr (true) {
      merge_kernel<W, true><<<blocks_for_warps(merges), kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half);
    } else {
      merge_kernel<W, false><<<(merges + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
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
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "warp_per_merge", [&] {
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
          merge_all<W>);
    };
    if (w.has_value()) {
      go(std::true_type{});
    } else {
      go(std::false_type{});
    }
  });
  return out;
}

}  // namespace tiexp_warp_per_merge

TORCH_LIBRARY(tiexp_warp_per_merge, m) {
  m.def("isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiexp_warp_per_merge, CUDA, m) {
  m.impl("isotonic", &tiexp_warp_per_merge::isotonic);
}
