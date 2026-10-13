// torchisotonic::isotonic_tiles and isotonic_grad on CUDA, for every shape of problem. The
// kernels and how they work are in tiles.cuh, and those of the gradient in grad.cuh.
#include "grad.cuh"
#include "tiles.cuh"

namespace torchisotonic {

namespace {

#ifdef TI_PHASES
// The cycles counted so far per phase (then reset), on the CPU; then, for the float,
// unweighted chunk_kernel: registers per thread, local memory bytes per thread, static shared
// memory bytes per block, warps per block and resident blocks per SM.
at::Tensor tiles_phases(const at::Tensor& like, int64_t stop) {
  const c10::cuda::CUDAGuard guard(like.device());
  at::Tensor out = at::zeros({16 + 2 + 3 * kStatRounds}, at::kLong);
  int64_t* o = out.data_ptr<int64_t>();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(o, phase_cycles, sizeof(phase_cycles)));
  const unsigned long long zeros[8] = {};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(phase_cycles, zeros, sizeof(zeros)));
  const int cut = static_cast<int>(stop);
  C10_CUDA_CHECK(cudaMemcpyToSymbol(phase_stop, &cut, sizeof(cut)));
  constexpr int warps = chunk_warps<float, false>();
  const void* kernel = reinterpret_cast<const void*>(&chunk_kernel<float, false, warps>);
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, warps * kLanes, 0));
  // [16], [17]: merges, and merges whose walk gave up; [18 + 3 r ..]: round r's nodes,
  // descriptors, vertices.
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(o + 16, merge_counts, sizeof(merge_counts)));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(o + 18, merge_level_stats, sizeof(merge_level_stats)));
  const unsigned long long zero_counts[2] = {};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(merge_counts, zero_counts, sizeof(zero_counts)));
  static const unsigned long long zero_levels[kStatRounds][3] = {};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(merge_level_stats, zero_levels, sizeof(zero_levels)));
  o[8] = attr.numRegs;
  o[9] = static_cast<int64_t>(attr.localSizeBytes);
  o[10] = static_cast<int64_t>(attr.sharedSizeBytes);
  o[11] = warps;
  o[12] = blocks;
  return out;
}
#endif

}  // namespace

#ifdef TI_PHASES
TORCH_LIBRARY_FRAGMENT(torchisotonic, m) {
  m.def("tiles_phases(Tensor like, int stop) -> Tensor");
}
#endif

TORCH_LIBRARY_IMPL(torchisotonic, CUDA, m) {
  m.impl("isotonic_tiles", &isotonic_tiles_cuda);
  m.impl("isotonic_grad", &isotonic_grad_cuda);
#ifdef TI_PHASES
  m.impl("tiles_phases", &tiles_phases);
#endif
}

}  // namespace torchisotonic
