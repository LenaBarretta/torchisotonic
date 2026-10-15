// torchisotonic::isotonic_tiles and isotonic_grad on CUDA, for every shape of problem. The
// kernels and how they work are in tiles.cuh, and those of the gradient in grad.cuh.
#include "grad.cuh"
#include "tiles.cuh"

namespace torchisotonic {

TORCH_LIBRARY_IMPL(torchisotonic, CUDA, m) {
  m.impl("isotonic_tiles", &isotonic_tiles_cuda);
  m.impl("isotonic_grad", &isotonic_grad_cuda);
}

}  // namespace torchisotonic
