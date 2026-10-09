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

}  // namespace

TORCH_LIBRARY_IMPL(torchisotonic, CUDA, m) {
  m.impl("isotonic", &isotonic_cuda);
}

}  // namespace torchisotonic
