// The torchisotonic::isotonic operator: its schema and its CPU implementation. The CUDA
// implementation registers itself for the same operator in ops_cuda.cu, so the dispatcher
// picks the right one from the input's device.
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/types.h>

#include <vector>

#include "pava.h"

namespace torchisotonic {

namespace {

// y, w: [batch, n], contiguous, same floating dtype; returns the increasing fit of each row.
at::Tensor isotonic_cpu(const at::Tensor& y, const at::Tensor& w) {
  TORCH_CHECK(y.dim() == 2 && y.sizes() == w.sizes(), "y and w must be [batch, n] and alike");
  TORCH_CHECK(y.is_contiguous() && w.is_contiguous(), "y and w must be contiguous");
  TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y and w must have the same dtype");
  const int64_t batch = y.size(0), n = y.size(1);
  at::Tensor out = at::empty_like(y);
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_cpu", [&] {
    const scalar_t* yp = y.data_ptr<scalar_t>();
    const scalar_t* wp = w.data_ptr<scalar_t>();
    scalar_t* op = out.data_ptr<scalar_t>();
    at::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
      std::vector<int64_t> start(n);
      std::vector<scalar_t> weight(n), total(n);
      for (int64_t b = begin; b < end; ++b) {
        pava<scalar_t, scalar_t>(
            yp + b * n, wp + b * n, op + b * n, n, 1, start.data(), weight.data(), total.data());
      }
    });
  });
  return out;
}

}  // namespace

TORCH_LIBRARY(torchisotonic, m) {
  m.def("isotonic(Tensor y, Tensor w) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchisotonic, CPU, m) {
  m.impl("isotonic", &isotonic_cpu);
}

}  // namespace torchisotonic
