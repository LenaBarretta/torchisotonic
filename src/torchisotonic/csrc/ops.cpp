// The operators' schemas, and torchisotonic::isotonic on the CPU: pool adjacent violators,
// rows in parallel. The CUDA operators are in ops_tiles.cu.
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/types.h>

#include <vector>

#include "pava.h"

namespace torchisotonic {

namespace {

// y, w: [batch, n], contiguous, same floating dtype. Returns the increasing fit of each row
// and, if with_starts, a bool [batch, n] marking where its blocks start (else empty).
std::vector<at::Tensor> isotonic_cpu(const at::Tensor& y, const at::Tensor& w, bool with_starts) {
  TORCH_CHECK(y.dim() == 2 && y.sizes() == w.sizes(), "y and w must be [batch, n] and alike");
  TORCH_CHECK(y.is_contiguous() && w.is_contiguous(), "y and w must be contiguous");
  TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y and w must have the same dtype");
  const int64_t batch = y.size(0), n = y.size(1);
  at::Tensor out = at::empty_like(y);
  at::Tensor starts = with_starts ? at::empty(y.sizes(), y.options().dtype(at::kBool))
                                  : at::empty({0}, y.options().dtype(at::kBool));
  uint8_t* sp = with_starts ? reinterpret_cast<uint8_t*>(starts.data_ptr<bool>()) : nullptr;
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_cpu", [&] {
    const scalar_t* yp = y.data_ptr<scalar_t>();
    const scalar_t* wp = w.data_ptr<scalar_t>();
    scalar_t* op = out.data_ptr<scalar_t>();
    at::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
      std::vector<int64_t> start(n);
      std::vector<scalar_t> weight(n), total(n);
      for (int64_t b = begin; b < end; ++b) {
        pava<scalar_t, scalar_t, int64_t>(
            yp + b * n, wp + b * n, op + b * n, sp ? sp + b * n : nullptr, n, 1, start.data(),
            weight.data(), total.data());
      }
    });
  });
  return {out, starts};
}

}  // namespace

TORCH_LIBRARY(torchisotonic, m) {
  m.def("isotonic(Tensor y, Tensor w, bool with_starts=False) -> Tensor[]");
  m.def(
      "isotonic_tiles(Tensor y, Tensor? w, bool increasing, bool exact=True, "
      "bool with_starts=False) -> Tensor[]");
  m.def("isotonic_grad(Tensor out, Tensor starts, Tensor g, Tensor? w, Tensor? y) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(torchisotonic, CPU, m) {
  m.impl("isotonic", &isotonic_cpu);
}

}  // namespace torchisotonic
