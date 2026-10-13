// The torchisotonic::isotonic operator: its schema and its CPU implementation. The CUDA
// implementation registers itself for the same operator in ops_cuda.cu, so the dispatcher
// picks the right one from the input's device.
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/types.h>

#include <limits>
#include <vector>

#include "hull.h"
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
        pava<scalar_t, scalar_t, int64_t>(
            yp + b * n, wp + b * n, op + b * n, n, 1, start.data(), weight.data(), total.data());
      }
    });
  });
  return out;
}

// A hull: its vertices' coordinates and point indices.
struct Hull {
  std::vector<double> x, y;
  std::vector<int32_t> k;
  int32_t size() const { return static_cast<int32_t>(k.size()); }
};

Hull merge_hulls(const Hull& l, const Hull& r) {
  const int32_t hl = l.size(), hr = r.size();
  int32_t a = hl - 1, b = 0;
  if (needs_bridge(l.x.data(), l.y.data(), hl, r.x.data(), r.y.data(), hr)) {
    double best = std::numeric_limits<double>::infinity();
    for (int32_t c = 0; c + 1 < hl; ++c) {
      int32_t bc;
      const double v = violation(l.x.data(), l.y.data(), c, r.x.data(), r.y.data(), hr, &bc);
      if (v < best) {
        best = v;
        a = c;
        b = bc;
      }
    }
  }
  const int32_t size = merged_size(a, b, hr);
  Hull m{std::vector<double>(size), std::vector<double>(size), std::vector<int32_t>(size)};
  for (int32_t v = 0; v < size; ++v) {
    int32_t pos;
    const Hull& src = merged_source(a, b, v, &pos) ? r : l;
    m.x[v] = src.x[pos];
    m.y[v] = src.y[pos];
    m.k[v] = src.k[pos];
  }
  return m;
}

// The hull algorithm, one step after another (the CUDA version runs the same steps in
// parallel). y, w: [batch, n]; centre: [batch] float64, subtracted from y first (the fit
// commutes with that shift, and it keeps the cumulative sums small). Returns the fit, float64.
at::Tensor isotonic_hull_cpu(
    const at::Tensor& y, const at::Tensor& w, const at::Tensor& centre, int64_t chunk) {
  TORCH_CHECK(y.dim() == 2 && y.sizes() == w.sizes(), "y and w must be [batch, n] and alike");
  TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y and w must have the same dtype");
  TORCH_CHECK(y.is_contiguous() && w.is_contiguous(), "y and w must be contiguous");
  TORCH_CHECK(centre.scalar_type() == at::kDouble && centre.numel() == y.size(0), "bad centre");
  TORCH_CHECK(chunk >= 1 && y.size(1) >= 1, "chunk must be >= 1 and n >= 1");
  const int64_t batch = y.size(0), n = y.size(1);
  at::Tensor out = at::empty({batch, n}, y.options().dtype(at::kDouble));
  const at::Tensor c = centre.contiguous();
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "isotonic_hull_cpu", [&] {
    at::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
      for (int64_t r = begin; r < end; ++r) {
        const scalar_t* yr = y.data_ptr<scalar_t>() + r * n;
        const scalar_t* wr = w.data_ptr<scalar_t>() + r * n;
        const double cr = c.data_ptr<double>()[r];
        std::vector<Hull> hulls;
        double base_x = 0, base_y = 0;  // where this chunk starts
        for (int64_t lo = 0; lo < n; lo += chunk) {
          const int64_t hi = lo + chunk < n ? lo + chunk : n;
          const int64_t points = hi - lo + 1;
          Hull h{std::vector<double>(points), std::vector<double>(points),
                 std::vector<int32_t>(points)};
          int32_t count = 0;
          double x = 0, s = 0;  // local cumulative sums
          hull_push(h.x.data(), h.y.data(), h.k.data(), count, 0.0, 0.0, int32_t(lo));
          for (int64_t i = lo; i < hi; ++i) {
            x += static_cast<double>(wr[i]);
            s += static_cast<double>(wr[i]) * (static_cast<double>(yr[i]) - cr);
            hull_push(h.x.data(), h.y.data(), h.k.data(), count, x, s, int32_t(i + 1));
          }
          h.x.resize(count);
          h.y.resize(count);
          h.k.resize(count);
          for (int32_t v = 0; v < count; ++v) {
            h.x[v] += base_x;
            h.y[v] += base_y;
          }
          base_x += x;
          base_y += s;
          hulls.push_back(std::move(h));
        }
        while (hulls.size() > 1) {
          std::vector<Hull> next;
          for (size_t j = 0; j < hulls.size(); j += 2) {
            next.push_back(j + 1 == hulls.size() ? std::move(hulls[j])
                                                 : merge_hulls(hulls[j], hulls[j + 1]));
          }
          hulls = std::move(next);
        }
        const Hull& hull = hulls[0];
        double* o = out.data_ptr<double>() + r * n;
        for (int64_t i = 0; i < n; ++i) {
          o[i] = fit(hull.x.data(), hull.y.data(), hull.k.data(), hull.size(), i) + cr;
        }
      }
    });
  });
  return out;
}

}  // namespace

TORCH_LIBRARY(torchisotonic, m) {
  m.def("isotonic(Tensor y, Tensor w) -> Tensor");
  m.def("isotonic_hull(Tensor y, Tensor w, Tensor centre, int chunk) -> Tensor");
  m.def("isotonic_tiles(Tensor y, Tensor? w, bool increasing, bool exact=True) -> Tensor");
  m.def("isotonic_grad(Tensor out, Tensor g, Tensor? w, Tensor? y) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(torchisotonic, CPU, m) {
  m.impl("isotonic", &isotonic_cpu);
  m.impl("isotonic_hull", &isotonic_hull_cpu);
}

}  // namespace torchisotonic
