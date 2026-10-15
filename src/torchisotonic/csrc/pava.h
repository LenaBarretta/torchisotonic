// Pool adjacent violators (PAVA) for one weighted, increasing isotonic regression problem,
// on the CPU. One problem's elements are `stride` apart in memory.
#pragma once

#include <cmath>
#include <cstdint>

namespace torchisotonic {

// Solves min sum_i w_i (x_i - y_i)^2 subject to x_0 <= x_1 <= ... <= x_{n-1}.
//
// y, w: the problem; out: the solution (may alias y); starts, if not null: 1 where a block
// of the solution starts, 0 elsewhere. The three scratch arrays hold the stack of blocks found
// so far, with `stride` like the data: their start index, total weight and weighted sum. A
// NaN anywhere in y makes the whole solution NaN.
template <typename scalar_t, typename acc_t, typename index_t>
void pava(
    const scalar_t* y,
    const scalar_t* w,
    scalar_t* out,
    uint8_t* starts,
    int64_t n,
    int64_t stride,
    index_t* start,
    acc_t* weight,
    acc_t* total) {
  int64_t top = -1;  // index of the last block on the stack
  for (int64_t i = 0; i < n; ++i) {
    acc_t yi = static_cast<acc_t>(y[i * stride]);
    if (yi != yi) {  // NaN: no meaningful fit for this problem
      for (int64_t j = 0; j < n; ++j) {
        out[j * stride] = static_cast<scalar_t>(yi);
        if (starts) {
          starts[j * stride] = j == 0;
        }
      }
      return;
    }
    acc_t wi = static_cast<acc_t>(w[i * stride]);
    ++top;
    start[top * stride] = static_cast<index_t>(i);
    weight[top * stride] = wi;
    total[top * stride] = wi * yi;
    // Merge while the previous block's mean is not below this one's. Comparing
    // total_a / weight_a >= total_b / weight_b without dividing (weights are positive).
    while (top > 0) {
      acc_t wa = weight[(top - 1) * stride], ta = total[(top - 1) * stride];
      acc_t wb = weight[top * stride], tb = total[top * stride];
      if (ta * wb < tb * wa) {
        break;
      }
      weight[(top - 1) * stride] = wa + wb;
      total[(top - 1) * stride] = ta + tb;
      --top;
    }
  }
  // Every element takes its block's mean.
  for (int64_t k = top; k >= 0; --k) {
    int64_t first = start[k * stride];
    int64_t last = k == top ? n : start[(k + 1) * stride];
    scalar_t mean = static_cast<scalar_t>(total[k * stride] / weight[k * stride]);
    for (int64_t j = first; j < last; ++j) {
      out[j * stride] = mean;
      if (starts) {
        starts[j * stride] = j == first;
      }
    }
  }
}

}  // namespace torchisotonic
