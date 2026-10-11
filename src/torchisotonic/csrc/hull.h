// Isotonic regression of one long problem as a lower convex hull, in parallel pieces.
//
// The fit at element i is the slope of the lower convex hull of the cumulative sum diagram
// P_k = (W_k, S_k), W_k = w_0 + ... + w_{k-1}, S_k = w_0 y_0 + ... + w_{k-1} y_{k-1}, over
// [W_i, W_{i+1}] (docs/hull_prototype.py has the reference version and the reasoning).
//
// A hull is stored as its vertices: coordinates (x, y) and the point index k, in three
// arrays. Every step below only touches vertices, never the n points, so a merge reads a few
// contiguous arrays instead of gathering coordinates from the whole series.
//
// The steps are shared by the CPU code, which runs them one after the other, and the CUDA
// code, which runs them side by side:
//   hull_push    - add the next point of a chunk to its hull (monotone chain);
//   needs_bridge / violation / merged_source - merge two neighbouring hulls;
//   fit          - the slope of the final hull edge above one element.
// Neighbouring hulls share their end point.
#pragma once

#include <cstdint>

#include "pava.h"  // TI_HD

namespace torchisotonic {

template <typename T, typename I>
TI_HD inline T slope(const T* x, const T* y, I a, I b) {
  return (y[b] - y[a]) / (x[b] - x[a]);
}

// Appends point (px, py, k) to the lower hull held in sx/sy/si (h vertices so far), first
// dropping the vertices it makes non-convex.
template <typename T, typename I>
TI_HD inline void hull_push(T* sx, T* sy, I* si, I& h, T px, T py, I k) {
  while (h >= 2) {
    const T cross =
        (sx[h - 1] - sx[h - 2]) * (py - sy[h - 2]) - (sy[h - 1] - sy[h - 2]) * (px - sx[h - 2]);
    if (cross > 0) {
      break;  // a strict left turn: the last vertex stays
    }
    --h;
  }
  sx[h] = px;
  sy[h] = py;
  si[h] = k;
  ++h;
}

// Whether two neighbouring hulls need a bridge at all (they don't when the shared point is
// convex, or one of them is a single point).
template <typename T, typename I>
TI_HD bool needs_bridge(const T* lx, const T* ly, I hl, const T* rx, const T* ry, I hr) {
  if (hl < 2 || hr < 2) {
    return false;
  }
  return slope(lx, ly, hl - 2, hl - 1) > slope(rx, ry, I(0), I(1));
}

// For left vertex a (not the shared last one): finds its lower tangent to right vertices
// 1..hr-1, stores the touching position in *b, and returns how far the tangent is from
// supporting the left hull at a. Exactly one a gives 0 in exact arithmetic; with rounding,
// the bridge is the a with the smallest violation.
template <typename T, typename I>
TI_HD T violation(
    const T* lx, const T* ly, I a, const T* rx, const T* ry, I hr, I* b) {
  const T px = lx[a], py = ly[a];
  I lo = 1, hi = hr - 1;
  while (lo < hi) {  // slopes from p fall, then rise, along the right hull
    const I mid = lo + (hi - lo) / 2;
    if ((ry[mid + 1] - py) / (rx[mid + 1] - px) < (ry[mid] - py) / (rx[mid] - px)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  *b = lo;
  const T s = (ry[lo] - py) / (rx[lo] - px);
  const T before = a == 0 ? T(0) : slope(lx, ly, I(a - 1), a) - s;
  const T after = s - slope(lx, ly, a, I(a + 1));
  const T v = before > after ? before : after;
  return v > 0 ? v : T(0);
}

// The hull of two neighbours with bridge (a, b) is left[0..a] then right[b..hr-1]. Without a
// bridge, pass a = hl - 1 and b = 0 (the shared point, counted once).
template <typename I>
TI_HD inline I merged_size(I a, I b, I hr) {
  return a + 1 + hr - (b == 0 ? 1 : b);
}

// Where vertex k of the merged hull comes from: (false, position) in the left hull or
// (true, position) in the right one.
template <typename I>
TI_HD inline bool merged_source(I a, I b, I k, I* position) {
  if (k <= a) {
    *position = k;
    return false;
  }
  *position = (b == 0 ? 1 : b) + (k - a - 1);
  return true;
}

// The fit at element i: the slope of the final hull edge [v_e, v_{e+1}] with
// index_e <= i < index_{e+1}.
template <typename T, typename I>
TI_HD T fit(const T* x, const T* y, const I* index, I h, int64_t i) {
  I lo = 0, hi = h - 2;
  while (lo < hi) {
    const I mid = lo + (hi - lo + 1) / 2;
    if (index[mid] <= i) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return slope(x, y, lo, I(lo + 1));
}

}  // namespace torchisotonic
