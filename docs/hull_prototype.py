"""Reference prototype of the parallel algorithm for one large problem, in plain numpy.

Isotonic regression is the slope of the greatest convex minorant of the cumulative sum
diagram (Barlow et al.; see also Kearsley, J. Res. NIST 111(2), 2006): with points
P_k = (W_k, S_k), W_k = w_0 + ... + w_{k-1} and S_k = w_0 y_0 + ... + w_{k-1} y_{k-1}, the fit
at element i is the slope of the lower convex hull of P_0..P_n over [W_i, W_{i+1}].

The GPU version does this in parallel:

1. Cut the problem into chunks of C elements; solve each with PAVA, independently. A chunk's
   blocks are the edges of its own lower hull, so its block boundaries are hull vertices.
2. Merge neighbouring hulls in log2(#chunks) rounds. Two neighbouring lower hulls share their
   boundary point. If the hull is already convex there, they just concatenate; otherwise
   there is one bridge, a common lower tangent from vertex p of the left hull to vertex q of
   the right one. The vertices strictly between p and q drop out, and the elements between
   them form one block.
3. Every element takes the slope of the final hull edge above it.

The bridge is found the way the GPU does it, all candidates at once: every left vertex p
finds its own lower tangent to the right hull by binary search (slopes from p to the right
hull's vertices fall, then rise), and the bridge is the p whose tangent also supports the
left hull at p (or, with rounding, comes closest to it).

This file only checks the logic (tests/test_hull_prototype.py runs it against scipy). The
coordinates here are plain float64 prefix sums; the CUDA kernel works in local coordinates.
"""

from __future__ import annotations

import itertools

import numpy as np

Hull = list[int]  # indices k of the points P_k on a lower hull, increasing


def chunk_hull(w: np.ndarray, y: np.ndarray, lo: int, hi: int) -> Hull:
    """PAVA on elements lo..hi-1; returns its hull as point indices lo..hi."""
    starts: list[int] = []
    weights: list[float] = []
    totals: list[float] = []
    for i in range(lo, hi):
        starts.append(i)
        weights.append(float(w[i]))
        totals.append(float(w[i] * y[i]))
        while len(starts) > 1 and totals[-2] * weights[-1] >= totals[-1] * weights[-2]:
            top_weight, top_total = weights.pop(), totals.pop()
            weights[-1] += top_weight
            totals[-1] += top_total
            starts.pop()
    return [*starts, hi]


def slope(xs: np.ndarray, ys: np.ndarray, a: int, b: int) -> float:
    return float((ys[b] - ys[a]) / (xs[b] - xs[a]))


def tangent(xs: np.ndarray, ys: np.ndarray, p: int, right: Hull) -> int:
    """Position in ``right`` of the vertex where the lower tangent from point p touches."""
    lo, hi = 0, len(right) - 1
    while lo < hi:  # slopes from p fall then rise along the right hull: find the minimum
        mid = (lo + hi) // 2
        if slope(xs, ys, p, right[mid + 1]) < slope(xs, ys, p, right[mid]):
            lo = mid + 1
        else:
            hi = mid
    return lo


def merge(xs: np.ndarray, ys: np.ndarray, left: Hull, right: Hull) -> Hull:
    """Lower hull of two neighbouring lower hulls (left[-1] == right[0])."""
    assert left[-1] == right[0]
    if len(left) == 1 or len(right) == 1:
        return left + right[1:]
    if slope(xs, ys, left[-2], left[-1]) <= slope(xs, ys, right[0], right[1]):
        return left + right[1:]  # already convex at the shared point
    # Every left vertex but the shared one is a candidate; on the GPU, one thread each. In
    # exact arithmetic exactly one has a tangent that also supports the left hull (violation
    # 0). Rounding can make the tests contradict each other on (nearly) collinear points, so
    # take the candidate that violates support least: an argmin, i.e. one GPU reduction.
    best, best_violation, best_b = 0, float("inf"), 0
    for a in range(len(left) - 1):
        p = left[a]
        b = tangent(xs, ys, p, right[1:]) + 1
        s = slope(xs, ys, p, right[b])
        before = 0.0 if a == 0 else slope(xs, ys, left[a - 1], p) - s
        after = s - slope(xs, ys, p, left[a + 1])
        violation = max(before, after, 0.0)
        if violation < best_violation:
            best, best_violation, best_b = a, violation, b
    return left[: best + 1] + right[best_b:]


def isotonic(y: np.ndarray, w: np.ndarray | None = None, chunk: int = 4) -> np.ndarray:
    """Increasing isotonic regression of y, computed the parallel way."""
    n = len(y)
    if n == 0:
        return y.astype(np.float64)
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xs = np.concatenate([[0.0], np.cumsum(w)])
    ys = np.concatenate([[0.0], np.cumsum(w * y)])
    hulls = [chunk_hull(w, y, lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    while len(hulls) > 1:  # one round of pairwise merges per level of the tree
        merged = [merge(xs, ys, hulls[i], hulls[i + 1]) for i in range(0, len(hulls) - 1, 2)]
        hulls = merged + ([hulls[-1]] if len(hulls) % 2 else [])
    vertices = hulls[0]
    out = np.empty(n)
    for a, b in itertools.pairwise(vertices):
        out[a:b] = slope(xs, ys, a, b)
    return out
