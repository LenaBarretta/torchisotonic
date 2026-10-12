"""Where the GPU fit of weighted, strictly increasing data deviates from the data.

python bench/debug_weighted.py

For such data the fit is the data itself (every point is a hull vertex). Prints, per case,
the largest |fit - y| and where they sit relative to lane (32), chunk (1024) and fit-span
boundaries, plus the KKT multipliers' extremes - to tell a rounding problem in one kernel
from a wrong hull.
"""

from __future__ import annotations

import sys

import torch

import torchisotonic

sys.path.insert(0, "bench")
from accuracy import consecutive, weights

for n in (1_000, 100_000, 10_000_000):
    for wkind in ("uniform", "extreme"):
        g = torch.Generator().manual_seed(n)
        y = consecutive(n)  # float64, exact in float32
        w = weights(wkind, n, g)
        assert w is not None
        fit = torchisotonic.isotonic(y.cuda(), w.cuda()).cpu()
        d = (fit - y).abs()
        lam = -2 * (w * (fit - y)).cumsum(0)
        worst = torch.topk(d, 5).indices.tolist()
        mono = int((fit[1:] < fit[:-1]).sum())
        print(
            f"n={n} {wkind}: max|fit-y| {float(d.max()):.2e}, non-zero {int((d > 0).sum())}, "
            f"monotone violations {mono}, |lam| max {float(lam.abs().max()):.2e} "
            f"(sum w|y| {float((w * y.abs()).sum()):.2e})",
            flush=True,
        )
        for i in worst:
            print(
                f"    i={i} (i%32={i % 32}, i%1024={i % 1024}) fit={float(fit[i]):.17g} "
                f"y={float(y[i]):.17g} w={float(w[i]):.4g} diff={float(fit[i] - y[i]):.3e}"
            )
