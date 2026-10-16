"""Data whose cumulative sums cancel: the largest step down and deviation from the CPU float64
fit, relative to max |y|, and the number of blocks, per length, scale, weighting and dtype.

    python paper/cancellation.py

Per 64 elements: 32 values of +B, one of -32 B, then 31 small, nearly collinear values.
"""

from __future__ import annotations

import torch

import torchisotonic


def data(n: int, big: float, g: torch.Generator) -> torch.Tensor:
    k = torch.arange(n) % 64
    y = torch.where(k < 32, big, 1.0).double()
    y[k == 32] = -32 * big
    return (
        y + 1e-3 * (-1.0) ** k * (k > 32) + 1e-6 * torch.randn(n, generator=g, dtype=torch.float64)
    )


def main() -> None:
    print("n | B | weights | dtype | step down / max|y| | deviation / max|y| | blocks gpu/cpu")
    for n in (500, 3000, 20_000, 100_000):
        for big in (1e3, 1e6, 1e9, 1e12):
            g = torch.Generator().manual_seed(n)
            y = data(n, big, g)
            weightings = (
                ("none", None),
                ("rand", torch.rand(n, generator=g, dtype=torch.float64) + 0.1),
            )
            for wname, w in weightings:
                for dtype in (torch.float64, torch.float32):
                    yy = y.to(dtype)[None]
                    ww = None if w is None else w.to(dtype)[None]
                    want = torchisotonic.isotonic(yy.double(), None if ww is None else ww.double())
                    got = torchisotonic.isotonic(yy.cuda(), None if ww is None else ww.cuda())
                    want, got = want[0], got[0].double().cpu()
                    scale = float(yy.double().abs().max())
                    step = float((got[:-1] - got[1:]).clamp(min=0).max()) / scale
                    dev = float((got - want).abs().max()) / scale
                    blocks_gpu = int((got[1:] != got[:-1]).sum()) + 1
                    blocks_cpu = int((want[1:] != want[:-1]).sum()) + 1
                    print(
                        f"DIAG {n} | {big:.0e} | {wname} | {str(dtype)[6:]} | {step:.2e} | "
                        f"{dev:.2e} | {blocks_gpu}/{blocks_cpu}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
