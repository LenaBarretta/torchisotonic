"""The fit and its gradient on the GPU, torchisotonic against torchsort.

    python paper/gradients.py

Decreasing unweighted float32 rows, noise around a trend; ms per call over 10 calls on a warm
GPU, with CUDA events around each call: best, median and quartiles. "fit": the fit alone;
"+grad": the fit and the gradient of sum(fit * g) for a random g (torchsort: its forward and
backward kernels).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torchsort.ops as ts

import torchisotonic

SHAPES = [(1_000_000, 16), (10_000, 1_000), (1, 10_000_000)]


def times_ms(fn: Callable[[], object], repeats: int = 10) -> np.ndarray:
    fn()
    torch.cuda.synchronize()
    torch.cuda._sleep(500_000_000)  # a warm GPU: see bench/bench.py
    out = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end))
    return np.array(out)


def ours_fit(x: torch.Tensor) -> object:
    return torchisotonic.isotonic(x, increasing=False)


def ours_grad(x: torch.Tensor, g: torch.Tensor) -> object:
    xr = x.detach().requires_grad_()
    return torch.autograd.grad(torchisotonic.isotonic(xr, increasing=False), xr, g)


def theirs_fit(x: torch.Tensor) -> object:
    return ts.isotonic_l2["cuda"](x)


def theirs_grad(x: torch.Tensor, g: torch.Tensor) -> object:
    return ts.isotonic_l2_backward["cuda"](x, ts.isotonic_l2["cuda"](x), g)


def main() -> None:
    device = torch.cuda.get_device_name(0)
    print(device, "torch", torch.__version__, "torchisotonic", torchisotonic.__version__)
    for batch, n in SHAPES:
        x = (torch.linspace(0, 3, n, device="cuda") + torch.randn(batch, n, device="cuda")).flip(1)
        g = torch.randn(batch, n, device="cuda")
        cases: dict[str, Callable[[], object]] = {
            "ours fit": lambda x=x: ours_fit(x),
            "ours +grad": lambda x=x, g=g: ours_grad(x, g),
            "torchsort fit": lambda x=x: theirs_fit(x),
            "torchsort +grad": lambda x=x, g=g: theirs_grad(x, g),
        }
        for name, fn in cases.items():
            t = times_ms(fn)
            q1, q3 = np.percentile(t, [25, 75])
            print(
                f"GRAD {batch}x{n} | {name} | best {t.min():.3f} | median {np.median(t):.3f} | "
                f"q1 {q1:.3f} | q3 {q3:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
