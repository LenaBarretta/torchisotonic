"""The fit and its gradient on the GPU: torchisotonic against torchsort.

python bench/grad.py [--profile]

--profile adds, per shape, the GPU time of every kernel of one fit + gradient call.

Decreasing, unweighted fits of float32 data shaped [batch, n] (what torchsort solves), noisy
around a trend; milliseconds, best of 10 on a warm GPU. "fit": the fit alone; "+ grad": the fit
and the gradient of sum(fit * g) for a random g (torchsort: its forward and backward kernels,
which its soft sort and soft rank call).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile

import torchisotonic

SHAPES = [(1_000_000, 16), (100_000, 128), (10_000, 1_000), (1_000, 10_000), (1, 1_000_000),
          (1, 10_000_000)]  # fmt: skip


def best_ms(fn: Callable[[], object], repeats: int = 10) -> float:
    fn()
    torch.cuda._sleep(500_000_000)  # a warm GPU: see bench/bench.py
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return min(times)


def ours_grad(x: torch.Tensor, g: torch.Tensor) -> object:
    xr = x.detach().requires_grad_()
    return torch.autograd.grad(torchisotonic.isotonic(xr, increasing=False), xr, g)


try:
    import torchsort.ops as ts
except ImportError:
    ts = None

print("| batch x n | torchisotonic fit | + grad | torchsort fit | + grad |")
print("| --- | ---: | ---: | ---: | ---: |")
for batch, n in SHAPES:
    x = (torch.linspace(0, 3, n, device="cuda") + torch.randn(batch, n, device="cuda")).flip(1)
    g = torch.randn(batch, n, device="cuda")
    cells = [
        best_ms(lambda x=x: torchisotonic.isotonic(x, increasing=False)),
        best_ms(lambda x=x, g=g: ours_grad(x, g)),
    ]
    text = [f"{c:,.2f}" for c in cells]
    if ts is not None and batch * n <= 10_000_000:  # beyond, a single call takes minutes
        fwd, bwd = ts.isotonic_l2["cuda"], ts.isotonic_l2_backward["cuda"]
        text.append(f"{best_ms(lambda x=x, fwd=fwd: fwd(x)):,.2f}")
        text.append(f"{best_ms(lambda x=x, g=g, f=fwd, b=bwd: b(x, f(x), g)):,.2f}")
    else:
        text += ["—", "—"]
    print(f"| {batch:,} x {n:,} | " + " | ".join(text) + " |", flush=True)
    if "--profile" in sys.argv:
        ours_grad(x, g)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            ours_grad(x, g)
            torch.cuda.synchronize()
        kernels: dict[str, float] = defaultdict(float)
        for event in prof.key_averages():
            if event.device_type.name == "CUDA":
                kernels[event.key[:70]] += event.self_device_time_total / 1e3
        for name, ms in sorted(kernels.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {ms:7.3f} ms  {name}", flush=True)
