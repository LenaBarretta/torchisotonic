"""Where the time goes, per CUDA kernel, for a few shapes.

python bench/profile_hull.py
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

import torchisotonic

KERNELS = ("chunk", "packed", "merge", "ends", "fit", "isotonic")
REPEATS = 5
SHAPES = [(1, 100_000_000), (1, 1_000_000), (100, 100_000), (10_000, 1_000)]

CASES = [(shape, None) for shape in SHAPES] + [  # chunk sizes compared
    (shape, warps)
    for shape in [(1, 100_000_000), (1, 10_000_000), (1000, 2000), (1000, 4000)]
    for warps in ("1", "2", "4")
]
for (batch, n), warps in CASES:
    if warps is None:
        os.environ.pop("TI_CHUNK_WARPS", None)
    else:
        os.environ["TI_CHUNK_WARPS"] = warps
        print(f"chunks of {warps} warps:", end=" ")
    x = torch.linspace(0, 3, n, device="cuda") + torch.randn(batch, n, device="cuda")
    for _ in range(2):  # compile and warm up
        torchisotonic.isotonic(x)
    torch.cuda._sleep(500_000_000)  # an idle GPU runs at a low clock until it is busy a while
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPEATS):
            torchisotonic.isotonic(x)
        torch.cuda.synchronize()
    times: dict[str, float] = defaultdict(float)
    for event in prof.key_averages():
        if event.device_type.name != "CUDA":
            continue
        name = next((k for k in KERNELS if f"{k}_kernel" in event.key), "torch")
        times[name] += event.self_device_time_total / 1e3 / REPEATS
    total = sum(times.values())
    ideal = 2 * batch * n * 4 / 320e9 * 1e3  # read y, write the fit, at 320 GB/s
    # What a plain copy of the input actually takes on this GPU: the practical bound.
    out = torch.empty_like(x)
    out.copy_(x)
    torch.cuda.synchronize()
    start_e, end_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_e.record()
    for _ in range(REPEATS):
        out.copy_(x)
    end_e.record()
    end_e.synchronize()
    copy_ms = start_e.elapsed_time(end_e) / REPEATS
    parts = ", ".join(f"{k} {v:7.2f}" for k, v in sorted(times.items(), key=lambda kv: -kv[1]))
    bounds = f"bandwidth bound {ideal:.2f}, copy {copy_ms:.2f}"
    print(f"{batch}x{n}: total {total:7.2f} ms ({bounds}) | {parts}", flush=True)
    rounds = sorted(
        (e for e in prof.events() if "merge_kernel" in e.name and e.device_type.name == "CUDA"),
        key=lambda e: e.time_range.start,
    )
    rounds = rounds[: len(rounds) // REPEATS]  # the first call's
    if rounds:
        each = " ".join(f"{e.device_time_total / 1e3:.2f}" for e in rounds)
        print(f"    merge rounds, ms: {each}", flush=True)
