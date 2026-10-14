"""Benchmark torchisotonic against scipy and torchsort.

    python bench/bench.py            # full grid
    python bench/bench.py --quick    # small grid, to check that everything runs
    python bench/bench.py --gpu-only # the GPU columns alone (a cloud container's CPU says
                                     # nothing, and the scipy loop would take hours on it)
    python bench/bench.py --sweep    # one row of 2^14 .. 2^27 elements (with --gpu-only)

Every implementation solves the same problems: a batch of decreasing, unweighted isotonic
regressions (the only kind torchsort solves) of float32 data shaped [batch, n], noisy
around a trend so that blocks of all sizes appear. GPU libraries get tensors already on the
GPU; only the computation is timed. Writes bench/results/<device>.json and .md.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

import torchisotonic

FULL = [(1_000_000, 16), (100_000, 128), (10_000, 1_000), (1_000, 10_000), (100, 100_000),
        (10, 1_000_000), (1, 10_000_000), (100, 1_000), (10, 10_000), (1, 100_000),
        (1, 1_000_000), (1, 100_000_000)]  # fmt: skip
QUICK = [
    (1_000_000, 16),
    (100_000, 128),
    (100, 1_000),
    (1, 100_000),
    (1_000, 10_000),
    (1, 10_000_000),
]
SWEEP = [(1, 1 << k) for k in range(14, 28)]
RESULTS = Path(__file__).parent / "results"
Fn = Callable[[], object]


def call_times(fn: Fn, sync: Callable[[], None], repeats: int) -> list[float]:
    """Seconds per call, over up to `repeats` calls after a warm-up."""
    fn()  # warm-up; also compiles the operators
    sync()
    if torch.cuda.is_available():
        # An idle GPU runs at a low clock and takes a while to raise it: keep it busy first,
        # or short GPU calls are timed at a fraction of the speed they run at under load.
        torch.cuda._sleep(1_000_000_000)
        sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
        if sum(times) > 10:  # slow enough that these runs say it
            break
    return times


def impls(data: np.ndarray, cuda: bool, gpu_only: bool = False) -> dict[str, Fn]:
    out: dict[str, Fn] = {}
    if cuda:
        x = torch.tensor(data, device="cuda")
        out["torchisotonic (cuda)"] = lambda: torchisotonic.isotonic(x, increasing=False)
        out["torchisotonic (cuda, exact=False)"] = lambda: torchisotonic.isotonic(
            x, increasing=False, exact=False
        )
    cpu = torch.tensor(data)
    if not gpu_only:
        out["torchisotonic (cpu)"] = lambda: torchisotonic.isotonic(cpu, increasing=False)
    try:
        import torchsort.ops as ts
    except ImportError:
        pass
    else:
        if cuda and data.shape[1] <= 10_000_000:  # a row of 10^8 takes minutes on one thread
            out["torchsort (cuda)"] = lambda: ts.isotonic_l2["cuda"](x)
        if not gpu_only:
            out["torchsort (cpu)"] = lambda: ts.isotonic_l2["cpu"](cpu)
    if gpu_only:
        return out
    from scipy.optimize import isotonic_regression

    def scipy_loop() -> object:
        return [isotonic_regression(row, increasing=False) for row in data]

    out["scipy (loop)"] = scipy_loop
    return out


def run(
    grid: list[tuple[int, int]],
    repeats: int,
    save: Callable[[list[dict[str, Any]]], None],
    gpu_only: bool = False,
) -> list[dict[str, Any]]:
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    rows = []
    for batch, n in grid:
        rng = np.random.default_rng(0)
        trend = np.linspace(0, 3, n, dtype=np.float32)
        data = (trend + rng.standard_normal((batch, n), dtype=np.float32))[:, ::-1].copy()
        for name, fn in impls(data, cuda, gpu_only).items():
            row: dict[str, Any] = {"batch": batch, "n": n, "impl": name}
            try:
                times = np.array(call_times(fn, sync, repeats))
                row["seconds"] = float(times.min())
                row["median"] = float(np.median(times))
                row["quartiles"] = [float(q) for q in np.percentile(times, [25, 75])]
            except torch.cuda.OutOfMemoryError:
                row["error"] = "out of memory"
            except (RuntimeError, MemoryError) as e:
                row["error"] = f"failed ({str(e).splitlines()[0][:60]})"
            if cuda:
                torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)
            save(rows)  # as it goes: a run cut short keeps what it measured
    return rows


def markdown(rows: list[dict[str, Any]], device: str) -> str:
    names = list(dict.fromkeys(r["impl"] for r in rows))
    lines = [f"Device: {device}. Best of several runs, in milliseconds (lower is better).", ""]
    lines.append("| batch x n | " + " | ".join(names) + " |")
    lines.append("| --- |" + " ---: |" * len(names))
    for batch, n in dict.fromkeys((r["batch"], r["n"]) for r in rows):
        cells = []
        for name in names:
            match = [r for r in rows if (r["batch"], r["n"], r["impl"]) == (batch, n, name)]
            if not match:
                cells.append("—")
            elif "seconds" in match[0]:
                cells.append(f"{match[0]['seconds'] * 1e3:,.2f}")
            else:
                cells.append(match[0]["error"])
        lines.append(f"| {batch:,} x {n:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small grid")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--gpu-only", action="store_true", help="skip the CPU columns")
    parser.add_argument("--sweep", action="store_true", help="one row of growing length")
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    else:
        device = f"CPU ({platform.processor() or platform.machine()})"
    RESULTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", device.lower()).strip("-") + ("-sweep" if args.sweep else "")
    meta = {
        "device": device,
        "torch": torch.__version__,
        "torchisotonic": torchisotonic.__version__,
    }

    def save(rows: list[dict[str, Any]]) -> None:
        (RESULTS / f"{slug}.json").write_text(json.dumps({**meta, "rows": rows}, indent=1))
        (RESULTS / f"{slug}.md").write_text(markdown(rows, device))

    grid = SWEEP if args.sweep else QUICK if args.quick else FULL
    run(grid, args.repeats, save, args.gpu_only)
    print(f"\nWrote {RESULTS / slug}.json and .md")


if __name__ == "__main__":
    main()
