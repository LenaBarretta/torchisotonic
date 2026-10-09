"""Benchmark torchisotonic against scipy and torchsort.

    python bench/bench.py            # full grid
    python bench/bench.py --quick    # small grid, to check that everything runs

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
        (10, 1_000_000), (1, 10_000_000)]  # fmt: skip
QUICK = [(10_000, 16), (100, 1_000), (1, 100_000)]
RESULTS = Path(__file__).parent / "results"
Fn = Callable[[], object]


def best_time(fn: Fn, sync: Callable[[], None], repeats: int) -> float:
    fn()  # warm-up; also compiles the operators
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
        if times[-1] > 5:  # slow enough that one run says it
            break
    return min(times)


def impls(data: np.ndarray, cuda: bool) -> dict[str, Fn]:
    out: dict[str, Fn] = {}
    if cuda:
        x = torch.tensor(data, device="cuda")
        out["torchisotonic (cuda)"] = lambda: torchisotonic.isotonic(x, increasing=False)
    cpu = torch.tensor(data)
    out["torchisotonic (cpu)"] = lambda: torchisotonic.isotonic(cpu, increasing=False)
    try:
        import torchsort.ops as ts
    except ImportError:
        pass
    else:
        if cuda:
            out["torchsort (cuda)"] = lambda: ts.isotonic_l2["cuda"](x)
        out["torchsort (cpu)"] = lambda: ts.isotonic_l2["cpu"](cpu)
    from scipy.optimize import isotonic_regression

    def scipy_loop() -> object:
        return [isotonic_regression(row, increasing=False) for row in data]

    out["scipy (loop)"] = scipy_loop
    return out


def run(grid: list[tuple[int, int]], repeats: int) -> list[dict[str, Any]]:
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    rows = []
    for batch, n in grid:
        rng = np.random.default_rng(0)
        trend = np.linspace(0, 3, n, dtype=np.float32)
        data = (trend + rng.standard_normal((batch, n), dtype=np.float32))[:, ::-1].copy()
        for name, fn in impls(data, cuda).items():
            row: dict[str, Any] = {"batch": batch, "n": n, "impl": name}
            try:
                row["seconds"] = best_time(fn, sync, repeats)
            except torch.cuda.OutOfMemoryError:
                row["error"] = "out of memory"
            except (RuntimeError, MemoryError) as e:
                row["error"] = f"failed ({str(e).splitlines()[0][:60]})"
            if cuda:
                torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)
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
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    else:
        device = f"CPU ({platform.processor() or platform.machine()})"
    rows = run(QUICK if args.quick else FULL, args.repeats)
    RESULTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", device.lower()).strip("-")
    meta = {
        "device": device,
        "torch": torch.__version__,
        "torchisotonic": torchisotonic.__version__,
    }
    (RESULTS / f"{slug}.json").write_text(json.dumps({**meta, "rows": rows}, indent=1))
    (RESULTS / f"{slug}.md").write_text(markdown(rows, device))
    print(f"\nWrote {RESULTS / slug}.json and .md")


if __name__ == "__main__":
    main()
