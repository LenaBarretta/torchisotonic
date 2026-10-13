"""The merge stage against the row length: segment lists against copying the vertices.

    python experiments/sweep.py [results.json]

One row of n = 2^14 .. 2^27 float32 elements, increasing (every point a vertex) and noisy;
for each, the GPU time of the merge stage of the package and of two controls that copy the
surviving vertices of every merge: `materialized`, where the thread or warp of a merge copies
its own, and `materialized_grid`, where the whole GPU copies every round's into a second
buffer. For the latter, also the time of its copies alone and, on increasing data, the rate
at which they move memory. Prints a table and writes the numbers as JSON (default:
experiments/sweep.json).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from run import CSRC, HERE, Op, stage_ms, timing_data
from torch.profiler import ProfilerActivity, profile
from torch.utils import cpp_extension

from torchisotonic import _ext

LENGTHS = [1 << k for k in range(14, 28)]
PATTERNS = ("increasing", "random")
CONTROLS = ("materialized", "materialized_grid")
CHUNK = 1024  # elements per chunk of unweighted float32
VERTEX_BYTES = 8 + 4 + 4  # a vertex of an unweighted row: y, point index, chunk


def kernel_ms(fn: Callable[[], object], name: str, repeats: int = 5) -> float:
    """GPU time of the kernels whose names contain `name` in one call, averaged."""
    fn()
    torch.cuda._sleep(500_000_000)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    times: dict[str, float] = defaultdict(float)
    for event in prof.key_averages():
        if event.device_type.name == "CUDA" and name in event.key:
            times[name] += event.self_device_time_total / 1e3 / repeats
    return times[name]


def main(out: Path) -> None:
    ref_ops: Any = _ext._load()
    ref: Op = ref_ops.isotonic_tiles
    controls: dict[str, Op] = {}
    for name in CONTROLS:
        cpp_extension.load(
            name=f"tiexp_{name}",
            sources=[str(HERE / name / "kernel.cu")],
            extra_include_paths=[str(CSRC)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
        )
        controls[name] = getattr(torch.ops, f"tiexp_{name}").isotonic
    rows = []
    print(
        "| data | n | merge ms: segments | per-merge copying | grid copying | grid copies alone "
        "| GB/s | totals ms: segments | per-merge | grid |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for data in PATTERNS:
        for n in LENGTHS:
            y = timing_data(data, 1, n)
            want = ref(y, None, True)
            for op in controls.values():
                torch.testing.assert_close(op(y, None, True), want, rtol=1e-5, atol=1e-5)
            seg = stage_ms(lambda y=y: ref(y, None, True))
            cop = stage_ms(lambda y=y: controls["materialized"](y, None, True))
            grid = stage_ms(lambda y=y: controls["materialized_grid"](y, None, True))
            copies = kernel_ms(
                lambda y=y: controls["materialized_grid"](y, None, True), "copy_merge"
            )
            rounds = math.ceil(math.log2(math.ceil(n / CHUNK)))
            # Increasing data: all n + 1 points are vertices, and every round reads and writes
            # each once.
            moved = 2 * rounds * (n + 1) * VERTEX_BYTES if data == "increasing" else 0
            row = {
                "data": data,
                "n": n,
                "segments_merge": seg.get("merge", 0.0),
                "copying_merge": cop.get("merge", 0.0),
                "grid_merge": grid.get("merge", 0.0),
                "grid_copies": copies,
                "grid_gbps": moved / (copies * 1e6) if moved and copies else 0.0,
                "segments_total": sum(seg.values()),
                "copying_total": sum(cop.values()),
                "grid_total": sum(grid.values()),
            }
            rows.append(row)
            print(
                f"| {data} | {n:,} | {row['segments_merge']:.3f} | {row['copying_merge']:.3f} | "
                f"{row['grid_merge']:.3f} | {row['grid_copies']:.3f} | {row['grid_gbps']:.0f} | "
                f"{row['segments_total']:.3f} | {row['copying_total']:.3f} | "
                f"{row['grid_total']:.3f} |",
                flush=True,
            )
    device = torch.cuda.get_device_name(0)
    out.write_text(json.dumps({"device": device, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sweep.json")
