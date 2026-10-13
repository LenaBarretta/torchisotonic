"""The merge stage against the row length: segment lists against copying the vertices.

    python experiments/sweep.py [results.json]

One row of n = 2^14 .. 2^27 float32 elements, increasing (every point a vertex) and noisy;
for each, the GPU time of the merge stage of the package and of the `materialized` variant,
which copies the surviving vertices of every merge. Prints a table and writes the numbers
as JSON (default: experiments/sweep.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from run import CSRC, HERE, Op, stage_ms, timing_data
from torch.utils import cpp_extension

from torchisotonic import _ext

LENGTHS = [1 << k for k in range(14, 28)]
PATTERNS = ("increasing", "random")


def main(out: Path) -> None:
    ref_ops: Any = _ext._load()
    ref: Op = ref_ops.isotonic_tiles
    cpp_extension.load(
        name="tiexp_materialized",
        sources=[str(HERE / "materialized" / "kernel.cu")],
        extra_include_paths=[str(CSRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
    )
    copying: Op = torch.ops.tiexp_materialized.isotonic
    rows = []
    print(
        "| data | n | segments, merge ms | copying, merge ms | segments, total | copying, total |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in PATTERNS:
        for n in LENGTHS:
            y = timing_data(name, 1, n)
            seg = stage_ms(lambda y=y: ref(y, None, True))
            cop = stage_ms(lambda y=y: copying(y, None, True))
            row = {
                "data": name,
                "n": n,
                "segments_merge": seg.get("merge", 0.0),
                "copying_merge": cop.get("merge", 0.0),
                "segments_total": sum(seg.values()),
                "copying_total": sum(cop.values()),
            }
            rows.append(row)
            print(
                f"| {name} | {n:,} | {row['segments_merge']:.3f} | {row['copying_merge']:.3f} | "
                f"{row['segments_total']:.3f} | {row['copying_total']:.3f} |",
                flush=True,
            )
    device = torch.cuda.get_device_name(0)
    out.write_text(json.dumps({"device": device, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sweep.json")
