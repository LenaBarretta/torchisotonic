"""Check and time every variant in experiments/ against the package's CUDA operator.

python experiments/run.py [name ...]

See experiments/README.md.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils import cpp_extension

import torchisotonic
from torchisotonic import _ext

HERE = Path(__file__).parent
CSRC = Path(torchisotonic.__file__).parent / "csrc"
# Shapes at lane, chunk and problem edges: packed rows, one chunk, several, many.
CHECK_SHAPES = [(1, 1), (3, 2), (70, 31), (70, 32), (5, 33), (100, 100), (7, 128), (9, 511),
                (9, 512), (3, 1000), (3, 1024), (3, 1025), (2, 4096), (2, 4097), (1, 100_000),
                (2, 1_000_003)]  # fmt: skip
TIME_SHAPES = [(1_000_000, 16), (100_000, 128), (10_000, 1_000), (1_000, 10_000),
               (100, 100_000), (1, 10_000_000), (1, 100_000_000)]  # fmt: skip
Op = Callable[[torch.Tensor, torch.Tensor | None, bool], torch.Tensor]


def pattern(name: str, batch: int, n: int, g: torch.Generator) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64).expand(batch, n)
    noise = torch.randn(batch, n, generator=g, dtype=torch.float64)
    return {
        "random": torch.linspace(0, 3, n, dtype=torch.float64) + noise,
        "increasing": i.clone(),
        "decreasing": -i,
        "saw": i % 7 + (i % 1500) * 0.1,
        "cliff": torch.where(i == n - 1, -1e9, i),
        "walk": noise.cumsum(1),
        "constant": torch.ones(batch, n, dtype=torch.float64),
        "valley": (i - n / 3) ** 2,
    }[name]


def check(op: Op, ref: Op) -> list[str]:
    failures = []
    g = torch.Generator().manual_seed(0)
    for batch, n in CHECK_SHAPES:
        for name in (
            "random",
            "increasing",
            "decreasing",
            "saw",
            "cliff",
            "walk",
            "constant",
            "valley",
        ):
            y = pattern(name, batch, n, g)
            w = torch.rand(batch, n, generator=g, dtype=torch.float64) + 0.1
            for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 1e-4)):
                for weights, increasing in ((False, True), (True, False)):
                    yy = y.to(dtype).cuda()
                    ww = w.to(dtype).cuda() if weights else None
                    want, got = ref(yy, ww, increasing), op(yy, ww, increasing)
                    if not torch.allclose(got, want, rtol=tol, atol=tol, equal_nan=True):
                        err = ((got - want).abs() / (1 + want.abs())).max().item()
                        failures.append(
                            f"{batch}x{n} {name} {dtype} w={weights} inc={increasing}: {err:.2e}"
                        )
    return failures


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


def main(names: list[str]) -> None:
    ref_ops: Any = _ext._load()
    ref: Op = ref_ops.isotonic_tiles
    folders = sorted(p.parent for p in HERE.glob("*/kernel.cu"))
    folders = [f for f in folders if not names or f.name in names]
    variants: dict[str, Op] = {}
    for folder in folders:
        cpp_extension.load(
            name=f"tiexp_{folder.name}",
            sources=[str(folder / "kernel.cu")],
            extra_include_paths=[str(CSRC)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
        )
        variants[folder.name] = getattr(torch.ops, f"tiexp_{folder.name}").isotonic
        failures = check(variants[folder.name], ref)
        status = "same fits" if not failures else f"{len(failures)} DIFFERENT"
        print(f"{folder.name}: {status}", flush=True)
        for f in failures[:10]:
            print(f"    {f}", flush=True)
    print("\nms (best of 10, warm T4), package / variant(s):", flush=True)
    print("| batch x n | package | " + " | ".join(variants) + " |")
    print("| --- | ---: |" + " ---: |" * len(variants))
    for batch, n in TIME_SHAPES:
        y = torch.linspace(0, 3, n, device="cuda") + torch.randn(batch, n, device="cuda")
        base = best_ms(lambda y=y: ref(y, None, True))
        cells = []
        for op in variants.values():
            t = best_ms(lambda y=y, op=op: op(y, None, True))
            cells.append(f"{t:.3f} ({base / t:.2f}x)")
        print(f"| {batch:,} x {n:,} | {base:.3f} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
