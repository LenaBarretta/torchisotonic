"""Check and time every variant in experiments/ against the package's CUDA operator.

python experiments/run.py [--profile] [name ...]

--profile adds, per timing shape, the GPU time of each stage (chunk hulls, merges, the fit,
anything else) for the package and every variant, from torch.profiler.

See experiments/README.md.
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile
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


def consecutive_floats(n: int, dtype: torch.dtype) -> torch.Tensor:
    """n strictly increasing values exactly representable in dtype: the consecutive floats
    from 1.0 (float32 has only 2^24 exact integers, so arange would repeat past that)."""
    if dtype == torch.float32:
        return (torch.arange(n, dtype=torch.int32) + 0x3F800000).view(torch.float32)
    return (torch.arange(n, dtype=torch.int64) + 0x3FF0000000000000).view(torch.float64)


def pattern(name: str, batch: int, n: int, g: torch.Generator) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64).expand(batch, n)
    noise = torch.randn(batch, n, generator=g, dtype=torch.float64)
    return {
        "random": torch.linspace(0, 3, n, dtype=torch.float64) + noise,
        "increasing": consecutive_floats(n, torch.float64).expand(batch, n).clone(),
        "decreasing": -i,
        "saw": i % 7 + (i % 1500) * 0.1,
        "cliff": torch.where(i == n - 1, -1e9, i),
        "walk": noise.cumsum(1),
        "constant": torch.ones(batch, n, dtype=torch.float64),
        "valley": (i - n / 3) ** 2,
    }[name]


ROUNDING = 1e-6  # relative: a float64 difference below this is rounding, not a bug


def check(op: Op, ref: Op) -> list[str]:
    """Shapes and patterns where the variant's fit differs from the package's beyond rounding."""
    failures = []
    roundings: list[str] = []
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
                        if dtype == torch.float64 and err < ROUNDING:
                            # Rounding-level: say which is closer to the float64 CPU fit.
                            truth = torchisotonic.isotonic(
                                yy.cpu(), None if ww is None else ww.cpu(), increasing=increasing
                            )
                            e_pkg = ((want.cpu() - truth).abs() / (1 + truth.abs())).max().item()
                            e_var = ((got.cpu() - truth).abs() / (1 + truth.abs())).max().item()
                            roundings.append(
                                f"{batch}x{n} {name} w={weights} inc={increasing}: differ by "
                                f"{err:.1e}; vs CPU float64: package {e_pkg:.1e}, "
                                f"variant {e_var:.1e}"
                            )
                            continue
                        failures.append(
                            f"{batch}x{n} {name} {dtype} w={weights} inc={increasing}: {err:.2e}"
                        )
    for r in roundings:
        print(f"    rounding: {r}", flush=True)
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


# Data for the timings: noisy (few hull vertices survive), increasing (every point is a vertex:
# the most to merge - but the fit is then the input, which a trivial check could return), and
# nearly monotone: consecutive floats jittered by k ulps, so that many vertices survive while
# the fit does pool; the surviving share falls as k grows. Also sawtooth (many short blocks).
TIME_PATTERNS = (
    "random",
    "increasing",
    "mono+1ulp",
    "mono+4ulp",
    "mono+16ulp",
    "mono+64ulp",
    "saw",
    "cliff",
)


def timing_data(name: str, batch: int, n: int) -> torch.Tensor:
    i = torch.arange(n, device="cuda", dtype=torch.float32)
    if name == "increasing":  # every point a hull vertex: strictly increasing, exactly
        return consecutive_floats(n, torch.float32).cuda().expand(batch, n).contiguous()
    if name.startswith("mono+"):
        k = float(name[len("mono+") : -len("ulp")])
        g = torch.Generator(device="cuda").manual_seed(7)
        jitter = (k * torch.randn(batch, n, device="cuda", generator=g)).round().to(torch.int32)
        bits = torch.arange(n, device="cuda", dtype=torch.int32) + 0x3F800000 + jitter
        return bits.view(torch.float32).contiguous()
    if name == "saw":
        return (i % 7 + (i % 1500) * 0.1).expand(batch, n).contiguous()
    if name == "cliff":  # increasing, then a last point far below: bridges across everything
        y = consecutive_floats(n, torch.float32).cuda().expand(batch, n).contiguous()
        y[:, -1] = -1e9
        return y
    return torch.linspace(0, 3, n, device="cuda") + torch.randn(batch, n, device="cuda")


STAGES = {
    "chunk": ("chunk_kernel", "packed_kernel"),
    "merge": ("merge_kernel", "merge_b3_kernel"),
    "fit": ("ends_kernel", "fit_kernel"),  # flat_fit_kernel too, by name
}


def resource_usage(name: str) -> None:
    """Registers, spills and shared memory of the merge kernels of a built extension."""
    so = next(Path(cpp_extension._get_build_directory(name, False)).glob("*.so"), None)
    if so is None:
        return
    dump = subprocess.run(["cuobjdump", "-res-usage", str(so)], capture_output=True, text=True)
    lines = dump.stdout.splitlines()
    for i, line in enumerate(lines):
        if "Function" in line and "merge" in line and i + 1 < len(lines):
            fn = line.split()[-1].rstrip(":")
            demangled = subprocess.run(["c++filt", fn], capture_output=True, text=True).stdout
            print(f"    {demangled.strip()[:90]}\n        {lines[i + 1].strip()}", flush=True)


def stage_ms(fn: Callable[[], object], repeats: int = 5) -> dict[str, float]:
    """GPU time per stage of one call, averaged over `repeats` calls."""
    fn()
    torch.cuda._sleep(500_000_000)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    times: dict[str, float] = defaultdict(float)
    for event in prof.key_averages():
        if event.device_type.name != "CUDA":
            continue
        stage = next(
            (s for s, names in STAGES.items() if any(n in event.key for n in names)), "other"
        )
        times[stage] += event.self_device_time_total / 1e3 / repeats
    return times


def main(names: list[str]) -> None:
    do_profile = "--profile" in names
    names = [n for n in names if n != "--profile"]
    ref_ops: Any = _ext._load()
    ref: Op = ref_ops.isotonic_tiles
    if do_profile:
        print("package: merge kernels", flush=True)
        resource_usage("torchisotonic_ops")
    folders = sorted(p.parent for p in HERE.glob("*/kernel.cu"))
    folders = [f for f in folders if not names or f.name in names]
    variants: dict[str, Op] = {}
    for folder in folders:
        try:
            cpp_extension.load(
                name=f"tiexp_{folder.name}",
                sources=[str(folder / "kernel.cu")],
                extra_include_paths=[str(CSRC)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-Xptxas", "-v"],
                is_python_module=False,
                verbose=True,
            )
        except RuntimeError as e:  # a build error: report it, test the others
            lines = [ln for ln in str(e).splitlines() if "error" in ln.lower()][:3]
            print(f"{folder.name}: DID NOT BUILD\n    " + "\n    ".join(lines), flush=True)
            continue
        variants[folder.name] = getattr(torch.ops, f"tiexp_{folder.name}").isotonic
        if do_profile:
            resource_usage(f"tiexp_{folder.name}")
        failures = check(variants[folder.name], ref)
        status = "same fits" if not failures else f"{len(failures)} DIFFERENT"
        print(f"{folder.name}: {status}", flush=True)
        for f in failures[:10]:
            print(f"    {f}", flush=True)
    for pattern_name in TIME_PATTERNS:
        # How much the fit pools on this data: the share of elements not starting a block.
        probe = timing_data(pattern_name, 1, 10_000_000)
        fit = ref(probe, None, True)[0]
        pooled = 1 - (1 + int((fit[1:] != fit[:-1]).sum())) / probe.numel()
        print(
            f"\n[{pattern_name}] ms (best of 10, warm GPU), package / variant(s); "
            f"pooled share of elements on 1x1e7: {100 * pooled:.1f}%",
            flush=True,
        )
        print("| batch x n | package | " + " | ".join(variants) + " |")
        print("| --- | ---: |" + " ---: |" * len(variants))
        for batch, n in TIME_SHAPES:
            y = timing_data(pattern_name, batch, n)
            base = best_ms(lambda y=y: ref(y, None, True))
            cells = []
            for op in variants.values():
                t = best_ms(lambda y=y, op=op: op(y, None, True))
                cells.append(f"{t:.3f} ({base / t:.2f}x)")
            print(f"| {batch:,} x {n:,} | {base:.3f} | " + " | ".join(cells) + " |", flush=True)
        if not do_profile:
            continue
        print(f"\n[{pattern_name}] GPU ms per stage (chunk / merge / fit / other):", flush=True)
        for batch, n in TIME_SHAPES:
            y = timing_data(pattern_name, batch, n)
            rows = [("package", stage_ms(lambda y=y: ref(y, None, True)))]
            rows += [
                (name, stage_ms(lambda y=y, op=op: op(y, None, True)))
                for name, op in variants.items()
            ]
            for name, t in rows:
                parts = " / ".join(
                    f"{t.get(k, 0.0):.3f}" for k in ("chunk", "merge", "fit", "other")
                )
                print(
                    f"| {batch:,} x {n:,} | {name} | {parts} | total {sum(t.values()):.3f} |",
                    flush=True,
                )


if __name__ == "__main__":
    main(sys.argv[1:])
