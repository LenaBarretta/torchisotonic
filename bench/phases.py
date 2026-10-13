"""Where chunk_kernel's time goes, by phase, from a build with -DTI_PHASES.

python bench/phases.py

Only for measuring: it loads its own build of the operators, so it does not import the
package's (both register the same operator names). Two measures: clock cycles each phase
takes per warp (the compiler may move work across phase boundaries), and the time of the
whole kernel cut after each phase (one chunk per problem, so nothing runs after it).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils import cpp_extension

# --csrc DIR: build another copy of the sources (e.g. an earlier commit's, extracted beside
# the tree) for the same measurements, in a separate process: the operator names are fixed.
CSRC = Path(__file__).parents[1] / "src" / "torchisotonic" / "csrc"
if "--csrc" in sys.argv:
    CSRC = Path(sys.argv[sys.argv.index("--csrc") + 1]).resolve()
PHASES = ["load", "lane hulls", "lane starts", "merge lanes", "write"]
EXT = "torchisotonic_phases_" + "".join(ch for ch in str(CSRC) if ch.isalnum())[-40:]

print(f"sources: {CSRC}", flush=True)
cpp_extension.load(
    name=EXT,
    sources=[str(CSRC / f) for f in ("ops.cpp", "ops_cuda.cu", "ops_tiles.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-DTI_PHASES"],
    is_python_module=False,
)
ops = torch.ops.torchisotonic


def timed(fn: Callable[[], object], reps: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


spin = 200_000_000
print(f"SM clock: about {spin / timed(lambda: torch.cuda._sleep(spin), 3) / 1e3:.0f} MHz")

CASES = {
    "random": lambda b, n: (
        torch.linspace(0, 3, n, device="cuda") + torch.randn(b, n, device="cuda")
    ),
    "increasing (no pops)": lambda b, n: (
        torch.arange(n, device="cuda").float().expand(b, n).contiguous()
    ),
    "decreasing (one pop per step)": lambda b, n: (
        -torch.arange(n, device="cuda").float().expand(b, n).contiguous()
    ),
}
for (name, make), (batch, n) in [((k, v), (100_000, 1000)) for k, v in CASES.items()]:
    print(f"--- {name}", flush=True)
    x = make(batch, n)
    ops.isotonic_tiles(x, None, True)
    ops.tiles_phases(x, 99)  # reset
    ops.isotonic_tiles(x, None, True)
    info = ops.tiles_phases(x, 99)
    cycles = info[: len(PHASES)].double()
    regs, local, smem, warps, blocks = info[8:13].tolist()
    print(
        f"chunk_kernel<float>: {regs} registers, {local} B local, {smem} B shared per block, "
        f"{warps} warps per block, {blocks} blocks per SM",
        flush=True,
    )
    share = ", ".join(
        f"{p} {100 * c / cycles.sum():4.1f}%" for p, c in zip(PHASES, cycles, strict=True)
    )
    print(f"{batch}x{n}: {cycles.sum().item() / batch:8.0f} cycles per warp | {share}")
    steps, longest = (info[5:7].double() / batch).tolist()
    print(f"    lane merges: {steps:.1f} walk steps per chunk, {longest:.1f} in the longest walks")
    cut = []
    for stop in range(len(PHASES)):
        print(f"    cut after {PHASES[stop]}...", flush=True)
        ops.tiles_phases(x, stop if stop < len(PHASES) - 1 else 99)
        cut.append(timed(lambda x=x: ops.isotonic_tiles(x, None, True)))
    ops.tiles_phases(x, 99)
    parts = ", ".join(
        f"{p} {t - (cut[i - 1] if i else 0.0):.2f}"
        for i, (p, t) in enumerate(zip(PHASES, cut, strict=True))
    )
    print(f"{batch}x{n}: cut after each phase, ms: {parts} (total {cut[-1]:.2f})", flush=True)

# Instructions per kernel in the binary (a big unrolled kernel can overflow the instruction cache).
import subprocess  # noqa: E402

so = next(Path(cpp_extension._get_build_directory(EXT, False)).glob("*.so"))
sass = subprocess.run(["cuobjdump", "-sass", str(so)], capture_output=True, text=True).stdout
sizes: dict[str, int] = {}
current = None
for line in sass.splitlines():
    if "Function :" in line:
        current = line.split("Function :")[1].strip()
        sizes[current] = 0
    elif current and line.strip().startswith("/*") and "*/" in line and ";" in line:
        sizes[current] += 1
for fn, count in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]:
    print(f"{count:7d} instructions  {fn[:150]}")

# The float, unweighted chunk_kernel's SASS, for reading (kaggle output: sass_chunk_kernel.txt).
keep, lines = False, []
for line in sass.splitlines():
    if "Function :" in line:
        keep = "chunk_kernelIfLb0ELi4E" in line
    if keep:
        lines.append(line)
Path("/kaggle/working" if Path("/kaggle/working").exists() else ".").joinpath(
    "sass_chunk_kernel.txt"
).write_text("\n".join(lines))
print(f"wrote {len(lines)} lines of SASS")


# Merge statistics (plan: figure 5): per round, nodes / descriptors / vertices alive; walks
# that gave up for a search.
def pattern_data(name: str, batch: int, n: int) -> torch.Tensor:
    i = torch.arange(n, device="cuda", dtype=torch.float32)
    if name == "increasing":
        return i.expand(batch, n).contiguous()
    if name == "saw":
        return (i % 7 + (i % 1500) * 0.1).expand(batch, n).contiguous()
    return torch.linspace(0, 3, n, device="cuda") + torch.randn(batch, n, device="cuda")


for name in ("random", "increasing", "saw"):
    for batch, n in [(1, 10_000_000), (100, 100_000)]:
        x = pattern_data(name, batch, n)
        ops.isotonic_tiles(x, None, True)
        ops.tiles_phases(x, 99)  # reset
        ops.isotonic_tiles(x, None, True)
        info = ops.tiles_phases(x, 99)
        merges, gave_up = info[16].item(), info[17].item()
        rounds = info[18:].view(-1, 3)
        rounds = rounds[rounds[:, 0] > 0]
        print(
            f"--- {name} {batch}x{n}: {merges} merges, {gave_up} walks gave up "
            f"({100 * gave_up / max(merges, 1):.2f}%)"
        )
        print("| round | nodes | descriptors | vertices alive | descriptors per node |")
        print("| ---: | ---: | ---: | ---: | ---: |")
        for r, (nodes, descs, verts) in enumerate(rounds.tolist()):
            print(f"| {r} | {nodes} | {descs} | {verts} | {descs / nodes:.2f} |", flush=True)
