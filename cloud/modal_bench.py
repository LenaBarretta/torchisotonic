"""Runs kaggle/kernel/run.py's steps on a Modal GPU, for a second architecture beside the T4.

    modal run cloud/modal_bench.py --gpu A100-40GB --steps "tests accuracy profile"
    modal run cloud/modal_bench.py --gpu H100 --steps "tests accuracy profile grad experiments-each"

The working tree (what git tracks or would track, so not notes/ or kaggle/output/) is copied
into the container; the log is printed and saved to cloud/output/<gpu>-<time>.log.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/torchisotonic"


def tracked_files() -> list[str]:
    """The files git would ship: tracked or untracked but not ignored."""
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line and (ROOT / line).is_file()]


def source_note() -> str:
    """Which commit, with which local changes: printed at the top of the log."""
    head = subprocess.run(
        ["git", "log", "-1", "--format=commit %h %s"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    changed = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    lines = [head.strip()] + [f"  changed: {line}" for line in changed.splitlines()]
    return "\n".join(lines) + "\n" + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + "\n"


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "scipy", "scikit-learn", "pytest", "pytest-cov", "hypothesis", "ninja")
)
if modal.is_local():
    for name in tracked_files():
        image = image.add_local_file(ROOT / name, f"{REMOTE}/{name}", copy=False)

app = modal.App("torchisotonic-bench", image=image)


@app.function(gpu="A100-40GB", timeout=3 * 3600)
def run(steps: str, source: str) -> str:
    import os

    Path(f"{REMOTE}/SOURCE").write_text(source)
    env = dict(os.environ, TI_SRC=REMOTE, TI_OUT="/root/out", TI_STEPS=steps)
    Path("/root/out").mkdir(exist_ok=True)
    done = subprocess.run(
        ["python", f"{REMOTE}/kaggle/kernel/run.py"],
        env=env,
        cwd=REMOTE,
        capture_output=True,
        text=True,
    )
    return done.stdout + done.stderr


@app.local_entrypoint()
def main(gpu: str = "A100-40GB", steps: str = "tests accuracy profile") -> None:
    log = run.with_options(gpu=gpu).remote(steps, source_note())
    out = ROOT / "cloud" / "output"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{gpu}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    path.write_text(log)
    print(log)
    print(f"saved to {path}")
