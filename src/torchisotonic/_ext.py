"""Builds the C++/CUDA operators on first use, with torch's JIT extension loader."""

from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils import cpp_extension

CSRC = Path(__file__).parent / "csrc"


def isotonic(y: Tensor, w: Tensor) -> Tensor:
    """The increasing isotonic fit of every row of ``y`` ([batch, n], CPU, weights ``w``)."""
    out: Tensor = _load().isotonic(y, w)
    return out


def isotonic_tiles(
    y: Tensor, w: Tensor | None, increasing: bool, exact: bool = True
) -> Tensor:  # pragma: no cover
    """The fit of every row of ``y`` ([batch, n], CUDA) by the tile pipeline, in y's dtype."""
    out: Tensor = _load().isotonic_tiles(y, w, increasing, exact)
    return out


def isotonic_grad(
    out: Tensor, g: Tensor, w: Tensor | None, y: Tensor | None
) -> list[Tensor]:  # pragma: no cover
    """The gradients of a fit ``out`` (CUDA) with respect to y and, given y, to the weights."""
    grads: list[Tensor] = _load().isotonic_grad(out, g, w, y)
    return grads


def _ninja_on_path() -> None:
    """Puts the ninja that came with the package on PATH, for torch's builder, which runs it
    from PATH: a Python started from an environment that is not activated (a notebook kernel,
    an IDE, ``.venv/bin/python``) would not find it."""
    if shutil.which("ninja") is None:
        import ninja

        os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")


@functools.cache
def _load() -> Any:
    """``torch.ops.torchisotonic``, compiled for this machine (and its GPU, if any) once."""
    _ninja_on_path()
    sources = [CSRC / "ops.cpp"]
    if torch.cuda.is_available() and cpp_extension.CUDA_HOME is not None:  # pragma: no cover
        sources += [CSRC / "ops_tiles.cu"]
    cpp_extension.load(
        name="torchisotonic_ops",
        sources=[str(s) for s in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.torchisotonic
