"""Builds the C++/CUDA operators on first use, with torch's JIT extension loader."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils import cpp_extension

CSRC = Path(__file__).parent / "csrc"


def isotonic(y: Tensor, w: Tensor) -> Tensor:
    """The increasing isotonic fit of every row of ``y`` ([batch, n], weights ``w``)."""
    out: Tensor = _load().isotonic(y, w)
    return out


@functools.cache
def _load() -> Any:
    """``torch.ops.torchisotonic``, compiled for this machine (and its GPU, if any) once."""
    sources = [CSRC / "ops.cpp"]
    if torch.cuda.is_available() and cpp_extension.CUDA_HOME is not None:  # pragma: no cover
        sources.append(CSRC / "ops_cuda.cu")
    cpp_extension.load(
        name="torchisotonic_ops",
        sources=[str(s) for s in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.torchisotonic
