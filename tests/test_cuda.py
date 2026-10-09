"""The CUDA operator must give what the CPU one does (which test_isotonic checks against scipy)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import torchisotonic

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def both(y: torch.Tensor, w: torch.Tensor | None = None, **kwargs: Any) -> None:
    want = torchisotonic.isotonic(y, w, **kwargs)
    got = torchisotonic.isotonic(y.cuda(), None if w is None else w.cuda(), **kwargs)
    assert got.is_cuda and got.dtype == want.dtype
    # The GPU may fuse multiply-adds, so the last bit can differ from the CPU.
    tol = {torch.float64: 1e-12, torch.float32: 1e-5}.get(y.dtype, 1e-2)
    torch.testing.assert_close(got.cpu(), want, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("batch", [1, 3, 127, 128, 129, 5000])
@pytest.mark.parametrize("n", [1, 2, 7, 100, 1000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_cpu(batch: int, n: int, dtype: torch.dtype) -> None:  # pragma: no cover
    g = torch.Generator().manual_seed(batch * 1000 + n)
    y = torch.linspace(0, 3, n, dtype=dtype) + torch.randn(batch, n, generator=g, dtype=dtype)
    w = torch.rand(batch, n, generator=g, dtype=dtype) + 0.1
    both(y)
    both(y, w)
    both(y, w, increasing=False, y_min=-0.5, y_max=2.0)


@pytest.mark.parametrize("pattern", ["increasing", "decreasing", "random"])
def test_long_problem(pattern: str) -> None:  # pragma: no cover
    n = 200_000
    i = torch.arange(n, dtype=torch.float64)
    y = {"increasing": i, "decreasing": -i, "random": torch.randn(n, dtype=torch.float64)}
    both(y[pattern].reshape(2, -1))


def test_nan_rows() -> None:  # pragma: no cover
    y = torch.randn(300, 50, dtype=torch.float64)
    y[7, 3] = math.nan
    y[299, 49] = math.nan
    both(y)


def test_other_dims_and_low_precision() -> None:  # pragma: no cover
    y = torch.randn(4, 60, 5)
    both(y, dim=1)
    both(y.half(), dim=1)
    both(y.bfloat16(), dim=1)
