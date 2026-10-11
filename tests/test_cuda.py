"""The CUDA operator must give what the CPU one does (which test_isotonic checks against scipy)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import torchisotonic
from torchisotonic import _api

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def both(
    y: torch.Tensor, w: torch.Tensor | None = None, tol: float | None = None, **kwargs: Any
) -> None:
    want = torchisotonic.isotonic(y, w, **kwargs)
    got = torchisotonic.isotonic(y.cuda(), None if w is None else w.cuda(), **kwargs)
    assert got.is_cuda and got.dtype == want.dtype
    # The GPU may fuse multiply-adds, so the last bit can differ from the CPU.
    if tol is None:
        # float32: the GPU merges hulls in float64, PAVA on the CPU sums blocks in float32.
        tol = {torch.float64: 1e-12, torch.float32: 1e-4}.get(y.dtype, 1e-2)
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
    both(y[pattern].reshape(2, -1), tol=1e-9)  # two long problems: the hull algorithm


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


@pytest.fixture
def chunk_hull(monkeypatch: pytest.MonkeyPatch) -> None:
    """The older chunked-hull CUDA path instead of the tile pipeline."""
    monkeypatch.setattr(_api, "CUDA_METHOD", "hull")


@pytest.mark.parametrize("chunk", [1, 3, 32, 33, 128, 129, 512])  # every stack size
@pytest.mark.parametrize(("batch", "n"), [(1, 1), (1, 2), (3, 1000), (2, 50_000), (40, 5000)])
@pytest.mark.usefixtures("chunk_hull")
def test_hull_matches_cpu(
    chunk: int, batch: int, n: int, monkeypatch: pytest.MonkeyPatch
) -> None:  # pragma: no cover
    monkeypatch.setattr(_api, "HULL_MIN_N", 0)
    monkeypatch.setattr(_api, "HULL_CHUNK", chunk)
    g = torch.Generator().manual_seed(chunk * 7 + n)
    y = torch.linspace(0, 3, n, dtype=torch.float64) + torch.randn(
        batch, n, generator=g, dtype=torch.float64
    )
    w = torch.rand(batch, n, generator=g, dtype=torch.float64) + 0.1
    # The hull's slopes are differences of cumulative sums, so it rounds differently from
    # the PAVA it is compared with here: the same tolerance as against scipy on the CPU.
    both(y, tol=1e-9)
    both(y, w, tol=1e-9, increasing=False)


@pytest.mark.usefixtures("chunk_hull")
def test_hull_float32(monkeypatch: pytest.MonkeyPatch) -> None:  # pragma: no cover
    # float32 input: the chunks' hulls are computed in float32 on the GPU, in float64 on the CPU.
    monkeypatch.setattr(_api, "HULL_MIN_N", 0)
    g = torch.Generator().manual_seed(3)
    y = torch.linspace(0, 3, 200_000) + torch.randn(3, 200_000, generator=g)
    both(y, tol=1e-4)


@pytest.fixture
def tiles_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tile pipeline for every length, short problems included."""
    monkeypatch.setattr(_api, "TILES_MIN_N", 0)


TILE_SHAPES = [1, 2, 31, 32, 33, 64, 100, 128, 255, 511, 512, 1000, 1023, 1024, 1025, 2048]
TILE_SHAPES += [4095, 4096, 4097, 8193, 100_000]


@pytest.mark.parametrize("n", TILE_SHAPES)
@pytest.mark.parametrize("batch", [1, 3, 70])
@pytest.mark.usefixtures("tiles_only")
def test_tiles_shapes(n: int, batch: int) -> None:  # pragma: no cover
    # The tile pipeline (the default on CUDA) against PAVA on the CPU: lane and chunk edges,
    # partial lanes and chunks, one chunk (finished in one kernel), many merge rounds.
    g = torch.Generator().manual_seed(n + batch)
    y = torch.linspace(0, 3, n, dtype=torch.float64) + torch.randn(
        batch, n, generator=g, dtype=torch.float64
    )
    w = torch.rand(batch, n, generator=g, dtype=torch.float64) + 0.1
    both(y, tol=1e-9)
    both(y, w, tol=1e-9, increasing=False)
    both(y.float(), tol=1e-4)


PATTERNS = ["increasing", "decreasing", "saw", "cliff", "walk", "constant", "valley"]


def pattern(name: str, n: int, g: torch.Generator) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64)
    if name == "increasing":  # every point is a hull vertex
        return i
    if name == "decreasing":  # one block
        return -i
    if name == "saw":  # convex runs with drops, not aligned to lanes or chunks
        return i % 7 + (i % 1500) * 0.1
    if name == "cliff":  # one bridge across almost everything
        return torch.where(i == n - 1, torch.tensor(-1e9, dtype=torch.float64), i)
    if name == "walk":
        return torch.randn(n, generator=g, dtype=torch.float64).cumsum(0)
    if name == "constant":  # collinear points everywhere
        return torch.ones(n, dtype=torch.float64)
    return (i - n / 3) ** 2  # valley: falls, then rises


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("n", [700, 1025, 3000, 5000, 9000, 70_000])
@pytest.mark.usefixtures("tiles_only")
def test_tiles_patterns(name: str, n: int) -> None:  # pragma: no cover
    g = torch.Generator().manual_seed(n)
    y = pattern(name, n, g)[None]
    w = torch.rand(1, n, generator=g, dtype=torch.float64) + 0.1
    both(y, tol=1e-9)
    both(y, w, tol=1e-9)
    both(y, w, tol=1e-9, increasing=False)
    both(y.float(), tol=1e-4)
