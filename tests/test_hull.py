"""The hull algorithm (one long problem in parallel pieces) must match scipy, on any chunking."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.optimize import isotonic_regression

import torchisotonic
from torchisotonic import _api

Array = npt.NDArray[np.float64]

values = st.integers(-20, 20).map(float) | st.floats(-1e3, 1e3, allow_nan=False)


@pytest.fixture(params=[1, 2, 3, 7, 128])
def hull(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """Send every problem to the hull algorithm, on the CPU too, with this chunk size."""
    monkeypatch.setattr(_api, "HULL_ON_CPU", True)
    monkeypatch.setattr(_api, "HULL_MIN_N", 0)
    monkeypatch.setattr(_api, "HULL_CHUNK", request.param)
    yield request.param


def reference(y: Array, w: Array | None, increasing: bool = True) -> torch.Tensor:
    return torch.tensor(isotonic_regression(y, weights=w, increasing=increasing).x.copy())


# The fixture only patches module settings, the same for every example.
@settings(
    max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    y=st.lists(values, min_size=1, max_size=80),
    weighted=st.booleans(),
    increasing=st.booleans(),
)
def test_matches_scipy(hull: int, y: list[float], weighted: bool, increasing: bool) -> None:
    yy = np.array(y)
    w = np.linspace(0.5, 2.0, len(y), dtype=np.float64) if weighted else None
    got = torchisotonic.isotonic(
        torch.tensor(yy), None if w is None else torch.tensor(w), increasing=increasing
    )
    torch.testing.assert_close(got, reference(yy, w, increasing), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("pattern", ["increasing", "decreasing", "walk", "convex", "offset"])
def test_long_problems(hull: int, pattern: str) -> None:
    n = 20_000
    i = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(1)
    y = {
        "increasing": i,
        "decreasing": -i,
        "walk": rng.standard_normal(n).cumsum(),
        "convex": (i / n) ** 2,  # every point a hull vertex
        "offset": 1e6 + rng.standard_normal(n) * 1e-3 + i * 1e-6,  # needs the centring
    }[pattern]
    got = torchisotonic.isotonic(torch.tensor(y))
    torch.testing.assert_close(got, reference(y, None), rtol=1e-9, atol=1e-6)


def test_batch_and_nan(hull: int) -> None:
    y = torch.randn(5, 300, dtype=torch.float64)
    y[2, 7] = math.nan
    got = torchisotonic.isotonic(y)
    assert got[2].isnan().all()
    for r in (0, 1, 3, 4):
        torch.testing.assert_close(got[r], reference(y[r].numpy(), None), rtol=1e-9, atol=1e-9)
