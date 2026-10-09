"""The parallel algorithm (docs/hull_prototype.py) must give the same fit as scipy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import isotonic_regression

sys.path.insert(0, str(Path(__file__).parents[1] / "docs"))
import hull_prototype

values = st.integers(-20, 20).map(float) | st.floats(-1e3, 1e3, allow_nan=False)


@st.composite
def problems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray | None, int]:
    y = np.array(draw(st.lists(values, max_size=80)), dtype=np.float64)
    w = draw(st.none() | st.lists(st.floats(0.01, 100), min_size=len(y), max_size=len(y)))
    chunk = draw(st.sampled_from([1, 2, 3, 4, 7, 16, 100]))
    return y, None if w is None else np.array(w), chunk


@settings(max_examples=2000, deadline=None)
@given(problem=problems())
def test_matches_scipy(problem: tuple[np.ndarray, np.ndarray | None, int]) -> None:
    y, w, chunk = problem
    want = isotonic_regression(y, weights=w).x
    got = hull_prototype.isotonic(y, w, chunk=chunk)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_long_and_structured() -> None:
    rng = np.random.default_rng(0)
    for y in [
        np.arange(5000.0),
        -np.arange(5000.0),
        np.zeros(5000),
        rng.standard_normal(5000).cumsum(),
        np.sin(np.arange(5000) / 50) + np.arange(5000) / 2000,
    ]:
        want = isotonic_regression(y).x
        for chunk in (1, 32, 1000):
            np.testing.assert_allclose(hull_prototype.isotonic(y, chunk=chunk), want, atol=1e-9)
