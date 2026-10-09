"""torchisotonic.isotonic must match scipy / scikit-learn (up to float rounding)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import isotonic_regression as scipy_isotonic
from sklearn.isotonic import isotonic_regression as sklearn_isotonic

import torchisotonic

values = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
weights = st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def problems(draw: st.DrawFn) -> tuple[list[float], list[float] | None]:
    y = draw(st.lists(values | st.sampled_from([0.0, 1.0, -1.0]), max_size=60))
    w = draw(st.none() | st.lists(weights, min_size=len(y), max_size=len(y)))
    return y, w


def reference(y: list[float], w: list[float] | None, increasing: bool) -> torch.Tensor:
    x = scipy_isotonic(np.asarray(y, dtype=np.float64), weights=w, increasing=increasing).x
    # scipy returns a reversed view for decreasing fits; torch wants positive strides
    return torch.tensor(x.copy(), dtype=torch.float64)


def check(got: torch.Tensor, want: torch.Tensor) -> None:
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-9, equal_nan=True)


@settings(max_examples=500, deadline=None)
@given(problem=problems(), increasing=st.booleans())
def test_matches_scipy(problem: tuple[list[float], list[float] | None], increasing: bool) -> None:
    y, w = problem
    weight = None if w is None else torch.tensor(w, dtype=torch.float64)
    got = torchisotonic.isotonic(
        torch.tensor(y, dtype=torch.float64), weight, increasing=increasing
    )
    check(got, reference(y, w, increasing))


@settings(max_examples=200, deadline=None)
@given(
    problem=problems(),
    y_min=st.none() | values,
    y_max=st.none() | values,
    increasing=st.booleans(),
)
def test_bounds_match_sklearn(
    problem: tuple[list[float], list[float] | None],
    y_min: float | None,
    y_max: float | None,
    increasing: bool,
) -> None:
    y, w = problem
    assume(y)  # sklearn rejects empty input; test_empty covers it
    want = sklearn_isotonic(
        np.asarray(y, dtype=np.float64),
        sample_weight=w,
        y_min=y_min,
        y_max=y_max,
        increasing=increasing,
    )
    weight = None if w is None else torch.tensor(w, dtype=torch.float64)
    got = torchisotonic.isotonic(
        torch.tensor(y, dtype=torch.float64),
        weight,
        increasing=increasing,
        y_min=y_min,
        y_max=y_max,
    )
    check(got, torch.tensor(want.copy(), dtype=torch.float64))


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_batch_along_any_dim(dim: int) -> None:
    torch.manual_seed(0)
    y = torch.randn(3, 4, 50, dtype=torch.float64).movedim(-1, dim)
    w = torch.rand_like(y) + 0.1
    got = torchisotonic.isotonic(y, w, dim=dim)
    rows_y, rows_w, rows_got = (t.movedim(dim, -1).reshape(-1, 50) for t in (y, w, got))
    for yi, wi, gi in zip(rows_y, rows_w, rows_got, strict=True):
        check(gi, reference(yi.tolist(), wi.tolist(), True))


def test_weight_broadcasts() -> None:
    y = torch.tensor([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]], dtype=torch.float64)
    w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    got = torchisotonic.isotonic(y, w)
    for yi, gi in zip(y, got, strict=True):
        check(gi, reference(yi.tolist(), w.tolist(), True))


@pytest.mark.parametrize(
    "pattern",
    ["increasing", "decreasing", "alternating", "constant", "sawtooth"],
)
def test_long_problems(pattern: str) -> None:
    n = 100_000
    i = np.arange(n, dtype=np.float64)
    y = {
        "increasing": i,
        "decreasing": -i,  # one block of everything
        "alternating": (-1.0) ** i * i,
        "constant": np.full(n, 2.5),
        "sawtooth": i % 97 - i / 1000,
    }[pattern]
    check(torchisotonic.isotonic(torch.tensor(y)), reference(y.tolist(), None, True))


def test_nan_makes_only_its_row_nan() -> None:
    y = torch.tensor([[3.0, math.nan, 2.0], [3.0, 1.0, 2.0]], dtype=torch.float64)
    got = torchisotonic.isotonic(y)
    assert got[0].isnan().all()
    check(got[1], reference([3.0, 1.0, 2.0], None, True))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_keeps_float_dtype(dtype: torch.dtype) -> None:
    y = torch.tensor([3.0, 1.0, 2.0, 5.0], dtype=dtype)
    got = torchisotonic.isotonic(y)
    assert got.dtype == dtype
    torch.testing.assert_close(
        got.double(), torch.tensor([2.0, 2.0, 2.0, 5.0], dtype=torch.float64)
    )


def test_integers_become_default_float() -> None:
    got = torchisotonic.isotonic(torch.tensor([3, 1, 2]))
    assert got.dtype == torch.get_default_dtype()
    torch.testing.assert_close(got, torch.tensor([2.0, 2.0, 2.0]))


@pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 5)])
def test_empty(shape: tuple[int, ...]) -> None:
    got = torchisotonic.isotonic(torch.empty(shape))
    assert got.shape == shape


def test_single_element() -> None:
    torch.testing.assert_close(torchisotonic.isotonic(torch.tensor([4.0])), torch.tensor([4.0]))


def test_bad_inputs() -> None:
    with pytest.raises(TypeError, match=r"y must be a torch\.Tensor, got list"):
        torchisotonic.isotonic([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="complex"):
        torchisotonic.isotonic(torch.zeros(3, dtype=torch.complex64))
    with pytest.raises(ValueError, match="at least one dimension"):
        torchisotonic.isotonic(torch.tensor(1.0))
    with pytest.raises(TypeError, match=r"weight must be a torch\.Tensor"):
        torchisotonic.isotonic(torch.zeros(3), [1.0, 1.0, 1.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not fit y"):
        torchisotonic.isotonic(torch.zeros(3), torch.ones(4))
    with pytest.raises(ValueError, match="strictly positive"):
        torchisotonic.isotonic(torch.zeros(3), torch.tensor([1.0, 0.0, 1.0]))
