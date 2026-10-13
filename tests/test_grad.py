"""Gradients of the fit: averaging over its blocks, checked against finite differences."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.autograd import gradcheck

import torchisotonic


def data(shape: tuple[int, ...], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(*shape, generator=g, dtype=torch.float64).requires_grad_()
    w = (torch.rand(*shape, generator=g, dtype=torch.float64) + 0.1).requires_grad_()
    return y, w


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"increasing": False}, {"y_min": -0.3, "y_max": 0.5}, {"dim": 0}],
)
def test_matches_finite_differences(seed: int, kwargs: dict[str, Any]) -> None:
    y, w = data((4, 13), seed)
    assert gradcheck(lambda y, w: torchisotonic.isotonic(y, w, **kwargs), (y, w))
    assert gradcheck(lambda y: torchisotonic.isotonic(y, **kwargs), (y,))


def test_block_average() -> None:
    # [3, 1, 2] is one block: every output is the mean, so every input moves each output by 1/3.
    y = torch.tensor([3.0, 1.0, 2.0], requires_grad=True)
    c = torch.tensor([1.0, 2.0, 6.0])
    (grad,) = torch.autograd.grad((torchisotonic.isotonic(y) * c).sum(), (y,))
    torch.testing.assert_close(grad, torch.full((3,), 3.0))


def close_values() -> list[torch.Tensor]:
    """Strictly increasing rows whose neighbours are a few units in the last place apart."""
    one = torch.tensor(1.0).view(torch.int32)
    consecutive = (one + torch.arange(1000, dtype=torch.int32)).view(torch.float32)  # from 1 up
    return [consecutive[None], torch.linspace(0, 1, 10**6)[None]]


@pytest.mark.parametrize("row", range(2))
def test_close_values_are_separate_blocks(row: int) -> None:
    # Increasing data is its own fit, every element a block: the gradient passes unchanged,
    # however close the neighbours.
    y = close_values()[row].requires_grad_()
    c = torch.randn(y.shape, generator=torch.Generator().manual_seed(row))
    out = torchisotonic.isotonic(y)
    assert torch.equal(out, y)
    (grad,) = torch.autograd.grad((out * c).sum(), (y,))
    assert torch.equal(grad, c)


def test_weights_alone_and_broadcast() -> None:
    y = torch.randn(3, 8, dtype=torch.float64)
    w = (torch.rand(8, dtype=torch.float64) + 0.1).requires_grad_()  # broadcast over rows
    assert gradcheck(lambda w: torchisotonic.isotonic(y, w), (w,))


def test_dtype_and_no_grad() -> None:
    y = torch.randn(2, 50, requires_grad=True)
    (grad,) = torch.autograd.grad(torchisotonic.isotonic(y).sum(), (y,))
    assert grad.dtype == torch.float32
    assert not torchisotonic.isotonic(torch.randn(2, 5)).requires_grad
