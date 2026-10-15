"""Batched isotonic regression with the semantics of ``sklearn.isotonic.isotonic_regression``."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from torchisotonic import _ext


def isotonic(
    y: Tensor,
    weight: Tensor | None = None,
    *,
    increasing: bool = True,
    y_min: float | None = None,
    y_max: float | None = None,
    dim: int = -1,
    exact: bool = True,
) -> Tensor:
    """Isotonic regression of every 1-D slice of ``y`` along ``dim``.

    For each slice, returns the non-decreasing (or, with ``increasing=False``,
    non-increasing) sequence closest to it in weighted least squares, like
    ``sklearn.isotonic.isotonic_regression``. ``weight`` must be positive and shaped like
    ``y`` (or broadcast to it). Results are clipped to ``[y_min, y_max]`` if given. A slice
    containing NaN comes back as NaN. Integer input becomes the default float dtype; float
    input keeps its dtype, with float16/bfloat16 computed in float32.

    Differentiable in ``y`` and ``weight``: the fit is constant on blocks of neighbouring
    elements, each the weighted mean of ``y`` over its block, so a gradient is averaged over
    the blocks (wherever the blocks do not change, that is everywhere but on a set of measure
    zero).

    ``exact=False`` makes float32 on the GPU about twice as fast for short and medium rows,
    at the cost of accuracy on hard inputs: with strongly varying weights, or increments near
    float32's resolution, the fit may be off by a few parts in a thousand, step down by up to
    about 1e-3 of the data's scale and pool points that should stay apart. It has no effect on
    float64 or on the CPU.
    """
    if not isinstance(y, Tensor):
        raise TypeError(f"y must be a torch.Tensor, got {type(y).__name__}")
    if y.is_complex():
        raise TypeError("complex tensors are not supported")
    if y.ndim == 0:
        raise ValueError("y must have at least one dimension")
    out_dtype = y.dtype if y.is_floating_point() else torch.get_default_dtype()
    acc = torch.float64 if out_dtype == torch.float64 else torch.float32
    rows = y.movedim(dim, -1)
    w: Tensor | None = None
    if weight is not None:
        if not isinstance(weight, Tensor):
            raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
        try:
            w = weight.to(device=y.device, dtype=acc).broadcast_to(y.shape).movedim(dim, -1)
        except RuntimeError as e:
            raise ValueError(f"weight of shape {tuple(weight.shape)} does not fit y") from e
        if not bool((w > 0).all()):
            raise ValueError("weight must be strictly positive")
    shape = rows.shape
    if rows.numel() == 0:
        return y.to(out_dtype, copy=True)
    flat_y = rows.reshape(-1, shape[-1]).to(acc)
    flat_w = None if w is None else w.reshape(-1, shape[-1])
    out: Tensor = _Isotonic.apply(flat_y, flat_w, increasing, exact)
    if y_min is not None or y_max is not None:
        out = out.clamp(min=y_min, max=y_max)
    return out.reshape(shape).movedim(-1, dim).to(out_dtype)


class _Isotonic(torch.autograd.Function):
    """_solve, with the gradient of the fit: averaging over its blocks."""

    @staticmethod
    def forward(
        ctx: Any, flat_y: Tensor, flat_w: Tensor | None, increasing: bool, exact: bool
    ) -> Tensor:
        # Where the fit's blocks start, from the solver itself, for the gradient only.
        out, starts = _solve(flat_y, flat_w, increasing, exact, any(ctx.needs_input_grad[:2]))
        ctx.save_for_backward(flat_y, flat_w, out, starts)
        return out

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor | None, Tensor | None, None, None]:
        flat_y, flat_w, out, starts = ctx.saved_tensors
        want_w = ctx.needs_input_grad[1]
        if out.is_cuda:  # pragma: no cover - needs a GPU
            gy, gw = _ext.isotonic_grad(out, starts, grad, flat_w, flat_y if want_w else None)
            return gy if ctx.needs_input_grad[0] else None, gw if want_w else None, None, None
        w = torch.ones_like(out) if flat_w is None else flat_w
        g_sum, w_sum = _block_sums(starts, grad, w)
        mean_grad = g_sum / w_sum  # d(loss) / d(block's mean), per unit of weight
        grad_y = w * mean_grad if ctx.needs_input_grad[0] else None
        grad_w = (flat_y - out) * mean_grad if want_w else None
        return grad_y, grad_w, None, None


def _block_sums(starts: Tensor, *values: Tensor) -> list[Tensor]:
    """For every element, the sums of each of ``values`` over its block, the blocks of a
    [batch, n] fit starting where the bool tensor ``starts`` is set (and at every row's start).

    No host synchronisation: the blocks are numbered by a prefix sum over all rows, and
    summed into a buffer as long as the whole batch.
    """
    heads = starts.clone()
    heads[:, 0] = True
    ids = heads.reshape(-1).cumsum(0) - 1
    sums = []
    for v in values:
        total = torch.zeros(starts.numel(), dtype=v.dtype, device=v.device)
        total.index_add_(0, ids, v.reshape(-1))
        sums.append(total[ids].reshape(starts.shape))
    return sums


def _solve(
    flat_y: Tensor,
    flat_w: Tensor | None,
    increasing: bool,
    exact: bool = True,
    with_starts: bool = False,
) -> tuple[Tensor, Tensor]:
    """The fit of [batch, n] problems (n >= 1), in flat_y's dtype - on the GPU by the tile
    pipeline, on the CPU by pool adjacent violators - and, if ``with_starts``, where its blocks
    start (a bool tensor; else an empty one)."""
    if flat_y.is_cuda:  # pragma: no cover - needs a GPU
        w = None if flat_w is None else flat_w.contiguous()
        return _ext.isotonic_tiles(flat_y.contiguous(), w, increasing, exact, with_starts)
    w = torch.ones_like(flat_y) if flat_w is None else flat_w
    y = flat_y if increasing else -flat_y  # a decreasing fit of y is minus the increasing fit of -y
    out, starts = _ext.isotonic(y.contiguous(), w.contiguous(), with_starts)
    return (out if increasing else -out), starts
