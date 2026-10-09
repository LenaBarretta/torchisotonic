"""Batched isotonic regression with the semantics of ``sklearn.isotonic.isotonic_regression``."""

from __future__ import annotations

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
) -> Tensor:
    """Isotonic regression of every 1-D slice of ``y`` along ``dim``.

    For each slice, returns the non-decreasing (or, with ``increasing=False``,
    non-increasing) sequence closest to it in weighted least squares, like
    ``sklearn.isotonic.isotonic_regression``. ``weight`` must be positive and shaped like
    ``y`` (or broadcast to it). Results are clipped to ``[y_min, y_max]`` if given. A slice
    containing NaN comes back as NaN. Integer input becomes the default float dtype; float
    input keeps its dtype, with float16/bfloat16 computed in float32.
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
    if weight is None:
        w = torch.ones_like(rows, dtype=acc)
    else:
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
    flat_w = w.reshape(-1, shape[-1])
    if not increasing:
        flat_y = -flat_y  # a decreasing fit of y is minus the increasing fit of -y
    out = _ext.isotonic(flat_y.contiguous(), flat_w.contiguous())
    if not increasing:
        out = -out
    if y_min is not None or y_max is not None:
        out = out.clamp(min=y_min, max=y_max)
    return out.reshape(shape).movedim(-1, dim).to(out_dtype)
