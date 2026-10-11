"""Batched isotonic regression with the semantics of ``sklearn.isotonic.isotonic_regression``."""

from __future__ import annotations

import torch
from torch import Tensor

from torchisotonic import _ext

# Few long problems go to the hull algorithm, where a whole GPU block works on each merge;
# many shorter ones to PAVA with one thread per problem. On the CPU, PAVA always wins.
HULL_MAX_BATCH = 1024
HULL_MIN_N = 4096
HULL_CHUNK = 512
HULL_ON_CPU = False  # the tests turn this on to check the hull algorithm without a GPU
# On CUDA: "tiles" (chunk hulls + hull merges, every shape), or the older "pava" / "hull"
# paths. Problems shorter than TILES_MIN_N would go to PAVA with one thread per problem.
CUDA_METHOD = "tiles"
TILES_MIN_N = 1


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
    out = _solve(flat_y, flat_w, increasing)
    if y_min is not None or y_max is not None:
        out = out.clamp(min=y_min, max=y_max)
    return out.reshape(shape).movedim(-1, dim).to(out_dtype)


def _solve(flat_y: Tensor, flat_w: Tensor | None, increasing: bool) -> Tensor:
    """The fit of [batch, n] problems (n >= 1), in flat_y's dtype."""
    if (
        flat_y.is_cuda and CUDA_METHOD == "tiles" and flat_y.shape[1] >= TILES_MIN_N
    ):  # pragma: no cover - needs a GPU
        return _ext.isotonic_tiles(
            flat_y.contiguous(), None if flat_w is None else flat_w.contiguous(), increasing
        )
    w = torch.ones_like(flat_y) if flat_w is None else flat_w
    y = flat_y if increasing else -flat_y  # a decreasing fit of y is minus the increasing fit of -y
    if _use_hull(y):
        out = _hull(y, w).to(flat_y.dtype)
    else:
        out = _ext.isotonic(y.contiguous(), w.contiguous())
    return out if increasing else -out


def _use_hull(flat: Tensor) -> bool:
    batch, n = flat.shape
    return (flat.is_cuda or HULL_ON_CPU) and batch <= HULL_MAX_BATCH and n >= HULL_MIN_N


def _hull(flat_y: Tensor, flat_w: Tensor) -> Tensor:
    """The fit of [batch, n] problems by the hull algorithm (float64 result).

    Isotonic regression commutes with shifting y, so the operator shifts each row by its
    weighted mean first: that keeps the cumulative sums, and their rounding errors, small.
    """
    # Any rough centre does; summing in float64 avoids float64 copies of the whole batch.
    centre = (flat_w * flat_y).sum(-1, dtype=torch.float64) / flat_w.sum(-1, dtype=torch.float64)
    return _ext.isotonic_hull(flat_y.contiguous(), flat_w.contiguous(), centre, HULL_CHUNK)
