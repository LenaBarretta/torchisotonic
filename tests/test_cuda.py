"""The CUDA operator must give what the CPU one does (which test_isotonic checks against scipy)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from test_grad import close_values

import torchisotonic
from torchisotonic import _api, _ext

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


TILE_SHAPES = [1, 2, 31, 32, 33, 64, 100, 128, 255, 511, 512, 1000, 1023, 1024, 1025, 2048]
TILE_SHAPES += [4095, 4096, 4097, 8193, 100_000]


@pytest.mark.parametrize("n", TILE_SHAPES)
@pytest.mark.parametrize("batch", [1, 3, 70])
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


@pytest.mark.parametrize("n", [100, 1000, 5000, 100_000])
@pytest.mark.parametrize("weighted", [False, True])
def test_exact_flag(n: int, weighted: bool) -> None:  # pragma: no cover
    # exact=True is the default; exact=False changes float32 only, and on ordinary data stays
    # within float32's tolerance of the CPU.
    g = torch.Generator().manual_seed(n)
    y = torch.linspace(0, 3, n, dtype=torch.float64) + torch.randn(
        8, n, generator=g, dtype=torch.float64
    )
    w = torch.rand(8, n, generator=g, dtype=torch.float64) + 0.1 if weighted else None
    yc, wc = y.cuda(), None if w is None else w.cuda()
    for dtype in (torch.float32, torch.float64):
        wd = None if wc is None else wc.to(dtype)
        default = torchisotonic.isotonic(yc.to(dtype), wd)
        assert torch.equal(default, torchisotonic.isotonic(yc.to(dtype), wd, exact=True))
        fast = torchisotonic.isotonic(yc.to(dtype), wd, exact=False)
        if dtype == torch.float64:
            assert torch.equal(fast, default)
    both(y.float(), None if w is None else w.float(), tol=1e-4, exact=False)


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
def test_tiles_patterns(name: str, n: int) -> None:  # pragma: no cover
    g = torch.Generator().manual_seed(n)
    y = pattern(name, n, g)[None]
    w = torch.rand(1, n, generator=g, dtype=torch.float64) + 0.1
    both(y, tol=1e-9)
    both(y, w, tol=1e-9)
    both(y, w, tol=1e-9, increasing=False)
    both(y.float(), tol=1e-4)


@pytest.mark.parametrize(
    ("batch", "n"),
    [(70, 31), (70, 32), (9, 33), (9, 64), (9, 65), (7, 128), (5, 512), (5, 513), (5, 1000),
     (5, 1024), (3, 1025), (3, 5000)],  # every tile size, and the long-row path
)  # fmt: skip
def test_gradients_match_cpu(batch: int, n: int) -> None:  # pragma: no cover
    g = torch.Generator().manual_seed(n)
    y = torch.linspace(0, 3, n, dtype=torch.float64) + torch.randn(
        batch, n, generator=g, dtype=torch.float64
    )
    w = torch.rand(batch, n, generator=g, dtype=torch.float64) + 0.1
    c = torch.randn(batch, n, generator=g, dtype=torch.float64)
    grads = []
    for device in ("cpu", "cuda"):
        yy = y.to(device).requires_grad_()
        ww = w.to(device).requires_grad_()
        loss = (torchisotonic.isotonic(yy, ww, increasing=False) * c.to(device)).sum()
        gy, gw = torch.autograd.grad(loss, (yy, ww))
        grads.append((gy.cpu(), gw.cpu()))
    torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(grads[1][1], grads[0][1], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("n", [1024, 1025, 3000, 70_000])
def test_gradient_blocks_across_tiles(name: str, n: int) -> None:  # pragma: no cover
    # The CUDA gradient sums blocks over tiles of 1024: blocks within, across and over them.
    # It must be the block average over the blocks the GPU's fit reports, whatever the data.
    g = torch.Generator().manual_seed(n)
    y = pattern(name, n, g).expand(3, n).contiguous()
    w = torch.rand(3, n, generator=g, dtype=torch.float64) + 0.1
    c = torch.randn(3, n, generator=g, dtype=torch.float64)
    for weights in (w, None):
        yy = y.cuda().requires_grad_()
        ww = None if weights is None else weights.cuda().requires_grad_()
        fit = torchisotonic.isotonic(yy, ww)
        inputs = (yy,) if ww is None else (yy, ww)
        got = torch.autograd.grad((fit * c.cuda()).sum(), inputs)
        raw_w = None if ww is None else ww.detach()
        _, starts = _ext.isotonic_tiles(y.cuda(), raw_w, True, True, True)
        ones = torch.ones_like(y) if weights is None else weights
        g_sum, w_sum = _api._block_sums(starts.cpu(), c, ones)
        mean = g_sum / w_sum
        torch.testing.assert_close(got[0].cpu(), ones * mean, rtol=1e-9, atol=1e-9)
        if ww is not None:
            want = (y - fit.detach().cpu()) * mean
            torch.testing.assert_close(got[1].cpu(), want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", [p for p in PATTERNS if p not in ("constant", "saw")])
@pytest.mark.parametrize("n", [1024, 1025, 3000, 70_000])
def test_gradients_match_cpu_patterns(name: str, n: int) -> None:  # pragma: no cover
    # Against the CPU, on data whose neighbouring blocks have clearly different means. Ties
    # and near-ties (the constant and sawtooth patterns) are left out: there the fit is not
    # differentiable, or nearly not, and each solver's blocks are its own rounding of them.
    g = torch.Generator().manual_seed(n)
    y = pattern(name, n, g).expand(3, n).contiguous()
    w = torch.rand(3, n, generator=g, dtype=torch.float64) + 0.1
    c = torch.randn(3, n, generator=g, dtype=torch.float64)
    grads: list[tuple[torch.Tensor, ...]] = []
    for device in ("cpu", "cuda"):
        yy = y.to(device).requires_grad_()
        ww = w.to(device).requires_grad_()
        loss = (torchisotonic.isotonic(yy, ww) * c.to(device)).sum()
        gy, gw = torch.autograd.grad(loss, (yy, ww))
        grads.append((gy.cpu(), gw.cpu()))
        (gy,) = torch.autograd.grad((torchisotonic.isotonic(yy) * c.to(device)).sum(), (yy,))
        grads.append((gy.cpu(),))
    torch.testing.assert_close(grads[2][0], grads[0][0], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(grads[3][0], grads[1][0], rtol=1e-9, atol=1e-9)
    # d/dw = (y - fit) G / W: where the fit is y (one-element blocks), the CPU's is y exactly
    # and the GPU's (a slope of cumulative sums) y up to rounding, so the error scales with y.
    scale = 1e-9 * (1 + float(y.detach().abs().max()))
    torch.testing.assert_close(grads[2][1], grads[0][1], rtol=1e-9, atol=scale)


@pytest.mark.parametrize("row", range(2))
@pytest.mark.parametrize("batch", [1, 3])
def test_gradient_close_values(row: int, batch: int) -> None:  # pragma: no cover
    # Neighbours a few units in the last place apart are still separate blocks on the GPU:
    # increasing data is its own fit, so the gradient passes unchanged.
    y = close_values()[row].expand(batch, -1).contiguous().cuda().requires_grad_()
    c = torch.randn(y.shape, generator=torch.Generator().manual_seed(row)).cuda()
    (grad,) = torch.autograd.grad((torchisotonic.isotonic(y) * c).sum(), (y,))
    assert torch.equal(grad, c)


@pytest.mark.parametrize("name", ["collinear", "offset", "consecutive"])
@pytest.mark.parametrize("n", [1000, 100_000])
def test_float32_fit_is_monotone(name: str, n: int) -> None:  # pragma: no cover
    # Where neighbouring slopes are within an ulp of each other, as rounded float32 values
    # they must still not step down.
    g = torch.Generator().manual_seed(n)
    i = torch.arange(n, dtype=torch.float64)
    one = torch.tensor(1.0).view(torch.int32)
    y = {
        "collinear": 1e6 + 1e-3 * i + 1e-9 * torch.randn(n, generator=g, dtype=torch.float64),
        "offset": 1e6 + torch.randn(n, generator=g, dtype=torch.float64),
        "consecutive": (one + torch.arange(n, dtype=torch.int32)).view(torch.float32).double(),
    }[name].float()
    u = torch.rand(n, generator=g)
    for w in (None, u + 0.1, torch.exp(8 * u)):
        fit = torchisotonic.isotonic(y.cuda(), None if w is None else w.cuda())
        assert bool((fit[1:] >= fit[:-1]).all())


@pytest.mark.parametrize("n", [500, 3000, 100_000])  # packed, one chunk, merged chunks
def test_gradient_blocks_from_the_hull(n: int) -> None:  # pragma: no cover
    # Blocks (1 + u, 1, ..., 1) of s elements, u the float32 spacing at 1, s falling from S
    # to 3: their means 1 + u / s increase, so they are separate blocks, and all round to 1;
    # the rest of the row is 2, 3, ..., a block per element. The gradient takes its blocks
    # from the hull, not from equal fitted values.
    sizes: list[int] = []
    while sum(sizes) + len(sizes) + 3 <= n:
        sizes.append(len(sizes) + 3)
    sizes.reverse()
    u = torch.finfo(torch.float32).eps
    y = torch.ones(n)
    starts = torch.ones(n, dtype=torch.bool)
    at = 0
    for size in sizes:
        y[at] = 1 + u
        starts[at + 1 : at + size] = False
        at += size
    y[at:] = torch.arange(2, 2 + n - at)
    y, starts = y[None], starts[None]
    c = torch.randn(1, n, generator=torch.Generator().manual_seed(n))
    yy = y.cuda().requires_grad_()
    fit = torchisotonic.isotonic(yy)
    assert bool((fit[:, :at] == 1).all())  # the case this test is about
    (grad,) = torch.autograd.grad((fit * c.cuda()).sum(), (yy,))
    g_sum, count = _api._block_sums(starts, c, torch.ones_like(c))
    torch.testing.assert_close(grad.cpu(), g_sum / count, rtol=1e-5, atol=1e-6)
