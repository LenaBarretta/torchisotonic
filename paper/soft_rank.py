"""Soft rank and a training step: torchsort's SoftRank against a copy of it in which only the
isotonic solver, forward and backward, is torchisotonic's; then the agreement of both with a
float64 reference, values and gradients, with the blocks each solver used.

    python paper/soft_rank.py

ms per call: median and quartiles of 20 calls after a warm-up, CUDA events around each call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torchsort
from torchsort.ops import _arange_like, _inv_permutation, isotonic_l2, isotonic_l2_backward

import torchisotonic
from torchisotonic import _api, _ext

SHAPES = [(1024, 100), (256, 1000), (64, 10_000), (8, 100_000), (1, 1_000_000)]
STEP_SHAPES = [(256, 1000), (32, 10_000), (4, 100_000)]
STRENGTHS: tuple[tuple[str, Callable[[int], float]], ...] = (
    ("1", lambda n: 1.0),
    ("1/n", lambda n: 1.0 / n),
)


class OursSoftRank(torch.autograd.Function):
    """torchsort.ops.SoftRank (l2), with torchisotonic as the isotonic solver."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, regularization_strength: float) -> torch.Tensor:
        ctx.scale = 1.0 / regularization_strength
        w = _arange_like(tensor, reverse=True) + 1
        theta = tensor * ctx.scale
        s, permutation = torch.sort(theta, descending=True)
        inv_permutation = _inv_permutation(permutation)
        with torch.enable_grad():
            z = (s - w).detach().requires_grad_()
            dual = torchisotonic.isotonic(z, increasing=False)
        ctx.z, ctx.dual = z, dual
        ctx.save_for_backward(permutation, inv_permutation)
        return (s - dual.detach()).gather(1, inv_permutation)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        permutation, inv_permutation = ctx.saved_tensors
        grad = grad_output.clone()
        sorted_grad = grad.gather(1, permutation)
        (iso,) = torch.autograd.grad(ctx.dual, ctx.z, sorted_grad, retain_graph=True)
        grad -= iso.gather(1, inv_permutation)
        return grad * ctx.scale, None


def ours(x: torch.Tensor, eps: float) -> torch.Tensor:
    return OursSoftRank.apply(x, eps)


def theirs(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torchsort.soft_rank(x, regularization_strength=eps)


SOLVERS = (("torchsort", theirs), ("ours", ours))


def times_ms(fn: Callable[[], object], repeats: int = 20) -> tuple[float, float, float]:
    fn()
    torch.cuda.synchronize()
    torch.cuda._sleep(500_000_000)  # a warm GPU: see bench/bench.py
    out = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end))
    t = np.array(out)
    return float(np.median(t)), float(np.percentile(t, 25)), float(np.percentile(t, 75))


def grad_of(fn: Callable[[torch.Tensor, float], torch.Tensor], x: torch.Tensor, eps: float,
            g: torch.Tensor) -> object:  # fmt: skip
    xr = x.detach().requires_grad_()
    return torch.autograd.grad(fn(xr, eps), xr, g)


def time_soft_rank() -> None:
    print("OP | batch x n | eps | impl | fwd median (q1-q3) | fwd+bwd median (q1-q3)", flush=True)
    for b, n in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(b, n, device="cuda")
        g = torch.randn(b, n, device="cuda")
        for eps_name, strength in STRENGTHS:
            eps = strength(n)
            for name, fn in SOLVERS:
                fw = times_ms(lambda fn=fn, x=x, eps=eps: fn(x, eps))
                bw = times_ms(lambda fn=fn, x=x, eps=eps, g=g: grad_of(fn, x, eps, g))
                print(
                    f"OP | {b}x{n} | {eps_name} | {name} | {fw[0]:.3f} ({fw[1]:.3f}-{fw[2]:.3f}) | "
                    f"{bw[0]:.3f} ({bw[1]:.3f}-{bw[2]:.3f})",
                    flush=True,
                )


class Model(torch.nn.Module):
    """A compact scoring model: one score per item from 32 features."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def spearman_loss(ranks: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus the Spearman correlation of soft ranks with normalized target ranks."""
    r = ranks - ranks.mean(1, keepdim=True)
    r = r / r.norm(dim=1, keepdim=True)
    return 1 - (r * target).sum(1).mean()


def time_training_step() -> None:
    print("\nSTEP | batch x n | eps | impl | step median (q1-q3) | soft rank fwd+bwd", flush=True)
    for b, n in STEP_SHAPES:
        torch.manual_seed(0)
        x = torch.randn(b, n, 32, device="cuda")
        target = x[..., :4].sum(-1) + 0.5 * torch.randn(b, n, device="cuda")
        t = target.argsort(1).argsort(1).float()
        t = t - t.mean(1, keepdim=True)
        t = t / t.norm(dim=1, keepdim=True)
        for eps_name, strength in STRENGTHS:
            eps = strength(n)
            for name, fn in SOLVERS:
                torch.manual_seed(1)
                model = Model().cuda()
                opt = torch.optim.Adam(model.parameters(), 1e-3)

                def step(fn: Callable[[torch.Tensor, float], torch.Tensor] = fn,
                         model: Model = model, opt: torch.optim.Optimizer = opt,
                         eps: float = eps, x: torch.Tensor = x,
                         t: torch.Tensor = t) -> None:  # fmt: skip
                    opt.zero_grad(set_to_none=True)
                    spearman_loss(fn(model(x), eps), t).backward()
                    opt.step()

                st = times_ms(step)
                with torch.no_grad():
                    scores = model(x)
                gg = torch.randn_like(scores)
                rank = times_ms(lambda fn=fn, s=scores, eps=eps, gg=gg: grad_of(fn, s, eps, gg))
                print(
                    f"STEP | {b}x{n} | {eps_name} | {name} | "
                    f"{st[0]:.3f} ({st[1]:.3f}-{st[2]:.3f}) | {rank[0]:.3f}",
                    flush=True,
                )


def starts_of(a: torch.Tensor) -> torch.Tensor:
    """Where a block starts, from a vector-Jacobian product of the isotonic step: A g is
    constant on each block and, for a random g, differs between blocks."""
    st = torch.ones_like(a, dtype=torch.bool)
    st[:, 1:] = a[:, 1:] != a[:, :-1]
    return st


def check_gradients() -> None:
    print(
        "\nGRAD | batch x n | eps | impl | linf | l2 | max|ref| | starts differ | "
        "...of which gap 0 / <=1e-12 / >1e-12 | max gap",
        flush=True,
    )
    for b, n in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(b, n, device="cuda")
        for eps_name, strength in STRENGTHS:
            eps = strength(n)
            w = _arange_like(x, reverse=True) + 1
            s, _ = torch.sort(x / eps, descending=True)
            z = s - w
            g = torch.randn_like(z)
            d_ts = isotonic_l2["cuda"](z)
            a_ts = isotonic_l2_backward["cuda"](s, d_ts, g)
            zr = z.detach().requires_grad_()
            d_ours = torchisotonic.isotonic(zr, increasing=False)
            (a_ours,) = torch.autograd.grad(d_ours, zr, g)
            # The reference: float64 PAVA on the CPU of the same float32 problem, with its blocks.
            z64 = z.double().cpu()
            ref, st_ref = _ext.isotonic(-z64, torch.ones_like(z64), True)
            ref = -ref
            g64 = g.double().cpu()
            g_sum, count = _api._block_sums(st_ref, g64, torch.ones_like(g64))
            a_ref = g_sum / count
            scale = float(ref.abs().max())
            for name, a, fit in (("torchsort", a_ts, d_ts), ("ours", a_ours, d_ours.detach())):
                a, fit = a.double().cpu(), fit.double().cpu()
                linf = float((a - a_ref).abs().max() / a_ref.abs().max())
                l2 = float((a - a_ref).norm() / a_ref.norm())
                st = starts_of(a)
                diff = st != st_ref
                diff[:, 0] = False
                # The gap between the means on the two sides of a start that one partition has
                # and the other does not: the reference's means if it starts a block there,
                # else the solver's.
                rows, cols = diff.nonzero(as_tuple=True)
                gap_ref = (ref[rows, cols - 1] - ref[rows, cols]).abs()
                gap_fit = (fit[rows, cols - 1] - fit[rows, cols]).abs()
                gap = torch.where(st_ref[rows, cols], gap_ref, gap_fit) / scale
                zero = int((gap == 0).sum())
                small = int(((gap > 0) & (gap <= 1e-12)).sum())
                big = int((gap > 1e-12).sum())
                most = float(gap.max()) if gap.numel() else 0.0
                print(
                    f"GRAD | {b}x{n} | {eps_name} | {name} | {linf:.1e} | {l2:.1e} | "
                    f"{float(a_ref.abs().max()):.2f} | {int(diff.sum())} of {int(st_ref.sum())} | "
                    f"{zero} / {small} / {big} | {most:.1e}",
                    flush=True,
                )


def main() -> None:
    time_soft_rank()
    time_training_step()
    check_gradients()


if __name__ == "__main__":
    main()
