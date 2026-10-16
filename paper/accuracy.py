"""How exact the fit is, on hard data: not only its distance from a reference.

    python paper/accuracy.py [--device cuda|cpu] [--max-n N] [--fast]

For every data pattern, length, dtype and weighting, the fit is checked for

- monotonicity: the largest step down, relative to the data's scale;
- error: the largest difference from the float64 CPU fit of the same data, relative to 1 + |fit|;
- objective: how much larger its weighted least-squares objective is than the reference's,
  relative to the reference's (or, where that is near zero, to what the dtype can resolve);
- KKT: the largest violation of the optimality conditions, computed from the fit alone in
  float64 (multipliers lambda_i = -2 * sum_{j <= i} w_j (fit_j - y_j) must be >= 0, zero at
  the end and zero where the fit steps up), relative to sum w |y|;
- blocks: the number of constant blocks, against the reference's.

"The same data" is the data as the solver saw it: cast to the dtype, then back to float64.
What a cast to float32 loses (a 1e-9 ripple on values near 1e6, say) is lost before the
solver runs, so it is not charged to it. Prints one markdown table. The reference is
torchisotonic's own float64 CPU PAVA, which tests/test_isotonic.py checks against scipy.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch

import torchisotonic

Pattern = Callable[[int, torch.Generator], torch.Tensor]


def consecutive(n: int) -> torch.Tensor:
    """n strictly increasing values exact in float32 (and float64): the consecutive floats from
    1.0; arange would repeat values in float32 past 2^24."""
    return (torch.arange(n, dtype=torch.int32) + 0x3F800000).view(torch.float32).double()


def patterns() -> dict[str, Pattern]:
    def i(n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.float64)

    return {
        "increasing": lambda n, g: consecutive(n),
        "decreasing": lambda n, g: -i(n),
        "constant": lambda n, g: torch.ones(n, dtype=torch.float64),
        "piecewise": lambda n, g: (
            (i(n) // max(1, n // 50)).float().double()
            + 0.01 * torch.randn(n, generator=g, dtype=torch.float64)
        ),
        "gaussian": lambda n, g: torch.randn(n, generator=g, dtype=torch.float64),
        "walk": lambda n, g: torch.randn(n, generator=g, dtype=torch.float64).cumsum(0),
        "saw": lambda n, g: i(n) % 7 + (i(n) % 1500) * 0.1,
        "cliff": lambda n, g: torch.where(i(n) == n - 1, -1e9, i(n)),
        "collinear": lambda n, g: (
            1e6 + 1e-3 * i(n) + 1e-9 * torch.randn(n, generator=g, dtype=torch.float64)
        ),
        "offset": lambda n, g: 1e6 + torch.randn(n, generator=g, dtype=torch.float64),
    }


def weights(kind: str, n: int, g: torch.Generator) -> torch.Tensor | None:
    if kind == "none":
        return None
    if kind == "uniform":
        return torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    return torch.exp(8 * torch.rand(n, generator=g, dtype=torch.float64))  # extreme: 1 .. e^8


def kkt_violation(y: torch.Tensor, w: torch.Tensor, fit: torch.Tensor) -> float:
    """Largest violation of the KKT conditions by `fit`, relative to sum w |y|."""
    r = w * (fit - y)
    lam = -2 * r.cumsum(0)  # multipliers of fit_i <= fit_{i+1}, i = 0..n-1; lam[-1] must be 0
    scale = 2 * float((w * y.abs()).sum()) + 1e-300
    dual = float((-lam).clamp(min=0).max()) if lam.numel() else 0.0
    end = abs(float(lam[-1]))
    steps = fit[1:] > fit[:-1]  # where the constraint is slack, lam must be 0
    slack = float(lam[:-1][steps].abs().max()) if bool(steps.any()) else 0.0
    return max(dual, end, slack) / scale


def blocks(fit: torch.Tensor) -> int:
    eps = 16 * torch.finfo(fit.dtype).eps
    a, b = fit[1:], fit[:-1]
    return 1 + int((~((a - b).abs() <= eps * torch.maximum(a.abs(), b.abs()))).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-n", type=int, default=10_000_000)
    parser.add_argument("--fast", action="store_true", help="the fit with exact=False")
    args = parser.parse_args()
    lengths = [n for n in (1_000, 100_000, 10_000_000) if n <= args.max_n]
    print(f"device {args.device}, torch {torch.__version__}, exact={not args.fast}")
    print("| pattern | n | dtype | weights | monotone | error | objective | KKT | blocks |")
    print("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |")
    worst: dict[str, float] = {}
    for name, make in patterns().items():
        for n in lengths:
            for dtype in (torch.float32, torch.float64):
                for wkind in ("none", "uniform", "extreme"):
                    g = torch.Generator().manual_seed(n)
                    y = make(n, g).to(dtype)  # the problem, as the solver sees it
                    w = None if (w0 := weights(wkind, n, g)) is None else w0.to(dtype)
                    y64, w64 = y.double(), None if w is None else w.double()
                    ref = torchisotonic.isotonic(y64, w64)
                    fit = torchisotonic.isotonic(
                        y.to(args.device),
                        None if w is None else w.to(args.device),
                        exact=not args.fast,
                    )
                    fit = fit.double().cpu()
                    ones = torch.ones(n, dtype=torch.float64)
                    ww = ones if w64 is None else w64
                    scale = float(y64.abs().max()) + 1e-300
                    mono = float((fit[:-1] - fit[1:]).clamp(min=0).max()) / scale if n > 1 else 0.0
                    err = float(((fit - ref).abs() / (1 + ref.abs())).max())
                    obj = float((ww * (fit - y64) ** 2).sum())
                    obj_ref = float((ww * (ref - y64) ** 2).sum())
                    resolvable = torch.finfo(dtype).eps * float((ww * y64**2).sum())
                    gap = (obj - obj_ref) / (max(obj_ref, resolvable) + 1e-300)
                    kkt = kkt_violation(y64, ww, fit)
                    nb, nb_ref = blocks(fit), blocks(ref)
                    tag = f"{dtype}".replace("torch.", "")
                    print(
                        f"| {name} | {n:,} | {tag} | {wkind} | {mono:.1e} | {err:.1e} | "
                        f"{gap:+.1e} | {kkt:.1e} | {nb} / {nb_ref} |",
                        flush=True,
                    )
                    for key, v in (
                        ("monotone", mono),
                        ("error", err),
                        ("objective", gap),
                        ("KKT", kkt),
                    ):
                        worst[f"{key} {tag}"] = max(
                            worst.get(f"{key} {tag}", 0.0), v if math.isfinite(v) else math.inf
                        )
    print(
        "\nworst per metric and dtype: "
        + ", ".join(f"{k} {v:.1e}" for k, v in sorted(worst.items()))
    )


if __name__ == "__main__":
    main()
