"""Calibrating a click-through-rate model with isotonic regression, on the Criteo data.

    python examples/ctr_calibration/calibrate.py --data .../dac/train.txt --out results/

The Criteo Display Advertising Challenge data has 45.8 million ad impressions from 7 days,
in chronological order, each with a click label, 13 integer and 26 hashed categorical
features. The script splits it into seven equal parts, "days", and:

1. trains a logistic regression with hashed features on days 1-5, keeping 10% of the
   non-clicks (negative downsampling, common in practice: it makes training cheaper, and
   leaves the model's probabilities far too high);
2. calibrates the model's scores on day 6 with: the analytic correction for the
   downsampling, Platt scaling, one isotonic regression over all of day 6, and one isotonic
   regression per segment (a categorical feature's frequent values) - the last two with
   scikit-learn on the CPU and with torchisotonic on the GPU, which must agree;
3. measures, on day 7, log-loss, Brier score and expected calibration error, per method;
4. times isotonic calibration on growing numbers of scores, up to all 45.8 million.

Writes results.json, results.md and reliability.png to --out.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import torchisotonic

DENSE = [f"I{i}" for i in range(1, 14)]
CATS = [f"C{i}" for i in range(1, 27)]
CRITEO_ROWS = 45_840_617
BUCKETS = 1 << 18  # hashed buckets per categorical feature
KEEP_NEGATIVES = 0.1
SEGMENT_MIN_ROWS = 5_000  # on day 6, for a segment of its own


# --- data -------------------------------------------------------------------------------------

_HEX = np.zeros(256, dtype=np.uint32)
for _i, _c in enumerate(b"0123456789abcdef"):
    _HEX[_c] = _i
for _i, _c in enumerate(b"ABCDEF"):
    _HEX[_c] = 10 + _i


def hex_codes(arr: pa.Array) -> np.ndarray:
    """The 32-bit values of an array of 8-digit hex strings (0 where missing), vectorized."""
    n = len(arr)
    out = np.zeros(n, dtype=np.uint32)
    buffers = arr.buffers()
    if buffers[2] is None:  # every value missing
        return out
    offsets = np.frombuffer(buffers[1], dtype=np.int32, count=arr.offset + n + 1)[arr.offset :]
    data = np.frombuffer(buffers[2], dtype=np.uint8)
    full = np.diff(offsets) == 8
    nibbles = _HEX[data[offsets[:-1][full, None] + np.arange(8)]]
    shifts = np.arange(28, -1, -4, dtype=np.uint32)
    out[full] = np.bitwise_or.reduce(nibbles << shifts, axis=1)
    return out


def read_criteo(path: Path, rows: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels [N] uint8, dense features [N, 13] float32 (log1p, 0 where missing) and
    categorical codes [N, 26] uint32 of the first `rows` rows (all if None)."""
    cap = rows or CRITEO_ROWS
    labels = np.empty(cap, dtype=np.uint8)
    dense = np.empty((cap, len(DENSE)), dtype=np.float32)
    cats = np.empty((cap, len(CATS)), dtype=np.uint32)
    names = ["label", *DENSE, *CATS]
    types = {
        "label": pa.int8(),
        **{c: pa.float32() for c in DENSE},
        **{c: pa.string() for c in CATS},
    }
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=names, block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(column_types=types, strings_can_be_null=True),
    )
    n = 0
    for batch in reader:
        m = min(batch.num_rows, cap - n)
        if m <= 0:
            break
        batch = batch.slice(0, m)
        labels[n : n + m] = batch.column(0).to_numpy(zero_copy_only=False)
        for j in range(len(DENSE)):
            x = batch.column(1 + j).to_numpy(zero_copy_only=False)
            dense[n : n + m, j] = np.log1p(np.maximum(np.nan_to_num(x), 0.0))
        for j in range(len(CATS)):
            cats[n : n + m, j] = hex_codes(batch.column(1 + len(DENSE) + j))
        n += m
    return labels[:n], dense[:n], cats[:n]


def days(n: int) -> list[slice]:
    """Seven equal consecutive parts of n rows: the data is in chronological order."""
    edges = np.linspace(0, n, 8).astype(int)
    return [slice(edges[d], edges[d + 1]) for d in range(7)]


# --- the model ---------------------------------------------------------------------------------


class HashedLogistic(torch.nn.Module):
    """Logistic regression on hashed categorical features and log-scaled counts."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = torch.nn.Embedding(len(CATS) * BUCKETS, 1)
        torch.nn.init.zeros_(self.weights.weight)
        self.dense = torch.nn.Linear(len(DENSE), 1)
        self.register_buffer("field", torch.arange(len(CATS)) * BUCKETS)

    def forward(self, dense: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        # codes: raw 32-bit values stored as int32; as unsigned, hashed into the field's buckets
        buckets = (codes.long() & 0xFFFFFFFF) % BUCKETS + self.field
        return self.weights(buckets).sum((1, 2)) + self.dense(dense).squeeze(1)


def train(
    model: HashedLogistic,
    dense: torch.Tensor,
    codes: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        order = torch.randperm(len(labels), device=labels.device)
        for start in range(0, len(order), 1 << 14):
            i = order[start : start + (1 << 14)]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(dense[i], codes[i]), labels[i]).backward()
            opt.step()


@torch.no_grad()
def scores(model: HashedLogistic, dense: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """The model's click probabilities, float64."""
    out = [
        torch.sigmoid(model(dense[s : s + (1 << 20)], codes[s : s + (1 << 20)]).double())
        for s in range(0, len(dense), 1 << 20)
    ]
    return torch.cat(out)


# --- calibration ---------------------------------------------------------------------------------


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """np.interp on the GPU: piecewise linear through (xp, fp), constant beyond its ends."""
    i = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    t = ((x - x0) / (x1 - x0).clamp_min(1e-300)).clamp(0, 1)
    return f0 + t * (f1 - f0)


def isotonic_fit(keys: torch.Tensor, y: torch.Tensor, offset: torch.Tensor | None = None):
    """The isotonic fit of y over sorted unique keys, as sklearn's IsotonicRegression does
    it: equal keys pooled into their mean, weighted by their count. `offset` (per element)
    is added to the means before the fit and subtracted after: segments placed this far
    apart never pool across, so one fit serves them all."""
    keys, order = torch.sort(keys)
    xp, inverse, counts = torch.unique_consecutive(keys, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(xp), dtype=torch.float64, device=keys.device)
    sums.index_add_(0, inverse, y[order])
    means = sums / counts
    shift = 0.0
    if offset is not None:
        shift = torch.zeros_like(means)
        shift[inverse] = offset[order]
        means = means + shift
    fit = torchisotonic.isotonic(means[None], counts.double()[None])[0]
    return xp, fit - shift


def platt(s: torch.Tensor, y: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Platt scaling: a logistic regression of y on logit(s) (scikit-learn's, on the CPU)."""
    z = torch.logit(s.clamp(1e-12, 1 - 1e-12)).cpu().numpy()[:, None]
    fit = LogisticRegression(C=1e6).fit(z, y.cpu().numpy())
    a, b = float(fit.coef_[0, 0]), float(fit.intercept_[0])
    return lambda t: torch.sigmoid(a * torch.logit(t.clamp(1e-12, 1 - 1e-12)) + b)


def metrics(p: torch.Tensor, y: torch.Tensor, bins: int = 15) -> dict[str, float]:
    q = p.clamp(1e-7, 1 - 1e-7)
    logloss = -(y * q.log() + (1 - y) * (1 - q).log()).mean()
    which = (p * bins).long().clamp(max=bins - 1)
    count = torch.bincount(which, minlength=bins).double()
    conf = torch.bincount(which, weights=p, minlength=bins)
    acc = torch.bincount(which, weights=y, minlength=bins)
    ece = (conf - acc).abs().sum() / len(p)
    return {
        "logloss": float(logloss),
        "brier": float(((p - y) ** 2).mean()),
        "ece": float(ece),
        "mean_p": float(p.mean()),
        "ctr": float(y.mean()),
        "nonempty_bins": int((count > 0).sum()),
    }


def auc(p: torch.Tensor, y: torch.Tensor) -> float:
    """Area under the ROC curve, from the ranks of the positives (ties take their mean)."""
    _, order = torch.sort(p)
    ps = p[order]
    _, inverse, counts = torch.unique_consecutive(ps, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    mean_rank = (ends - (counts.double() - 1) / 2)[inverse]
    pos = y[order] > 0.5
    n1, n0 = float(pos.sum()), float((~pos).sum())
    return (float(mean_rank[pos].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_logloss_gain(p_a: torch.Tensor, p_b: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """How much lower b's log-loss is than a's on the same rows: the mean of the per-row
    differences, with its standard error and a normal 95% interval."""

    def per_row(p: torch.Tensor) -> torch.Tensor:
        q = p.clamp(1e-7, 1 - 1e-7)
        return -(y * q.log() + (1 - y) * (1 - q).log())

    d = per_row(p_a) - per_row(p_b)
    mean, se = float(d.mean()), float(d.std() / len(d) ** 0.5)
    return {"mean": mean, "se": se, "low": mean - 1.96 * se, "high": mean + 1.96 * se}


def log(*parts: object) -> None:
    print(*parts, flush=True)


def best_time(fn: Callable[[], object], repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


# --- the experiment ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="Criteo's dac/train.txt")
    parser.add_argument("--rows", type=int, default=None, help="only the first rows")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--segment-min-rows", type=int, default=SEGMENT_MIN_ROWS)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)  # the model's initialization and batch order: reproducible results
    res: dict[str, object] = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    }

    t0 = time.perf_counter()
    labels, dense, cats = read_criteo(args.data, args.rows)
    n = len(labels)
    log(f"read {n:,} rows in {time.perf_counter() - t0:.0f} s; click rate {labels.mean():.4f}")
    part = days(n)
    train_rows, cal_rows, test_rows = slice(0, part[4].stop), part[5], part[6]

    def to_dev(rows: slice | np.ndarray):
        return (
            torch.from_numpy(dense[rows]).to(dev),
            torch.from_numpy(cats[rows].view(np.int32)).to(dev),
            torch.from_numpy(labels[rows].astype(np.float64)).to(dev),
        )

    # 1. Training on days 1-5 with 10% of the non-clicks.
    rng = np.random.default_rng(0)
    idx = np.arange(train_rows.stop)
    keep = (labels[idx] == 1) | (rng.random(len(idx)) < KEEP_NEGATIVES)
    xd, xc, yt = to_dev(idx[keep])
    model = HashedLogistic().to(dev)
    t0 = time.perf_counter()
    train(model, xd, xc, yt.float(), args.epochs)
    res["train"] = {"rows": int(keep.sum()), "seconds": time.perf_counter() - t0}
    log(f"trained on {int(keep.sum()):,} rows in {res['train']['seconds']:.0f} s")
    del xd, xc, yt

    cal_d, cal_c, cal_y = to_dev(cal_rows)
    test_d, test_c, test_y = to_dev(test_rows)
    cal_s, test_s = scores(model, cal_d, cal_c), scores(model, test_d, test_c)
    res["auc_test"] = auc(test_s, test_y)

    # 2. Calibration on day 6, evaluation on day 7.
    calibrated: dict[str, torch.Tensor] = {"none": test_s}
    w = KEEP_NEGATIVES  # q = p / (p + (1 - p) / w) undoes negative downsampling
    calibrated["downsampling correction"] = test_s / (test_s + (1 - test_s) / w)
    calibrated["Platt"] = platt(cal_s, cal_y)(test_s)
    xp, fp = isotonic_fit(cal_s, cal_y)
    calibrated["isotonic"] = interp(test_s, xp, fp)

    # The same with scikit-learn, which must agree.
    cal_np, cal_y_np, test_np = cal_s.cpu().numpy(), cal_y.cpu().numpy(), test_s.cpu().numpy()
    sk = IsotonicRegression(out_of_bounds="clip").fit(cal_np, cal_y_np)
    res["global_isotonic"] = {
        "rows": len(cal_np),
        "unique_scores": len(xp),
        "max_diff_vs_sklearn": float(
            np.abs(sk.predict(test_np) - calibrated["isotonic"].cpu().numpy()).max()
        ),
        "sklearn_seconds": best_time(
            lambda: IsotonicRegression(out_of_bounds="clip").fit(cal_np, cal_y_np)
        ),
        "torchisotonic_seconds": best_time(lambda: isotonic_fit(cal_s, cal_y)),
        # The same with the scores and labels starting in host memory, as scikit-learn's do.
        "torchisotonic_from_host_seconds": best_time(
            lambda: isotonic_fit(
                torch.from_numpy(cal_np).to(dev), torch.from_numpy(cal_y_np).to(dev)
            )
        ),
    }
    log("global isotonic:", res["global_isotonic"])

    # Per segment: the categorical feature with the most values frequent on day 6.
    cal_codes = cats[cal_rows]
    best = (0, 0, np.empty(0, dtype=np.uint32))
    for f in range(len(CATS)):
        values, counts = np.unique(cal_codes[:, f], return_counts=True)
        frequent = values[counts >= args.segment_min_rows]
        if len(frequent) > best[1]:
            best = (f, len(frequent), frequent)
    field, k, frequent = best
    if k == 0:
        raise SystemExit(f"no categorical value has {args.segment_min_rows} rows on day 6")
    seg_cal = np.searchsorted(frequent, cal_codes[:, field])
    seg_cal[(seg_cal >= k) | (frequent[np.minimum(seg_cal, k - 1)] != cal_codes[:, field])] = k
    test_codes = cats[test_rows][:, field]
    seg_test = np.searchsorted(frequent, test_codes)
    seg_test[(seg_test >= k) | (frequent[np.minimum(seg_test, k - 1)] != test_codes)] = k
    in_cal, in_test = seg_cal < k, seg_test < k  # rows of the frequent values; the rest: global

    # torchisotonic: one fit over all segments, keys segment + score, means offset by 2 segment.
    scal = torch.from_numpy(seg_cal).to(dev)
    stest = torch.from_numpy(seg_test).to(dev)
    mc, mt = torch.from_numpy(in_cal).to(dev), torch.from_numpy(in_test).to(dev)
    s_cal = cal_s.clamp(max=1 - 1e-12)[mc]
    keys_cal = scal[mc].double() + s_cal

    def per_segment_fit():
        return isotonic_fit(keys_cal, cal_y[mc], offset=2.0 * scal[mc].double())

    kp, kf = per_segment_fit()
    seg_of_key = kp.floor().long()
    first = torch.searchsorted(seg_of_key, torch.arange(k, device=dev))
    last = torch.searchsorted(seg_of_key, torch.arange(k, device=dev), right=True) - 1
    tk = stest[mt].double() + test_s.clamp(max=1 - 1e-12)[mt]
    lo, hi = kp[first[stest[mt]]], kp[last[stest[mt]]]
    per_seg = calibrated["isotonic"].clone()
    per_seg[mt] = interp(torch.minimum(torch.maximum(tk, lo), hi), kp, kf)
    calibrated["isotonic per segment"] = per_seg

    # scikit-learn: a loop over the segments.
    def sklearn_per_segment() -> np.ndarray:
        out = sk.predict(test_np)
        for g in range(k):
            a, b = seg_cal == g, seg_test == g
            if b.any():
                fit = IsotonicRegression(out_of_bounds="clip").fit(cal_np[a], cal_y_np[a])
                out[b] = fit.predict(test_np[b])
        return out

    # For the timing, the fits alone, as per_segment_fit: each segment's rows gathered first.
    order = np.argsort(seg_cal, kind="stable")
    bounds = np.searchsorted(seg_cal[order], np.arange(k + 1))
    groups = [order[bounds[g] : bounds[g + 1]] for g in range(k)]
    seg_x, seg_y = [cal_np[i] for i in groups], [cal_y_np[i] for i in groups]

    def sklearn_per_segment_fit() -> None:
        for x, y in zip(seg_x, seg_y, strict=True):
            IsotonicRegression(out_of_bounds="clip").fit(x, y)

    sk_seg = sklearn_per_segment()
    res["per_segment_isotonic"] = {
        "feature": CATS[field],
        "segments": int(k),
        "rows_in_segments": int(in_cal.sum()),
        "max_diff_vs_sklearn": float(np.abs(sk_seg - per_seg.cpu().numpy()).max()),
        "sklearn_seconds": best_time(sklearn_per_segment_fit, repeats=1),
        "torchisotonic_seconds": best_time(per_segment_fit),
    }
    log("per-segment isotonic:", res["per_segment_isotonic"])

    res["day7"] = {name: metrics(p, test_y) for name, p in calibrated.items()}
    res["day7_segments"] = {
        name: metrics(p[mt], test_y[mt]) for name, p in calibrated.items() if name != "none"
    }
    # Is the per-segment gain over one curve more than noise? Paired over the same rows.
    res["per_segment_gain"] = paired_logloss_gain(
        calibrated["isotonic"][mt], calibrated["isotonic per segment"][mt], test_y[mt]
    )
    res["isotonic_over_platt_gain"] = paired_logloss_gain(
        calibrated["Platt"], calibrated["isotonic"], test_y
    )
    log("log-loss gains:", res["per_segment_gain"], res["isotonic_over_platt_gain"])

    # 3. Isotonic calibration on growing numbers of scores (all days, for the timing only).
    all_d, all_c, all_y = to_dev(slice(0, n))
    all_s = scores(model, all_d, all_c)
    del all_d, all_c
    scaling = []
    for m in [10**5, 10**6, 10**7, n]:
        if m > n:
            continue
        s, y = all_s[:m], all_y[:m]
        sn, yn = s.cpu().numpy(), y.cpu().numpy()
        row = {
            "scores": m,
            "torchisotonic_seconds": best_time(lambda s=s, y=y: isotonic_fit(s, y)),
            "torchisotonic_from_host_seconds": best_time(
                lambda sn=sn, yn=yn: isotonic_fit(
                    torch.from_numpy(sn).to(dev), torch.from_numpy(yn).to(dev)
                )
            ),
            "sklearn_seconds": best_time(
                lambda sn=sn, yn=yn: IsotonicRegression(out_of_bounds="clip").fit(sn, yn),
                repeats=1,
            ),
        }
        scaling.append(row)
        log("scaling:", row)
    res["scaling"] = scaling

    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    (args.out / "results.md").write_text(markdown(res))
    reliability(calibrated, test_y, args.out / "reliability.png")
    log(markdown(res))


def markdown(res: dict) -> str:
    lines = [f"Device: {res['device']}; test AUC {res['auc_test']:.4f}.", ""]
    lines += ["| calibration (fit on day 6) | log-loss | Brier | ECE | mean p | CTR |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]  # fmt: skip
    for name, m in res["day7"].items():
        lines.append(
            f"| {name} | {m['logloss']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} | "
            f"{m['mean_p']:.4f} | {m['ctr']:.4f} |"
        )
    seg = res["per_segment_isotonic"]
    title = f"Rows of the {seg['segments']} frequent values of {seg['feature']} on day 7:"
    lines += ["", title, "", "| calibration | log-loss | ECE |", "| --- | ---: | ---: |"]
    for name, m in res["day7_segments"].items():
        lines.append(f"| {name} | {m['logloss']:.4f} | {m['ece']:.4f} |")
    gains = [
        ("per_segment_gain", "per segment over one curve, on those rows"),
        ("isotonic_over_platt_gain", "isotonic over Platt, on all of day 7"),
    ]
    for key, what in gains:
        gain = res[key]
        lines.append(
            f"\nLog-loss gain, {what}: {gain['mean']:.2e} "
            f"(95% interval {gain['low']:.2e} to {gain['high']:.2e})."
        )
    g = res["global_isotonic"]
    lines += ["", "| fit | rows | scikit-learn (CPU), s | torchisotonic, s | "
              "from host memory, s | max difference |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]  # fmt: skip
    lines.append(
        f"| one curve | {g['rows']:,} | {g['sklearn_seconds']:.3f} | "
        f"{g['torchisotonic_seconds']:.4f} | {g['torchisotonic_from_host_seconds']:.4f} | "
        f"{g['max_diff_vs_sklearn']:.1e} |"
    )
    lines.append(
        f"| {seg['segments']} curves | {seg['rows_in_segments']:,} | "
        f"{seg['sklearn_seconds']:.3f} | {seg['torchisotonic_seconds']:.4f} | - | "
        f"{seg['max_diff_vs_sklearn']:.1e} |"
    )
    lines += ["", "| scores | scikit-learn (CPU), s | torchisotonic, s | from host memory, s |",
              "| ---: | ---: | ---: | ---: |"]  # fmt: skip
    for row in res["scaling"]:
        lines.append(
            f"| {row['scores']:,} | {row['sklearn_seconds']:.3f} | "
            f"{row['torchisotonic_seconds']:.4f} | {row['torchisotonic_from_host_seconds']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def reliability(calibrated: dict[str, torch.Tensor], y: torch.Tensor, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="0.7", lw=1)
    for name, p in calibrated.items():
        order = torch.argsort(p)
        groups = torch.tensor_split(order, 50)  # 50 bins of equal mass
        ax.plot(
            [float(p[g].mean()) for g in groups],
            [float(y[g].mean()) for g in groups],
            marker=".",
            lw=1,
            label=name,
        )
    ax.set_xlabel("predicted click probability")
    ax.set_ylabel("observed click rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
