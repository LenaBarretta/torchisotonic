# torchisotonic

Batched isotonic regression for PyTorch tensors, fast on the GPU.

Isotonic regression finds the non-decreasing (or non-increasing) sequence closest to the
data in weighted least squares. It is what `sklearn.isotonic.isotonic_regression` and
`scipy.optimize.isotonic_regression` compute for one sequence; torchisotonic computes it for
every row of a tensor at once, on the CPU or with CUDA kernels on the GPU, from a few
elements per row to a hundred million.

```python
import torch
import torchisotonic

y = torch.randn(1000, 5000, device="cuda")
fit = torchisotonic.isotonic(y)  # every row, non-decreasing

w = torch.rand(1000, 5000, device="cuda") + 0.1
fit = torchisotonic.isotonic(y, w, increasing=False, y_min=0.0, y_max=1.0)
```

## What it is for

Whenever a monotone fit has to run often, on long data, for many curves at once, or inside a
model during training.

**Calibrating a classifier.** A model's scores rank well but are not probabilities; isotonic
regression maps each score to the rate of positives observed at that score, as scikit-learn's
`IsotonicRegression` and `CalibratedClassifierCV(method="isotonic")` do. A curve over 6.5 million
held-out scores takes 42 ms on a T4, against 2 s for scikit-learn. These two functions give
scikit-learn's curve and predictions, equal scores pooled into their mean:

```python
def fit_calibration(scores, labels):
    """The calibration curve, as scikit-learn's IsotonicRegression fits it."""
    s, order = torch.sort(scores)
    xs, inverse, counts = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(xs), dtype=torch.float64, device=scores.device)
    sums.index_add_(0, inverse, labels[order].double())
    return xs, torchisotonic.isotonic(sums / counts, counts.double())


def calibrate(new_scores, xs, fit):
    """Calibrated probabilities: linear between the curve's points, constant beyond them."""
    i = torch.searchsorted(xs, new_scores).clamp(1, len(xs) - 1)
    t = ((new_scores - xs[i - 1]) / (xs[i] - xs[i - 1])).clamp(0, 1)
    return fit[i - 1] + t * (fit[i] - fit[i - 1])
```

[`examples/ctr_calibration`](examples/ctr_calibration) calibrates a click model on 45.8
million Criteo impressions this way, with one curve per segment for 280 segments in one call.

**Many curves at once.** Every row of the input is its own problem, so a thousand curves cost
little more than one: per segment, per threshold, or per bootstrap sample. For instance, a
band of uncertainty around a calibration curve, from a Bayesian bootstrap of the labels `y`
ordered by score:

```python
w = torch.empty(1000, len(y), device=y.device).exponential_()  # one resample per row
curves = torchisotonic.isotonic(y.expand(1000, -1), w)
low, high = curves.quantile(0.05, dim=0), curves.quantile(0.95, dim=0)
```

**Inside a model.** The fit is differentiable in `y` and in the weights, so it can enforce
monotonicity in a network and train through it: predicted quantiles that never cross,
dose-response or survival curves that must not go the wrong way, or the projection step of
differentiable sorting and ranking.

```python
q = model(x)  # [batch, levels]: quantiles at increasing levels
q = torchisotonic.isotonic(q)  # never crossing, with gradients
loss = pinball(q, target, levels)  # any quantile loss
```

## Semantics

The semantics are scikit-learn's: weights must be strictly positive (a weight of the shape of
`y`, or one that broadcasts to it), the result is clipped to `[y_min, y_max]` if they are
given, `dim` picks the axis, and a row holding a NaN comes back as NaN. Float input keeps its
dtype (float16 and bfloat16 are computed in float32), integer input becomes the default
float dtype. The fit is differentiable in `y` and in the weights.

On the GPU, float32 is accurate to the format's rounding by default, and the fit never steps
down. `exact=False` makes it about 1.5–2 times faster on short and medium rows, at the cost of
accuracy on hard inputs (strongly varying weights, increments near float32's resolution): errors
of a few parts in a thousand, and steps down of up to about 1e-3 of the data's scale; float64
is unaffected.

Rows of unequal length go in one batch padded at the end with a value above each row's maximum
(below its minimum for `increasing=False`): the padding never pools with the data, so the fit
of each row is exactly its own. Rows of very different lengths fit faster laid end to end in a single row,
as [`examples/ctr_calibration`](examples/ctr_calibration) does.

## Installation

```bash
pip install torchisotonic
```

The C++/CUDA code is compiled the first time it is used (about a minute), which needs `ninja`,
a C++ compiler and, for the GPU, the CUDA toolkit. **Status: alpha**; the API may still change.

## Performance

On an NVIDIA T4 (Kaggle), float32, unweighted, decreasing fits of data noisy around a trend,
in milliseconds; best of ten calls on a warmed-up GPU, data already on the device for the GPU
columns:

| batch × n | torchisotonic (cuda) | `exact=False` | torchsort (cuda) | torchisotonic (cpu) | torchsort (cpu) | scipy, a loop over rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 × 16 | **2.16** | 1.21 | 72.0 | 407 | 677 | 14,363 |
| 100,000 × 128 | **1.71** | 1.30 | 136 | 280 | 492 | 1,613 |
| 10,000 × 1,000 | **2.29** | 1.22 | 99.2 | 160 | 354 | 318 |
| 1,000 × 10,000 | **1.72** | 1.12 | 197 | 160 | 358 | 178 |
| 100 × 100,000 | **1.82** | 1.23 | 396 | 158 | 352 | 166 |
| 10 × 1,000,000 | **1.95** | 1.35 | 1,656 | 157 | 357 | 170 |
| 1 × 10,000,000 | **2.07** | 1.47 | 5,493 | 236 | 355 | 176 |
| 100 × 1,000 | **0.09** | 0.07 | 3.0 | 1.7 | 3.5 | 2.9 |
| 10 × 10,000 | **0.22** | 0.20 | 14.0 | 1.7 | 3.5 | 1.6 |
| 1 × 100,000 | **0.26** | 0.25 | 51.7 | 1.8 | 3.5 | 1.6 |
| 1 × 1,000,000 | **0.51** | 0.44 | 550 | 17.9 | 35.0 | 15.5 |
| 1 × 100,000,000 | **16.9** | 10.8 | — | 3,552 | — | — |

torchsort solves each row with one GPU thread, so a long row is slow; it was not run on 10⁸
elements. The CPU columns are on Kaggle's CPU, so they say how the libraries compare, not how
fast a good CPU is. On an L4, an A100 and an H100 one row of 10⁷ elements takes 1.2, 0.8 and
0.5 ms, and 10⁸ elements 8.2, 4.3 and 2.7 ms. `bench/bench.py` reproduces the table.

## How it works

The fit at each element is the slope of the lower convex hull of the cumulative sums of the
data (the "greatest convex minorant"), so the GPU code builds that hull in parallel:

1. Every lane of a warp builds the hull of 32 neighbouring points with the monotone chain,
   its stack kept as a bit mask; then the lane hulls of a chunk of 1024 to 4096 elements merge
   pairwise in shared memory, each merge walking to its bridge and clearing the bits under it.
   Short rows are packed several per warp. A row that fits in one chunk is finished there.
2. Longer rows merge their chunks' hulls pairwise, in rounds, as lists of segments of the
   chunks' hulls, so that a merge costs segments rather than vertices.
3. Every element takes the slope of the hull edge above it.

[`docs/hull_prototype.py`](docs/hull_prototype.py) is a plain-numpy reference of the parallel
algorithm; [`src/torchisotonic/csrc/tiles.cuh`](src/torchisotonic/csrc/tiles.cuh)
describes the kernels. On the CPU, rows are solved with PAVA, in parallel.

## Development

```bash
uv sync
uv run pytest
```

The tests compare against scipy and scikit-learn (with hypothesis) and, on a machine with a
GPU, the CUDA operator against the CPU one; without a GPU the CUDA tests are skipped.

## License

MIT
