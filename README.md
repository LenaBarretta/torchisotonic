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

The semantics are scikit-learn's: weights must be strictly positive (a weight of the shape of
`y`, or one that broadcasts to it), the result is clipped to `[y_min, y_max]` if they are
given, `dim` picks the axis, and a row holding a NaN comes back as NaN. Float input keeps its
dtype (float16 and bfloat16 are computed in float32), integer input becomes the default
float dtype. The fit is differentiable in `y` and in the weights.

On the GPU, float32 is exact to the format's rounding by default. `exact=False` makes it about
1.5–2 times faster on short and medium rows, at the cost of errors up to about 1e-3 of the
data's scale on hard inputs (strongly varying weights, increments near float32's resolution);
float64 is unaffected.

Rows of unequal length go in one batch padded at the end with a value above each row's maximum
(below its minimum for `increasing=False`): the padding never pools with the data, so the fit
of each row is exactly its own. Rows of very different lengths fit faster laid end to end in a single row,
as [`examples/ctr_calibration`](examples/ctr_calibration) does.

**Status: pre-alpha.** Not on PyPI yet; the API may change. The C++/CUDA code is compiled the
first time it is used (about a minute), which needs `ninja`, a C++ compiler and, for the GPU,
the CUDA toolkit:

```bash
pip install git+https://github.com/LenaBarretta/torchisotonic
```

## Performance

On an NVIDIA T4 (Kaggle), float32, unweighted, decreasing fits of data noisy around a trend,
in milliseconds; best of ten calls on a warmed-up GPU, data already on the device for the GPU
columns:

| batch × n | torchisotonic (cuda) | `exact=False` | torchsort (cuda) | torchisotonic (cpu) | torchsort (cpu) | scipy, a loop over rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 × 16 | **2.15** | 1.20 | — | 407 | 677 | 14,363 |
| 100,000 × 128 | **1.66** | 1.29 | — | 280 | 492 | 1,613 |
| 10,000 × 1,000 | **2.15** | 1.18 | 101 | 160 | 354 | 318 |
| 1,000 × 10,000 | **1.70** | 1.15 | 197 | 160 | 358 | 178 |
| 100 × 100,000 | **1.82** | 1.22 | 396 | 158 | 352 | 166 |
| 10 × 1,000,000 | **1.94** | 1.35 | 1,654 | 157 | 357 | 170 |
| 1 × 10,000,000 | **2.06** | 1.47 | 5,482 | 236 | 355 | 176 |
| 100 × 1,000 | **0.09** | 0.07 | 3.0 | 1.7 | 3.5 | 2.9 |
| 10 × 10,000 | **0.21** | 0.20 | 14.0 | 1.7 | 3.5 | 1.6 |
| 1 × 100,000 | **0.26** | 0.26 | 51.6 | 1.8 | 3.5 | 1.6 |
| 1 × 1,000,000 | **0.50** | 0.44 | 548 | 17.9 | 35.0 | 15.5 |
| 1 × 100,000,000 | **16.7** | 10.8 | — | 3,552 | — | — |

torchsort (cuda) was not run with more than 10⁷ elements (a single call would take minutes);
the CPU columns are on Kaggle's CPU, so they say how the libraries compare, not how fast a
good CPU is. A plain copy of 10⁸ float32 elements on the T4 takes 3.4 ms: the GPU code is a few
times above that, bound by the instructions the hull takes rather than by memory. On an L4 and
an A100 one row of 10⁷ elements takes 1.2 and 0.8 ms, and 10⁸ elements 8.0 and 4.3 ms.
`bench/bench.py` reproduces the table.

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
GPU, the CUDA operator against the CPU one.

## License

MIT
