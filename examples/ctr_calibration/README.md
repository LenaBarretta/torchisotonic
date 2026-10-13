# Calibrating a click-through-rate model

An ad-ranking model predicts the probability that an ad is clicked. To train it cheaply, the
non-clicks are usually downsampled - here to 10% - which leaves the model's probabilities far
too high. Before they can be used to price ads or to compare with real click rates, they are
calibrated on held-out traffic: one monotone curve maps every score to the click rate observed
at that score. Isotonic regression fits that curve; it is what scikit-learn's
`CalibratedClassifierCV(method="isotonic")` does.

[`calibrate.py`](calibrate.py) does it end to end on the
[Criteo Display Advertising Challenge](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset)
data: 45.8 million impressions over seven days, split into seven equal parts in time order.

1. A logistic regression with hashed features is trained on days 1-5, with 10% of the
   non-clicks.
2. Its scores on day 6 calibrate it, in four ways: the analytic correction for the
   downsampling, Platt scaling, one isotonic regression over the whole day, and one per
   segment - the frequent values of a categorical feature, which the script picks. The two
   isotonic fits run with scikit-learn on the CPU and with torchisotonic on the GPU, and must
   give the same curves.
3. Day 7 measures each: log-loss, Brier score, expected calibration error, and a reliability
   diagram.
4. Isotonic calibration is timed on growing numbers of scores, up to all 45.8 million.

The per-segment fit is a single call: the scores are keyed by segment plus score, and each
segment's click rates are offset by twice its index, so that no block of the fit can reach
across a segment boundary. One long problem, rather than a batch of rows of unequal length.

```bash
pip install torchisotonic scikit-learn pyarrow matplotlib
python examples/ctr_calibration/calibrate.py --data path/to/dac/train.txt --out results/
```

It needs about 8 GB of memory for the data and as much on the GPU. `--rows` takes only the
first rows, for a quick try. On Kaggle, `kaggle/push.sh example-ctr` runs it with the dataset
attached.

This folder is not part of the installed package.

## Results

To be filled in from the Kaggle run (T4).
