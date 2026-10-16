# Paper experiments

The code behind the measurements of *Merge Hulls Without Copying Vertices: GPU Isotonic
Regression*. It is not part of the package: it needs a CUDA GPU, the package installed from this
repository, and torchsort for the comparisons with it (built against the installed PyTorch:
`pip install --no-build-isolation torchsort`).

| Paper | Command |
| --- | --- |
| Table 1, Figure 2 | `python bench/bench.py --gpu-only` and `python bench/bench.py --gpu-only --sweep` |
| CPU baselines (Section 7.2) | `python bench/bench.py` |
| Tables 2–4, Section 7.5 | `python paper/run.py --profile` |
| Figure 3 | `python paper/sweep.py` |
| Section 7.6 | `python paper/accuracy.py`, `python paper/accuracy.py --fast`, `python paper/cancellation.py` |
| Table 5 | `python paper/gradients.py` |
| Table 6, Appendix C | `python paper/soft_rank.py` |

## Variants

`run.py` builds each folder of `variants/` as its own PyTorch extension, checks that it gives
the package's fits on shapes at lane, chunk and row edges, and times it against the package;
with `--profile` it also gives the GPU time of each stage (chunk hulls, merges, reconstruction,
the rest). `python paper/run.py --profile copy_grid` runs one variant. Each variant includes the
package's `tiles.cuh` and replaces one part of the pipeline:

| Variant | What it changes | In the paper |
| --- | --- | --- |
| `copy_per_merge` | after every merge, the thread or warp of the merge copies the surviving vertices into one array | per-merge copying, Tables 2–3 |
| `copy_grid` | every level's merged hulls are copied into a second buffer by the whole GPU | grid copying, Tables 2–3 |
| `local_stack` | lane hulls by the monotone chain over a stack in local memory instead of a bit mask | Table 4 |
| `three_case` | the three-case bridge search, with a walk cap that grows with the level | Table 4, Appendix A |
| `plain_predicates` | merge-level turns as plain double predicates on absolute coordinates | Table 4 |
| `warp_per_merge` | a warp for every merge at every level | Table 4 |
| `warps_for_capped` | threads walk; merges whose walk reaches its cap are finished by a warp each | Table 4 |

A variant defines its kernels in a named namespace of its own, as an unnamed one beside the
header's confuses nvcc.
