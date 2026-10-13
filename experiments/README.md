# Experiments

Variants of the CUDA pipeline, built and checked next to the package's own operator. Each
variant is one folder, `experiments/<name>/kernel.cu`, that defines in its own torch library,
`tiexp_<name>`, the operator

    isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor

for CUDA, the way `torchisotonic::isotonic_tiles` is defined. A variant includes the package's
`tiles.cuh` (on the include path) to reuse the parts it does not change, and defines the parts
it does under new names, in a namespace of its own (a named one: an unnamed one beside the
header's confuses nvcc).

    python experiments/run.py            # every variant
    python experiments/run.py fast_chain # some
    python experiments/run.py --profile  # ... and GPU time per stage (chunk / merge / fit)

builds them, checks that each gives the package's fits (on shapes and data patterns chosen to
hit lane, chunk and problem edges), and times each against the package on the benchmark's
shapes. A variant that wins is moved into `tiles.cuh`; one that does not is deleted with its
folder, unless it is kept as a control for a comparison (`materialized`: the package's
segment lists against copying the merged hulls' vertices; `merge_v9`: the merges with plain
double predicates on absolute coordinates, against the package's filtered exact ones - the
cost of exactness). Nothing outside `experiments/` refers to a variant. `kaggle/push.sh experiments` (or
`experiments-profile`, `experiments-each`, or `variant:<name>` for one variant with the
profile) runs it on a T4.
