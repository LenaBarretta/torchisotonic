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

builds them, checks that each gives the package's fits (on shapes and data patterns chosen to
hit lane, chunk and problem edges), and times each against the package on the benchmark's
shapes. A variant that wins is moved into `tiles.cuh`; one that does not is deleted with its
folder. Nothing outside `experiments/` refers to a variant. `kaggle/push.sh experiments` runs
it on a T4.
