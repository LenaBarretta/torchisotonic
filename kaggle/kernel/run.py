"""Runs on Kaggle (pushed by kaggle/push.sh): unpack the code, build it, test it, benchmark it.

STEPS is filled in by push.sh. Everything prints to the log, and the results land in
/kaggle/working, which `kaggle kernels output` downloads.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

STEPS = ["tests"]  # replaced by push.sh
SRC = "/kaggle/working/torchisotonic"
OUT = "/kaggle/working"
START = time.perf_counter()


def sh(cmd: str) -> int:
    print(f"\n[{time.perf_counter() - START:6.0f}s] $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


# Kaggle mounts datasets under /kaggle/input (lately /kaggle/input/datasets/<owner>/<name>)
# and may unpack archives itself, so accept the archive or the unpacked folder.
archives = glob.glob("/kaggle/input/**/torchisotonic.tar.gz", recursive=True)
projects = glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
shutil.rmtree(SRC, ignore_errors=True)
if archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall(SRC, filter="data")
elif projects:
    shutil.copytree(os.path.dirname(projects[0]), SRC)
else:
    for root, _, files in os.walk("/kaggle/input"):
        print(root, files[:10])
    sys.exit("found neither torchisotonic.tar.gz nor pyproject.toml under /kaggle/input")
with open(f"{SRC}/SOURCE") as f:
    print(f.read(), flush=True)

sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
sh("pip install -q pytest pytest-cov hypothesis scipy scikit-learn ninja")
sh(f"pip install -q --no-deps --force-reinstall {SRC}")
smoke = (
    "import time, torch, torchisotonic; "
    "print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); "
    "t = time.perf_counter(); "
    "y = torch.tensor([[5., 3, 1, 2, 8, 10, 7, 9, 6, 4]], device='cuda'); "
    "print(torchisotonic.isotonic(y)); "
    "print(f'first call, compiling included: {time.perf_counter() - t:.0f} s')"
)
failed = sh(f'python -c "{smoke}"')
if not failed and "tests" in STEPS:
    failed = sh(f"cd {SRC} && python -m pytest -q -p no:cacheprovider --no-cov -x tests")
if not failed and "torchsort" in STEPS:
    sh("pip install -q torchsort")
if not failed and ("bench" in STEPS or "bench-quick" in STEPS):
    quick = "--quick" if "bench-quick" in STEPS else ""
    failed = sh(f"cd {SRC} && python bench/bench.py {quick}")
    sh(f"cp -r {SRC}/bench/results {OUT}/results")
print(f"\n[{time.perf_counter() - START:6.0f}s] DONE, failed={failed}", flush=True)
