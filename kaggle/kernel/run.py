"""Runs on Kaggle (pushed by kaggle/push.sh): unpack the code, build it, test it, benchmark it.

STEPS is filled in by push.sh. Everything prints to the log, and the results land in
/kaggle/working, which `kaggle kernels output` downloads.

Elsewhere (cloud/modal_bench.py) the code is already in place: TI_SRC names it, TI_OUT where
results go, and TI_STEPS the steps, separated by spaces.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

STEPS = ["tests"]  # replaced by push.sh
if os.environ.get("TI_STEPS"):
    STEPS = os.environ["TI_STEPS"].split()
SRC = os.environ.get("TI_SRC", "/kaggle/working/torchisotonic")
OUT = os.environ.get("TI_OUT", "/kaggle/working")
START = time.perf_counter()


STEP_TIMEOUT = 1500  # seconds: a hung step (say, a kernel that never ends) is killed


def sh(cmd: str) -> int:
    print(f"\n[{time.perf_counter() - START:6.0f}s] $ {cmd}", flush=True)
    try:
        return subprocess.run(cmd, shell=True, timeout=STEP_TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {STEP_TIMEOUT} s", flush=True)
        subprocess.run(
            "pkill -9 -f 'python bench/|python experiments/|python -m pytest'", shell=True
        )
        return 1


if "TI_SRC" not in os.environ:
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
if os.path.exists(f"{SRC}/SOURCE"):
    with open(f"{SRC}/SOURCE") as f:
        print(f.read(), flush=True)

for step in STEPS:  # csrc:<dir>: build and run the package with these sources instead
    if step.startswith("csrc:"):
        csrc = f"{SRC}/src/torchisotonic/csrc"
        shutil.rmtree(csrc)
        shutil.copytree(f"{SRC}/{step[5:]}", csrc)
        print(f"sources: {step[5:]}", flush=True)
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
    # Without build isolation, so that it compiles against the torch that is installed.
    sh("pip install -q --no-build-isolation torchsort")
if not failed and "profile" in STEPS:
    # The SM clock, power and temperature while it runs: a T4 at its power limit slows down.
    with open("/tmp/clocks.csv", "w") as log:
        query = "clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
        clocks = subprocess.Popen(  # no shell, so that kill() reaches nvidia-smi itself
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader", "-lms", "50"],
            stdout=log,
        )
        try:
            sh(f"cd {SRC} && python bench/profile_hull.py")
        finally:
            clocks.kill()
            clocks.wait()
    sh("sort /tmp/clocks.csv | uniq -c | sort -rn | head -12")
if not failed and "grad" in STEPS:
    sh(f"cd {SRC} && python bench/grad.py")
if not failed and "grad-profile" in STEPS:
    sh(f"cd {SRC} && python bench/grad.py --profile")
if not failed and "grad-micro" in STEPS:
    sh(f"cd {SRC} && python bench/grad.py --micro")
if not failed and "debug-weighted" in STEPS:
    sh(f"cd {SRC} && python bench/debug_weighted.py")
if not failed and "accuracy" in STEPS:
    sh(f"cd {SRC} && python bench/accuracy.py")
if not failed and "accuracy-fast" in STEPS:
    sh(f"cd {SRC} && python bench/accuracy.py --fast")
if not failed and "experiments" in STEPS:
    sh(f"cd {SRC} && python experiments/run.py")
if not failed and "experiments-profile" in STEPS:
    sh(f"cd {SRC} && python experiments/run.py --profile")
if not failed and "experiments-each" in STEPS:
    # One variant per process, each under the step timeout: a hanging variant shows by name.
    for folder in sorted(glob.glob(f"{SRC}/experiments/*/kernel.cu")):
        name = os.path.basename(os.path.dirname(folder))
        sh(f"cd {SRC} && timeout 600 python experiments/run.py --profile {name}")
for step in STEPS:  # variant:<name>: that variant alone, with the profile
    if not failed and step.startswith("variant:"):
        sh(f"cd {SRC} && timeout 600 python experiments/run.py --profile {step[8:]}")
if not failed and "example-ctr" in STEPS:
    # The Criteo data (mrkmakr/criteo-dataset), attached by push.sh for this step.
    data = glob.glob("/kaggle/input/**/dac/train.txt", recursive=True)
    if data:
        script = "examples/ctr_calibration/calibrate.py"
        sh(f"cd {SRC} && python {script} --data {data[0]} --out {OUT}/ctr")
    else:
        print("example-ctr: the Criteo dataset is not attached", flush=True)
if not failed and "sweep" in STEPS:  # for the paper's figures: one row of growing length
    sh(f"cd {SRC} && python bench/bench.py --gpu-only --sweep")
    sh(f"cat {SRC}/bench/results/*-sweep.md")
    sh(f"cd {SRC} && python experiments/sweep.py {OUT}/merge-sweep.json")
    sh(f"cp -r {SRC}/bench/results {OUT}/results")
if not failed and "merge-sweep" in STEPS:  # the merge stage alone, of the same sweep
    sh(f"cd {SRC} && python experiments/sweep.py {OUT}/merge-sweep.json")
if not failed and "phases" in STEPS:
    sh(f"cd {SRC} && python bench/phases.py")
for step in STEPS:  # phases:<dir>: the phases of another copy of the sources
    if not failed and step.startswith("phases:"):
        sh(f"cd {SRC} && python bench/phases.py --csrc {step[7:]}")
if not failed and ("bench" in STEPS or "bench-quick" in STEPS or "bench-gpu" in STEPS):
    flags = "--quick" if "bench-quick" in STEPS else "--gpu-only" if "bench-gpu" in STEPS else ""
    failed = sh(f"cd {SRC} && python bench/bench.py {flags}")
    sh(f"cat {SRC}/bench/results/*.md")
    sh(f"cp -r {SRC}/bench/results {OUT}/results")
print(f"\n[{time.perf_counter() - START:6.0f}s] DONE, failed={failed}", flush=True)
# Nothing left behind may keep the Kaggle session alive.
subprocess.run("pkill -9 -f nvidia-smi", shell=True)
os._exit(0)
