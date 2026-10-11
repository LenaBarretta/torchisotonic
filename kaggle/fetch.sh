#!/usr/bin/env bash
# Wait for the run started by kaggle/push.sh, download its output into kaggle/output/ and
# print the log.
set -euo pipefail
cd "$(dirname "$0")"
KERNEL=lenabarretta/torchisotonic-gpu
until status=$(kaggle kernels status "$KERNEL" 2>&1); echo "$status" | grep -qiE "complete|error|cancel"; do
  sleep 30
done
echo "$status"
rm -rf output && mkdir output
kaggle kernels output "$KERNEL" -p output -q
python3 - <<'PY'
import glob, json
for path in glob.glob("output/*.log"):
    raw = open(path).read()
    try:
        print("".join(event.get("data", "") for event in json.loads(raw)))
    except ValueError:
        print(raw)
PY
