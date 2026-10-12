#!/usr/bin/env bash
# Upload the working tree to the private Kaggle dataset and run kaggle/kernel/run.py on a T4.
#
#   kaggle/push.sh tests              # build + tests
#   kaggle/push.sh tests bench-quick  # ... + a small benchmark
#   kaggle/push.sh tests torchsort bench
#
# Then: kaggle kernels status lenabarretta/torchisotonic-bench
#       kaggle kernels output lenabarretta/torchisotonic-bench -p <dir>
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=lenabarretta/torchisotonic
KERNEL=lenabarretta/torchisotonic-bench
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/data" "$STAGE/kernel"
# Tracked and untracked (not ignored) files: what is on disk now, committed or not.
{ git log -1 --format='commit %h %s'; git status --short | sed 's/^/  changed: /'; date -u; } \
  > "$STAGE/SOURCE"
git ls-files -co --exclude-standard -z | tar --null -T - -cf "$STAGE/src.tar"
tar -rf "$STAGE/src.tar" -C "$STAGE" SOURCE
gzip -c "$STAGE/src.tar" > "$STAGE/data/torchisotonic.tar.gz"
cat > "$STAGE/data/dataset-metadata.json" <<META
{"title": "torchisotonic", "id": "$DATASET", "licenses": [{"name": "other"}]}
META

if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/data" -m "push $(date -u +%FT%TZ)" -q
else
  kaggle datasets create -p "$STAGE/data" -q
fi
echo "waiting for the dataset to be ready..."
until kaggle datasets status "$DATASET" 2>/dev/null | grep -qi ready; do sleep 5; done

cp kaggle/kernel/kernel-metadata.json "$STAGE/kernel/"
steps=$(printf '"%s", ' "${@:-tests}")
sed "s/^STEPS = .*/STEPS = [${steps%, }]/" kaggle/kernel/run.py > "$STAGE/kernel/run.py"
kaggle kernels push -p "$STAGE/kernel" --accelerator NvidiaTeslaT4  # Kaggle's only GPU now
