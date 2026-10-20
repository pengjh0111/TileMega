#!/usr/bin/env bash
# Usage: mx_gpu_job.sh <gpu> <timeout_s> <out_dir> <command...>
#
# MX-1 appendix B guard wrapper. Lived outside the tree as /root/deps/mxjob.sh
# through MX-1 and MX-2, which left the measurements unreproducible from a
# clone (MX3-0 item 1). The repository root is derived from this file, so a
# clone anywhere works.
set -e
g=$1; t=$2; out=$3; shift 3
case "$g" in 4|5|6|7) ;; *) echo "refusing GPU $g (only 7,6,5,4)"; exit 2;; esac
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TILEMEGA_DEVICE_INDEX=$g CUDA_VISIBLE_DEVICES=$g MACA_VISIBLE_DEVICES=$g
export TILEMEGA_GPU_LOCK=/tmp/tilemega_mx_gpu${g}.lock
cd "$ROOT"
export PYTHONPATH=$ROOT/python${PYTHONPATH:+:$PYTHONPATH}
LOG=${TILEMEGA_GPU_JOB_LOG:-$ROOT/runs/mx1/gpu_jobs.log}
mkdir -p "$out" "$(dirname "$LOG")"
echo "$(date -Is) gpu=$g timeout=$t out=$out cmd=$*" >> "$LOG"
set +e
python -m tilemega.bench.gpu_guard --policy configs/bench/guard_policy_c550.json \
  --needs-free-mib ${NEEDS_MIB:-12288} --out "$out" --timeout-s "$t" -- "$@"
rc=$?
echo "$(date -Is) gpu=$g rc=$rc out=$out" >> "$LOG"
exit $rc
