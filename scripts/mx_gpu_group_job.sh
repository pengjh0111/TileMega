#!/usr/bin/env bash
set -eu
[ "$#" -ge 4 ] || { echo 'usage: mx_gpu_group_job.sh <g1,g2[,g3,g4]> <timeout_s> <out_dir> <command...>' >&2; exit 2; }
group=$(python - "$1" <<'PY'
import sys
try:g=sorted(set(map(int,sys.argv[1].split(','))))
except ValueError:raise SystemExit('GPU group must contain integer indices')
if len(g) not in (2,4) or not set(g)<=set(range(8)):
 raise SystemExit('GPU group requires 2 or 4 cards from 0 through 7')
print(','.join(map(str,g)))
PY
)
t=$2; out=$3; shift 3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TILEMEGA_DEVICE_GROUP="$group" CUDA_VISIBLE_DEVICES="$group" MACA_VISIBLE_DEVICES="$group"
export TILEMEGA_DEVICE_INDEX="${group%%,*}"
unset TILEMEGA_GPU_LOCK
cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
LOG=${MX_GPU_JOB_LOG:-$ROOT/runs/mx7/gpu_jobs.log}
mkdir -p "$out" "$(dirname "$LOG")"
printf '%s group=%s timeout=%s out=%s command=' "$(date --iso-8601=seconds)" "$group" "$t" "$out" >> "$LOG"
printf '%q ' "$@" >> "$LOG"
printf '\n' >> "$LOG"
set +e
python scripts/mx_job_duration.py --out "$out/monotonic_duration.json" -- \
  python -m tilemega.bench.gpu_guard --policy configs/bench/guard_policy_c550.json \
  --needs-free-mib "${NEEDS_MIB:-12288}" --out "$out" --timeout-s "$t" -- "$@"
rc=$?
printf '%s group=%s out=%s rc=%s\n' "$(date --iso-8601=seconds)" "$group" "$out" "$rc" >> "$LOG"
exit "$rc"
