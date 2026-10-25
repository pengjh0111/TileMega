#!/usr/bin/env bash
# Run as the child of scripts/mx_gpu_job.sh; every sample is a fresh process.
set -euo pipefail
if [[ ${TILEMEGA_GPU_LOCK_HELD:-0} != 1 ]]; then
  echo 'run_probes.sh requires scripts/mx_gpu_job.sh' >&2
  exit 2
fi
sync_binary=$1
load_binary=$2
output=$3
mkdir -p "$output"
for round in 1 2 3 4 5; do
  "$sync_binary" 256 "$output/sync_round${round}.csv"
  "$load_binary" 4096 "$output/load_round${round}.csv"
  echo "round=$round complete"
done
