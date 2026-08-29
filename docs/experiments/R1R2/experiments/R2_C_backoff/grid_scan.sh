#!/bin/bash
# R2-C grid-scan hang statistics: for each (backoff-tier variant, grid) pair, run N
# single-shot launches (fresh process + fresh CUDA context each time, no warmup memset --
# same "worst case" harness pattern as R2-B's grid_scan.sh / round-1 E5's no-warmup finding),
# tally pass vs timeout (hang).
#
# Reduced matrix vs R2-B's full 4-grid x 50-run scan (time budget): grid in {30, 170} (the
# two extremes: smallest tested / SM count) x 30 runs x 5s timeout, across 4 variants
# (backoff64, backoff256, backoff1024, relaxed_backoff256). This is intentionally a reduced
# but still statistically meaningful (M=30, not M=50) matrix -- documented as such in result.md.
set -u
cd "$(dirname "$0")"
LOG=$1
TIMEOUT_S=${2:-5}
RUNS=${3:-30}
: > "$LOG"

declare -A VARIANTS=(
  [backoff64]="spin_wait_backoff64.cubin spin_wait_backoff64"
  [backoff256]="spin_wait_backoff256.cubin spin_wait_backoff256"
  [backoff1024]="spin_wait_backoff1024.cubin spin_wait_backoff1024"
  [relaxed_backoff256]="spin_wait_relaxed_backoff256.cubin spin_wait_relaxed_backoff256"
)

for vname in backoff64 backoff256 backoff1024 relaxed_backoff256; do
  read -r cubin entry <<< "${VARIANTS[$vname]}"
  for grid in 30 170; do
    hang=0; ok=0
    for i in $(seq 1 $RUNS); do
      timeout ${TIMEOUT_S} ./host_test_single "$cubin" "$entry" "$grid" > /tmp/r2c_run.out 2>&1
      rc=$?
      if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        hang=$((hang+1)); status="HANG"
      elif [ $rc -eq 0 ] && grep -q "completed without hang" /tmp/r2c_run.out; then
        ok=$((ok+1)); status="OK"
      else
        hang=$((hang+1)); status="ERR(rc=$rc)"
      fi
      echo "$vname grid=$grid run=$i/$RUNS -> $status (running_tally hang=$hang ok=$ok)" >> "$LOG"
    done
    echo "SUMMARY $vname grid=$grid: 挂起 ${hang}/${RUNS} 次" >> "$LOG"
  done
done
echo "ALL DONE" >> "$LOG"
