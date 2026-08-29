#!/bin/bash
# Grid-scan hang statistics: for each (variant, grid) pair, run N single-shot launches
# (fresh process + fresh CUDA context each time, via the no-warmup host_test_single harness),
# tally pass vs timeout (hang). Appends one line per run to the given log file so progress
# can be monitored live.
set -u
cd "$(dirname "$0")"
LOG=$1
TIMEOUT_S=${2:-6}
RUNS=${3:-50}
shift 3 || true
: > "$LOG"

declare -A VARIANTS=(
  [relaxed_acquire]="spin_wait_relaxed_acquire.cubin spin_wait_relaxed_acquire"
  [baseline_acquire_in_loop]="../E4_spin_wait/variants/spin_wait_tokenchain.cubin spin_wait_tokenchain"
)

for vname in "${!VARIANTS[@]}"; do
  read -r cubin entry <<< "${VARIANTS[$vname]}"
  for grid in 30 80 120 170; do
    hang=0; ok=0
    for i in $(seq 1 $RUNS); do
      timeout ${TIMEOUT_S} ./host_test_single "$cubin" "$entry" "$grid" > /tmp/r2b_run.out 2>&1
      rc=$?
      if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        hang=$((hang+1)); status="HANG"
      elif [ $rc -eq 0 ] && grep -q "completed without hang" /tmp/r2b_run.out; then
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
