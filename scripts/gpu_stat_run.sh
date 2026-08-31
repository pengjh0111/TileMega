#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# TileMega -- scripts/gpu_stat_run.sh
# Statistical executor (skeleton §7 P0.3: "统计化执行 / 挂起检测").
#
# Runs a GPU test N times, each in a FRESH PROCESS, and reports pass / fail /
# hang counts.  A single successful run of a synchronisation test is not
# evidence -- races are probabilistic, and a persistent kernel that deadlocks
# does so only under particular scheduling.
#
# Fresh processes matter for two reasons:
#   * the CUDA context (and therefore the device allocator's state) is rebuilt,
#     so buffer reuse patterns vary between runs;
#   * a hung kernel wedges its context, so the runs must be isolated.
#
# Usage:
#   gpu_stat_run.sh -n 50 -t 10 -l label -o outdir -- ./prog --args...
#
# Exit code: 0 iff every run passed.
#
# Output: a one-line machine-readable summary on stdout:
#   STAT label=<l> runs=<n> pass=<p> fail=<f> hang=<h> error=<e> rate=<p/n>
# plus per-run logs under <outdir>/, and, for hangs, the preserved scene
# (stdout/stderr of the run + nvidia-smi + the exact command to replay).

set -u -o pipefail

RUNS=50
TIMEOUT=20
LABEL="unnamed"
OUTDIR=""
KEEP_ALL=0

usage() { sed -n '2,30p' "$0"; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--runs)    RUNS="$2"; shift 2;;
    -t|--timeout) TIMEOUT="$2"; shift 2;;
    -l|--label)   LABEL="$2"; shift 2;;
    -o|--outdir)  OUTDIR="$2"; shift 2;;
    -k|--keep-all) KEEP_ALL=1; shift;;
    -h|--help)    usage;;
    --) shift; break;;
    *) echo "unknown option $1" >&2; usage;;
  esac
done

if [[ $# -eq 0 ]]; then echo "no command given" >&2; usage; fi

if [[ -z "$OUTDIR" ]]; then OUTDIR="$(mktemp -d /tmp/tilemega_stat.XXXXXX)"; fi
mkdir -p "$OUTDIR"

CMD=("$@")

pass=0; fail=0; hang=0; err=0
first_fail_log=""
first_hang_log=""

# nvidia-smi may not exist in a CPU-only container; degrade gracefully.
have_smi=0; command -v nvidia-smi >/dev/null 2>&1 && have_smi=1

for ((i=1; i<=RUNS; i++)); do
  log="$OUTDIR/run_${i}.log"
  # `timeout --foreground -k` : SIGTERM then SIGKILL, so a wedged CUDA context
  # cannot keep the process alive and stall the whole sweep.
  timeout --foreground -k 5 "$TIMEOUT" "${CMD[@]}" >"$log" 2>&1
  rc=$?
  case $rc in
    0)   pass=$((pass+1));;
    124|137)
         hang=$((hang+1))
         if [[ -z "$first_hang_log" ]]; then
           first_hang_log="$OUTDIR/HANG_scene_run${i}.txt"
           {
             echo "=== TileMega hang scene ==="
             echo "label   : $LABEL"
             echo "run     : $i / $RUNS"
             echo "timeout : ${TIMEOUT}s (rc=$rc)"
             echo "date    : $(date -Is)"
             echo "replay  : ${CMD[*]}"
             echo
             echo "--- process stdout/stderr up to the kill ---"
             cat "$log"
             echo
             if [[ $have_smi -eq 1 ]]; then
               echo "--- nvidia-smi at kill time ---"
               nvidia-smi 2>&1
               echo
               echo "--- utilisation / power sample ---"
               nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm \
                          --format=csv 2>&1
             fi
           } > "$first_hang_log"
           echo "  [hang] scene preserved: $first_hang_log" >&2
         fi
         ;;
    1)   fail=$((fail+1))
         if [[ -z "$first_fail_log" ]]; then
           first_fail_log="$OUTDIR/FAIL_run${i}.log"
           cp "$log" "$first_fail_log"
           echo "  [fail] first mismatch: $first_fail_log" >&2
           grep -h '^RESULT' "$log" >&2 || true
         fi
         ;;
    *)   err=$((err+1))
         if [[ ! -f "$OUTDIR/ERROR_first.log" ]]; then
           cp "$log" "$OUTDIR/ERROR_first.log"
           echo "  [error rc=$rc] $OUTDIR/ERROR_first.log" >&2
           tail -3 "$log" >&2
         fi
         ;;
  esac
  # Keep the log only if it is interesting, else the sweep leaves 1000s of files.
  if [[ $KEEP_ALL -eq 0 && $rc -eq 0 ]]; then rm -f "$log"; fi
done

rate=$(awk -v p="$pass" -v n="$RUNS" 'BEGIN{ printf "%.4f", (n>0)? p/n : 0 }')
echo "STAT label=$LABEL runs=$RUNS pass=$pass fail=$fail hang=$hang error=$err rate=$rate outdir=$OUTDIR"

[[ $pass -eq $RUNS ]]
