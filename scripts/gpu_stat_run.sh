#!/usr/bin/env bash
# ==============================================================================
# Statistical GPU test runner (skeleton P0.3)
#
# Three disciplines distilled from the R1/R2 lessons, all built in here:
#
#  1. A single success is not evidence. R2's grid scan showed the hang rate is a
#     probabilistic quantity (39/50 vs 48/50 at grid=30), so one run proves
#     nothing. N defaults to 50.
#
#  2. Hangs must time out, count as failures, and leave their artifacts behind.
#     A GPU test without a timeout wedges the whole CI pipeline.
#
#  3. EXCLUSIVE GPU ACCESS. The hang is a timing-sensitive race (R1-E5's central
#     finding: warm-up could "rescue" grid=80). If another process is loading the
#     GPU or the CPU at the same time, the measured hang rate is meaningless.
#     This serialises through flock and checks that the machine is idle before
#     starting.
#
# Emits CSV so results can be compared across versions and variants.
#
# Usage:
#   gpu_stat_run.sh --label <name> --runs 50 --timeout 6 -- <command> [args...]
# ==============================================================================
set -uo pipefail

LOCK_FILE="${TILEMEGA_GPU_LOCK:-/tmp/tilemega-gpu.lock}"
RUNS=50
TIMEOUT_S=6
LABEL="unnamed"
CSV=""
SUCCESS_PATTERN=""
ALLOW_BUSY=0

usage() {
  cat <<'EOF'
Usage: gpu_stat_run.sh [options] -- <command> [args...]

Runs <command> repeatedly and reports how often it passes, hangs, or errors.

Options:
  --label <name>     label for this cell, used in output and CSV (default: unnamed)
  --runs <n>         number of runs (default: 50)
  --timeout <sec>    per-run timeout in seconds; exceeding it counts as a hang
                     (default: 6)
  --csv <file>       append one summary row to this CSV file
  --expect <text>    a run only counts as OK if its output contains <text>
  --allow-busy       run even when the machine is not idle (the data must then
                     be reported as collected under load)
  -h, --help         this help

Environment:
  TILEMEGA_GPU_LOCK      lock file serialising GPU access
  TILEMEGA_ARTIFACT_DIR  where failure artifacts are kept
  TILEMEGA_TILEIRAS      recorded in the CSV for traceability

Exit status is non-zero if any run hung or errored.
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)   LABEL="$2"; shift 2;;
    --runs)    RUNS="$2"; shift 2;;
    --timeout) TIMEOUT_S="$2"; shift 2;;
    --csv)     CSV="$2"; shift 2;;
    --expect)  SUCCESS_PATTERN="$2"; shift 2;;
    --allow-busy) ALLOW_BUSY=1; shift;;
    -h|--help) usage 0;;
    --) shift; break;;
    *) echo "unknown option: $1" >&2; usage 2;;
  esac
done
[[ $# -gt 0 ]] || { echo "error: no command given" >&2; usage 2; }

# --- machine idle check ----------------------------------------------------
if [[ $ALLOW_BUSY -eq 0 ]]; then
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  load1=$(cut -d' ' -f1 /proc/loadavg)
  ncpu=$(nproc)
  busy=0
  [[ -n "$gpu_util" && "$gpu_util" -gt 10 ]] && { echo "warning: GPU utilisation ${gpu_util}%, not idle" >&2; busy=1; }
  awk -v l="$load1" -v n="$ncpu" 'BEGIN{exit !(l > n*0.5)}' && {
    echo "warning: CPU load $load1 across $ncpu cores, not idle" >&2; busy=1; }
  if [[ $busy -eq 1 ]]; then
    cat >&2 <<'EOF'
error: machine is not idle, refusing to run.
  The hang is a timing-sensitive race (R1-E5: warm-up could "rescue" grid=80),
  so a hang rate measured on a loaded machine supports no conclusion at all.
  Wait until the machine is idle, or pass --allow-busy explicitly and state in
  the report that the data was collected under load.
EOF
    exit 3
  fi
fi

exec 9>"$LOCK_FILE"
flock 9 || { echo "error: could not acquire GPU lock $LOCK_FILE" >&2; exit 3; }

ARTIFACT_DIR="${TILEMEGA_ARTIFACT_DIR:-/tmp/tilemega-gpu-$LABEL-$$}"
mkdir -p "$ARTIFACT_DIR"

ok=0; hang=0; err=0
echo "=== $LABEL : $RUNS runs, ${TIMEOUT_S}s timeout each ==="
for ((i=1; i<=RUNS; i++)); do
  out="$ARTIFACT_DIR/run_$i.out"
  timeout --signal=KILL "$TIMEOUT_S" "$@" > "$out" 2>&1
  rc=$?
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    hang=$((hang+1)); status=HANG
  elif [[ $rc -ne 0 ]]; then
    err=$((err+1)); status="ERR(rc=$rc)"
  elif [[ -n "$SUCCESS_PATTERN" ]] && ! grep -q "$SUCCESS_PATTERN" "$out"; then
    err=$((err+1)); status="ERR(output lacks '$SUCCESS_PATTERN')"
  else
    ok=$((ok+1)); status=OK
    rm -f "$out"   # discard successful output; keep only failure artifacts
  fi
  printf "\r  %3d/%d  ok=%d hang=%d err=%d  last=%s        " \
         "$i" "$RUNS" "$ok" "$hang" "$err" "$status"
done
echo

hang_rate=$(awk -v h="$hang" -v n="$RUNS" 'BEGIN{printf "%.1f", 100.0*h/n}')
echo "=== $LABEL result: ok=$ok/$RUNS  hang=$hang/$RUNS (${hang_rate}%)  err=$err/$RUNS ==="
[[ $((hang+err)) -gt 0 ]] && echo "    failure artifacts kept in: $ARTIFACT_DIR"

if [[ -n "$CSV" ]]; then
  [[ -f "$CSV" ]] || echo "label,runs,ok,hang,err,hang_rate_pct,timeout_s,timestamp,tileiras,driver" > "$CSV"
  echo "$LABEL,$RUNS,$ok,$hang,$err,$hang_rate,$TIMEOUT_S,$(date -Is),${TILEMEGA_TILEIRAS:-default},$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1)" >> "$CSV"
fi

# Any hang is a failure: a hang must never be waved away as "flaky".
[[ $hang -eq 0 && $err -eq 0 ]]
