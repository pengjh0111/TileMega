#!/usr/bin/env bash
# Finish the user-narrowed Llama B=1/16 acceptance only after its active run ends.
set -uo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_PID RUN_DIRECTORY" >&2
  exit 2
fi
run_pid=$1
run_dir=$2
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$root"
export PYTHONPATH="$root/python"
export TILEMEGA_BIN=${TILEMEGA_BIN:-/root/r11_work/build/tools/tilemega}
export TILEMEGA_PYTHON=${TILEMEGA_PYTHON:-/root/venvs/tilemega-torch213-cu126/bin/python}
build_dir=$(dirname "$(dirname "$TILEMEGA_BIN")")
queue_dir=${TILEMEGA_QUEUE_DIR:-/root/r11_work/acceptance_queue}
mkdir -p "$queue_dir"
trap 'rc=$?; printf "finished_utc=%s\nexit_code=%s\n" "$(date -u +%FT%TZ)" "$rc" > "$queue_dir/status.txt"' EXIT
printf 'waiting_for_pid=%s\nstarted_utc=%s\n' "$run_pid" "$(date -u +%FT%TZ)" > "$queue_dir/status.txt"
tail --pid="$run_pid" -f /dev/null
failed=0
run_step() {
  local name=$1
  shift
  "$@" > "$queue_dir/$name.log" 2>&1
  local rc=$?
  printf '%s\texit=%s\n' "$name" "$rc" >> "$queue_dir/steps.tsv"
  if [[ $rc -ne 0 ]]; then failed=1; fi
}
printf 'step\tresult\n' > "$queue_dir/steps.tsv"
if [[ -s "$run_dir/report.json" ]]; then
  run_step archive python3 docs/experiments/SERVING_R11/archive_ev2.py --run "$run_dir"
  run_step diagnostics python3 docs/experiments/SERVING_R11/collect_selected_diagnostics.py --run "$run_dir"
else
  printf 'acceptance\tmissing report.json\n' >> "$queue_dir/steps.tsv"
  failed=1
fi
run_step build cmake --build "$build_dir" -j 8
run_step synthetic_handoff "$build_dir/tilemega-unit" handoff_ir smem_direct
run_step ctest ctest --test-dir "$build_dir" -j 1 --output-on-failure
run_step verify python3 docs/experiments/SERVING_R11/verify.py
printf 'Queued acceptance and independent checks ended; see steps.tsv and logs.\n' > "$queue_dir/complete.txt"
exit "$failed"
