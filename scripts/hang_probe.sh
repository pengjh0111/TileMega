#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <output-dir> -- <command> [args...]" >&2
  exit 2
fi

out=$1
shift
[[ ${1:-} == -- ]] || { echo "missing --" >&2; exit 2; }
shift
mkdir -p "$out"

"$@" >"$out/process.log" 2>&1 &
pid=$!
echo "$pid" >"$out/pid"

cleanup() {
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  wait "$pid" || rc=$?
  echo "process exited before sampling: rc=${rc:-0}" >&2
  exit 1
fi

for sample in 1 2 3; do
  cuda-gdb -q -batch -p "$pid" \
    -ex 'set pagination off' -ex 'info cuda kernels' \
    -ex 'info cuda threads' -ex 'where' \
    >"$out/gdb_${sample}.txt" 2>&1 || true
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm \
    --format=csv,noheader >"$out/smi_${sample}.csv" 2>&1 || true
  sleep 1
done

# Equal normalized backtraces/PC listings indicate no observed forward
# progress. Different samples indicate a live lock or merely a slow kernel.
for sample_file in "$out"/gdb_*.txt; do
  sed -E 's/0x[0-9a-f]+/0xADDR/g' "$sample_file" | sha256sum
done >"$out/sample_hashes.txt"
if [[ $(cut -d' ' -f1 "$out/sample_hashes.txt" | sort -u | wc -l) -eq 1 ]]; then
  echo "HANG_PROBE classification=no-observed-progress"
else
  echo "HANG_PROBE classification=pc-progress-observed"
fi
