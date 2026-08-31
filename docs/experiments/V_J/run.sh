#!/usr/bin/env bash
set -u -o pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
here="$root/docs/experiments/V_J"
va="$root/docs/experiments/V_A"
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
mkdir -p "$here/raw"

"$nvcc" -O2 -std=c++17 -arch=sm_89 -lineinfo -Xptxas=-v \
  -I"$root/test/harness" "$va/event_sync.cu" -o "$here/event_sync" \
  2>"$here/raw/build.ptxas.txt" || exit 2

: >"$here/raw/tile_scan.txt"
for tile in 512 1024 2048 4096 8192 16384; do
  "$root/scripts/gpu_stat_run.sh" -n 50 -t 10 \
    -l "tile_${tile}" -o "$here/raw/tile_${tile}" -- \
    "$here/event_sync" --variant reduce --deps circular --sync no_fence \
      --fill alt --grid 128 --reuse-data --no-backoff --tile "$tile" \
    | tee -a "$here/raw/tile_scan.txt"
done

occupancy=$($here/event_sync --variant reduce --deps circular --sync correct \
  --grid 2 -v 2>&1 | sed -n 's/.*resident_cap=\([0-9][0-9]*\).*/\1/p' | head -1)
num_sms=$($here/event_sync --variant reduce --deps circular --sync correct \
  --grid 2 -v 2>&1 | sed -n 's/.*nsm=\([0-9][0-9]*\).*/\1/p' | head -1)
[[ -n "$occupancy" && -n "$num_sms" ]] || exit 2

: >"$here/raw/backward_scan.txt"
for grid in $((occupancy - 1)) "$occupancy" $((occupancy + 1)) $((2 * occupancy)); do
  "$root/scripts/gpu_stat_run.sh" -n 20 -t 2 \
    -l "backward_${grid}" -o "$here/raw/backward_${grid}" -- \
    "$here/event_sync" --variant reduce --deps backward --sync correct \
      --fill alt --grid "$grid" | tee -a "$here/raw/backward_scan.txt"
done

echo "resident_limit=$occupancy num_sms=$num_sms" | tee "$here/raw/target.txt"
