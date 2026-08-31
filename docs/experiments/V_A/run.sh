#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
here="$root/docs/experiments/V_A"
nvcc=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}
mode=${1:-base}

"$nvcc" -O2 -std=c++17 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -I "$root/test/harness" \
  -o "$here/event_sync" "$here/event_sync.cu" 2>"$here/ptxas.log"

if [[ "$mode" == "--supplement" ]]; then
  "$nvcc" -O2 -std=c++17 -arch=sm_89 -cubin -I "$root/test/harness" \
    -o "$here/event_sync.cubin" "$here/event_sync.cu"
  kernel='_Z12event_kernelILi1ELi0EEvPfS0_P12EventCounterS2_iiiiiijii'
  occupancy=$($root/tools/tilemega-occupancy --cubin "$here/event_sync.cubin" \
    --kernel "$kernel" --dynamic-smem 32)
  echo "$occupancy" | tee "$here/supplement_occupancy.txt"
  num_sms=$(sed -n 's/.* num_sms=\([0-9][0-9]*\).*/\1/p' <<<"$occupancy")
  limit=$(sed -n 's/.* resident_limit=\([0-9][0-9]*\).*/\1/p' <<<"$occupancy")

  for sync in correct_hostile no_fence_hostile; do
    for grid in $((num_sms/2)) "$num_sms" $((2*num_sms)); do
      "$root/scripts/gpu_stat_run.sh" -n 50 -t 20 -l "${sync}_g${grid}" -- \
        "$here/event_sync" --variant reduce --deps circular --sync "$sync" \
        --fill alt --grid "$grid"
    done
  done

  for case_name in reuse_only no_backoff_only large_tile_only; do
    for grid in $((num_sms/2)) "$num_sms" $((2*num_sms)); do
      args=(--variant reduce --deps circular --sync no_fence --fill alt --grid "$grid")
      case "$case_name" in
        reuse_only) args+=(--reuse-data);;
        no_backoff_only) args+=(--no-backoff);;
        large_tile_only) args+=(--tile 8192);;
      esac
      "$root/scripts/gpu_stat_run.sh" -n 50 -t 20 -l "${case_name}_g${grid}" -- \
        "$here/event_sync" "${args[@]}" || true
    done
  done
  for grid in $((num_sms/2)) "$num_sms" $((2*num_sms)); do
    "$root/scripts/gpu_stat_run.sh" -n 50 -t 20 -l "correct_reuse_only_g${grid}" -- \
      "$here/event_sync" --variant reduce --deps circular --sync correct \
      --fill alt --grid "$grid" --reuse-data
  done

  for grid in $((num_sms/2)) "$num_sms" $((2*num_sms)); do
    "$root/scripts/gpu_stat_run.sh" -n 50 -t 1 -l "barrier_in_spin_g${grid}" -- \
      "$here/event_sync" --variant reduce --deps circular \
      --sync barrier_in_spin --fill alt --grid "$grid" || true
  done

  timing_raw="$here/supplement_polling_timing_raw.txt"
  timing_summary="$here/supplement_polling_timing_summary.txt"
  : >"$timing_raw"
  : >"$timing_summary"
  for grid in $((num_sms/2)) "$num_sms" $((2*num_sms)); do
    for sync in correct allthread; do
      for run in $(seq 1 20); do
        ms=$($here/event_sync --variant reduce --deps circular --sync "$sync" \
          --fill alt --grid "$grid" | awk '/^KERNEL_MS/{print $2}')
        echo "TIMING grid=$grid sync=$sync run=$run kernel_ms=$ms" >>"$timing_raw"
      done
      median=$(awk -v g="$grid" -v s="$sync" \
        '$2=="grid="g && $3=="sync="s {sub("kernel_ms=","",$5); print $5}' \
        "$timing_raw" | sort -n | awk 'NR==10{a=$1} NR==11{printf "%.6f", (a+$1)/2}')
      echo "MEDIAN grid=$grid sync=$sync kernel_ms=$median" >>"$timing_summary"
    done
    correct=$(awk -v g="$grid" '$2=="grid="g && $3=="sync=correct" {sub("kernel_ms=","",$4); print $4}' "$timing_summary")
    allthread=$(awk -v g="$grid" '$2=="grid="g && $3=="sync=allthread" {sub("kernel_ms=","",$4); print $4}' "$timing_summary")
    ratio=$(awk -v a="$allthread" -v c="$correct" 'BEGIN{printf "%.3f", a/c}')
    echo "RATIO grid=$grid allthread_over_correct=$ratio" >>"$timing_summary"
  done

  for grid in $((limit-num_sms)) $((limit-1)) "$limit" $((limit+1)) \
              $((limit+num_sms)) $((2*limit)); do
    "$root/scripts/gpu_stat_run.sh" -n 50 -t 1 -l "boundary_g${grid}" -- \
      "$here/event_sync" --variant reduce --deps circular --sync correct \
      --fill alt --grid "$grid" || true
  done
  exit 0
fi

num_sms=$($here/event_sync -v --grid 2 2>&1 | sed -n 's/.* nsm=\([0-9][0-9]*\).*/\1/p' | head -1)
for spec in 'A1 elementwise circular alt' 'A2 reduce circular const' \
            'A3 reduce circular alt' 'A4 reduce shared alt' \
            'A5 reduce exclusive alt'; do
  read -r label variant deps fill <<<"$spec"
  for mul in 1/4 1/2 1 2 4; do
    case "$mul" in
      1/4) grid=$((num_sms / 4));;
      1/2) grid=$((num_sms / 2));;
      *) grid=$((num_sms * mul));;
    esac
    "$root/scripts/gpu_stat_run.sh" -n 50 -t 20 \
      -l "${label}_g${grid}" -o "$here/raw/${label}_g${grid}" -- \
      "$here/event_sync" --variant "$variant" --deps "$deps" \
      --sync correct --fill "$fill" --grid "$grid"
  done
done
