#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
here="$root/docs/experiments/V_A"
nvcc=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}

"$nvcc" -O2 -std=c++17 -arch=sm_89 -lineinfo \
  --ptxas-options=-v -I "$root/test/harness" \
  -o "$here/event_sync" "$here/event_sync.cu" 2>"$here/ptxas.log"

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
