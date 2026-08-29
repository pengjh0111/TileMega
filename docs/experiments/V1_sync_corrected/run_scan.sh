#!/usr/bin/env bash
# V1 grid scan: hang rate AND per-slot correctness, 50 runs per cell.
cd "$(dirname "$0")"
CSV=${1:-v1_gridscan.csv}; rm -f "$CSV"
run() { # name cubin entry variant [expect]
  local name=$1 cubin=$2 entry=$3 var=$4 exp=${5:-}
  for g in 3 30 80 120 170 340; do
    [ -n "$exp" ] && export V1_EXPECT=$exp || unset V1_EXPECT
    TILEMEGA_ARTIFACT_DIR=/tmp/v1_${name}_${g} \
    /data/tilemega/scripts/gpu_stat_run.sh --label "${name}_grid${g}" --runs 50 \
      --timeout 8 --csv "$CSV" --expect "all slots verified" \
      -- ./v1_host "$cubin" "$entry" "$g" "$var" 2>&1 | tail -1
  done
}
run ctrl V1_ctrl/spin_wait_tokenchain.cubin spin_wait_tokenchain ctrl
run a    V1_a/v1_a.cubin  v1_a  a
run b    V1_b/v1_b.cubin  v1_b  b
run min  /tmp/v1_min.cubin v1_min b 1024
echo "SCAN DONE"
