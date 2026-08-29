#!/usr/bin/env bash
cd "$(dirname "$0")"
CSV=v0_gridscan.csv; rm -f $CSV
for ti in 131 133; do
  for grid in 30 80 120 170 340; do
    TILEMEGA_TILEIRAS="tileiras_$ti" TILEMEGA_ARTIFACT_DIR=/tmp/v0_${ti}_${grid} \
    /data/tilemega/scripts/gpu_stat_run.sh \
      --label "ti${ti}_grid${grid}" --runs 50 --timeout 6 --csv $CSV \
      --expect "completed without hang" \
      -- ./host_test_single sw_ti${ti}.cubin spin_wait_relaxed_acquire $grid 2>&1 | tail -1
  done
done
echo "V0 DONE"
