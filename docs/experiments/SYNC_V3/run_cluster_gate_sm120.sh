#!/usr/bin/env bash
# Cluster-only extract of run_sm120.sh's cluster_off/cluster_on pair (EX-E3
# step 6, TILEMEGA_CLUSTER_ARRIVE). The full run_sm120.sh reaches this pair
# only after baseline/c1/c2/c2_dependency/window2/local2/window4/local4 and
# the litmus scan, several hours of unrelated non-cluster work. This script
# builds and correctness-gates just the two cluster arms so the one
# sm_90+-only measurement in that script can be prioritized on its own.
# It is not a substitute for run_sm120.sh: the ablation timing
# (measure_matrix.py) and realwidth arms are not run here.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120_cluster_gate}")"
case "${raw}" in "${here}/raw"|"${here}/raw/"*|"${here}/sass_identity"*) echo 'refusing sm_89 evidence' >&2; exit 2;; esac
label=sync_v3_cluster_gate
NEED_MIB=4096
python_sources=("${repo}/docs/experiments/FENCE/run.py")
source "${here}/sm120_common.sh"
if [[ "${SELF_CHECK:-0}" == 1 ]]; then
  r4_init
  rg -q "define TILEMEGA_CLUSTER_ARRIVE 0" "${repo}/include/tilemega/Codegen/tasks/ModelRuntime.h"
  exit 0
fi
r4_init
runner="${repo}/docs/experiments/FENCE/run.py"
# Identical to run_sm120.sh's c2 and cluster flag composition -- kept in sync
# by hand since this script does not source the other one's variables.
c2='-DTILEMEGA_RELEASE_AFTER_BARRIER=1 -DTILEMEGA_ASYNC_PUBLISH=1'
cluster="${c2} -DTILEMEGA_EVENT_SHARDED=1 -DTILEMEGA_EVENT_CLUSTER_FANIN=1 -DTILEMEGA_GENERATED_CLUSTER_DIM=2"
for name_flags in "cluster_off|${cluster} -DTILEMEGA_CLUSTER_ARRIVE=0" \
                   "cluster_on|${cluster} -DTILEMEGA_CLUSTER_ARRIVE=1"; do
  IFS='|' read -r name flags <<<"${name_flags}"
  # --arms full --placements 0 because correctness() only ever exercises the
  # p0_full binary; the other 4 arms/placement-5 exist only for run_sm120.sh's
  # later measure_matrix.py ablation, which this script does not run.
  python3 "${runner}" build --arch sm_120 --raw "${raw}/${name}" --window 1 \
    --arms full --placements 0 --extra-flags="${flags}"
  python3 "${runner}" correctness --arch sm_120 --raw "${raw}/${name}"
done
echo PASS > "${raw}/status.txt"
