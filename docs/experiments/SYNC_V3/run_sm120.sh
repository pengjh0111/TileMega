#!/usr/bin/env bash
# R4 complete-protocol ablation. Written on sm_89; NOT RUN on sm_120.
# All controls are unmaterialized sources. Window and cluster plans are built
# by the host on the target's measured resident grid; no 4090 table is copied.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120}")"
case "${raw}" in "${here}/raw"|"${here}/raw/"*|"${here}/sass_identity"*) echo 'refusing sm_89 evidence' >&2; exit 2;; esac
label=sync_v3
NEED_MIB=16384
python_sources=("${repo}/docs/experiments/FENCE/run.py" "${here}/audit_raw.py" "${here}/measure_matrix.py" "${here}/run_litmus.py" "${here}/trace_dependency.py" "${here}/realwidth.py")
source "${here}/sm120_common.sh"
if [[ "${SELF_CHECK:-0}" == 1 ]]; then
  r4_init
  for switch in RELEASE_AFTER_BARRIER ASYNC_PUBLISH LOCAL_DEP_SMEM CLUSTER_ARRIVE; do
    rg -q "define TILEMEGA_${switch} 0" "${repo}/include/tilemega/Codegen/tasks/ModelRuntime.h"
  done
  exit 0
fi
r4_init
runner="${repo}/docs/experiments/FENCE/run.py"
c1='-DTILEMEGA_RELEASE_AFTER_BARRIER=1'
c2="${c1} -DTILEMEGA_ASYNC_PUBLISH=1"
c3="${c2} -DTILEMEGA_LOCAL_DEP_SMEM=1"
config_names=()
run_configuration() {
  local name="$1" window="$2" flags="$3"
  python3 "${runner}" build --arch sm_120 --raw "${raw}/${name}" --window "${window}" --extra-flags="${flags}"
  python3 "${runner}" correctness --arch sm_120 --raw "${raw}/${name}"
  config_names+=("${name}")
}
python3 "${here}/run_litmus.py" build --arch sm_120 --raw "${raw}/litmus"
python3 "${here}/run_litmus.py" pilot --arch sm_120 --raw "${raw}/litmus"
python3 "${here}/run_litmus.py" scan --arch sm_120 --raw "${raw}/litmus"
run_configuration baseline 1 ''
run_configuration c1 1 "${c1}"
python3 "${runner}" seqscan --arch sm_120 --raw "${raw}/c1"
run_configuration c2 1 "${c2}"
python3 "${runner}" seqscan --arch sm_120 --raw "${raw}/c2"
python3 "${runner}" build --arch sm_120 --raw "${raw}/c2_dependency" --kappa 2 --arms full --placements 0 --extra-flags="${c2} -DTILEMEGA_TRACE_V2=1"
python3 "${runner}" correctness --arch sm_120 --raw "${raw}/c2_dependency"
python3 "${here}/trace_dependency.py" run --raw "${raw}/c2_dependency"
run_configuration window2 2 "${c2}"
run_configuration local2 2 "${c3}"
run_configuration window4 4 "${c2}"
run_configuration local4 4 "${c3}"
# The cluster arrival itself is the sole difference in this pair. DSMEM
# reservation, shard layout, kernel cluster geometry and protocol are equal.
cluster="${c2} -DTILEMEGA_EVENT_SHARDED=1 -DTILEMEGA_EVENT_CLUSTER_FANIN=1 -DTILEMEGA_GENERATED_CLUSTER_DIM=2"
run_configuration cluster_off 1 "${cluster} -DTILEMEGA_CLUSTER_ARRIVE=0"
run_configuration cluster_on 1 "${cluster} -DTILEMEGA_CLUSTER_ARRIVE=1"
python3 -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); (p/"matrix_manifest.json").write_text(json.dumps({n:str(p/n) for n in sys.argv[2:]},indent=2)+"\n")' "${raw}" "${config_names[@]}"
python3 "${here}/measure_matrix.py" --manifest "${raw}/matrix_manifest.json" --out "${raw}/ablation"
for name in "${config_names[@]}"; do
  python3 "${runner}" summarize --arch sm_120 --raw "${raw}/ablation/${name}" > "${raw}/ablation/${name}/summary.txt"
done
python3 "${here}/realwidth.py" build --arch sm_120 --raw "${raw}/realwidth"
python3 "${here}/realwidth.py" measure --arch sm_120 --raw "${raw}/realwidth"
echo PASS > "${raw}/status.txt"
