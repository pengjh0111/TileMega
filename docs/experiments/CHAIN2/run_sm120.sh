#!/usr/bin/env bash
# R4 cost-aware extension on Blackwell. Written on sm_89; NOT RUN on sm_120.
# Probe the target's resident geometry and solve every materialized Plan there.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="$(realpath -m "${OUT_DIR:-${here}/raw_sm120}")"
case "${raw}" in "${here}/raw"|"${here}/raw/"*|"${here}/final"|"${here}/final/"*) echo 'refusing sm_89 evidence' >&2; exit 2;; esac
label=chain2
NEED_MIB=16384
python_sources=("${here}/run.py" "${here}/build.py" "${here}/prepare_sm120.py" "${repo}/docs/experiments/SIMULATOR/hop_fit.py")
source "${repo}/docs/experiments/SYNC_V3/sm120_common.sh"
if [[ "${SELF_CHECK:-0}" == 1 ]]; then
  r4_init
  "${repo}/build-portable/chain_placement_test"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["feedback_rounds"]==4 and d["cap_fill"]' "${here}/design.json"
  python3 - "${repo}" <<'PY'
from pathlib import Path
import subprocess,sys,tempfile
source=Path(sys.argv[1])/'docs/experiments/SIMULATOR'
with tempfile.TemporaryDirectory(prefix='r4-hop-refit-') as tmp:
    output=Path(tmp)/'hop_ns.tsv'
    subprocess.run([sys.executable,str(source/'hop_fit.py'),str(source/'contention.tsv'),
        str(source/'contention_load.tsv'),str(output)],stdout=subprocess.DEVNULL,check=True)
    assert output.read_bytes()==(source/'hop_ns.tsv').read_bytes()
print('HOP_REFIT committed sm_89 inputs PASS; no GPU execution')
PY
  exit 0
fi
r4_init
# All six cells are required. Missing fixtures fail here, before compilation.
for model in gqa2 mha4; do
  for seq in 4 128; do
    test -d "${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
  done
done
for seq in 4 128; do
  test -d "${repo}/docs/experiments/REALMODEL/raw/work/r2sim_s${seq}/export/fixture"
done
python3 "${here}/build.py" > "${raw}/driver_build.log" 2>&1
# Measure this machine's hop curve in an R4-owned directory. The old
# SIMULATOR/raw_sm120 table remains a historical comparison only.
mkdir -p "${raw}/calibration/bin"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
for arm in rmw load; do
  load_poll=0
  [[ "${arm}" != load ]] || load_poll=1
  "${nvcc}" -std=c++17 -O2 -arch=sm_120 -I "${repo}/include" \
    -DTILEMEGA_EVENT_LOAD_POLL="${load_poll}" \
    "${repo}/docs/experiments/SIMULATOR/contention.cu" -o "${raw}/calibration/bin/contention_${arm}"
  "${raw}/calibration/bin/contention_${arm}" 4096 20000 "${raw}/calibration/${arm}.tsv"
done
python3 "${repo}/docs/experiments/SIMULATOR/hop_fit.py" \
  "${raw}/calibration/rmw.tsv" "${raw}/calibration/load.tsv" \
  "${raw}/calibration/hop_ns.tsv" > "${raw}/calibration/hop_fit.txt"
python3 "${here}/prepare_sm120.py" --out "${raw}/prepare_local" --realwidth 1
target="${TARGET_JSON:-${raw}/prepare_local/sm_120.json}"
hop="${raw}/calibration/hop_ns.tsv"
test -s "${hop}"
for variant in on off original; do
  mkdir -p "${raw}/${variant}/plan"
  aware=0; feedback=4
  [[ "${variant}" != on ]] || aware=1
  [[ "${variant}" != original ]] || feedback=0
  env TILEMEGA_CHAIN_COST_AWARE="${aware}" TILEMEGA_CHAIN_FEEDBACK="${feedback}" \
    "${here}/raw/place_chain" "${repo}" "${raw}/prepare_local/manifest.tsv" "${raw}/${variant}" \
    --target "${target}" --hop "${hop}" > "${raw}/${variant}/solver.log" 2>&1
done
# The matched on/off pair fixes feedback and capacity limits. The original
# arm preserves the old greedy as well, so the cheaper-hop prediction is
# falsifiable from raw hop counts, rejected price bins and paired makespans.
python3 "${here}/run.py" build --arch sm_120 --raw "${raw}"
python3 "${here}/run.py" correctness --arch sm_120 --raw "${raw}"
python3 "${here}/run.py" measure --arch sm_120 --raw "${raw}"
python3 "${here}/run.py" summarize --arch sm_120 --raw "${raw}"
echo PASS > "${raw}/status.txt"
