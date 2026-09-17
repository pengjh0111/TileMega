#!/usr/bin/env bash
# R6 shared runner. Written/self-checked on RTX 4090; NOT RUN on sm_120.
# Usage: sm120_common.sh cost|joint|rebase|models
# SELF_CHECK=1: guard/parser and sm_120 compile checks, NO sm_120 GPU launch.
# OUT_DIR: fresh artifact directory; NEED_MIB=131072 minimum free disk.
# SEARCH_CAPACITY=18: explicit reduced outer candidate budget (not optimality).
# LOCAL_R5_ROOT: optional same-GPU-UUID baseline made by this runner; otherwise
# regenerate local inputs, target calibration, R5 controls and local Plans.
# Dependencies: CUDA with sm_120 support, configured build-portable for this
# target, MLIR/ISL/Barvinok, Python/PyTorch and the repository export bridge.
# Artifacts: status.txt, pipeline.log, device.json, local_r5/{inputs,calibration,
# joint}, kloop/{build,bin,correctness,phase}, body_fit, target.json, replay.tsv,
# tools, joint/*/{auto.cu,auto.mlir,top-3,pilot,measure,correctness,trace},
# symbolic (local CG only), optional rebase and llama_covered maximal supported graph artifacts.
set -euo pipefail
mode="$1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${OUT_DIR:-${here}/raw_sm120_${mode}}"
need_mib="${NEED_MIB:-131072}"
mkdir -p "${raw}"
status="${raw}/status.txt"
trap 'echo FAIL > "${status}"' ERR
reject_inherited() {
  local name
  while IFS='=' read -r name _; do
    if [[ "${name}" == TILEMEGA_* ]]; then echo "refusing inherited ${name}" >&2; return 1; fi
  done < <(env)
}
cap_ok() { [[ "$1" == 12.0 ]]; }
check_disk() {
  local free
  free="$(df -Pm "${raw}" | awk 'NR==2 {print $4}')"
  echo "DISK NEED_MIB=${need_mib} FREE_MIB=${free}"
  (( free >= need_mib ))
}
if [[ "${SELF_CHECK:-0}" == 1 ]]; then
  cap_ok 12.0; ! cap_ok 8.9; ! cap_ok ''
  (unset "${!TILEMEGA_@}"; reject_inherited)
  ! (export TILEMEGA_TRACE_KLOOP=1; reject_inherited)
  check_disk
  (need_mib=1099511627776; ! check_disk)
  python3 - "${repo}" <<'PY'
import ast,sys
from pathlib import Path
repo=Path(sys.argv[1])
for name in ('COSTMODEL','JOINT2','REBASE','MODELS','SYMBOLIC'):
 for p in (repo/'docs/experiments'/name).glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for name in ('COSTMODEL/kloop.py','COSTMODEL/fit_body.py','JOINT2/run.py','JOINT2/build_tools.py','REBASE/generate.cpp','SYMBOLIC/r6_templates.cpp','MODELS/export_covered.py','MODELS/run_covered.py'):
 assert (repo/'docs/experiments'/name).is_file(),name
PY
  /usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_120 \
    -DTILEMEGA_TRACE_PHASE=1 -DTILEMEGA_TRACE_KLOOP=1 \
    -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
    -I"${repo}/third_party/cutlass/tools/util/include" -I"${repo}/third_party/cutlass/test" \
    -c "${repo}/docs/experiments/PLAN_CONTRACT/legacy_identity/plan/gqa2.cu" \
    -o "${raw}/selfcheck_sm120.o" > "${raw}/compile.log" 2>&1
  echo 'SELF_CHECK PASS; sm_120 not run' | tee "${status}"
  exit 0
fi
reject_inherited
check_disk
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
cap_ok "${cap}"
args=()
if [[ -n "${LOCAL_R5_ROOT:-}" ]]; then args+=(--local-r5-root "${LOCAL_R5_ROOT}"); fi
python3 "${here}/target_pipeline.py" --mode "${mode}" --out "${raw}" \
  --capacity "${SEARCH_CAPACITY:-18}" "${args[@]}" > "${raw}/pipeline.log" 2>&1
echo PASS > "${status}"
