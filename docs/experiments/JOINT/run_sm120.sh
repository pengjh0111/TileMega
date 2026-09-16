#!/usr/bin/env bash
# R5 joint search. Written/self-checked on RTX 4090; NOT RUN on sm_120.
# SELF_CHECK=1: CPU guard/parser checks and sm_120 compilation, no GPU launch.
# OUT_DIR: fresh output directory (default raw_sm120); NEED_MIB: minimum free
# disk, default 131072 MiB. Requires configured build-portable + Ninja, CUDA
# supporting sm_120, Python/PyTorch and repository export dependencies.
# Artifacts: inputs/{export,fixture,src}, phase/{build,measure,trace}, target.json,
# calibration/{fence,hop_ns.tsv,publication.json}, joint/*/{plans,top3.tsv,
# pilot,choice.json,measure,correctness,trace}, logs/status.txt. No foreign Plan.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${OUT_DIR:-${here}/raw_sm120}"
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
  cap_ok 12.0
  ! cap_ok 8.9
  ! cap_ok ''
  (unset "${!TILEMEGA_@}"; reject_inherited)
  ! (export TILEMEGA_TRACE_PHASE=1; reject_inherited)
  check_disk
  (need_mib=1099511627776; ! check_disk)
  python3 - "${here}" "${repo}/docs/experiments/PHASE" <<'PY'
import ast,sys
from pathlib import Path
for root in sys.argv[1:]:
 for p in Path(root).glob('*.py'):ast.parse(p.read_text(),filename=str(p))
PY
  tmp="$(mktemp -d)"
  /usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_120 -I"${repo}/include" -c \
    "${repo}/docs/experiments/SIMULATOR/contention.cu" -o "${tmp}/contention.o"
  /usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_120 -DTILEMEGA_TRACE_PHASE=1 \
    -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
    -I"${repo}/third_party/cutlass/tools/util/include" -I"${repo}/third_party/cutlass/test" \
    -c "${repo}/docs/experiments/PLAN_CONTRACT/legacy_identity/plan/gqa2.cu" \
    -o "${tmp}/phase_gqa2.o"
  rm -rf "${tmp}"
  echo 'SELF_CHECK PASS; sm_120 not run' | tee "${status}"
  exit 0
fi
reject_inherited
check_disk
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
cap_ok "${cap}"
args=()
if [[ "${PHASE_ONLY:-0}" == 1 ]]; then args+=(--phase-only); fi
python3 "${here}/target_pipeline.py" --out "${raw}" "${args[@]}" > "${raw}/pipeline.log" 2>&1
echo PASS > "${status}"
