#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Rebuild and validate the queue-driven L2 path in fresh processes.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-50}"
raw="${here}/raw"
mkdir -p "${raw}/src" "${raw}/bin" "${raw}/log"

declare -A export fixture
export[gqa2]="${repo}/docs/experiments/ORACLE/raw_bf16/export/gqa2.json"
fixture[gqa2]="${repo}/docs/experiments/ORACLE/raw_bf16/fixture/gqa2"
export[mha4]="${repo}/docs/experiments/ORACLE/raw_bf16/export/mha4.json"
fixture[mha4]="${repo}/docs/experiments/ORACLE/raw_bf16/export/mha4/fixture"

"${build}/tools/tilemega-target-audit" \
  "${repo}" > "${raw}/target_audit.txt"
ctest --test-dir "${build}" --output-on-failure > "${raw}/ctest.txt"

common=(-std=c++17 -O2 -arch=native -lineinfo
  -DTILEMEGA_EVENT_KAPPA=1 -I"${repo}/include"
  -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")

printf 'model\tpasses\tprocesses\ttask_refs\ttask_ref_bytes\twaits\twait_bytes\twaiting_tasks\tnormalization_dummies_lb\tmax_worker_span\tresident_limit\ti3_overresident\n' \
  > "${raw}/correctness.tsv"
for model in gqa2 mha4; do
  [[ -f "${export[$model]}" && -f "${fixture[$model]}/manifest.json" ]] || {
    echo "missing BF16 export/fixture; run docs/experiments/ORACLE/run_bf16.sh first" >&2
    exit 77
  }
  "${build}/tools/tilemega-compile" "${export[$model]}" \
    "${raw}/src/${model}.cu" --variants \
    "${repo}/docs/experiments/OWNERSHIP/plan_structured.json" \
    > "${raw}/log/${model}.codegen" 2>&1
  "${nvcc}" "${common[@]}" "${raw}/src/${model}.cu" \
    "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart \
    -o "${raw}/bin/${model}" 2> "${raw}/log/${model}.ptxas"
  log="${raw}/log/${model}.fresh.txt"
  : > "${log}"
  for unused in $(seq 1 "${runs}"); do
    timeout 30s "${raw}/bin/${model}" "${fixture[$model]}" >> "${log}" 2>&1
  done
  pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
  [[ "${pass}" == "${runs}" ]] || { echo "${model}: ${pass}/${runs}" >&2; exit 1; }
  schedule="$(grep -m1 '^E2E_SCHEDULE' "${log}")"
  field() { sed -n "s/.* $1=\\([^ ]*\\).*/\\1/p" <<< "${schedule}"; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}" "${pass}" "${runs}" "$(field task_refs)" \
    "$(field task_ref_bytes)" "$(field waits)" "$(field wait_bytes)" \
    "$(field waiting_tasks)" "$(field normalization_dummies_lb)" \
    "$(field max_worker_span)" "$(field resident_limit)" \
    "$(field i3_overresident)" >> "${raw}/correctness.tsv"
  TILEMEGA_TASK_TRACE=1 "${raw}/bin/${model}" "${fixture[$model]}" \
    > "${raw}/log/${model}.overlap.txt"
done
cat "${raw}/correctness.tsv"
grep '^E2E_OVERLAP' "${raw}"/log/*.overlap.txt
