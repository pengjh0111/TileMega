#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# Part 1 evidence: the declarative matcher is normalization invariant.
# The same exported program is serialized twice -- as exported, and after
# torch's own run_decompositions() to Core ATen -- and both are pushed through
# the whole frontend. The model plan (stage kinds and operand wiring) must be
# the same on both, which is what "semantics recovered from use-def structure"
# means operationally.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${repo}/build-portable"
export PYTHONPATH="${repo}/.venv/site${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${raw}"

for model in e2e mha; do
  case "${model}" in
    e2e) program="${repo}/docs/experiments/V_H/raw/exported_program.pt2" ;;
    mha) program="${repo}/docs/experiments/P3_GENERALIZATION/raw/exported_program.pt2" ;;
  esac
  if [[ ! -e "${program}" ]]; then
    echo "BLOCKED: ${program} missing" | tee "${raw}/run_status.txt" >&2
    exit 77
  fi
  python3 "${repo}/python/tilemega/export_bridge.py" "${program}" \
    --out "${raw}/${model}_plain.json" \
    --normalization-report "${raw}/${model}_plain_report.json" \
    > "${raw}/${model}_plain_summary.txt"
  python3 "${repo}/python/tilemega/export_bridge.py" "${program}" --decompose \
    --out "${raw}/${model}_core.json" \
    --normalization-report "${raw}/${model}_core_report.json" \
    > "${raw}/${model}_core_summary.txt"
  for form in plain core; do
    "${build}/tools/tilemega-compile" "${raw}/${model}_${form}.json" \
      "${raw}/${model}_${form}.cu" 2> "${raw}/${model}_${form}_codegen.txt" \
      || echo "COMPILE_FAILED ${model} ${form}" >> "${raw}/${model}_${form}_codegen.txt"
    # The stage table is the model plan the matcher produced; comparing it
    # rather than the whole file keeps tile-size noise out of the diff.
    sed -n '/kStages\[\]/,/};/p' "${raw}/${model}_${form}.cu" \
      > "${raw}/${model}_${form}_stages.txt" || true
  done
  if diff -q "${raw}/${model}_plain_stages.txt" "${raw}/${model}_core_stages.txt" \
       > /dev/null 2>&1; then
    echo "STAGES_IDENTICAL ${model}"
  else
    echo "STAGES_DIFFER ${model}"
  fi
  if diff -q "${raw}/${model}_plain.cu" "${raw}/${model}_core.cu" > /dev/null 2>&1; then
    echo "CODEGEN_IDENTICAL ${model}"
  else
    echo "CODEGEN_DIFFERS ${model} $(diff "${raw}/${model}_plain.cu" \
      "${raw}/${model}_core.cu" | grep -c '^[<>]') lines"
  fi
done | tee "${raw}/normalization.txt"

"${build}/degradation_test" > "${raw}/degradation.txt" 2>&1 \
  && echo "DEGRADATION pass" || echo "DEGRADATION fail"
echo PASS > "${raw}/run_status.txt"
