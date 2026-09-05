#!/usr/bin/env bash
# Reproduce the runtime-variant resource curve, two-interval correctness and
# the former COARSEN 0/50 regression. Models and fixtures are regenerated.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-phase12}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
raw="${here}/raw"
mkdir -p "${raw}"/{plans,export,fixture,src,bin,log}
cmake --build "${build}" --target tilemega-compile --parallel "$(nproc)" >/dev/null
python3 "${here}/generate_plans.py" "${raw}/plans"

for dtype in bf16 f32; do
  python3 "${repo}/docs/experiments/V_H/export_probe.py" \
    --out "${raw}/export/gqa2_${dtype}" --dtype "${dtype}" \
    --seq-max 2048 --past-max 512 >/dev/null
  python3 "${repo}/python/tilemega/export_bridge.py" \
    "${raw}/export/gqa2_${dtype}/exported_program.pt2" \
    --out "${raw}/export/gqa2_${dtype}.json" >/dev/null
  python3 "${repo}/docs/experiments/P3_GENERALIZATION/export_second.py" \
    --repo "${repo}" --out "${raw}/export/mha4_${dtype}" --dtype "${dtype}" \
    --seq-max 2048 --past-max 512 --fixture-seq 4 --fixture-past 3 >/dev/null
  python3 "${repo}/python/tilemega/export_bridge.py" \
    "${raw}/export/mha4_${dtype}/exported_program.pt2" \
    --out "${raw}/export/mha4_${dtype}.json" >/dev/null
done
for seq in 4 2048; do
  python3 "${repo}/docs/experiments/E2E/prepare_e2e.py" \
    --vh-raw "${raw}/export/gqa2_bf16" \
    --out "${raw}/fixture/gqa2_bf16_s${seq}" --seq "${seq}" --past 3 >/dev/null
  python3 "${repo}/docs/experiments/P3_GENERALIZATION/prepare_fixture.py" \
    --repo "${repo}" --program "${raw}/export/mha4_bf16/exported_program.pt2" \
    --out "${raw}/fixture/mha4_bf16_s${seq}" --seq "${seq}" --past 3 >/dev/null
done
python3 "${repo}/docs/experiments/E2E/prepare_e2e.py" \
  --vh-raw "${raw}/export/gqa2_f32" --out "${raw}/fixture/gqa2_f32" \
  --seq 4 --past 3 >/dev/null

common=(-std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v
  -DTILEMEGA_EVENT_KAPPA=1 -I"${repo}/include"
  -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
build_one() {
  local export_json="$1" plan="$2" tag="$3"
  "${build}/tools/tilemega-compile" "${export_json}" "${raw}/src/${tag}.cu" \
    --variants "${plan}" >"${raw}/log/${tag}.codegen" 2>&1
  "${nvcc}" "${common[@]}" "${raw}/src/${tag}.cu" \
    "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart \
    -o "${raw}/bin/${tag}" 2>"${raw}/log/${tag}.ptxas"
}

printf 'composition\tvariants\ttask_union_bytes\tgemm_union_bytes\tctas_per_sm\tregisters\tstatus\n' \
  > "${raw}/curve.tsv"
for composition in ${COMPOSITIONS:-same multi}; do
  for count in ${COUNTS:-1 2 4 8 16}; do
    tag="${composition}_${count}"
    build_one "${raw}/export/gqa2_bf16.json" "${raw}/plans/${tag}.json" "${tag}"
    "${raw}/bin/${tag}" "${raw}/fixture/gqa2_bf16_s4" >"${raw}/log/${tag}.run"
    resource="$(grep '^E2E_RESOURCE' "${raw}/log/${tag}.run")"
    task="$(sed -n 's/.* smem=\([0-9]*\).*/\1/p' <<<"${resource}")"
    gemm="$(sed -n 's/.* gemm_union=\([0-9]*\).*/\1/p' <<<"${resource}")"
    ctas="$(sed -n 's/.* ctas_per_sm=\([0-9]*\).*/\1/p' <<<"${resource}")"
    regs="$(grep -o 'Used [0-9]* registers' "${raw}/log/${tag}.ptxas" \
      | awk '{print $2}' | sort -rn | awk 'NR==1')"
    status="$(sed -n 's/^RESULT status=//p' "${raw}/log/${tag}.run")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${composition}" "${count}" \
      "${task}" "${gemm}" "${ctas}" "${regs}" "${status}" | tee -a "${raw}/curve.tsv"
  done
done

printf 'model\tdtype\tseq\tselected_variant\ttile_m\ttile_n\ttile_k\tpasses\tprocesses\n' \
  > "${raw}/multi_variant.tsv"
for model in gqa2 mha4; do
  tag="${model}_multi"
  build_one "${raw}/export/${model}_bf16.json" "${here}/plan_two.json" "${tag}"
  for seq in 4 2048; do
    log="${raw}/log/${tag}_s${seq}.txt"; : > "${log}"
    for unused in $(seq 1 50); do
      "${raw}/bin/${tag}" "${raw}/fixture/${model}_bf16_s${seq}" >> "${log}"
    done
    pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
    variant="$(sed -n 's/^E2E_VARIANT index=\([0-9]*\).*/\1/p' "${log}" | sort -u)"
    if [[ "${variant}" == 0 ]]; then m=32; n=16; else m=32; n=32; fi
    printf '%s\tbf16\t%s\t%s\t%s\t%s\t16\t%s\t50\n' \
      "${model}" "${seq}" "${variant}" "${m}" "${n}" "${pass}" \
      | tee -a "${raw}/multi_variant.tsv"
    [[ "${pass}" == 50 ]] || exit 1
  done
done

printf 'model\tdtype\ttile_m\ttile_n\ttile_k\tstages\tsplit_k\twait_table\tpasses\tprocesses\n' \
  > "${raw}/coarsen_regression.tsv"
for model in gqa2 mha4; do
  tag="${model}_coarsen"
  build_one "${raw}/export/${model}_f32.json" \
    "${here}/plan_coarsen_regression.json" "${tag}"
  fixture="${raw}/fixture/gqa2_f32"
  [[ "${model}" == mha4 ]] && fixture="${raw}/export/mha4_f32/fixture"
  log="${raw}/log/${tag}.txt"; : > "${log}"
  for unused in $(seq 1 50); do "${raw}/bin/${tag}" "${fixture}" >> "${log}"; done
  pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
  table="$(sed -n 's/^E2E_KAPPA.*dependency_table=//p' "${log}" | sort -u)"
  printf '%s\tf32\t16\t64\t16\t2\t16\t%s\t%s\t50\n' \
    "${model}" "${table}" "${pass}" | tee -a "${raw}/coarsen_regression.tsv"
  [[ "${pass}" == 50 && "${table}" == variant_exact ]] || exit 1
done
