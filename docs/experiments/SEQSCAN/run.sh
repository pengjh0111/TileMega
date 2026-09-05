#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-phase12}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
raw="${here}/raw"
mkdir -p "${raw}"/{export,fixture,src,bin,log}

python3 "${repo}/docs/experiments/V_H/export_probe.py" \
  --out "${raw}/export/gqa2" --dtype bf16 --seq-max 2048 --past-max 512
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/gqa2/exported_program.pt2" --out "${raw}/export/gqa2.json"
python3 "${repo}/docs/experiments/P3_GENERALIZATION/export_second.py" \
  --repo "${repo}" --out "${raw}/export/mha4" --dtype bf16 \
  --seq-max 2048 --past-max 512 --fixture-seq 4 --fixture-past 3
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/mha4/exported_program.pt2" --out "${raw}/export/mha4.json"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
for model in gqa2 mha4; do
  "${build}/tools/tilemega-compile" "${raw}/export/${model}.json" \
    "${raw}/src/${model}.cu" --variants \
    "${repo}/docs/experiments/OWNERSHIP/plan_structured.json"
  "${nvcc}" "${common[@]}" "${raw}/src/${model}.cu" \
    "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart \
    -o "${raw}/bin/${model}"
done
"${nvcc}" "${common[@]}" -DTILEMEGA_NEGATIVE_OLD_CLAMP=1 \
  "${raw}/src/gqa2.cu" "${build}/libtilemega.a" \
  -L/usr/local/cuda/lib64 -lcudart -o "${raw}/bin/gqa2_old_clamp"

printf 'model\tseq\tpast\tpasses\tprocesses\n' > "${raw}/matrix.tsv"
for model in gqa2 mha4; do
  for seq in 1 4 128 512 2048; do
    for past in 0 3 512; do
      fixture="${raw}/fixture/${model}_s${seq}_p${past}"
      if [[ "${model}" == gqa2 ]]; then
        python3 "${repo}/docs/experiments/E2E/prepare_e2e.py" \
          --vh-raw "${raw}/export/gqa2" --out "${fixture}" \
          --seq "${seq}" --past "${past}"
      else
        python3 "${repo}/docs/experiments/P3_GENERALIZATION/prepare_fixture.py" \
          --repo "${repo}" --program "${raw}/export/mha4/exported_program.pt2" \
          --out "${fixture}" --seq "${seq}" --past "${past}"
      fi
      log="${raw}/log/${model}_s${seq}_p${past}.txt"; : > "${log}"
      for unused in $(seq 1 50); do
        "${raw}/bin/${model}" "${fixture}" >> "${log}"
      done
      pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
      printf '%s\t%s\t%s\t%s\t50\n' \
        "${model}" "${seq}" "${past}" "${pass}" | tee -a "${raw}/matrix.tsv"
    done
  done
done

negative="${raw}/log/gqa2_old_clamp_s2048_p0.txt"; : > "${negative}"
for unused in $(seq 1 50); do
  "${raw}/bin/gqa2_old_clamp" "${raw}/fixture/gqa2_s2048_p0" \
    >> "${negative}" || true
done
pass="$(grep -c '^RESULT status=PASS' "${negative}" || true)"
[[ "${pass}" == 0 ]] || { echo 'old clamp unexpectedly passed' >&2; exit 1; }
