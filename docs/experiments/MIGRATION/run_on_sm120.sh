#!/usr/bin/env bash
# BF16 + structured-ownership migration smoke test for the configured cluster
# target. TargetSpec::Probe must match the target JSON and report clusters.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-migrate}"
raw="${here}/raw"
runs="${RUNS:-50}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
mkdir -p "${raw}/export" "${raw}/src" "${raw}/bin" "${raw}/fixture" "${raw}/log"

command -v python3 >/dev/null || { echo 'FAIL: python3 not found' >&2; exit 77; }
[[ -x "${nvcc}" ]] || { echo "FAIL: nvcc not found at ${nvcc}" >&2; exit 77; }
python3 -c 'import torch' || {
  echo 'FAIL: PyTorch 2.13.x or 2.14.x is required to regenerate BF16 fixtures' >&2
  exit 77
}

cmake -S "${repo}" -B "${build}" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTILEMEGA_TARGET_ARCH=auto >/dev/null
cmake --build "${build}" --target tilemega-compile tilemega-migrate \
  --parallel "$(nproc)" >/dev/null
"${build}/tools/tilemega-migrate" --repo "${repo}" --probe \
  | tee "${raw}/probe.txt"

python3 "${repo}/docs/experiments/V_H/export_probe.py" \
  --out "${raw}/export/gqa2" --dtype bf16 --seq-max 2048 --past-max 512
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/gqa2/exported_program.pt2" --out "${raw}/export/gqa2.json"
python3 "${repo}/docs/experiments/E2E/prepare_e2e.py" \
  --vh-raw "${raw}/export/gqa2" --out "${raw}/fixture/gqa2" --seq 4 --past 3

python3 "${repo}/docs/experiments/P3_GENERALIZATION/export_second.py" \
  --repo "${repo}" --out "${raw}/export/mha4" --dtype bf16 \
  --seq-max 2048 --past-max 512 --fixture-seq 4 --fixture-past 3
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/mha4/exported_program.pt2" --out "${raw}/export/mha4.json"

common=(-std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v
  -DTILEMEGA_EVENT_KAPPA=1 -I"${repo}/include"
  -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
printf 'model\tpass\tprocesses\tvariant\tl05_median_ms\tl1_median_ms\tl2_median_ms\n' \
  > "${raw}/summary.tsv"
median() { sort -n | awk '{v[NR]=$1} END{printf "%.6f",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
for model in gqa2 mha4; do
  fixture="${raw}/fixture/gqa2"
  [[ "${model}" == mha4 ]] && fixture="${raw}/export/mha4/fixture"
  "${build}/tools/tilemega-compile" "${raw}/export/${model}.json" \
    "${raw}/src/${model}.cu" --variants \
    "${repo}/docs/experiments/OWNERSHIP/plan_structured.json"
  "${nvcc}" "${common[@]}" "${raw}/src/${model}.cu" \
    "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart \
    -o "${raw}/bin/${model}" 2> "${raw}/log/${model}.ptxas"
  log="${raw}/log/${model}_fresh.txt"; : > "${log}"
  for unused in $(seq 1 "${runs}"); do
    "${raw}/bin/${model}" "${fixture}" >> "${log}"
  done
  pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
  [[ "${pass}" == "${runs}" ]] || { echo "FAIL: ${model} ${pass}/${runs}" >&2; exit 1; }
  variant="$(sed -n 's/^E2E_VARIANT index=\([0-9]*\).*/\1/p' "${log}" | sort -u | paste -sd, -)"
  l05="$(sed -n 's/^E2E_TIME l05_ms=\([0-9.]*\).*/\1/p' "${log}" | median)"
  l1="$(sed -n 's/.* l1_ms=\([0-9.]*\).*/\1/p' "${log}" | median)"
  l2="$(sed -n 's/.* l2_ms=\([0-9.]*\).*/\1/p' "${log}" | median)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}" "${pass}" "${runs}" "${variant}" "${l05}" "${l1}" "${l2}" \
    | tee -a "${raw}/summary.tsv"
done

# ---------------------------------------------------------------------------
# Rank transfer.  The arm above answers "does the shipped form run correctly
# there"; this one answers the question Part 4.4 was written for: do the
# sm_89 coefficients still rank configurations on a GPU with 170 SMs instead
# of 128, where wave quantization moves?  It re-measures an ORACLE subset (the
# 50 fastest plus 50 random, fixed seed) with the calibration target's
# coefficients left untouched, and scores both machines through the same model.
#
# Granularity is no longer a -D override, so each subset point is generated as
# its own single-interval runtime plan -- the same path the BF16 sweep used.
top="${TOP:-50}"; sample="${SAMPLE:-50}"; seed="${SEED:-20260904}"
jobs="${JOBS:-$(nproc)}"
label="${LABEL:-sm120}"
sweep="${repo}/docs/experiments/ORACLE/raw_bf16"
if [[ ! -s "${sweep}/screen_gqa2.tsv" ]]; then
  echo "SKIPPED rank transfer: ${sweep}/screen_gqa2.tsv is missing;" \
       "run docs/experiments/ORACLE/run.sh on the calibration target first" >&2
else
  cmake --build "${build}" --target tilemega-migrate --parallel "${jobs}" >/dev/null
  "${build}/tools/tilemega-migrate" --repo "${repo}" --out "${raw}" \
    --dtype bf16 --top "${top}" --sample "${sample}" --seed "${seed}" >/dev/null
  header='tile_m\ttile_n\ttile_k\tstages\tsplit_k\tl05_ms\tl1_ms\tl2_ms\tstatus\tl05_hash\tsmem\tctas_per_sm\tgrid\n'
  build_point() {  # model export_json m n k s split
    local tag="$1_$3x$4x$5s$6k$7"
    python3 "${repo}/docs/experiments/ORACLE/make_plan.py" \
      --out "${raw}/plan/${tag}.json" --tile-m "$3" --tile-n "$4" \
      --tile-k "$5" --stages "$6" --split-k "$7"
    if "${build}/tools/tilemega-compile" "$2" "${raw}/src/${tag}.cu" \
         --variants "${raw}/plan/${tag}.json" > "${raw}/log/${tag}.codegen" 2>&1 \
       && "${nvcc}" ${common_str} "${raw}/src/${tag}.cu" "${build}/libtilemega.a" \
         -L/usr/local/cuda/lib64 -lcudart -o "${raw}/bin/${tag}" \
         2> "${raw}/log/${tag}.ptxas"; then echo "OK ${tag}"; else echo "FAIL ${tag}"; fi
  }
  export -f build_point
  export repo raw build nvcc common_str="${common[*]}"
  mkdir -p "${raw}/plan"
  for model in gqa2 mha4; do
    fixture="${raw}/fixture/gqa2"
    [[ "${model}" == mha4 ]] && fixture="${raw}/export/mha4/fixture"
    subset="${raw}/subset_${model}.txt"
    echo "== ${model}: generating $(wc -l < "${subset}") migration points =="
    while read -r m n k s kc; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" \
        "${raw}/export/${model}.json" "${m}" "${n}" "${k}" "${s}" "${kc}"
    done < "${subset}" \
      | xargs -P "${jobs}" -n 7 bash -c \
          'build_point "$0" "$1" "$2" "$3" "$4" "$5" "$6"' \
      > "${raw}/compile_status_${label}_${model}.txt"
    {
      printf "${header}"
      while read -r m n k s kc; do
        tag="${model}_${m}x${n}x${k}s${s}k${kc}"; bin="${raw}/bin/${tag}"
        [[ -x "${bin}" ]] || continue
        l05=(); l1=(); l2=(); status=PASS; hash=; grid=; smem=; ctas=
        for ((i = 0; i < ${TRANSFER_RUNS:-3}; ++i)); do
          out="$(timeout 120 "${bin}" "${fixture}" 2>/dev/null)" || { status=RUNFAIL; break; }
          grep -q '^RESULT status=PASS' <<<"${out}" || status=MISMATCH
          l05+=("$(sed -n 's/^E2E_TIME l05_ms=\([0-9.]*\).*/\1/p' <<<"${out}")")
          l1+=("$(sed -n 's/.*l1_ms=\([0-9.]*\).*/\1/p' <<<"${out}")")
          l2+=("$(sed -n 's/.* l2_ms=\([0-9.]*\).*/\1/p' <<<"${out}")")
          hash="$(sed -n 's/^E2E_HASH l05=\([0-9a-f]*\).*/\1/p' <<<"${out}")"
          grid="$(sed -n 's/.*grid=\([0-9]*\).*/\1/p' <<<"${out}")"
          smem="$(sed -n 's/^E2E_RESOURCE.*smem=\([0-9]*\).*/\1/p' <<<"${out}")"
          ctas="$(sed -n 's/.*ctas_per_sm=\([0-9]*\).*/\1/p' <<<"${out}")"
        done
        if [[ "${status}" != PASS ]]; then
          printf '%s\t%s\t%s\t%s\t%s\tnan\tnan\tnan\t%s\t-\t-\t-\t-\n' \
            "${m}" "${n}" "${k}" "${s}" "${kc}" "${status}"; continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\n' \
          "${m}" "${n}" "${k}" "${s}" "${kc}" \
          "$(printf '%s\n' "${l05[@]}" | median)" \
          "$(printf '%s\n' "${l1[@]}" | median)" \
          "$(printf '%s\n' "${l2[@]}" | median)" \
          "${hash}" "${smem}" "${ctas}" "${grid}"
      done < "${subset}"
    } > "${raw}/screen_${label}_${model}.tsv"
    # Registers are a property of this compiler target, so they are read out of
    # this machine's own ptxas logs and never carried over from sm_89.
    {
      echo -e "# shape\tmax_registers  (max over entry points, ${label})"
      for f in "${raw}/log/${model}"_*.ptxas; do
        [[ -e "${f}" ]] || continue
        base=${f##*/}; base=${base%.ptxas}; base=${base#${model}_}
        regs=$( { grep -o 'Used [0-9]* registers' "${f}" || true; } |
                awk '{print $2}' | sort -rn | awk 'NR==1')
        [[ -n "${regs}" ]] && echo -e "${base%%k*}\t${regs}"
      done | sort -u -k1,1 -k2,2rn | awk '!seen[$1]++'
    } > "${raw}/registers_${label}_${model}.tsv"
  done
  echo "== scoring the unchanged sm_89 BF16 coefficients on both machines =="
  "${build}/tools/tilemega-migrate" --repo "${repo}" --out "${raw}" \
    --dtype bf16 --label "${label}" --top "${top}" --sample "${sample}" \
    --seed "${seed}"
fi
