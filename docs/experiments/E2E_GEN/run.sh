#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
fixture="${repo}/docs/experiments/E2E/fixture"
build="${repo}/build-phase12"
mkdir -p "${raw}" "${fixture}"
export PYTHONPATH="${repo}/.venv/site${PYTHONPATH:+:${PYTHONPATH}}"

mlir_dir="${MLIR_DIR:-${repo}/build-cutlass-compiler/lib/cmake/mlir}"
if [[ ! -f "${mlir_dir}/MLIRConfig.cmake" ]]; then
  echo "BLOCKED: MLIRConfig.cmake not found; set MLIR_DIR (see V_F)" \
    | tee "${raw}/run_status.txt"
  exit 77
fi

python3 "${repo}/python/tilemega/export_bridge.py" \
  "${repo}/docs/experiments/V_H/raw/exported_program.pt2" \
  --out "${raw}/export_bridge.json" | tee "${raw}/bridge_summary.txt"

cmake -S "${repo}" -B "${build}" -G Ninja \
  -DMLIR_DIR="${mlir_dir}" -DTILEMEGA_BUILD_VERIFY=OFF \
  > "${raw}/cmake.txt"
ninja -C "${build}" tilemega-opt tilemega-import tilemega-compile \
  > "${raw}/build.txt"
"${build}/tools/tilemega-import" "${raw}/export_bridge.json" \
  > "${raw}/cg.mlir" 2> "${raw}/import_summary.txt"
"${build}/tools/tilemega-opt" "${raw}/cg.mlir" \
  > "${raw}/cg_roundtrip.mlir"
"${build}/tools/tilemega-compile" "${raw}/export_bridge.json" \
  "${raw}/generated_e2e.cu" 2> "${raw}/codegen_summary.txt"

python3 "${repo}/docs/experiments/E2E/prepare_e2e.py" \
  --vh-raw "${repo}/docs/experiments/V_H/raw" --out "${fixture}" \
  > "${raw}/fixture.txt"

nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
common=(-std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test"
  "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
"${nvcc}" "${raw}/generated_e2e.cu" "${common[@]}" \
  -o "${here}/generated_e2e" 2> "${raw}/generated_ptxas.txt"
"${nvcc}" "${repo}/docs/experiments/E2E/e2e.cu" "${common[@]}" \
  -o "${here}/handwritten_e2e" 2> "${raw}/handwritten_ptxas.txt"

timeout 30 "${here}/generated_e2e" "${fixture}" | tee "${raw}/generated_first.txt"
timeout 30 "${here}/handwritten_e2e" "${fixture}" | tee "${raw}/handwritten_first.txt"
generated_hash="$(sed -n 's/^E2E_HASH l05=\([^ ]*\).*/\1/p' "${raw}/generated_first.txt")"
handwritten_hash="$(sed -n 's/^E2E_HASH l05=\([^ ]*\).*/\1/p' "${raw}/handwritten_first.txt")"
if [[ -z "${generated_hash}" || "${generated_hash}" != "${handwritten_hash}" ]]; then
  echo "generated/handwritten L0.5 hash mismatch" | tee "${raw}/hash_compare.txt"
  exit 1
fi
echo "generated_l05=${generated_hash} handwritten_l05=${handwritten_hash} bitwise=PASS" \
  | tee "${raw}/hash_compare.txt"

"${repo}/scripts/gpu_stat_run.sh" -n 50 -t 30 -l e2e_gen_l1 \
  -k -o "${raw}/fresh_processes" -- "${here}/generated_e2e" "${fixture}" \
  | tee "${raw}/fresh_process_summary.txt"
grep -H -E '^(E2E_RESOURCE|E2E_TIME|E2E_DIFF|E2E_HASH|RESULT)' \
  "${raw}"/fresh_processes/run_*.log > "${raw}/fresh_process_raw.txt"
grep '^E2E_TIME' "${raw}"/fresh_processes/run_*.log \
  | sed -E 's/.*l05_ms=([0-9.]+) l1_ms=([0-9.]+) ratio=([0-9.]+).*/\1\t\2\t\3/' \
  | sort -n -k3,3 > "${raw}/timings_sorted.tsv"
{
  cut -f1 "${raw}/timings_sorted.tsv" | sort -n | awk 'NR==25{a=$1} NR==26{printf "l05_median_ms\t%.6f\n",(a+$1)/2}'
  cut -f2 "${raw}/timings_sorted.tsv" | sort -n | awk 'NR==25{a=$1} NR==26{printf "l1_median_ms\t%.6f\n",(a+$1)/2}'
  cut -f3 "${raw}/timings_sorted.tsv" | sort -n | awk 'NR==25{a=$1} NR==26{printf "l1_over_l05_median\t%.6f\n",(a+$1)/2}'
} > "${raw}/timing_median.txt"
grep '^E2E_TIME' "${raw}"/fresh_processes/run_*.log \
  | sed -E 's/.*l2_ms=([0-9.]+) l2_over_l1=([0-9.]+).*/\1\t\2/' \
  | sort -n -k2,2 > "${raw}/l2_timings_sorted.tsv"
{
  cut -f1 "${raw}/l2_timings_sorted.tsv" | sort -n | awk 'NR==25{a=$1} NR==26{printf "l2_median_ms\t%.6f\n",(a+$1)/2}'
  cut -f2 "${raw}/l2_timings_sorted.tsv" | sort -n | awk 'NR==25{a=$1} NR==26{printf "l2_over_l1_median\t%.6f\n",(a+$1)/2}'
} > "${raw}/l2_timing_median.txt"
echo PASS > "${raw}/run_status.txt"
