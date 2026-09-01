#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
fixture="${here}/fixture"
mkdir -p "${raw}" "${fixture}"
export PYTHONPATH="${repo}/.venv/site${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${repo}/docs/experiments/V_H/raw/report.json" ]]; then
  echo "BLOCKED: V-H has not produced a real ExportedProgram" | tee "${raw}/run_status.txt"
  exit 77
fi

python3 "${here}/prepare_e2e.py" \
  --vh-raw "${repo}/docs/experiments/V_H/raw" --out "${fixture}" \
  | tee "${raw}/frontend.txt"

nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
"${nvcc}" -std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v \
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
  -I"${repo}/third_party/cutlass/tools/util/include" \
  -I"${repo}/third_party/cutlass/test" \
  "${here}/e2e.cu" "${repo}/build/libtilemega.a" \
  -L/usr/local/cuda/lib64 -lcudart -o "${here}/e2e" \
  2> "${raw}/ptxas.txt"

timeout 30 "${here}/e2e" "${fixture}" | tee "${raw}/first_run.txt"
"${repo}/scripts/gpu_stat_run.sh" -n 50 -t 30 -l e2e_l1 \
  -k -o "${raw}/fresh_processes" -- "${here}/e2e" "${fixture}" \
  | tee "${raw}/fresh_process_summary.txt"
grep '^E2E_TIME' "${raw}"/fresh_processes/run_*.log \
  | sed -E 's/.*l05_ms=([0-9.]+) l1_ms=([0-9.]+) ratio=([0-9.]+).*/\1\t\2\t\3/' \
  | sort -n -k3,3 > "${raw}/timings_sorted.tsv"
{
  cut -f1 "${raw}/timings_sorted.tsv" | sort -n \
    | awk 'NR == 25 {a=$1} NR == 26 {printf "l05_median_ms\t%.6f\n", (a+$1)/2}'
  cut -f2 "${raw}/timings_sorted.tsv" | sort -n \
    | awk 'NR == 25 {a=$1} NR == 26 {printf "l1_median_ms\t%.6f\n", (a+$1)/2}'
  cut -f3 "${raw}/timings_sorted.tsv" | sort -n \
    | awk 'NR == 25 {a=$1} NR == 26 {printf "l1_over_l05_median\t%.6f\n", (a+$1)/2}'
} > "${raw}/timing_median.txt"
grep -H -E '^(E2E_RESOURCE|E2E_TIME|E2E_DIFF|RESULT)' \
  "${raw}"/fresh_processes/run_*.log > "${raw}/fresh_process_raw.txt"
echo PASS > "${raw}/run_status.txt"
