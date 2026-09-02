#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# L2 acceptance (skeleton §7 P3.5 / Part 5.3): median timings for the L1
# global barrier vs. the L2 per-edge event graph, on BOTH accepted models,
# in fresh processes.
#
# This script does not regenerate anything.  It runs the binaries that
# `docs/experiments/E2E_GEN/run.sh` (2-layer GQA) and
# `docs/experiments/P3_GENERALIZATION/run.sh` (4-layer MHA) already built, so
# run those first.  E2E_GEN already does its own 50-fresh-process sweep and
# writes medians; this script reuses those and adds the equivalent sweep for
# the MHA model, which the P3_GENERALIZATION pipeline only ran once.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
runs="${RUNS:-25}"
mkdir -p "${raw}"

gqa_raw="${repo}/docs/experiments/E2E_GEN/raw"
mha_bin="${repo}/docs/experiments/P3_GENERALIZATION/generated"
mha_fixture="${repo}/docs/experiments/P3_GENERALIZATION/raw/fixture"
for needed in "${gqa_raw}/l2_timing_median.txt" "${mha_bin}" "${mha_fixture}"; do
  if [[ ! -e "${needed}" ]]; then
    echo "BLOCKED: ${needed} missing; run E2E_GEN/run.sh and P3_GENERALIZATION/run.sh first" >&2
    exit 77
  fi
done

# 2-layer GQA: copy the medians E2E_GEN's own 50-run sweep produced.
cp "${gqa_raw}/timing_median.txt"    "${raw}/gqa2_l1_timing_median.txt"
cp "${gqa_raw}/l2_timing_median.txt" "${raw}/gqa2_l2_timing_median.txt"
cp "${gqa_raw}/fresh_process_summary.txt" "${raw}/gqa2_fresh_process_summary.txt"
cp "${gqa_raw}/fresh_process_raw.txt" "${raw}/gqa2_fresh_process_raw.txt"

# 4-layer MHA: its pipeline only runs once, so do the sweep here.
"${repo}/scripts/gpu_stat_run.sh" -n "${runs}" -t 60 -l e2e_l2_mha4 \
  -k -o "${raw}/mha4_fresh_processes" -- "${mha_bin}" "${mha_fixture}" \
  | tee "${raw}/mha4_fresh_process_summary.txt"
grep -H -E '^(E2E_RESOURCE|E2E_TIME|E2E_DIFF|E2E_HASH|E2E_ITER|RESULT)' \
  "${raw}"/mha4_fresh_processes/run_*.log > "${raw}/mha4_fresh_process_raw.txt"

median() {  # median of stdin's numbers, works for odd and even n
  sort -n | awk '{v[NR]=$1} END {
    if (NR == 0) exit 1
    printf "%s\t%.6f\n", label, (NR % 2) ? v[(NR+1)/2] : (v[NR/2] + v[NR/2+1]) / 2
  }' label="$1"
}
timings="${raw}/mha4_timings.tsv"
grep -h '^E2E_TIME' "${raw}"/mha4_fresh_processes/run_*.log \
  | sed -E 's/.*l05_ms=([0-9.]+) l1_ms=([0-9.]+) ratio=([0-9.]+) l2_ms=([0-9.]+) l2_over_l1=([0-9.]+).*/\1\t\2\t\3\t\4\t\5/' \
  > "${timings}"
{
  cut -f1 "${timings}" | median l05_median_ms
  cut -f2 "${timings}" | median l1_median_ms
  cut -f3 "${timings}" | median l1_over_l05_median
  cut -f4 "${timings}" | median l2_median_ms
  cut -f5 "${timings}" | median l2_over_l1_median
  printf "samples\t%d\n" "$(wc -l < "${timings}")"
} > "${raw}/mha4_timing_median.txt"
cat "${raw}/mha4_timing_median.txt"

# Part 5.2: is any derived edge a `kIdentity` candidate at all?  Built the
# same way P3_ISL builds its probes (see docs/DEPENDENCIES.md for the isl /
# barvinok trees).
isl_build="${TILEMEGA_ISL_BUILD_DIR:-${repo}/build-isl}"
polylib_build="${TILEMEGA_POLYLIB_BUILD_DIR:-${repo}/build-polylib}"
barvinok_build="${TILEMEGA_BARVINOK_BUILD_DIR:-${repo}/build-barvinok}"
mlir_src="${TILEMEGA_MLIR_SRC:-/tmp/tilemega-vf-llvm-project}"
mlir_build="${TILEMEGA_MLIR_BUILD:-/tmp/tilemega-vf-build/llvm-project}"
if [[ -d "${barvinok_build}" && -d "${mlir_build}/lib" ]]; then
  g++ -std=c++17 -I"${repo}/include" -I"${repo}/build-portable/include" \
    -I"${mlir_src}/llvm/include" -I"${mlir_build}/include" \
    -I"${repo}/third_party/barvinok/isl/include" -I"${isl_build}/include" \
    "${here}/identity_probe.cpp" \
    -L"${repo}/build-portable" -ltilemega \
    "${barvinok_build}/.libs/libbarvinok.a" "${isl_build}/.libs/libisl.a" \
    "${polylib_build}/.libs/libpolylibgmp.a" \
    -L"${mlir_build}/lib" -lLLVMSupport -lLLVMDemangle \
    -L/usr/local/cuda/lib64 -lcudart -lntl -lgmp -lpthread \
    -o "${raw}/identity_probe" 2>&1 | tee "${raw}/identity_build.txt"
  "${raw}/identity_probe" > "${raw}/identity.txt" 2>&1
  grep '^SUMMARY' "${raw}/identity.txt"
else
  echo "SKIPPED: isl/barvinok or MLIR build tree missing; identity probe not run" \
    | tee "${raw}/identity.txt"
fi
