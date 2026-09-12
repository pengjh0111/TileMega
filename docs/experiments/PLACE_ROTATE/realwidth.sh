#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# §4.5 real-width arm: the same headroom analysis on a production-shaped graph,
# because a two-layer reference model can hide both the parallelism and the
# imbalance a real one has.  The export and codegen are driven through
# REALMODEL/run.sh unchanged; only the trace v2 rebuild is done here.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw/realwidth"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
# build-phase12 still holds a tilemega-compile from before the schedule table
# was added to RuntimeVariantDesc; its output no longer compiles.  The
# current-tree build is the only generator whose emission matches this
# header, so it is the default here.
build="${BUILD_DIR:-${repo}/build-portable}"
tick="$(awk -F'\t' '$1=="globaltimer_resolution_ns"{print $2}' \
  "${repo}/docs/experiments/TRACE_V2/resolution.tsv")"
mkdir -p "${raw}"
export TILEMEGA_COMMIT="$(git -C "${repo}" rev-parse HEAD)"
export TILEMEGA_GLOBALTIMER_NS="${tick}"

for seq in 4 128; do
  label="r1w_s${seq}"
  work="${repo}/docs/experiments/REALMODEL/raw/work/${label}"
  if ! LAYERS=4 HIDDEN=4096 INTERMEDIATE=14336 HEADS=32 KV_HEADS=8 \
       FIXTURE_SEQ="${seq}" FIXTURE_PAST=3 LABEL="${label}" RUNS=1 \
       BUILD_DIR="${build}" \
       bash "${repo}/docs/experiments/REALMODEL/run.sh" \
       > "${raw}/build_s${seq}.log" 2>&1; then
    # A failure is recorded as a failure; reference-model numbers are not a
    # substitute for the real-width cell.
    {
      printf 'seq\t%s\nstatus\tFAIL\n' "${seq}"
      printf 'command\tLAYERS=4 HIDDEN=4096 INTERMEDIATE=14336 HEADS=32 KV_HEADS=8 FIXTURE_SEQ=%s FIXTURE_PAST=3 LABEL=%s RUNS=1 bash docs/experiments/REALMODEL/run.sh\n' \
        "${seq}" "${label}"
      # REALMODEL/run.sh keeps its compiler output in the work directory, so the
      # driver log alone would record a torch warning as the cause.
      printf 'tail\t%s\n' "$(tail -3 "${raw}/build_s${seq}.log" | tr '\n' ' ')"
      for extra in ptxas.log codegen.log; do
        [[ -f "${work}/${extra}" ]] && printf '%s\t%s\n' "${extra}" \
          "$(tail -5 "${work}/${extra}" | tr '\n' ' ')"
      done
    } > "${raw}/status_s${seq}.tsv"
    echo "REALWIDTH seq=${seq} status=FAIL log=${raw}/build_s${seq}.log"
    continue
  fi
  "${nvcc}" "${work}/model.cu" -std=c++17 -O2 -arch=native -lineinfo \
    -DTILEMEGA_TRACE_V2=1 \
    -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
    -I"${repo}/third_party/cutlass/tools/util/include" \
    -I"${repo}/third_party/cutlass/test" \
    "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart \
    -o "${raw}/model_s${seq}" 2> "${raw}/ptxas_s${seq}.log"
  out="${raw}/dump/real_s${seq}"
  rm -rf "${out}"
  TILEMEGA_MODEL_NAME="real_l4_h4096" TILEMEGA_TRACE_V2=1 TILEMEGA_TRACE_V2_OUT="${out}" \
    "${raw}/model_s${seq}" "${work}/export/fixture" > "${raw}/run_s${seq}.log" 2>&1
  grep -q '^RESULT status=PASS' "${raw}/run_s${seq}.log" || {
    printf 'seq\t%s\nstatus\tFAIL_RUN\n' "${seq}" > "${raw}/status_s${seq}.tsv"
    echo "REALWIDTH seq=${seq} status=FAIL_RUN log=${raw}/run_s${seq}.log"
    continue
  }
  printf 'seq\t%s\nstatus\tPASS\n' "${seq}" > "${raw}/status_s${seq}.tsv"
  grep -h '^E2E_TRACE_V2' "${raw}/run_s${seq}.log"
done

dumps=("${raw}/dump"/real_s*)
if [[ -d "${dumps[0]}" ]]; then
  python3 "${here}/headroom.py" "${dumps[@]}" --out "${here}" --label "real width l4 h4096"
fi
