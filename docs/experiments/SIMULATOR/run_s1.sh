#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S1 gates S1-a (trace agreement), S1-b (ranking), S1-c (speed), sm_89.
#
# Deliberately a separate script from run.sh: run.sh produces the *calibration*
# set (the contention microbenchmark that fixes hop_ns), this one produces the
# *evaluation* set.  H8 requires the two to be disjoint and named, and two
# scripts make that a property of the layout rather than a claim in a README.
#
# Each (model, seq) cell is measured three ways -- placement 0, 4 and 5 -- and
# simulated with every candidate the current tree can materialize.  Modes 0 and
# 5 are both in the set, as S1-b demands, because they are the two the round-one
# result actually separated (l2/l1 0.807 against 1.0).
#
# Timing and tracing are separate builds.  Trace-on perturbs the number it is
# measuring (D1-c), so per-task start times come from the traced runs and the
# l2_ms used for the ranking comes from trace-off runs, paired in one session
# with a rotating arm order per H6.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
rounds="${ROUNDS:-15}"
gen="${repo}/docs/experiments/PLAN_CONTRACT/legacy_identity/plan"

if env | grep -q '^TILEMEGA_'; then
  echo "refusing to run with TILEMEGA_* inherited from the environment:" >&2
  env | grep '^TILEMEGA_' >&2
  exit 2
fi

tick="$(awk -F'\t' '$1=="globaltimer_resolution_ns"{print $2}' \
  "${repo}/docs/experiments/TRACE_V2/resolution.tsv")"
# Not `out`: .gitignore excludes docs/experiments/**/out/, and these per-cell
# stdouts are the evidence S1-a and verify.py re-read.
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/dump" "${raw}/run" "${raw}/time"

cmake --build "${build}" --target tilemega-simulate -j "$(nproc)" \
  > "${raw}/log/build.txt" 2>&1

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

export TILEMEGA_GLOBALTIMER_NS="${tick}"

# SKIP_MEASURE=1 re-runs only the simulate and report tail against the dumps
# already on disk.  It exists because the tail is the part that changes while
# the measurements do not; it must never be used to present a result as
# freshly measured (H6).
if [ "${SKIP_MEASURE:-0}" != 1 ]; then
# Mode 4 needs the variant to request the balanced writeback, so it compiles a
# different generated source; 0 and 5 are the macro alone.
for model in gqa2 mha4; do
  for place in 0 4 5; do
    src="${gen}/${model}.cu"
    [ "${place}" = 4 ] && src="${gen}/${model}_balanced.cu"
    for arm in trace time; do
      flags=()
      [ "${arm}" = trace ] && flags=(-DTILEMEGA_TRACE_V2=1)
      "${nvcc}" "${common[@]}" "${flags[@]}" -DTILEMEGA_PLACEMENT="${place}" \
        "${src}" "${link[@]}" -o "${raw}/bin/${arm}_${model}_p${place}" \
        2>> "${raw}/log/nvcc.log"
    done
  done
done

# --- traced runs: the per-task start times S1-a compares against ----------
for model in gqa2 mha4; do
  for place in 0 4 5; do
    for seq in 4 128 512; do
      out="${raw}/dump/${model}_s${seq}_p${place}"
      rm -rf "${out}"
      env TILEMEGA_MODEL_NAME="${model}" TILEMEGA_TRACE_V2=1 \
        TILEMEGA_TRACE_V2_OUT="${out}" TILEMEGA_PLACEMENT_BASE_DUMP=1 \
        "${raw}/bin/trace_${model}_p${place}" \
        "${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3" \
        > "${raw}/run/${model}_p${place}_s${seq}.out" 2>&1
      grep -q '^RESULT status=PASS' "${raw}/run/${model}_p${place}_s${seq}.out" || {
        echo "trace ${model} s${seq} p${place}: not PASS" >&2; exit 1; }
    done
  done
done

# --- untraced timing, paired in one session with a rotating arm order -----
places=(0 4 5)
: > "${raw}/time/l2.tsv"
printf 'model\tseq\tplace\tround\tl2_ms\tl1_ms\n' >> "${raw}/time/l2.tsv"
for model in gqa2 mha4; do
  for seq in 4 128 512; do
    fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
    for ((round=0; round<rounds; ++round)); do
      for ((slot=0; slot<${#places[@]}; ++slot)); do
        place="${places[$(((round + slot) % ${#places[@]}))]}"
        line="$(env TILEMEGA_MODEL_NAME="${model}" \
          "${raw}/bin/time_${model}_p${place}" "${fixture}" 2>&1 \
          | grep '^E2E_TIME' || true)"
        [ -n "${line}" ] || { echo "no E2E_TIME ${model} s${seq} p${place}" >&2; exit 1; }
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" "${place}" "${round}" \
          "$(sed 's/.*l2_ms=\([0-9.]*\).*/\1/' <<<"${line}")" \
          "$(sed 's/.*l1_ms=\([0-9.]*\).*/\1/' <<<"${line}")" >> "${raw}/time/l2.tsv"
      done
    done
  done
done

# --- real width, for the S1-c budget that is stated separately -----------
# S1-c gives reference models < 1 ms and real width < 10 ms, so a run that only
# ever evaluates a two-layer model has not checked half the gate.  Codegen and
# export go through REALMODEL/run.sh unchanged, as PLACE_ROTATE/realwidth.sh
# does; only modes 0 and 5 are built, which is the pair S1-b requires and
# enough for the provenance check.  Mode 4 needs its own generated source and
# is not what this arm is for.
for seq in 4 128; do
  label="r2sim_s${seq}"
  work="${repo}/docs/experiments/REALMODEL/raw/work/${label}"
  if ! env LAYERS=4 HIDDEN=4096 INTERMEDIATE=14336 HEADS=32 KV_HEADS=8 \
         FIXTURE_SEQ="${seq}" FIXTURE_PAST=3 LABEL="${label}" RUNS=1 \
         BUILD_DIR="${build}" \
         bash "${repo}/docs/experiments/REALMODEL/run.sh" \
         > "${raw}/log/realwidth_build_s${seq}.log" 2>&1; then
    printf 'seq\t%s\nstatus\tFAIL_BUILD\ntail\t%s\n' "${seq}" \
      "$(tail -3 "${raw}/log/realwidth_build_s${seq}.log" | tr '\n' ' ')" \
      > "${raw}/run/realwidth_s${seq}.status"
    echo "REALWIDTH seq=${seq} status=FAIL_BUILD" >&2
    continue
  fi
  ok=1
  for place in 0 5; do
    "${nvcc}" "${common[@]}" -DTILEMEGA_PLACEMENT="${place}" \
      "${work}/model.cu" "${link[@]}" -o "${raw}/bin/time_real_p${place}_s${seq}" \
      2>> "${raw}/log/nvcc_realwidth.log" || { ok=0; break; }
    env TILEMEGA_MODEL_NAME="real_l4_h4096" TILEMEGA_PLACEMENT_BASE_DUMP=1 \
      "${raw}/bin/time_real_p${place}_s${seq}" "${work}/export/fixture" \
      > "${raw}/run/real_p${place}_s${seq}.out" 2>&1 || { ok=0; break; }
    grep -q '^RESULT status=PASS' "${raw}/run/real_p${place}_s${seq}.out" || { ok=0; break; }
  done
  if [ "${ok}" != 1 ]; then
    printf 'seq\t%s\nstatus\tFAIL_RUN\n' "${seq}" > "${raw}/run/realwidth_s${seq}.status"
    echo "REALWIDTH seq=${seq} status=FAIL_RUN" >&2
    continue
  fi
  printf 'seq\t%s\nstatus\tPASS\nexport\t%s\n' "${seq}" "${work}/model.json" \
    > "${raw}/run/realwidth_s${seq}.status"
done
fi

# --- simulate every cell --------------------------------------------------
manifest="${raw}/manifest.tsv"
printf 'model\tseq\tpast\tgenerated_cu\tout_prefix\ttrace_dir\n' > "${manifest}"
for model in gqa2 mha4; do
  for seq in 4 128 512; do
    printf '%s\t%s\t3\t%s\t%s\t%s\n' "${model}" "${seq}" \
      "${gen}/${model}.cu" "${raw}/run/${model}" \
      "${raw}/dump/${model}_s${seq}_p5" >> "${manifest}"
  done
done
for seq in 4 128; do
  status="${raw}/run/realwidth_s${seq}.status"
  [ -f "${status}" ] && [ "$(awk -F'\t' '$1=="status"{print $2}' "${status}")" = PASS ] || continue
  work="${repo}/docs/experiments/REALMODEL/raw/work/r2sim_s${seq}"
  # No trace dir: the real-width arm exists for the S1-c budget, and mapping
  # worker -> SM as w % sms costs only the co-residency grouping, not the queue.
  printf 'real\t%s\t3\t%s\t%s\t-\t%s\n' "${seq}" "${work}/model.cu" \
    "${raw}/run/real" "${work}/model.json" >> "${manifest}"
done
"${build}/tools/tilemega-simulate" "${repo}" "${manifest}" "${raw}" \
  2>&1 | tee "${raw}/log/simulate.txt"

python3 "${here}/s1_report.py" "${raw}" --out "${here}" | tee "${here}/s1_report.txt"
cd "${repo}"
sha256sum docs/experiments/SIMULATOR/raw/predicted.tsv \
  docs/experiments/SIMULATOR/{s1_start_error.tsv,s1_ranking.tsv} \
  >> "${here}/sha256.txt"
echo PASS > "${raw}/status.txt"
