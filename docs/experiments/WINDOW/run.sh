#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-E2 gates E2-a, E2-b, E2-c and E2-d.
#
# W reaches the executor from the Plan and nothing else (§8.11).  The reference
# sources carry no plan brace at all -- their RuntimePlanDesc is default
# constructed, which is where today's W = 1 comes from -- so `plan_window.py`
# appends one whose only non-default field is the window.  Its mode stays 0,
# the legacy grid-stride placement the source already uses, so W is the single
# difference between the arms.  A W = 1 stamp is inert: its SASS is byte
# identical to the unstamped source (`sass_identity/`, checked by this script).
#
# Phases, via PHASES=build,identity,refusal,seqscan,paired,negative,trace:
#   identity   the W = 1 stamp against the unstamped source, SASS and bytes
#   refusal    a W = 4 plan under a W = 2 build must exit 2 rather than run
#              under lifting rules it was not solved for (H4)
#   seqscan    E2-a: the full 30-cell matrix per W, 50 processes a cell
#   paired     E2-b: W = 1 against today, 25 paired rounds, rotated (H6)
#   negative   E2-c: the W = 1 host rules under a W > 1 executor must FAIL
#   trace      E2-d: trace v2 per W, for head-of-line reclamation
#   analyze    TRACE_V2/analyze.py over those dumps, for HOL and the ceiling
#   summary    the four gates, computed from the raw logs above
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${RAW_DIR:-${here}/raw}"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cuobjdump="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"
seqscan="${repo}/docs/experiments/SEQSCAN/raw"
runs="${RUNS:-25}"
correctness_runs="${CORRECTNESS_RUNS:-50}"
phases="${PHASES:-build,identity,refusal,seqscan,paired,negative,trace,analyze,summary}"
windows="${WINDOWS:-1 2 4}"
has() { [[ ",${phases}," == *",$1,"* ]]; }

mkdir -p "${raw}/bin" "${raw}/log" "${raw}/src" "${raw}/final" "${raw}/sass_identity"
# Written only at the very end: a leftover PASS from a partial invocation reads
# as a pass with no data behind it, which happened once in SYNC_V2.
rm -f "${raw}/status.txt"
common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

if has build; then
  for model in gqa2 mha4; do
    for w in ${windows}; do
      python3 "${here}/plan_window.py" "${seqscan}/src/${model}.cu" "${w}" \
        "${raw}/src/${model}_w${w}.cu" > "${raw}/log/stamp_${model}_w${w}.txt"
      "${nvcc}" "${common[@]}" "-DTILEMEGA_SLOT_WINDOW=${w}" \
        "${raw}/src/${model}_w${w}.cu" "${link[@]}" \
        -o "${raw}/bin/${model}_w${w}" 2> "${raw}/log/${model}_w${w}.ptxas"
      "${nvcc}" "${common[@]}" "-DTILEMEGA_SLOT_WINDOW=${w}" \
        -DTILEMEGA_TRACE_V2=1 "${raw}/src/${model}_w${w}.cu" "${link[@]}" \
        -o "${raw}/bin/${model}_w${w}_trace" \
        2> "${raw}/log/${model}_w${w}_trace.ptxas"
    done
    # `today`: the unstamped source in a build that knows nothing of the window,
    # which is E2-b's reference and not merely W = 1 under the new executor.
    "${nvcc}" "${common[@]}" "${seqscan}/src/${model}.cu" "${link[@]}" \
      -o "${raw}/bin/${model}_today" 2> "${raw}/log/${model}_today.ptxas"
    # E2-c: the W = 1 host rules against a W = 2 executor.  Must fail.
    "${nvcc}" "${common[@]}" -DTILEMEGA_SLOT_WINDOW=2 \
      -DTILEMEGA_NEGATIVE_WINDOW_W1_RULES=1 "${raw}/src/${model}_w2.cu" \
      "${link[@]}" -o "${raw}/bin/${model}_negative" \
      2> "${raw}/log/${model}_negative.ptxas"
  done
  # The refusal probe: a plan asking for more window than the build implements.
  "${nvcc}" "${common[@]}" -DTILEMEGA_SLOT_WINDOW=2 "${raw}/src/gqa2_w4.cu" \
    "${link[@]}" -o "${raw}/bin/gqa2_w4_under_w2" \
    2> "${raw}/log/gqa2_w4_under_w2.ptxas"
fi

# --- identity: the W = 1 stamp changes no instruction
if has identity; then
  printf 'model\tsource_bytes_differ\tsass_identical\n' > "${raw}/identity.tsv"
  for model in gqa2 mha4; do
    "${nvcc}" "${common[@]}" -cubin "${seqscan}/src/${model}.cu" \
      -o "${raw}/sass_identity/${model}_today.cubin"
    "${nvcc}" "${common[@]}" -DTILEMEGA_SLOT_WINDOW=1 -cubin \
      "${raw}/src/${model}_w1.cu" -o "${raw}/sass_identity/${model}_w1.cubin"
    for tag in today w1; do
      "${cuobjdump}" --dump-sass "${raw}/sass_identity/${model}_${tag}.cubin" \
        | grep -v '^\s*$' > "${raw}/sass_identity/${model}_${tag}.sass"
    done
    if diff -q "${raw}/sass_identity/${model}_today.sass" \
        "${raw}/sass_identity/${model}_w1.sass" > /dev/null; then same=1; else same=0; fi
    printf '%s\t1\t%s\n' "${model}" "${same}" >> "${raw}/identity.tsv"
    [[ "${same}" == 1 ]] || {
      echo "FAIL: ${model} W=1 stamp moved an instruction" >&2; exit 1; }
  done
  cat "${raw}/identity.tsv"
fi

# --- refusal: W from the Plan, bounded by what the build implements
if has refusal; then
  set +e
  "${raw}/bin/gqa2_w4_under_w2" "${seqscan}/fixture/gqa2_s4_p3" \
    > "${raw}/log/refusal.txt" 2>&1
  code=$?
  set -e
  echo "REFUSAL exit=${code} expect=2"
  grep -h 'executor implements' "${raw}/log/refusal.txt" || true
  [[ "${code}" == 2 ]] || {
    echo 'a plan wider than the build implements was accepted' >&2; exit 1; }
fi

# --- E2-a: the full SEQSCAN matrix per W
if has seqscan; then
  printf 'w\tmodel\tseq\tpast\tpasses\tprocesses\n' > "${raw}/matrix.tsv"
  for w in ${windows}; do
    for model in gqa2 mha4; do
      for seq in 1 4 128 512 2048; do
        for past in 0 3 512; do
          log="${raw}/log/matrix_w${w}_${model}_s${seq}_p${past}.txt"; : > "${log}"
          for ((i=0; i<correctness_runs; ++i)); do
            timeout 120s "${raw}/bin/${model}_w${w}" \
              "${seqscan}/fixture/${model}_s${seq}_p${past}" >> "${log}" || true
          done
          pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${w}" "${model}" "${seq}" "${past}" \
            "${pass}" "${correctness_runs}" | tee -a "${raw}/matrix.tsv"
        done
      done
    done
  done
  # Asserted on the finished table so every row survives for diagnosis.
  short="$(awk -F'\t' 'NR>1 && $5 != $6' "${raw}/matrix.tsv")"
  [[ -z "${short}" ]] || {
    printf 'E2-a cells short of full pass:\n%s\n' "${short}" >&2; exit 1; }
fi

# --- E2-b: W = 1 against today, paired and rotated
if has paired; then
  pair=(today w1)
  for model in gqa2 mha4; do
    for seq in 4 128; do
      fixture="${seqscan}/fixture/${model}_s${seq}_p3"
      for ((round=0; round<runs; ++round)); do
        for ((slot=0; slot<${#pair[@]}; ++slot)); do
          arm="${pair[$(((round + slot) % ${#pair[@]}))]}"
          "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
            -l "window_${model}_${seq}_${arm}" -k \
            -o "${raw}/final/${model}_s${seq}_${arm}/r${round}" -- \
            "${raw}/bin/${model}_${arm}" "${fixture}" \
            >> "${raw}/log/${model}_s${seq}_${arm}.runner" 2>&1
        done
      done
    done
  done
fi

# --- E2-c: the H4 negative control.  A pass here is a stop condition (§10),
# and never evidence that the rule is unnecessary.
if has negative; then
  printf 'model\tseq\tpasses\tprocesses\n' > "${raw}/negative.tsv"
  for model in gqa2 mha4; do
    for seq in 4 128; do
      log="${raw}/log/negative_${model}_s${seq}.txt"; : > "${log}"
      for ((i=0; i<correctness_runs; ++i)); do
        timeout 120s "${raw}/bin/${model}_negative" \
          "${seqscan}/fixture/${model}_s${seq}_p3" >> "${log}" 2>&1 || true
      done
      pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
      printf '%s\t%s\t%s\t%s\n' "${model}" "${seq}" "${pass}" \
        "${correctness_runs}" | tee -a "${raw}/negative.tsv"
    done
  done
  clean="$(awk -F'\t' 'NR>1 && $3 == $4' "${raw}/negative.tsv")"
  [[ -z "${clean}" ]] || {
    printf 'E2-c did NOT fail in:\n%s\nH4 is unverified; this is a stop condition, not a licence to drop the rule.\n' \
      "${clean}" >&2; exit 1; }
fi

# --- E2-d: trace v2 per W, for head-of-line reclamation
if has trace; then
  for w in ${windows}; do
    for model in gqa2 mha4; do
      for seq in 4 128; do
        out="${raw}/final/trace_w${w}_${model}_s${seq}"
        mkdir -p "${out}"
        env TILEMEGA_MODEL_NAME="${model}" TILEMEGA_TRACE_V2=1 \
          TILEMEGA_TRACE_V2_OUT="${out}" timeout 300s \
          "${raw}/bin/${model}_w${w}_trace" \
          "${seqscan}/fixture/${model}_s${seq}_p3" \
          > "${raw}/log/trace_w${w}_${model}_s${seq}.txt" 2>&1 || true
      done
    done
  done
fi

# --- E2-d's numbers come from TRACE_V2/analyze.py, called on the W dumps
# rather than recomputed here: `hol_reclaimable_ns` is the quantity F-134
# measured and a second definition of it would not be comparable (H7).  The
# same table carries `cp_lb_nosync_ms`, which is this configuration's own
# ceiling (H8).
if has analyze; then
  dumps=()
  for w in ${windows}; do
    for model in gqa2 mha4; do
      for seq in 4 128; do
        d="${raw}/final/trace_w${w}_${model}_s${seq}"
        [[ -f "${d}/slots.tsv" ]] && dumps+=("${d}")
      done
    done
  done
  ((${#dumps[@]})) || { echo 'no trace dumps to analyze' >&2; exit 1; }
  python3 "${repo}/docs/experiments/TRACE_V2/analyze.py" "${dumps[@]}" \
    --out "${raw}/analysis"
  cut -f1 "${raw}/analysis/analysis.tsv" | head -5
fi

if has summary; then
  python3 "${here}/summarize.py" "${raw}" > "${raw}/summary.tsv"
  cat "${raw}/summary.tsv"
fi
echo "PASS phases=${phases}" > "${raw}/status.txt"
