#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# EX-S2r: every placement candidate remeasured against the ceiling, under each
# of this round's mechanism configurations (R3 §7.2, §7.3).
#
# Two axes, and they are not the same kind of thing.  A *configuration* is a set
# of compile flags; a *candidate* is a Plan.  §7.2 names configuration C
# "protocol + chaining", but chaining is a candidate rather than a flag, so C is
# the (B-flags, chain-candidate) cell of this table and needs no binary of its
# own -- compiling one would produce a file byte-identical to B's chain arm.
# §8's "chaining only" row is likewise the (A-flags, chain) cell, and its
# "window only" row is configuration `w`.
#
#   a  no flags at all.  The headers default to the wait the generator has
#      always emitted, so this is the round's baseline and not a fourth variant.
#   b  the calibrated wait policy plus E3 steps 1-3, W = 1.
#   d  b plus the window, at the W that WINDOW's E2-d picked.
#   w  the window alone, for the additivity row.
#
# Phases, via PHASES=generate,provenance,stamp,build,correctness,paired,trace,analyze:
#   generate     both placement tools into one plan directory
#   provenance   each tool's legacy source against the committed control, and
#                the two tools against each other
#   stamp        W into the Plan, per candidate
#   build        configuration x candidate x cell
#   correctness  50 fresh processes per cell x candidate (H6)
#   paired       25 rounds, configuration and candidate rotated together
#   trace        trace v2 per configuration, for its own ceiling (H8)
#   analyze      TRACE_V2/analyze.py over those dumps
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${RAW_DIR:-${here}/raw}"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
seqscan="${repo}/docs/experiments/SEQSCAN/raw"
sim="${repo}/docs/experiments/SIMULATOR/raw"
control="${repo}/docs/experiments/PLAN_CONTRACT/legacy_identity/plan"
runs="${RUNS:-25}"
correctness_runs="${CORRECTNESS_RUNS:-50}"
phases="${PHASES:-generate,provenance,stamp,build,correctness,paired,trace,analyze}"
configs="${CONFIGS:-a b d w}"
# W is not chosen here: it is the window E2-d measured as best (§7.2 "best
# W in {2,4}").  Passed in so this script never disagrees with WINDOW's result.
W="${WINDOW:-2}"
correctness_configs="${CORRECTNESS_CONFIGS:-b d}"
# The arms whose ceiling is recomputed: rotate is round two's best candidate and
# chain is this round's, so the pair brackets the comparison S2r-d reports.
trace_arms="${TRACE_ARMS:-rotate chain}"
realwidth="${REALWIDTH:-1}"
has() { [[ ",${phases}," == *",$1,"* ]]; }

mkdir -p "${raw}/bin" "${raw}/log" "${raw}/plan" "${raw}/src" "${raw}/final"
# Written only at the very end: a leftover PASS from a partial invocation reads
# as a pass with no data behind it (SYNC_V2 lost a seqscan that way).
rm -f "${raw}/status.txt"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

# The policy flags are read from the target rather than spelled out, so this
# script carries no calibrated constant of its own (H5).  The tag is a variable
# because the two architectures calibrate to different policies -- sm_89 wants
# `spin_iters=64`, sm_120 wants 0 against its ~400 ns hop floor -- so compiling
# a Blackwell part with this default would carry the 4090's constant onto
# another machine, which is what H9 forbids.
arch_tag="${ARCH_TAG:-sm_89}"
policy_flags=""
if [[ "${configs}" == *b* || "${configs}" == *d* ]]; then
  policy_flags="$("${build}/tools/tilemega-wait-policy" "${arch_tag}" bf16 "${repo}" \
    2> "${raw}/log/wait_policy.log")"
  echo "WAIT_POLICY ${policy_flags}"
fi
protocol="${policy_flags} -DTILEMEGA_BARRIER_V2=1 -DTILEMEGA_EVENT_SOLO=1"
protocol="${protocol} -DTILEMEGA_EVENT_RED_PUBLISH=1"

config_flags() {
  case "$1" in
    a) echo '' ;;
    b) echo "${protocol}" ;;
    d) echo "${protocol} -DTILEMEGA_SLOT_WINDOW=${W}" ;;
    w) echo "-DTILEMEGA_SLOT_WINDOW=${W}" ;;
  esac
}

cells() { echo "gqa2:4 gqa2:128 mha4:4 mha4:128"; }
real_cells() { [[ "${realwidth}" == 1 ]] && echo "real:4 real:128" || true; }

# `band` is not an arm this round: it was byte-identical to legacy_grid_stride
# in all six round-two cells (§7.2), so measuring it again would only re-time
# the same binary.  `balanced` needs its own variant plan and the real-width
# model has none, which is why the real cells carry five arms and not six.
arms_for() {
  case "$1" in
    real) echo "legacy_grid_stride rotate eft wavefront chain" ;;
    *) echo "legacy_grid_stride balanced rotate eft wavefront chain" ;;
  esac
}

work_for() { echo "${repo}/docs/experiments/REALMODEL/raw/work/r2sim_s$1"; }

fixture_for() {
  if [[ "$1" == real ]]; then echo "$(work_for "$2")/export/fixture"
  else echo "${seqscan}/fixture/$1_s$2_p3"; fi
}

arm_macro() {
  case "$1" in legacy_grid_stride) echo 0 ;; balanced) echo 4 ;; rotate) echo 5 ;;
    *) echo 0 ;; esac
}

# legacy and rotate ride the committed control source and differ only in the
# placement macro; everything else travels as an emitted Plan.
base_source() {
  local arm="$1" model="$2" seq="$3"
  case "${arm}" in
    legacy_grid_stride|rotate)
      if [[ "${model}" == real ]]; then echo "$(work_for "${seq}")/model.cu"
      else echo "${control}/${model}.cu"; fi ;;
    balanced) echo "${control}/${model}_balanced.cu" ;;
    *) echo "${raw}/plan/${model}_s${seq}_${arm}.cu" ;;
  esac
}

# Which sources already carry a plan brace, and therefore need the window
# rewritten in place rather than appended.  `balanced` is on this list: it
# looks like a control but emits `{2u, nullptr, 0u, 1u, 0u}`.
plan_carrying() {
  case "$1" in balanced|eft|wavefront|chain) return 0 ;; *) return 1 ;; esac
}

source_for() {
  local config="$1" arm="$2" model="$3" seq="$4"
  case "${config}" in
    a|b) base_source "${arm}" "${model}" "${seq}" ;;
    *) echo "${raw}/src/${model}_s${seq}_${arm}_w${W}.cu" ;;
  esac
}

# --- generate: both tools into one plan directory
#
# `tilemega-place-eft` emits legacy_grid_stride, band, eft and wavefront;
# `tilemega-place-chain` emits legacy_grid_stride, band, wavefront and chain.
# The three they share are byte-identical between the two, which is asserted
# into `tool_agreement.tsv` below before either directory is used rather than
# assumed, so one plan directory needs no collision handling.
if has generate && [[ "${SKIP_GENERATE:-0}" != 1 ]]; then
  {
    printf 'model\tseq\tpast\tgenerated_cu\tout_prefix\ttrace_dir\texport_json\n'
    for seq in 4 128; do
      for model in gqa2 mha4; do
        printf '%s\t%s\t3\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" \
          "${control}/${model}.cu" "${sim}/run/${model}" \
          "${sim}/dump/${model}_s${seq}_p5" \
          "${seqscan}/export/${model}.json"
      done
      if [[ "${realwidth}" == 1 ]]; then
        work="$(work_for "${seq}")"
        printf 'real\t%s\t3\t%s\t%s\t-\t%s\n' "${seq}" \
          "${work}/model.cu" "${sim}/run/real" "${work}/model.json"
      fi
    done
  } > "${raw}/manifest.tsv"
  for tool in eft chain; do
    # The tool writes into `plan/` but does not create it.
    mkdir -p "${raw}/gen_${tool}/plan"
    "${build}/tools/tilemega-place-${tool}" "${repo}" "${raw}/manifest.tsv" \
      "${raw}/gen_${tool}" 2> "${raw}/log/place_${tool}.log"
  done
  grep -h '^CHAIN_SUMMARY' "${raw}/log/place_chain.log" || true
  # Both tools reach legacy_grid_stride, band and wavefront through their own
  # import and codegen path, so a difference there would mean one of them is not
  # emitting the placement it names.  Checked before the two directories are
  # merged, because merging is what would hide it.
  printf 'file\tidentical_between_tools\n' > "${raw}/tool_agreement.tsv"
  for f in "${raw}/gen_eft/plan"/*.cu; do
    name="$(basename "${f}")"
    other="${raw}/gen_chain/plan/${name}"
    [[ -e "${other}" ]] || continue
    if diff -q "${f}" "${other}" > /dev/null; then same=1; else same=0; fi
    printf '%s\t%s\n' "${name}" "${same}" >> "${raw}/tool_agreement.tsv"
    [[ "${same}" == 1 ]] || {
      echo "FAIL: ${name} differs between the two placement tools" >&2; exit 1; }
  done
  # One `cp` with both directories is refused by GNU cp the moment two
  # sources land on the same destination ("will not overwrite
  # just-created"), and under `set -e` that ends the run.  Copied one tool
  # at a time instead: the second pass rewrites the three shared files with
  # content the agreement check above just proved byte-identical.
  cp "${raw}/gen_eft/plan"/*.cu "${raw}/plan/"
  cp "${raw}/gen_chain/plan"/*.cu "${raw}/plan/"
  echo "TOOL_AGREEMENT shared=$(($(wc -l < "${raw}/tool_agreement.tsv") - 1))"
fi

# --- provenance: the emitted legacy source against the committed control, and
# the two tools against each other.  If legacy is byte-identical to the control
# then the other candidates differ from it only in the Plan.
if has provenance; then
  printf 'cell\tarm\tbytes\tidentical_to_control\n' > "${raw}/provenance.tsv"
  for entry in $(cells) $(real_cells); do
    model="${entry%%:*}"; seq="${entry##*:}"
    emitted="${raw}/plan/${model}_s${seq}_legacy_grid_stride.cu"
    base="$(base_source legacy_grid_stride "${model}" "${seq}")"
    if diff -q "${base}" "${emitted}" > /dev/null; then same=1; else same=0; fi
    printf '%s_s%s\tlegacy_grid_stride\t%s\t%s\n' "${model}" "${seq}" \
      "$(stat -c%s "${emitted}")" "${same}" >> "${raw}/provenance.tsv"
    [[ "${same}" == 1 ]] || {
      echo "FAIL: ${model} s${seq} legacy source differs from the control" >&2
      exit 1; }
  done
  cat "${raw}/provenance.tsv"
fi

# --- stamp: W into the Plan.  Configurations d and w share one stamped source
# per candidate, because W is the only thing either adds.
if has stamp; then
  for entry in $(cells) $(real_cells); do
    model="${entry%%:*}"; seq="${entry##*:}"
    for arm in $(arms_for "${model}"); do
      base="$(base_source "${arm}" "${model}" "${seq}")"
      out="${raw}/src/${model}_s${seq}_${arm}_w${W}.cu"
      if plan_carrying "${arm}"; then
        python3 "${repo}/docs/experiments/WINDOW/plan_window.py" \
          "${base}" "${W}" "${out}" --in-plan
      else
        python3 "${repo}/docs/experiments/WINDOW/plan_window.py" \
          "${base}" "${W}" "${out}"
      fi
    done
  done > "${raw}/log/stamp.txt"
  tail -3 "${raw}/log/stamp.txt"
fi

# --- build
if has build; then
  need_mib="${NEED_MIB:-12000}"
  avail="$(df -Pm "${raw}" | awk 'NR==2{print $4}')"
  [[ "${avail}" -ge "${need_mib}" ]] || {
    echo "FAIL: need ${need_mib} MiB free, have ${avail} MiB" >&2; exit 1; }
  for config in ${configs}; do
    for entry in $(cells) $(real_cells); do
      model="${entry%%:*}"; seq="${entry##*:}"
      for arm in $(arms_for "${model}"); do
        src="$(source_for "${config}" "${arm}" "${model}" "${seq}")"
        # shellcheck disable=SC2046
        "${nvcc}" "${common[@]}" $(config_flags "${config}") \
          -DTILEMEGA_PLACEMENT="$(arm_macro "${arm}")" "${src}" "${link[@]}" \
          -o "${raw}/bin/${model}_s${seq}_${arm}_${config}" \
          2> "${raw}/log/nvcc_${model}_s${seq}_${arm}_${config}.log"
      done
    done
  done
fi

# --- correctness: 50 fresh processes per cell x candidate
if has correctness; then
  printf 'config\tcell\tarm\tpasses\tprocesses\n' > "${raw}/correctness.tsv"
  for config in ${correctness_configs}; do
    for entry in $(cells) $(real_cells); do
      model="${entry%%:*}"; seq="${entry##*:}"
      fixture="$(fixture_for "${model}" "${seq}")"
      for arm in $(arms_for "${model}"); do
        out="${raw}/final/correct_${config}_${model}_s${seq}_${arm}"
        "${repo}/scripts/gpu_stat_run.sh" -n "${correctness_runs}" -t 300 \
          -l "eft2_${config}_${model}_${seq}_${arm}" -k -o "${out}" -- \
          "${raw}/bin/${model}_s${seq}_${arm}_${config}" "${fixture}" \
          > "${raw}/log/correct_${config}_${model}_s${seq}_${arm}.runner" 2>&1
        pass="$(grep -lc '^RESULT status=PASS' "${out}"/run_*.log 2>/dev/null \
          | wc -l)"
        printf '%s\t%s_s%s\t%s\t%s\t%s\n' "${config}" "${model}" "${seq}" \
          "${arm}" "${pass}" "${correctness_runs}" | tee -a "${raw}/correctness.tsv"
      done
    done
  done
  short="$(awk -F'\t' 'NR>1 && $4 != $5' "${raw}/correctness.tsv")"
  [[ -z "${short}" ]] || {
    printf 'cells short of a full pass:\n%s\n' "${short}" >&2; exit 1; }
fi

# --- paired: configuration and candidate rotated together, so neither axis
# sits at a fixed point in a session's drift (H6).
# See the note in SYNC_V2/run_barrier.sh: a fresh process can lose `cudaMalloc`
# to the previous run's context teardown.  Round three lost this phase at r1 of
# real_s4_rotate_w, which aborted every paired cell after it and left the real
# model unmeasured.  H1 fences out `scripts/gpu_stat_run.sh`, so the guard is
# repeated here rather than shared.
retry_oom() {
  local out="$1"; shift
  local rc=0
  "$@" || rc=$?
  [[ ${rc} -eq 0 ]] && return 0
  grep -qs 'out of memory' "${out}/ERROR_first.log" || return "${rc}"
  echo "  [retry] transient OOM, resettling: ${out}" >&2
  rm -rf "${out}"
  sleep 5
  "$@"
}

if has paired; then
  for entry in $(cells) $(real_cells); do
    model="${entry%%:*}"; seq="${entry##*:}"
    fixture="$(fixture_for "${model}" "${seq}")"
    pairs=()
    for config in ${configs}; do
      for arm in $(arms_for "${model}"); do pairs+=("${config}:${arm}"); done
    done
    n="${#pairs[@]}"
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<n; ++slot)); do
        item="${pairs[$(((round + slot) % n))]}"
        config="${item%%:*}"; arm="${item##*:}"
        attempt="${raw}/final/${model}_s${seq}_${arm}_${config}/r${round}"
        retry_oom "${attempt}" \
          "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 300 \
          -l "eft2_${config}_${model}_${seq}_${arm}" -k \
          -o "${attempt}" -- \
          "${raw}/bin/${model}_s${seq}_${arm}_${config}" "${fixture}" \
          >> "${raw}/log/${model}_s${seq}_${arm}_${config}.runner" 2>&1
      done
    done
  done
fi

# --- trace: each configuration's own ceiling, from its own trace (H8).
#
# Configurations b and d carry the RED publish, which used to refuse a traced
# build outright: with no identified last arriver there was nobody to stamp
# `event_publish`.  The stamp is now an atomicMax over the publishers, whose
# maximum is that same instant, so these rows are the real configuration and
# not a stand-in -- measured beside each arrival rather than after the last one,
# which is the one way it differs from the plain store.
if has trace; then
  for config in ${configs}; do
    for entry in $(cells); do
      model="${entry%%:*}"; seq="${entry##*:}"
      for arm in ${trace_arms}; do
        src="$(source_for "${config}" "${arm}" "${model}" "${seq}")"
        bin="${raw}/bin/${model}_s${seq}_${arm}_${config}_trace"
        # shellcheck disable=SC2046
        "${nvcc}" "${common[@]}" $(config_flags "${config}") \
          -DTILEMEGA_PLACEMENT="$(arm_macro "${arm}")" -DTILEMEGA_TRACE_V2=1 \
          "${src}" "${link[@]}" -o "${bin}" \
          2> "${raw}/log/nvcc_trace_${model}_s${seq}_${arm}_${config}.log"
        out="${raw}/final/trace_${config}_${model}_s${seq}_${arm}"
        mkdir -p "${out}"
        env TILEMEGA_MODEL_NAME="${model}" TILEMEGA_TRACE_V2=1 \
          TILEMEGA_TRACE_V2_OUT="${out}" timeout 300s "${bin}" \
          "$(fixture_for "${model}" "${seq}")" \
          > "${raw}/log/trace_${config}_${model}_s${seq}_${arm}.txt" 2>&1 || true
      done
    done
  done
fi

# --- analyze: the ceiling and the hop counts come from TRACE_V2/analyze.py,
# the tool that measured F-134 and F-145.  A second definition of either would
# not be comparable to the numbers this round has to move (H7).
if has analyze; then
  dumps=()
  for d in "${raw}"/final/trace_*; do
    [[ -f "${d}/slots.tsv" ]] && dumps+=("${d}")
  done
  ((${#dumps[@]})) || { echo 'no trace dumps to analyze' >&2; exit 1; }
  python3 "${repo}/docs/experiments/TRACE_V2/analyze.py" "${dumps[@]}" \
    --out "${raw}/analysis"
fi

echo "PASS phases=${phases}" > "${raw}/status.txt"
