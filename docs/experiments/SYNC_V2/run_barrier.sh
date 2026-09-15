#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# EX-E3: one compile-time switch, measured.  Defaults to step 1's: the
# executor's per-task barriers, where source counts five (§5.5.1 says "at most
# 5"; R3 §1.2 says 6 -- the difference is recorded in FINDINGS, not reconciled)
# and `TILEMEGA_BARRIER_V2=1` keeps the two that carry an ordering nothing else
# carries.  SWITCH/TAG/OUT_DIR retarget the same phases at another step's switch
# -- step 2 is `SWITCH=TILEMEGA_EVENT_SOLO TAG=solo OUT_DIR=.../raw_solo` -- so
# both steps are gated by one script rather than by a copy that can drift.
# `v0`/`v1` in every path name mean the switch off and on, whichever it is.
#
# Phases, via PHASES=build,sass,correctness,seqscan,attrib:
#   sass         BAR.SYNC counts, whole-file and split at each
#                `Function :` boundary, because RunTask inlines the task
#                bodies' barriers into tilemega_l2_kernel and the E3-1
#                subset is invisible in a whole-kernel count
#   correctness  50 fresh processes per model x seq in {4,128}, v2 on
#   seqscan      the full 30-cell matrix under v2, plus SEQSCAN's own two
#                negative controls -- the task-wait clamp must still FAIL, or
#                fewer barriers have made the wait rows vacuous
#   attrib       four arms x v2 off/on, paired and rotated (H6)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${OUT_DIR:-${here}/raw_barrier}"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cuobjdump="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"
seqscan="${repo}/docs/experiments/SEQSCAN/raw"
runs="${RUNS:-25}"
correctness_runs="${CORRECTNESS_RUNS:-50}"
phases="${PHASES:-build,sass,correctness,seqscan,attrib}"
switch="${SWITCH:-TILEMEGA_BARRIER_V2}"
tag="${TAG:-barrier}"
# Most steps turn one macro on, so the two arms are `-D<switch>=1` and `=0`.
# E3-0 is not one macro: its wait policy is five calibrated values read from the
# target (H5), and `-DTILEMEGA_WAIT_POLICY=1` alone would compile the header
# defaults -- EventSync.cuh:22 leaves spin_iters at 0, not the calibrated 64 --
# and file the result under the right name.  ON_FLAGS/OFF_FLAGS carry the whole
# set for such a step; unset, they are exactly the single-macro arms above.
on_flags="${ON_FLAGS:--D${switch}=1}"
off_flags="${OFF_FLAGS:--D${switch}=0}"
arm_switch() { if [[ "$1" == 1 ]]; then echo "${on_flags}"; else echo "${off_flags}"; fi; }
# The trace census is built under `-DTILEMEGA_TRACE_V2=1`.  The RED publish used
# to refuse that build outright -- with no identified last arriver there was
# nobody to stamp `event_publish` -- so this knob existed to skip it.  The stamp
# is now an `atomicMax` over the publishers, whose maximum is that same instant,
# so the RED census counts a real build; the knob remains for a future switch
# that genuinely cannot combine.  Set 0 for such a switch; plain census runs.
trace_census="${TRACE_CENSUS:-1}"
has() { [[ ",${phases}," == *",$1,"* ]]; }

mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final"
# Written only at the very end, so a leftover from an earlier partial invocation
# reads as a pass with no data behind it.  One did: a 10:32 `PASS` outlived a
# seqscan that stopped at cell 27 of 30.
rm -f "${raw}/status.txt"
common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
arms=(neither nowait full l1nosync)
flags() {
  case "$1" in
    neither) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1 -DTILEMEGA_UNSAFE_NO_EVENT_NOTIFY=1' ;;
    nowait) echo '-DTILEMEGA_UNSAFE_NO_EVENT_WAIT=1' ;;
    full) echo '' ;;
    l1nosync) echo '-DTILEMEGA_UNSAFE_NO_GRID_SYNC=1' ;;
  esac
}

# A denied CUDA context is not a failed run.  An external workload once held
# enough of the device that `cudaFuncGetAttributes` -- ModelHarness.cuh:2657,
# the first CUDA call the harness makes -- returned `out of memory` in 354 of
# 600 processes.  Each died before flushing stdout, so it left no RESULT line
# and was indistinguishable from a silent mismatch; the matrix it produced read
# as a correctness regression and was void data.  Counted apart, retried, and
# never folded into the pass count.
oom_retries="${OOM_RETRIES:-5}"
cell_oom=0
run_cell() {
  local bin="$1" fixture="$2" log="$3" n="$4" i try out
  cell_oom=0
  for ((i=0; i<n; ++i)); do
    for ((try=0; try<=oom_retries; ++try)); do
      out="$(timeout 120s "${bin}" "${fixture}" 2>&1)" || true
      if grep -q 'out of memory' <<<"${out}"; then
        if ((try < oom_retries)); then sleep 5; continue; fi
        cell_oom=$((cell_oom + 1))
      fi
      break
    done
    printf '%s\n' "${out}" >> "${log}"
  done
}

if has build; then
  for model in gqa2 mha4; do
    src="${seqscan}/src/${model}.cu"
    for v2 in 0 1; do
      # shellcheck disable=SC2206
      sw=($(arm_switch "${v2}"))
      for arm in "${arms[@]}"; do
        # shellcheck disable=SC2206
        extra=($(flags "${arm}"))
        "${nvcc}" "${common[@]}" "${sw[@]}" \
          "${extra[@]+"${extra[@]}"}" "${src}" "${link[@]}" \
          -o "${raw}/bin/${model}_v${v2}_${arm}" \
          2> "${raw}/log/${model}_v${v2}_${arm}.ptxas"
      done
      # Census only, never run: under trace v2 the `run_begin`/`run_end` stamps
      # come from thread 0, so the pair around RunTask has to stay and the
      # reduction is 5 -> 4 rather than 5 -> 2.  Built here so that count is
      # reproducible from this script instead of a scratch directory.
      if [[ "${trace_census}" == 1 ]]; then
        "${nvcc}" "${common[@]}" "${sw[@]}" \
          -DTILEMEGA_TRACE_V2=1 "${src}" "${link[@]}" \
          -o "${raw}/bin/${model}_v${v2}_trace" \
          2> "${raw}/log/${model}_v${v2}_trace.ptxas"
      fi
    done
  done
  # The two SEQSCAN negatives, rebuilt under v2 rather than reused: the clamps
  # are compile-time and the point is how they behave with fewer barriers.
  # shellcheck disable=SC2206
  sw1=($(arm_switch 1))
  for negative in NEGATIVE_OLD_CLAMP NEGATIVE_TASK_WAIT_CLAMP; do
    "${nvcc}" "${common[@]}" "${sw1[@]}" "-DTILEMEGA_${negative}=1" \
      "${seqscan}/src/gqa2.cu" "${link[@]}" \
      -o "${raw}/bin/gqa2_v1_$(echo "${negative}" | tr 'A-Z' 'a-z')"
  done
fi

if has sass; then
  # sass_report.sh greps with rg, and an absent rg makes every section empty
  # while the script still exits 0 -- which once produced four "PASS" reports
  # carrying no barrier data at all.  A missing tool is a failure, not a count
  # of zero.  grep is not a substitute: the ATOMICS pattern uses (?:...), which
  # POSIX ERE cannot express.
  command -v rg >/dev/null || {
    echo 'sass_report.sh needs ripgrep on PATH' >&2; exit 1; }
  printf 'model\tv2\tvariant\tkernel\tbar_sync\tmembar\tnanosleep\tatomics\n' \
    > "${raw}/census.tsv"
  for model in gqa2 mha4; do
    for v2 in 0 1; do
     variants=(plain)
     [[ "${trace_census}" == 1 ]] && variants+=(trace)
     for variant in "${variants[@]}"; do
      if [[ "${variant}" == plain ]]; then bin="${raw}/bin/${model}_v${v2}_full"
      else bin="${raw}/bin/${model}_v${v2}_trace"; fi
      report="${raw}/log/${model}_v${v2}_${variant}.sassreport"
      bash "${repo}/scripts/sass_report.sh" "${bin}" > "${report}" 2>&1
      grep -A100000 '^BARRIERS$' "${report}" | sed -n '2,/^NANOSLEEP$/p' \
        | grep -q 'BAR' || {
        echo "empty BARRIERS section for ${model} v${v2} ${variant}" >&2
        exit 1; }
      # The gate is on the executor subset, so the whole-file count is split at
      # the `Function :` boundaries; awk over the same dump silently produced no
      # rows here, so the ranges are cut with sed and stay auditable by hand.
      dump="${raw}/log/${model}_v${v2}_${variant}.sass"
      "${cuobjdump}" --dump-sass "${bin}" > "${dump}"
      total="$(wc -l < "${dump}")"
      starts=(); names=()
      while IFS= read -r entry; do
        starts+=("${entry%%:*}")
        names+=("$(printf '%s' "${entry#*Function : }" | tr -d ' \r')")
      done < <(grep -n 'Function : ' "${dump}")
      for ((k=0; k<${#starts[@]}; ++k)); do
        a="${starts[$k]}"
        if ((k + 1 < ${#starts[@]})); then b=$((${starts[$((k + 1))]} - 1)); else b="${total}"; fi
        # MEMBAR and NANOSLEEP are carried next to the barrier count as the
        # control: v2 must move BAR.SYNC and nothing else, and a fence or a
        # backoff that moved with it would mean the edit reached further than
        # the three barriers it claims to drop.
        slice="$(mktemp)"
        sed -n "${a},${b}p" "${dump}" > "${slice}"
        n="$(grep -c 'BAR\.SYNC' "${slice}" || true)"
        mb="$(grep -c 'MEMBAR' "${slice}" || true)"
        ns="$(grep -c 'NANOSLEEP' "${slice}" || true)"
        at="$(grep -cE 'ATOM|RED' "${slice}" || true)"
        rm -f "${slice}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${v2}" \
          "${variant}" "${names[$k]}" "${n}" "${mb}" "${ns}" "${at}" \
          >> "${raw}/census.tsv"
      done
     done
    done
  done
fi

if has correctness; then
  printf 'model\tseq\tpast\tv2\tpasses\tprocesses\toom\n' \
    > "${raw}/correctness.tsv"
  for model in gqa2 mha4; do
    for seq in 4 128; do
      log="${raw}/log/correct_${model}_s${seq}.txt"; : > "${log}"
      # A non-zero run must lower the measured pass count, not kill the phase:
      # the harness returns 1 on MISMATCH and the log keeps the line.
      run_cell "${raw}/bin/${model}_v1_full" \
        "${seqscan}/fixture/${model}_s${seq}_p3" "${log}" "${correctness_runs}"
      pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
      printf '%s\t%s\t3\t1\t%s\t%s\t%s\n' "${model}" "${seq}" "${pass}" \
        "${correctness_runs}" "${cell_oom}" | tee -a "${raw}/correctness.tsv"
    done
  done
  # Ordered: a cell the GPU refused did not run, and reporting it as a
  # correctness failure is the confusion this guard exists to prevent.
  denied="$(awk -F'\t' 'NR>1 && $7 > 0' "${raw}/correctness.tsv")"
  [[ -z "${denied}" ]] || {
    printf 'cells the GPU refused a context (did NOT run, not a failure):\n%s\n' \
      "${denied}" >&2
    exit 2; }
  short="$(awk -F'\t' 'NR>1 && $5 != $6' "${raw}/correctness.tsv")"
  [[ -z "${short}" ]] || {
    printf 'correctness cells short of full pass:\n%s\n' "${short}" >&2
    exit 1; }
fi

if has seqscan; then
  printf 'model\tseq\tpast\tpasses\tprocesses\toom\n' > "${raw}/matrix.tsv"
  for model in gqa2 mha4; do
    for seq in 1 4 128 512 2048; do
      for past in 0 3 512; do
        log="${raw}/log/matrix_${model}_s${seq}_p${past}.txt"; : > "${log}"
        run_cell "${raw}/bin/${model}_v1_full" \
          "${seqscan}/fixture/${model}_s${seq}_p${past}" "${log}" \
          "${correctness_runs}"
        pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" "${past}" \
          "${pass}" "${correctness_runs}" "${cell_oom}" | tee -a "${raw}/matrix.tsv"
      done
    done
  done
  # Asserted on the finished table so all 30 rows survive for diagnosis; an
  # abort mid-loop leaves a truncated matrix that looks like a short run.
  denied="$(awk -F'\t' 'NR>1 && $6 > 0' "${raw}/matrix.tsv")"
  [[ -z "${denied}" ]] || {
    printf 'matrix cells the GPU refused a context (did NOT run):\n%s\n' \
      "${denied}" >&2
    exit 2; }
  short="$(awk -F'\t' 'NR>1 && $4 != $5' "${raw}/matrix.tsv")"
  [[ -z "${short}" ]] || {
    printf 'seqscan cells short of full pass:\n%s\n' "${short}" >&2
    exit 1; }
  log="${raw}/log/negative_old_clamp.txt"; : > "${log}"
  for ((i=0; i<correctness_runs; ++i)); do
    timeout 120s "${raw}/bin/gqa2_v1_negative_old_clamp" \
      "${seqscan}/fixture/gqa2_s2048_p0" >> "${log}" || true
  done
  pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
  echo "NEGATIVE old_clamp pass=${pass}/${correctness_runs} expect=${correctness_runs}"
  [[ "${pass}" == "${correctness_runs}" ]] || {
    echo 'retired stage clamp changed the active queue path under v2' >&2; exit 1; }
  log="${raw}/log/negative_task_wait_clamp.txt"; : > "${log}"
  for ((i=0; i<correctness_runs; ++i)); do
    timeout 120s "${raw}/bin/gqa2_v1_negative_task_wait_clamp" \
      "${seqscan}/fixture/gqa2_s2048_p0" >> "${log}" || true
  done
  pass="$(grep -c '^RESULT status=PASS' "${log}" || true)"
  echo "NEGATIVE task_wait_clamp pass=${pass}/${correctness_runs} expect=0"
  [[ "${pass}" == 0 ]] || {
    echo 'task wait clamp passed under v2: the wait rows have gone vacuous' >&2
    exit 1; }
fi

# A fresh process can lose `cudaMalloc` to the previous run's context teardown.
# Round three lost this stage at r13 of mha4_s128_v1_full after thirteen clean
# rounds, and the `full` arm is the one whose failure is fatal below, so the
# transient landed exactly where it could not be tolerated.  Retry that single
# signature once from a cleared output directory, so the first attempt's
# ERROR_first.log cannot be counted against the retry.  Anything else stays
# fatal.  H1 holds `scripts/gpu_stat_run.sh` outside this round's fence, so the
# same guard is repeated in PLACE_EFT2/run.sh rather than shared.
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

if has attrib; then
  for model in gqa2 mha4; do
    for seq in 4 128; do
      fixture="${seqscan}/fixture/${model}_s${seq}_p3"
      for ((round=0; round<runs; ++round)); do
        for ((slot=0; slot<${#arms[@]}; ++slot)); do
          arm="${arms[$(((round + slot) % ${#arms[@]}))]}"
          for v2 in $(((round % 2))) $((1 - (round % 2))); do
            attempt="${raw}/final/${model}_s${seq}_v${v2}_${arm}/r${round}"
            retry_oom "${attempt}" \
              "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
              -l "${tag}_${model}_${seq}_v${v2}_${arm}" -k \
              -o "${attempt}" -- \
              "${raw}/bin/${model}_v${v2}_${arm}" "${fixture}" \
              >> "${raw}/log/${model}_s${seq}_v${v2}_${arm}.runner" 2>&1 \
              || [[ "${arm}" != full ]]
          done
        done
      done
    done
  done
fi

echo PASS > "${raw}/status.txt"
