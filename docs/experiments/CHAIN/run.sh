#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# EX-S2c gates S2c-a, S2c-b and S2c-d.
#
# The lever this round is the number of hops on the critical path, which is the
# one term that falls on both sm_89 and sm_120: on sm_120 the per-hop cost is
# already at a ~400 ns floor (F-145 and its sm_120 companion), so nothing but
# fewer hops can move it.  `tilemega-place-chain` places the longest chain of
# the runtime task DAG whole on one worker, where its internal edges cost no hop
# at all, and this script measures what that is worth.
#
# Three arms are measured, not the six of PLACE_EFT: S2c-a is the chain arm's
# correctness, S2c-d is defined as chain against rotate, and S2c-b's
# `critical_path_hops` comparison across all six candidates is computed offline
# into `predicted.tsv` -- it needs no GPU.  The measured six-candidate x A/B/C/D
# matrix is EX-S2r's job (§7) and running it twice would buy nothing.
#
# `legacy_grid_stride` is carried as the control the provenance diff pins: the
# source this tool emits for it must be byte-identical to the committed one, so
# a `chain` arm differs from the control only in the Plan (H2).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${RAW_DIR:-${here}/raw}"
build="${BUILD_DIR:-${repo}/build-portable}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
runs="${RUNS:-25}"
correctness="${CORRECTNESS_RUNS:-50}"
# S2c-b is computed offline, so the CPU half of this script can run while the
# GPU is busy with another experiment.  Same knob shape as SYNC_V2/run_barrier.sh
# rather than a second convention for the same idea.
phases="${PHASES:-generate,provenance,compile,stats,correctness,seqscan,paired,real,summary}"
has() { [[ ",${phases}," == *",$1,"* ]]; }
rm -f "${raw}/status.txt"
mkdir -p "${raw}/bin" "${raw}/log" "${raw}/final" "${raw}/plan"

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)
control="${repo}/docs/experiments/PLAN_CONTRACT/legacy_identity/plan"
arms=(legacy_grid_stride rotate chain)

arm_source() {
  local arm="$1" model="$2" seq="$3"
  case "${arm}" in
    legacy_grid_stride|rotate) echo "${control}/${model}.cu" ;;
    *) echo "${raw}/plan/${model}_s${seq}_${arm}.cu" ;;
  esac
}
arm_macro() {
  case "$1" in legacy_grid_stride) echo 0 ;; rotate) echo 5 ;; *) echo 0 ;; esac
}

# The manifest is written here rather than committed so that H9's sm_120 runner
# regenerates it against its own tree: a materialized Plan binds (worker, slot)
# to a specific resident grid and must never be carried across machines.  Under
# SKIP_GENERATE the manifest is an input, not an output: `prepare_sm120.py` has
# already written one naming its own control probes, and rewriting it here would
# leave a file in the sm_120 tree that names a 4090's dumps.
sim="${repo}/docs/experiments/SIMULATOR/raw"
if has generate && [[ "${SKIP_GENERATE:-0}" != 1 ]]; then
{
  printf 'model\tseq\tpast\tgenerated_cu\tout_prefix\ttrace_dir\texport_json\n'
  for seq in 4 128; do
    for model in gqa2 mha4; do
      printf '%s\t%s\t3\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" \
        "${control}/${model}.cu" "${sim}/run/${model}" \
        "${sim}/dump/${model}_s${seq}_p5" \
        "${repo}/docs/experiments/SEQSCAN/raw/export/${model}.json"
    done
    work="${repo}/docs/experiments/REALMODEL/raw/work/r2sim_s${seq}"
    printf 'real\t%s\t3\t%s\t%s\t-\t%s\n' "${seq}" \
      "${work}/model.cu" "${sim}/run/real" "${work}/model.json"
  done
} > "${raw}/manifest.tsv"
  "${build}/tools/tilemega-place-chain" "${repo}" "${raw}/manifest.tsv" "${raw}" \
    2> "${raw}/log/place_chain.log"
fi
grep -h '^CHAIN_CELL\|^CHAIN_SUMMARY' "${raw}/log/place_chain.log" || true

# --- provenance: the driver's legacy source against the committed control
if has provenance; then
  : > "${raw}/provenance.tsv"
  printf 'model\tseq\tbytes\tidentical\n' >> "${raw}/provenance.tsv"
  for seq in 4 128; do
    for model in gqa2 mha4; do
      emitted="${raw}/plan/${model}_s${seq}_legacy_grid_stride.cu"
      if diff -q "${control}/${model}.cu" "${emitted}" > /dev/null; then same=1; else same=0; fi
      printf '%s\t%s\t%s\t%s\n' "${model}" "${seq}" "$(stat -c%s "${emitted}")" "${same}" \
        >> "${raw}/provenance.tsv"
      [[ "${same}" == 1 ]] || {
        echo "FAIL: ${model} s${seq} legacy source differs from the control" >&2; exit 1; }
    done
  done
  cat "${raw}/provenance.tsv"
fi

# --- compile every arm of every reference cell
if has compile; then
  for seq in 4 128; do
    for model in gqa2 mha4; do
      for arm in "${arms[@]}"; do
        "${nvcc}" "${common[@]}" -DTILEMEGA_PLACEMENT="$(arm_macro "${arm}")" \
          "$(arm_source "${arm}" "${model}" "${seq}")" "${link[@]}" \
          -o "${raw}/bin/${model}_s${seq}_${arm}" \
          2> "${raw}/log/nvcc_${model}_s${seq}_${arm}.log"
      done
    done
  done
fi

# --- the queue statistics each arm reports about its own plan
if has stats; then
  : > "${raw}/place_stats.txt"
  for seq in 4 128; do
    for model in gqa2 mha4; do
      fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
      for arm in "${arms[@]}"; do
        "${raw}/bin/${model}_s${seq}_${arm}" "${fixture}" \
          > "${raw}/log/stats_${model}_s${seq}_${arm}.out" 2>&1
        printf '%s\tseq=%s\t%s\t' "${model}" "${seq}" "${arm}" >> "${raw}/place_stats.txt"
        grep -h '^E2E_PLACE_STATS' "${raw}/log/stats_${model}_s${seq}_${arm}.out" \
          >> "${raw}/place_stats.txt"
      done
    done
  done
  cat "${raw}/place_stats.txt"
fi

# --- S2c-a: 50 fresh processes per arm per reference cell
if has correctness; then
  printf 'model\tseq\tarm\tpasses\tprocesses\n' > "${raw}/correctness.tsv"
  for seq in 4 128; do
    for model in gqa2 mha4; do
      fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
      for arm in "${arms[@]}"; do
        "${repo}/scripts/gpu_stat_run.sh" -n "${correctness}" -t 120 \
          -l "chain_${model}_s${seq}_${arm}" -k \
          -o "${raw}/final/correct_${model}_s${seq}_${arm}" -- \
          "${raw}/bin/${model}_s${seq}_${arm}" "${fixture}" \
          > "${raw}/log/correct_${model}_s${seq}_${arm}.runner" 2>&1 || true
        pass="$({ grep -rh -c '^RESULT status=PASS' \
          "${raw}/final/correct_${model}_s${seq}_${arm}" || true; } \
          | awk '{n+=$1} END{print n+0}')"
        printf '%s\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" "${arm}" "${pass}" \
          "${correctness}" >> "${raw}/correctness.tsv"
      done
    done
  done
  cat "${raw}/correctness.tsv"
  awk -F'\t' 'NR>1 && $4 != $5 {bad++} END {exit bad>0}' "${raw}/correctness.tsv" || {
    echo "FAIL: S2c-a short of ${correctness}/${correctness} in at least one cell" >&2; exit 1; }
fi

# --- S2c-a SEQSCAN subset: the chain plan is pinned to one theta, so the host
# must refuse every seq its table was not solved for rather than run it wrong.
# That refusal is the gate here; a silently-accepted foreign seq is a failure.
if has seqscan; then
  printf 'model\tseq\tpast\tarm\texpect\tpasses\terrors\treason\tprocesses\n' \
    > "${raw}/seqscan.tsv"
  for seq in 4 128; do
    for model in gqa2 mha4; do
      for past in 0 3 512; do
        fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p${past}"
        [[ -d "${fixture}" ]] || continue
        "${repo}/scripts/gpu_stat_run.sh" -n "${correctness}" -t 120 \
          -l "chainscan_${model}_s${seq}_p${past}" \
          -o "${raw}/final/scan_${model}_s${seq}_p${past}" -- \
          "${raw}/bin/${model}_s${seq}_chain" "${fixture}" \
          > "${raw}/log/scan_${model}_s${seq}_p${past}.runner" 2>&1 || true
        out="${raw}/final/scan_${model}_s${seq}_p${past}"
        runner="${raw}/log/scan_${model}_s${seq}_p${past}.runner"
        # The plan table is solved for past=3, so that is the only cell the host
        # may run.  The other two MUST be refused: a refusal is the gate, and a
        # zero PASS count is therefore evidence, not an error to abort on.
        if [[ "${past}" == 3 ]]; then expect=run; else expect=refuse; fi
        # `gpu_stat_run.sh` deletes a passing run's log unless `-k` is given, so
        # counting PASS lines under `${out}` reports 0 for a cell that actually
        # passed 50/50.  The STAT line is the runner's own tally and survives
        # either way, which is where `errors` below already reads from.
        pass="$(awk '/^STAT /{for(i=1;i<=NF;i++) if ($i ~ /^pass=/) {
          sub(/^pass=/,"",$i); p=$i}} END{print p+0}' "${runner}")"
        errors="$(awk '/^STAT /{for(i=1;i<=NF;i++) if ($i ~ /^error=/) {
          sub(/^error=/,"",$i); e=$i}} END{print e+0}' "${runner}")"
        # `error=` alone does not say the host refused for the right reason -- a
        # segfault or a missing fixture counts the same.  The guard's own message
        # is what separates a correct refusal from a broken binary.
        if grep -q 'plan table is pinned' "${out}/ERROR_first.log" 2>/dev/null
        then reason=pinned; else reason=-; fi
        printf '%s\t%s\t%s\tchain\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${seq}" \
          "${past}" "${expect}" "${pass}" "${errors}" "${reason}" "${correctness}" \
          >> "${raw}/seqscan.tsv"
      done
    done
  done
  cat "${raw}/seqscan.tsv"
  awk -F'\t' 'NR>1 {
      if ($5 == "run"    && $6 != $9)                              { bad++; print }
      if ($5 == "refuse" && ($6 != 0 || $7 != $9 || $8 != "pinned")) { bad++; print }
    } END { exit bad>0 }' "${raw}/seqscan.tsv" || {
    echo "FAIL: a seqscan cell neither ran clean nor refused on the pinned plan" >&2
    exit 1; }
fi

# See the note in SYNC_V2/run_barrier.sh: a fresh process can lose `cudaMalloc`
# to the previous run's context teardown.  Both paired loops below assert a full
# sample count per arm, so one transient anywhere in them discards the whole
# phase's runs.  H1 holds `scripts/gpu_stat_run.sh` outside this round's fence,
# so the same guard is repeated here rather than shared.
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

# --- S2c-d: paired rounds, rotating arm order
if has paired; then
  for seq in 4 128; do
    for model in gqa2 mha4; do
      fixture="${repo}/docs/experiments/SEQSCAN/raw/fixture/${model}_s${seq}_p3"
      for ((round=0; round<runs; ++round)); do
        for ((slot=0; slot<${#arms[@]}; ++slot)); do
          arm="${arms[$(((round + slot) % ${#arms[@]}))]}"
          attempt="${raw}/final/${model}_s${seq}_${arm}/r${round}"
          retry_oom "${attempt}" \
            "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 120 \
            -l "chain_${model}_${seq}_${arm}" -k \
            -o "${attempt}" -- \
            "${raw}/bin/${model}_s${seq}_${arm}" "${fixture}" \
            >> "${raw}/log/${model}_s${seq}_${arm}.runner" 2>&1
        done
      done
      for arm in "${arms[@]}"; do
        logs=("${raw}/final/${model}_s${seq}_${arm}"/r*/run_*.log)
        samples="$({ grep -h -c '^E2E_TIME' "${logs[@]}" || true; } \
          | awk '{n+=$1} END{print n+0}')"
        pass="$({ grep -h -c '^RESULT status=PASS' "${logs[@]}" || true; } \
          | awk '{n+=$1} END{print n+0}')"
        [[ "${samples}" == "${runs}" && "${pass}" == "${runs}" ]] || {
          echo "${model} s${seq} ${arm}: ${samples}/${runs} timed, ${pass}/${runs} correct" >&2
          exit 1; }
      done
    done
  done
fi

# --- S2c-d real-width cells, same rotation, their own fixtures
realwidth="${REALWIDTH:-1}"
if has real && [[ "${realwidth}" == 1 ]]; then
  for seq in 4 128; do
    work="${repo}/docs/experiments/REALMODEL/raw/work/r2sim_s${seq}"
    fixture="${work}/export/fixture"
    [[ -d "${fixture}" ]] || { echo "FAIL: no real-width fixture at ${fixture}" >&2; exit 1; }
    for arm in "${arms[@]}"; do
      case "${arm}" in
        legacy_grid_stride|rotate) src="${work}/model.cu" ;;
        *) src="${raw}/plan/real_s${seq}_${arm}.cu" ;;
      esac
      "${nvcc}" "${common[@]}" -DTILEMEGA_PLACEMENT="$(arm_macro "${arm}")" \
        "${src}" "${link[@]}" -o "${raw}/bin/real_s${seq}_${arm}" \
        2> "${raw}/log/nvcc_real_s${seq}_${arm}.log"
    done
    for arm in "${arms[@]}"; do
      "${repo}/scripts/gpu_stat_run.sh" -n "${correctness}" -t 300 \
        -l "chain_real_s${seq}_${arm}" -k \
        -o "${raw}/final/correct_real_s${seq}_${arm}" -- \
        "${raw}/bin/real_s${seq}_${arm}" "${fixture}" \
        > "${raw}/log/correct_real_s${seq}_${arm}.runner" 2>&1 || true
      pass="$({ grep -rh -c '^RESULT status=PASS' \
        "${raw}/final/correct_real_s${seq}_${arm}" || true; } \
        | awk '{n+=$1} END{print n+0}')"
      printf 'real\t%s\t%s\t%s\t%s\n' "${seq}" "${arm}" "${pass}" "${correctness}" \
        >> "${raw}/correctness.tsv"
    done
    for ((round=0; round<runs; ++round)); do
      for ((slot=0; slot<${#arms[@]}; ++slot)); do
        arm="${arms[$(((round + slot) % ${#arms[@]}))]}"
        attempt="${raw}/final/real_s${seq}_${arm}/r${round}"
        retry_oom "${attempt}" \
          "${repo}/scripts/gpu_stat_run.sh" -n 1 -t 300 \
          -l "chain_real_${seq}_${arm}" -k \
          -o "${attempt}" -- \
          "${raw}/bin/real_s${seq}_${arm}" "${fixture}" \
          >> "${raw}/log/real_s${seq}_${arm}.runner" 2>&1
      done
    done
  done
fi

if has summary; then
  python3 "${here}/summarize.py" "${raw}" > "${raw}/summary.tsv"
  cat "${raw}/summary.tsv"
fi
echo "PASS phases=${phases}" > "${raw}/status.txt"
