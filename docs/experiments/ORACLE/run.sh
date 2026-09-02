#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# Partition oracle (skeleton Part 6): fix Place and the event structure, sweep
# the GEMM granularity `g` exhaustively, and measure end-to-end latency per
# configuration on both reference models.
#
# Depends on the binaries/fixtures produced by E2E_GEN/run.sh and
# P3_GENERALIZATION/run.sh; this script recompiles their generated .cu with a
# different `g` but regenerates nothing.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
build="${repo}/build-phase12"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
jobs="${JOBS:-16}"
screen_runs="${SCREEN_RUNS:-3}"
final_runs="${FINAL_RUNS:-25}"
topk="${TOPK:-8}"
# §2.4's Split applied to every GEMM's k. 1 is the current fixed g (no split).
splits="${SPLITS:-1 2 4 8 16}"
mkdir -p "${raw}/bin" "${raw}/log"

gqa_src="${repo}/docs/experiments/E2E_GEN/raw/generated_e2e.cu"
gqa_fixture="${repo}/docs/experiments/E2E/fixture"
mha_src="${repo}/docs/experiments/P3_GENERALIZATION/raw/generated.cu"
mha_fixture="${repo}/docs/experiments/P3_GENERALIZATION/raw/fixture"
for needed in "${build}/libtilemega.a" "${gqa_src}" "${gqa_fixture}/manifest.json" \
              "${mha_src}" "${mha_fixture}/manifest.json"; do
  if [[ ! -e "${needed}" ]]; then
    echo "BLOCKED: ${needed} missing; run E2E_GEN/run.sh and P3_GENERALIZATION/run.sh first" \
      | tee "${raw}/run_status.txt" >&2
    exit 77
  fi
done

common=(-std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
link=("${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart)

# ---------------------------------------------------------------- tier 1 ----
# Traits-only enumeration: constexpr legality + smem, no device code, no ptxas.
# Each stage is guarded by its own output so a re-run resumes rather than
# repeating half an hour of nvcc.
if [[ -s "${raw}/survivors.txt" && -s "${raw}/tier1_summary.txt" ]]; then
  echo "== tier 1: cached =="
  probe_build=$(awk -F'\t' '$1=="tier1_build_s"{print $2}' "${raw}/tier1_summary.txt")
  cat "${raw}/tier1_summary.txt"
else
echo "== tier 1: traits enumeration =="
probe_start=$(date +%s.%N)
"${nvcc}" -std=c++17 -O2 -arch=native \
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
  -I"${repo}/third_party/cutlass/tools/util/include" \
  "${here}/candidate_probe.cu" -o "${raw}/candidate_probe" \
  2> "${raw}/candidate_probe_build.txt"
probe_build=$(awk -v a="$(date +%s.%N)" -v b="${probe_start}" 'BEGIN{printf "%.2f", a-b}')
"${raw}/candidate_probe" > "${raw}/candidates.txt"
total=$(grep -c '^CANDIDATE' "${raw}/candidates.txt")
shape_legal=$(grep -c 'shape_legal=1' "${raw}/candidates.txt")
fits=$(grep -c 'fits=1' "${raw}/candidates.txt")
printf 'tier1_total\t%s\ntier1_shape_legal\t%s\ntier1_fits\t%s\ntier1_build_s\t%.2f\n' \
  "${total}" "${shape_legal}" "${fits}" "${probe_build}" \
  | tee "${raw}/tier1_summary.txt"

grep 'fits=1' "${raw}/candidates.txt" \
  | sed -E 's/^CANDIDATE m=([0-9]+) n=([0-9]+) k=([0-9]+) stages=([0-9]+).*/\1 \2 \3 \4/' \
  > "${raw}/survivors.txt"
fi

# ---------------------------------------------------------------- tier 3 ----
# One real nvcc+ptxas compile per survivor per model.
compile_one() {  # $1=model $2=src $3=M $4=N $5=K $6=S $7=Kc
  local tag="$1_$3x$4x$5s$6k$7"
  # shellcheck disable=SC2086
  "${nvcc}" "$2" ${common_str} \
    -DTILEMEGA_GEMM_TILE_M="$3" -DTILEMEGA_GEMM_TILE_N="$4" \
    -DTILEMEGA_GEMM_TILE_K="$5" -DTILEMEGA_GEMM_STAGES="$6" \
    -DTILEMEGA_GEMM_SPLIT_K="$7" \
    ${link_str} -o "${raw}/bin/${tag}" 2> "${raw}/log/${tag}.ptxas" \
    && echo "OK ${tag}" || echo "FAIL ${tag}"
}
export -f compile_one
export nvcc raw
export common_str="${common[*]}" link_str="${link[*]}"

# The full candidate list: every smem-legal tile shape crossed with every
# split factor. Kc is a granularity choice, not a backend trait, so tier 1
# cannot prune it -- its legality is `Kc <= k_tiles`, clamped at run time.
: > "${raw}/configs.txt"
while read -r m n k s; do
  for kc in ${splits}; do printf '%s %s %s %s %s\n' "${m}" "${n}" "${k}" "${s}" "${kc}"; done
done < "${raw}/survivors.txt" > "${raw}/configs.txt"

if [[ -s "${raw}/tier3_summary.txt" ]]; then
  echo "== tier 3: cached =="
  compile_s=$(awk -F'\t' '$1=="tier3_compile_wall_s"{print $2}' "${raw}/tier3_summary.txt")
  cat "${raw}/tier3_summary.txt"
else
echo "== tier 3: compiling $(wc -l < "${raw}/configs.txt") x 2 candidates =="
compile_start=$(date +%s.%N)
: > "${raw}/compile_status.txt"
for model in gqa2 mha4; do
  src="${gqa_src}"; [[ "${model}" == mha4 ]] && src="${mha_src}"
  while read -r m n k s kc; do printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${model}" "${src}" "${m}" "${n}" "${k}" "${s}" "${kc}"; done \
    < "${raw}/configs.txt"
done | xargs -P "${jobs}" -n 7 bash -c 'compile_one "$0" "$1" "$2" "$3" "$4" "$5" "$6"' \
     >> "${raw}/compile_status.txt"
compile_s=$(awk -v a="$(date +%s.%N)" -v b="${compile_start}" 'BEGIN{printf "%.2f", a-b}')
ok=$(grep -c '^OK' "${raw}/compile_status.txt" || true)
failed=$(grep -c '^FAIL' "${raw}/compile_status.txt" || true)
printf 'tier3_compiled_ok\t%s\ntier3_compile_failed\t%s\ntier3_compile_wall_s\t%.2f\n' \
  "${ok}" "${failed}" "${compile_s}" | tee "${raw}/tier3_summary.txt"
fi

# --------------------------------------------------------------- measure ----
median() { sort -n | awk '{v[NR]=$1} END {if(NR==0){print "nan";exit} printf "%.6f\n",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }

screen_model() {  # $1=model $2=fixture $3=runs -> tsv on stdout
  local model="$1" fixture="$2" runs="$3"
  while read -r m n k s kc; do
    local tag="${model}_${m}x${n}x${k}s${s}k${kc}"
    local bin="${raw}/bin/${tag}"
    [[ -x "${bin}" ]] || continue
    local l05=() l1=() l2=() status=PASS hash= grid= smem= ctas=
    local i out
    for ((i = 0; i < runs; ++i)); do
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
      printf '%s\t%s\t%s\t%s\t%s\tnan\tnan\tnan\t%s\t-\t-\t-\t-\n' "${m}" "${n}" "${k}" "${s}" "${kc}" "${status}"
      continue
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\n' \
      "${m}" "${n}" "${k}" "${s}" "${kc}" \
      "$(printf '%s\n' "${l05[@]}" | median)" \
      "$(printf '%s\n' "${l1[@]}" | median)" \
      "$(printf '%s\n' "${l2[@]}" | median)" \
      "${hash}" "${smem}" "${ctas}" "${grid}"
  done < "${raw}/configs.txt"
}

header='tile_m\ttile_n\ttile_k\tstages\tsplit_k\tl05_ms\tl1_ms\tl2_ms\tstatus\tl05_hash\tsmem\tctas_per_sm\tgrid\n'
if [[ -s "${raw}/screen_gqa2.tsv" && -s "${raw}/screen_mha4.tsv" && -s "${raw}/screen_summary.txt" ]]; then
  echo "== screening: cached =="
  screen_s=$(awk -F'\t' '$1=="screen_wall_s"{print $2}' "${raw}/screen_summary.txt")
else
measure_start=$(date +%s.%N)
echo "== screening: ${screen_runs} runs per candidate, 2-layer GQA =="
{ printf "${header}"; screen_model gqa2 "${gqa_fixture}" "${screen_runs}"; } \
  > "${raw}/screen_gqa2.tsv"
echo "== screening: ${screen_runs} runs per candidate, 4-layer MHA =="
{ printf "${header}"; screen_model mha4 "${mha_fixture}" "${screen_runs}"; } \
  > "${raw}/screen_mha4.tsv"
screen_s=$(awk -v a="$(date +%s.%N)" -v b="${measure_start}" 'BEGIN{printf "%.2f", a-b}')
printf 'screen_wall_s\t%.2f\n' "${screen_s}" | tee "${raw}/screen_summary.txt"
fi

# ----------------------------------------------------- fresh-process final ---
# The control plus the top-K screened candidates, re-measured the way every
# other timing claim in this repo is: N fresh processes, one timed launch each.
final_start=$(date +%s.%N)
for model in gqa2 mha4; do
  fixture="${gqa_fixture}"; [[ "${model}" == mha4 ]] && fixture="${mha_fixture}"
  # `awk 'NR<=k'` and not `head`: under `pipefail` a `head` that closes the
  # pipe early kills `sort` with SIGPIPE and aborts the whole selection.
  # Top-K under *both* metrics, not one: the L0.5 and L1 rankings disagree
  # (the L0.5 winner sits at rank 12 by L1 on gqa2), and a finals set drawn
  # from one of them would silently drop the other's optimum.
  {
    awk -F'\t' 'NR>1 && $9=="PASS" && $1==128 && $2==128 && $3==16 && $4==3 && $5==1 {print $6"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5}' \
      "${raw}/screen_${model}.tsv"
    awk -F'\t' 'NR>1 && $9=="PASS" {print $6"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5}' \
      "${raw}/screen_${model}.tsv" | sort -n | awk -v k="${topk}" 'NR<=k'
    awk -F'\t' 'NR>1 && $9=="PASS" {print $7"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5}' \
      "${raw}/screen_${model}.tsv" | sort -n | awk -v k="${topk}" 'NR<=k'
  } | sort -u -k2,6 > "${raw}/final_set_${model}.txt"
  : > "${raw}/final_${model}.tsv"
  printf "${header}" > "${raw}/final_${model}.tsv"
  while read -r _ m n k s kc; do
    tag="${model}_${m}x${n}x${k}s${s}k${kc}"
    "${repo}/scripts/gpu_stat_run.sh" -n "${final_runs}" -t 120 -l "oracle_${tag}" \
      -k -o "${raw}/final/${tag}" -- "${raw}/bin/${tag}" "${fixture}" \
      > "${raw}/log/${tag}.final" 2>&1
    l05=$(grep -h '^E2E_TIME' "${raw}/final/${tag}"/run_*.log | sed -E 's/.*l05_ms=([0-9.]+).*/\1/' | median)
    l1=$(grep -h '^E2E_TIME' "${raw}/final/${tag}"/run_*.log | sed -E 's/.*l1_ms=([0-9.]+).*/\1/' | median)
    l2=$(grep -h '^E2E_TIME' "${raw}/final/${tag}"/run_*.log | sed -E 's/.* l2_ms=([0-9.]+).*/\1/' | median)
    pass=$(grep -h -c '^RESULT status=PASS' "${raw}/final/${tag}"/run_*.log | awk '{t+=$1} END{print t+0}')
    runs=$(ls "${raw}/final/${tag}"/run_*.log | wc -l)
    hash=$(grep -h '^E2E_HASH' "${raw}/final/${tag}"/run_*.log | sed -E 's/.*l05=([0-9a-f]+).*/\1/' | sort -u | paste -sd,)
    smem=$(grep -h -m1 '^E2E_RESOURCE' "${raw}/final/${tag}"/run_*.log | sed -E 's/.*smem=([0-9]+).*/\1/' | head -1)
    ctas=$(grep -h -m1 '^E2E_RESOURCE' "${raw}/final/${tag}"/run_*.log | sed -E 's/.*ctas_per_sm=([0-9]+).*/\1/' | head -1)
    grid=$(grep -h -m1 '^E2E_RESOURCE' "${raw}/final/${tag}"/run_*.log | sed -E 's/.*grid=([0-9]+).*/\1/' | head -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s/%s\t%s\t%s\t%s\t%s\n' \
      "${m}" "${n}" "${k}" "${s}" "${kc}" "${l05}" "${l1}" "${l2}" "${pass}" "${runs}" \
      "${hash}" "${smem}" "${ctas}" "${grid}" >> "${raw}/final_${model}.tsv"
  done < "${raw}/final_set_${model}.txt"
  cat "${raw}/final_${model}.tsv"
done
final_s=$(awk -v a="$(date +%s.%N)" -v b="${final_start}" 'BEGIN{printf "%.2f", a-b}')

# ------------------------------------------------------- per-stage detail ---
# Where does the delta land? Control vs. best, per stage, on both models.
for model in gqa2 mha4; do
  fixture="${gqa_fixture}"; [[ "${model}" == mha4 ]] && fixture="${mha_fixture}"
  best=$(awk -F'\t' 'NR>1 {print $6"\t"$1"x"$2"x"$3"s"$4"k"$5}' "${raw}/final_${model}.tsv" | sort -n | awk 'NR==1{print $2}')
  for tag in "128x128x16s3k1" "${best}"; do
    TILEMEGA_STAGE_PROFILE=1 "${raw}/bin/${model}_${tag}" "${fixture}" \
      | grep '^E2E_STAGE' > "${raw}/stages_${model}_${tag}.txt" || true
  done
  echo "${best}" > "${raw}/best_${model}.txt"
done

printf 'tier1_build_s\t%.2f\ntier3_compile_wall_s\t%.2f\nscreen_wall_s\t%.2f\nfinal_wall_s\t%.2f\n' \
  "${probe_build}" "${compile_s}" "${screen_s}" "${final_s}" > "${raw}/wall_time.txt"
cat "${raw}/wall_time.txt"
echo PASS > "${raw}/run_status.txt"
