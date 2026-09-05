#!/usr/bin/env bash
# Exhaustive BF16 + structured-ownership oracle. Every point is regenerated
# from an exact ModelSpec plan; no compile-time -D granularity override exists.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-phase12}"
raw="${RAW_DIR:-${here}/raw_bf16}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
jobs="${JOBS:-6}"
screen_runs="${SCREEN_RUNS:-3}"
final_runs="${FINAL_RUNS:-25}"
topk="${TOPK:-8}"
splits="${SPLITS:-1 2 4 8 16}"
mkdir -p "${raw}"/{export,fixture,plan,src,bin,log,final,cost}

cmake --build "${build}" --target tilemega-compile tilemega-costmodel \
  --parallel "${jobs}" >/dev/null

# The sweep is long enough that it is resumed rather than restarted. Export is
# the one non-idempotent stage: regenerating a fixture would silently change
# what every already-compiled binary is measured against, so it is skipped when
# its outputs are already on disk.
if [[ -s "${raw}/export/gqa2.json" && -s "${raw}/export/mha4.json" \
      && -d "${raw}/fixture/gqa2" && -d "${raw}/export/mha4/fixture" ]]; then
  echo "REUSED export + fixtures"
else
python3 "${repo}/docs/experiments/V_H/export_probe.py" \
  --out "${raw}/export/gqa2" --dtype bf16 --seq-max 2048 --past-max 512 >/dev/null
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/gqa2/exported_program.pt2" --out "${raw}/export/gqa2.json" >/dev/null
python3 "${repo}/docs/experiments/E2E/prepare_e2e.py" \
  --vh-raw "${raw}/export/gqa2" --out "${raw}/fixture/gqa2" --seq 4 --past 3 >/dev/null
python3 "${repo}/docs/experiments/P3_GENERALIZATION/export_second.py" \
  --repo "${repo}" --out "${raw}/export/mha4" --dtype bf16 \
  --seq-max 2048 --past-max 512 --fixture-seq 4 --fixture-past 3 >/dev/null
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/mha4/exported_program.pt2" --out "${raw}/export/mha4.json" >/dev/null
fi

common=(-std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v
  -DTILEMEGA_EVENT_KAPPA=1 -I"${repo}/include"
  -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")

if [[ -s "${raw}/configs.txt" && -s "${raw}/tier1.tsv" ]]; then
  echo "REUSED tier-1 enumeration ($(wc -l < "${raw}/configs.txt") configs)"
else
"${nvcc}" -std=c++17 -O2 -arch=native -DTILEMEGA_MODEL_BF16=1 \
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
  "${here}/candidate_probe.cu" "${build}/libtilemega.a" \
  -L/usr/local/cuda/lib64 -lcudart -o "${raw}/candidate_probe"
"${raw}/candidate_probe" > "${raw}/candidates.txt"
grep 'fits=1' "${raw}/candidates.txt" | sed -E \
  's/^CANDIDATE m=([0-9]+) n=([0-9]+) k=([0-9]+) stages=([0-9]+).*/\1 \2 \3 \4/' \
  > "${raw}/survivors.txt"
: > "${raw}/configs.txt"
while read -r m n k s; do
  for split in ${splits}; do printf '%s %s %s %s %s\n' "$m" "$n" "$k" "$s" "$split"; done
done < "${raw}/survivors.txt" > "${raw}/configs.txt"
printf 'enumerated\t300\nshape_legal\t%s\nfits_smem\t%s\nconfigs\t%s\n' \
  "$(grep -c 'shape_legal=1' "${raw}/candidates.txt")" \
  "$(wc -l < "${raw}/survivors.txt")" "$(wc -l < "${raw}/configs.txt")" \
  > "${raw}/tier1.tsv"
fi

compile_one() {
  local model="$1" export_json="$2" m="$3" n="$4" k="$5" s="$6" split="$7"
  local tag="${model}_${m}x${n}x${k}s${s}k${split}"
  local plan="${raw}/plan/${tag}.json" src="${raw}/src/${tag}.cu"
  if [[ -x "${raw}/bin/${tag}" ]]; then
    echo "CACHED ${tag}"
    return 0
  fi
  python3 "${here}/make_plan.py" --out "${plan}" --tile-m "$m" \
    --tile-n "$n" --tile-k "$k" --stages "$s" --split-k "$split"
  if "${build}/tools/tilemega-compile" "${export_json}" "${src}" \
       --variants "${plan}" > "${raw}/log/${tag}.codegen" 2>&1 && \
     "${nvcc}" "${common[@]}" "${src}" "${build}/libtilemega.a" \
       -L/usr/local/cuda/lib64 -lcudart -o "${raw}/bin/${tag}" \
       2> "${raw}/log/${tag}.ptxas"; then
    echo "OK ${tag}"
  else
    echo "FAIL ${tag}"
  fi
}
export -f compile_one
export here repo build raw nvcc
export common_q="$(printf '%q ' "${common[@]}")"

# Exported Bash arrays do not survive xargs' child shell. Reconstruct common
# from its shell-quoted representation inside the worker.
compile_worker='eval "common=($common_q)"; compile_one "$@"'
for model in gqa2 mha4; do
  while read -r m n k s split; do
    printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0' "$model" \
      "${raw}/export/${model}.json" "$m" "$n" "$k" "$s" "$split"
  done < "${raw}/configs.txt"
done | xargs -0 -P "${jobs}" -n 7 bash -c "${compile_worker}" _ \
  >> "${raw}/compile_status.txt"

minimum() { sort -n | awk 'NR==1{printf "%.6f",$1}'; }
median() { sort -n | awk '{v[NR]=$1} END{printf "%.6f",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
header='tile_m\ttile_n\ttile_k\tstages\tsplit_k\tl05_ms\tl1_ms\tl2_ms\tstatus\thash\tsmem\tctas_per_sm\tgrid\n'
screen_model() {
  local model="$1" fixture="$2" runs="$3"
  while read -r m n k s split; do
    # `local a=$x b=$a` does not work: bash expands every word of the builtin
    # before assigning any of them, so `bin` would read an unset `tag` and
    # `set -u` aborts the whole sweep on its first configuration.
    local tag="${model}_${m}x${n}x${k}s${s}k${split}"
    local bin="${raw}/bin/${tag}"
    [[ -x "${bin}" ]] || continue
    local l05=() l1=() l2=() status=PASS output hash= smem= ctas= grid=
    for unused in $(seq 1 "${runs}"); do
      output="$(timeout 120 "${bin}" "${fixture}" 2>/dev/null)" || { status=RUNFAIL; break; }
      grep -q '^RESULT status=PASS' <<<"${output}" || status=MISMATCH
      l05+=("$(sed -n 's/^E2E_TIME l05_ms=\([0-9.]*\).*/\1/p' <<<"${output}")")
      l1+=("$(sed -n 's/.* l1_ms=\([0-9.]*\).*/\1/p' <<<"${output}")")
      l2+=("$(sed -n 's/.* l2_ms=\([0-9.]*\).*/\1/p' <<<"${output}")")
      hash="$(sed -n 's/^E2E_HASH l05=\([0-9a-f]*\).*/\1/p' <<<"${output}")"
      smem="$(sed -n 's/^E2E_RESOURCE.* smem=\([0-9]*\).*/\1/p' <<<"${output}")"
      ctas="$(sed -n 's/.*ctas_per_sm=\([0-9]*\).*/\1/p' <<<"${output}")"
      grid="$(sed -n 's/.*grid=\([0-9]*\).*/\1/p' <<<"${output}")"
    done
    if [[ "${status}" == PASS ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\n' \
        "$m" "$n" "$k" "$s" "$split" \
        "$(printf '%s\n' "${l05[@]}" | minimum)" \
        "$(printf '%s\n' "${l1[@]}" | minimum)" \
        "$(printf '%s\n' "${l2[@]}" | minimum)" "$hash" "$smem" "$ctas" "$grid"
    else
      printf '%s\t%s\t%s\t%s\t%s\tnan\tnan\tnan\t%s\t-\t-\t-\t-\n' \
        "$m" "$n" "$k" "$s" "$split" "$status"
    fi
  done < "${raw}/configs.txt"
}
for model in gqa2 mha4; do
  fixture="${raw}/fixture/gqa2"; [[ "$model" == mha4 ]] && fixture="${raw}/export/mha4/fixture"
  { printf "${header}"; screen_model "$model" "$fixture" "$screen_runs"; } \
    > "${raw}/screen_${model}.tsv"
done

# ptxas register table used by the dtype-aware cost model.
for model in gqa2 mha4; do
  { echo -e '# shape\tmax_registers';
    for f in "${raw}/log/${model}"_*.ptxas; do
      [[ -e "$f" ]] || continue
      base="${f##*/}"; base="${base%.ptxas}"; base="${base#${model}_}"
      regs="$(grep -o 'Used [0-9]* registers' "$f" | awk '{print $2}' | sort -rn | awk 'NR==1')"
      [[ -n "$regs" ]] && printf '%s\t%s\n' "${base%%k*}" "$regs"
    done | sort -u -k1,1 -k2,2rn | awk '!seen[$1]++';
  } > "${raw}/cost/registers_${model}.tsv"
done

for model in gqa2 mha4; do
  fixture="${raw}/fixture/gqa2"; [[ "$model" == mha4 ]] && fixture="${raw}/export/mha4/fixture"
  { awk -F'\t' 'NR>1 && $9=="PASS" && $1==128 && $2==128 && $3==16 && $4==3 && $5==1 {print $6"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5}' "${raw}/screen_${model}.tsv"
    awk -F'\t' 'NR>1 && $9=="PASS" {print $6"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5}' "${raw}/screen_${model}.tsv" | sort -n | awk -v k="$topk" 'NR<=k'
    awk -F'\t' 'NR>1 && $9=="PASS" {print $8"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5}' "${raw}/screen_${model}.tsv" | sort -n | awk -v k="$topk" 'NR<=k';
  } | sort -u -k2,6 > "${raw}/final_set_${model}.txt"
  printf "${header}" > "${raw}/final_${model}.tsv"
  while read -r unused m n k s split; do
    tag="${model}_${m}x${n}x${k}s${s}k${split}"; log="${raw}/final/${tag}.txt"; : > "$log"
    for run in $(seq 1 "$final_runs"); do "${raw}/bin/${tag}" "$fixture" >> "$log"; done
    pass="$(grep -c '^RESULT status=PASS' "$log" || true)"
    [[ "$pass" == "$final_runs" ]] || { echo "FAIL final ${tag} ${pass}/${final_runs}" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s/%s\t%s\t%s\t%s\t%s\n' \
      "$m" "$n" "$k" "$s" "$split" \
      "$(sed -n 's/^E2E_TIME l05_ms=\([0-9.]*\).*/\1/p' "$log" | median)" \
      "$(sed -n 's/.* l1_ms=\([0-9.]*\).*/\1/p' "$log" | median)" \
      "$(sed -n 's/.* l2_ms=\([0-9.]*\).*/\1/p' "$log" | median)" \
      "$pass" "$final_runs" \
      "$(sed -n 's/^E2E_HASH l05=\([0-9a-f]*\).*/\1/p' "$log" | sort -u | paste -sd, -)" \
      "$(sed -n 's/^E2E_RESOURCE.* smem=\([0-9]*\).*/\1/p' "$log" | head -1)" \
      "$(sed -n 's/.*ctas_per_sm=\([0-9]*\).*/\1/p' "$log" | head -1)" \
      "$(sed -n 's/.*grid=\([0-9]*\).*/\1/p' "$log" | head -1)" \
      >> "${raw}/final_${model}.tsv"
  done < "${raw}/final_set_${model}.txt"
done

"${build}/tools/tilemega-costmodel" --repo "${repo}" --dtype bf16 \
  --screen-dir "${raw}" --out "${raw}/cost" \
  --gqa-cu "${raw}/src/gqa2_128x128x16s3k1.cu" \
  --mha-cu "${raw}/src/mha4_128x128x16s3k1.cu" \
  > "${raw}/cost/model.log"
echo PASS > "${raw}/run_status.txt"
