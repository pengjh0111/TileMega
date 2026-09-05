#!/usr/bin/env bash
# Reproduce the RoPE-only decision experiment and the promoted ownership plan
# on both accepted BF16 models. Every comparison is 25 paired fresh processes.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-phase12}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
raw="${here}/raw"
mkdir -p "${raw}"/{export,fixture,src,bin,log}
cmake --build "${build}" --target tilemega-compile --parallel "$(nproc)" >/dev/null

python3 "${repo}/docs/experiments/V_H/export_probe.py" \
  --out "${raw}/export/gqa2" --dtype bf16 --seq-max 2048 --past-max 512 >/dev/null
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/gqa2/exported_program.pt2" --out "${raw}/export/gqa2.json" >/dev/null
python3 "${repo}/docs/experiments/P3_GENERALIZATION/export_second.py" \
  --repo "${repo}" --out "${raw}/export/mha4" --dtype bf16 \
  --seq-max 2048 --past-max 512 --fixture-seq 4 --fixture-past 3 >/dev/null
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/export/mha4/exported_program.pt2" --out "${raw}/export/mha4.json" >/dev/null
for seq in 4 128; do
  python3 "${repo}/docs/experiments/E2E/prepare_e2e.py" \
    --vh-raw "${raw}/export/gqa2" --out "${raw}/fixture/gqa2_s${seq}" \
    --seq "${seq}" --past 3 >/dev/null
  python3 "${repo}/docs/experiments/P3_GENERALIZATION/prepare_fixture.py" \
    --repo "${repo}" --program "${raw}/export/mha4/exported_program.pt2" \
    --out "${raw}/fixture/mha4_s${seq}" --seq "${seq}" --past 3 >/dev/null
done

common=(-std=c++17 -O2 -arch=native -lineinfo -DTILEMEGA_EVENT_KAPPA=1
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include"
  -I"${repo}/third_party/cutlass/tools/util/include"
  -I"${repo}/third_party/cutlass/test")
for model in gqa2 mha4; do
  for arm in chunk rope structured; do
    plan="${here}/plan_element_chunk.json"
    [[ "${arm}" == rope ]] && plan="${here}/plan_tile_per_block.json"
    [[ "${arm}" == structured ]] && plan="${here}/plan_structured.json"
    "${build}/tools/tilemega-compile" "${raw}/export/${model}.json" \
      "${raw}/src/${model}_${arm}.cu" --variants "${plan}"
    "${nvcc}" "${common[@]}" "${raw}/src/${model}_${arm}.cu" \
      "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart \
      -o "${raw}/bin/${model}_${arm}"
  done
done

paired_run() {
  local seq="$1" changed="$2" destination="$3"
  for model in gqa2 mha4; do
    for round in $(seq 1 25); do
      order=(chunk tile); (( round % 2 == 0 )) && order=(tile chunk)
      for label in "${order[@]}"; do
        arm=chunk; [[ "${label}" == tile ]] && arm="${changed}"
        dir="${destination}/${model}_${label}/r${round}"; mkdir -p "${dir}"
        "${raw}/bin/${model}_${arm}" "${raw}/fixture/${model}_s${seq}" \
          > "${dir}/run_1.log"
        grep -q '^RESULT status=PASS' "${dir}/run_1.log"
      done
    done
  done
  python3 "${here}/summarize.py" "${destination}"
}
paired_run 128 rope "${raw}/rope_s128" | tee "${raw}/rope_only_s128.tsv"
for seq in 4 128; do
  paired_run "${seq}" structured "${raw}/structured_s${seq}" \
    | tee "${raw}/paired_s${seq}.tsv"
done

for arm in chunk rope structured; do
  TILEMEGA_WAIT_PROFILE=128 "${raw}/bin/gqa2_${arm}" \
    "${raw}/fixture/gqa2_s128" > "${raw}/log/${arm}_wait_profile.txt"
done
python3 - "${raw}" <<'PY'
import pathlib, re, sys
root=pathlib.Path(sys.argv[1])
print('ownership\tseq\tall_edges\tidentity_edges\twindow_edges\tpolls\tfully_relaxed_polls\tpoll_ratio')
for arm in ('chunk','rope','structured'):
    maps={0:0,1:0,2:0}; polls=relaxed=0
    for line in (root/'log'/f'{arm}_wait_profile.txt').read_text().splitlines():
        if not line.startswith('E2E_WAITSET'): continue
        p=dict(re.findall(r'(\w+)=([^ ]+)',line)); maps[int(p['map'])]+=1
        polls+=int(p['polls']); relaxed+=int(p['relaxed'])
    print(f'{arm}\t128\t{maps[1]}\t{maps[0]}\t{maps[2]}\t{polls}\t{relaxed}\t{polls/relaxed:.6f}')
PY
