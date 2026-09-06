#!/usr/bin/env bash
# Part 4 -- a production-shaped model end to end.
#
#   bash docs/experiments/REALMODEL/run.sh            # Llama-3.2-1B shape
#   LAYERS=4 HIDDEN=4096 bash .../run.sh              # width-only variant
#
# Every stage is timed and recorded, because "how long does the compiler take
# on a real graph" is one of the questions this experiment exists to answer.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-phase12}"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
raw="${here}/raw"
runs="${RUNS:-50}"
layers="${LAYERS:-16}"
hidden="${HIDDEN:-2048}"
intermediate="${INTERMEDIATE:-8192}"
heads="${HEADS:-32}"
kv_heads="${KV_HEADS:-8}"
seq="${FIXTURE_SEQ:-4}"
past="${FIXTURE_PAST:-3}"
label="${LABEL:-l${layers}h${hidden}}"
mkdir -p "${raw}"
work="${raw}/${label}"
mkdir -p "${work}"

command -v python3 >/dev/null || { echo 'FAIL: python3 not found' >&2; exit 77; }
python3 -c 'import torch' >/dev/null 2>&1 || {
  echo 'FAIL: PyTorch is required to export the model' >&2; exit 77; }
[[ -x "${nvcc}" ]] || { echo "FAIL: no nvcc at ${nvcc}" >&2; exit 77; }

stamp() { date +%s; }
row() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "${raw}/timing_${label}.tsv"; }
printf 'stage\tseconds\tnote\n' > "${raw}/timing_${label}.tsv"

t=$(stamp)
python3 "${here}/export_real.py" --repo "${repo}" --out "${work}/export" \
  --dtype bf16 --layers "${layers}" --hidden "${hidden}" \
  --intermediate "${intermediate}" --heads "${heads}" --kv-heads "${kv_heads}" \
  --fixture-seq "${seq}" --fixture-past "${past}" > "${work}/export.json"
row export "$(( $(stamp) - t ))" "$(cat "${work}/export.json")"

t=$(stamp)
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${work}/export/exported_program.pt2" --out "${work}/model.json" \
  > "${work}/bridge.json"
row bridge "$(( $(stamp) - t ))" "$(tail -1 "${work}/bridge.json")"

t=$(stamp)
"${build}/tools/tilemega-compile" "${work}/model.json" "${work}/model.cu" \
  --variants "${repo}/docs/experiments/OWNERSHIP/plan_structured.json" \
  > "${work}/codegen.log" 2>&1
row codegen "$(( $(stamp) - t ))" "$(grep -o 'CODEGEN_SUMMARY.*' "${work}/codegen.log" || true)"

t=$(stamp)
"${nvcc}" "${work}/model.cu" -std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v \
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
  -I"${repo}/third_party/cutlass/tools/util/include" \
  -I"${repo}/third_party/cutlass/test" \
  "${build}/libtilemega.a" -L/usr/local/cuda/lib64 -lcudart \
  -o "${work}/model" 2> "${work}/ptxas.log"
row nvcc "$(( $(stamp) - t ))" "$(grep -o 'Used [0-9]* registers' "${work}/ptxas.log" | sort -u | paste -sd, -)"

# Size of what the generator emitted, which is the other half of "does it scale".
printf 'metric\tvalue\n' > "${raw}/size_${label}.tsv"
{
  printf 'model_json_bytes\t%s\n' "$(stat -c%s "${work}/model.json")"
  printf 'generated_cu_bytes\t%s\n' "$(stat -c%s "${work}/model.cu")"
  printf 'generated_cu_lines\t%s\n' "$(wc -l < "${work}/model.cu")"
  printf 'dependency_rows\t%s\n' "$(grep -c '{ *[0-9]\+u\?,' "${work}/model.cu" || true)"
  for map in kAll kIdentity kWindow; do
    printf 'map_%s\t%s\n' "$map" "$(grep -c "Map::${map}" "${work}/model.cu" || true)"
  done
} >> "${raw}/size_${label}.tsv"

printf 'model\tpasses\tprocesses\tl05_ms\tl1_ms\tl2_ms\thash\n' \
  > "${raw}/correctness_${label}.tsv"
log="${work}/fresh.txt"; : > "${log}"
for _ in $(seq 1 "${runs}"); do
  "${work}/model" "${work}/export/fixture" >> "${log}" 2>&1 || true
done
median() { sort -n | awk '{v[NR]=$1} END{if(!NR){print "nan";exit} printf "%.6f",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${label}" \
  "$(grep -c '^RESULT status=PASS' "${log}" || true)" "${runs}" \
  "$(sed -n 's/^E2E_TIME l05_ms=\([0-9.]*\).*/\1/p' "${log}" | median)" \
  "$(sed -n 's/.* l1_ms=\([0-9.]*\).*/\1/p' "${log}" | median)" \
  "$(sed -n 's/.* l2_ms=\([0-9.]*\).*/\1/p' "${log}" | median)" \
  "$(sed -n 's/^E2E_HASH l05=\([0-9a-f]*\).*/\1/p' "${log}" | sort -u | paste -sd, -)" \
  | tee -a "${raw}/correctness_${label}.tsv"
cat "${raw}/timing_${label}.tsv" "${raw}/size_${label}.tsv"
