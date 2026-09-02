#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${repo}/build-portable"
raw="${here}/raw"
mkdir -p "${raw}"
export PYTHONPATH="${repo}/.venv/site${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${here}/export_second.py" --repo "${repo}" --out "${raw}" \
  | tee "${raw}/export_summary.txt"
python3 "${repo}/python/tilemega/export_bridge.py" \
  "${raw}/exported_program.pt2" --out "${raw}/export_bridge.json" \
  | tee "${raw}/bridge_summary.txt"
"${build}/tools/tilemega-import" "${raw}/export_bridge.json" \
  > "${raw}/cg.mlir" 2> "${raw}/import_summary.txt"
"${build}/tools/tilemega-opt" "${raw}/cg.mlir" > "${raw}/cg_roundtrip.mlir"
"${build}/tools/tilemega-compile" "${raw}/cg_roundtrip.mlir" \
  "${raw}/generated.cu" 2> "${raw}/codegen_summary.txt"

nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
"${nvcc}" -std=c++17 -O2 -arch=native -lineinfo -Xptxas=-v \
  -I"${repo}/include" -I"${repo}/third_party/cutlass/include" \
  -I"${repo}/third_party/cutlass/tools/util/include" \
  -I"${repo}/third_party/cutlass/test" "${raw}/generated.cu" \
  "${build}/libtilemega.a" -lcudart -o "${here}/generated" \
  2> "${raw}/ptxas.txt"
timeout 60 "${here}/generated" "${raw}/fixture" | tee "${raw}/run.txt"

if grep -nE '% 12|TILEMEGA_GENERATED_TASK_COUNT 179|TILEMEGA_GENERATED_COUPLING_COUNT 222|GeneratedLlamaRuntime' \
    "${raw}/generated.cu" > "${raw}/forbidden.txt"; then
  echo "generated source contains model-specific control flow" >&2
  exit 1
fi
echo PASS > "${raw}/status.txt"
