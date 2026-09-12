#!/usr/bin/env bash
# EX-D1 §3.5: %globaltimer resolution and cross-SM spread on the local GPU.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
raw="${here}/raw"
nvcc="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
reads="${READS:-20000}"
blocks="${BLOCKS:-128}"

rounds="${ROUNDS:-256}"
spin="${SPIN_CYCLES:-20000}"

mkdir -p "${raw}/bin"
"${nvcc}" -std=c++17 -O2 -arch=native -o "${raw}/bin/resolution" \
  "${here}/resolution.cu" -lcudart
"${nvcc}" -std=c++17 -O2 -arch=native -o "${raw}/bin/calibrate" \
  "${here}/calibrate.cu" -lcudart

"${raw}/bin/resolution" "${reads}" "${blocks}" \
  "${here}/resolution.tsv" "${here}/resolution_ctas.tsv" \
  | tee "${raw}/resolution.out"

# The clock64 calibration is required only when the globaltimer tick exceeds
# 100 ns (§3.5); run it unconditionally so the decision is visible in the data.
"${raw}/bin/calibrate" "${blocks}" "${rounds}" "${spin}" \
  "${here}/calibration.tsv" | tee "${raw}/calibration.out"

printf 'commit\t%s\n' "$(cd "${repo}" && git rev-parse HEAD)" >> "${here}/resolution.tsv"
