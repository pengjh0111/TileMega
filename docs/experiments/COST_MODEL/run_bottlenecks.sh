#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# T2.1: read the existing dtype-specific ORACLE sweeps; no GPU required.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-portable}"
out="${OUT_DIR:-${here}/t2_before}"
fp32="${repo}/docs/experiments/ORACLE/raw"
bf16="${repo}/docs/experiments/ORACLE/raw_bf16"
cmake --build "${build}" --target tilemega-costmodel check-policy -j "${JOBS:-4}"
mkdir -p "${out}/f32" "${out}/bf16"
"${build}/tools/tilemega-costmodel" --repo "${repo}" --histogram-only \
  --out "${out}/f32" --screen-dir "${fp32}" \
  --register-dir "${here}/raw" | tee "${out}/f32/run.txt"
"${build}/tools/tilemega-costmodel" --repo "${repo}" --histogram-only \
  --dtype bf16 --out "${out}/bf16" --screen-dir "${bf16}" \
  --register-dir "${bf16}/cost" \
  --gqa-cu "${bf16}/src/gqa2_128x128x16s3k1.cu" \
  --mha-cu "${bf16}/src/mha4_128x128x16s3k1.cu" | tee "${out}/bf16/run.txt"
(
  cd "${repo}"
  sha256sum configs/targets/sm_89.json \
    docs/experiments/ORACLE/raw/screen_{gqa2,mha4}.tsv \
    docs/experiments/COST_MODEL/raw/registers_{gqa2,mha4}.tsv \
    docs/experiments/E2E_GEN/raw/generated_e2e.cu \
    docs/experiments/P3_GENERALIZATION/raw/generated.cu \
    docs/experiments/ORACLE/raw_bf16/screen_{gqa2,mha4}.tsv \
    docs/experiments/ORACLE/raw_bf16/cost/registers_{gqa2,mha4}.tsv \
    docs/experiments/ORACLE/raw_bf16/src/{gqa2,mha4}_128x128x16s3k1.cu
) > "${out}/inputs.sha256"
