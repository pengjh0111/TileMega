#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-portable}"
cmake --build "${build}" --target tilemega-parametric -j "${JOBS:-4}"
TILEMEGA_ISL_AUDIT=1 "${build}/tools/tilemega-parametric" "${repo}" bf16 \
  > "${here}/input_gate_bf16.tsv" 2> "${here}/input_gate_bf16.log"
tail -3 "${here}/input_gate_bf16.log"
TILEMEGA_ISL_AUDIT=1 "${build}/tools/tilemega-parametric" "${repo}" f32 \
  > "${here}/input_gate.tsv" 2> "${here}/input_gate.log"
tail -3 "${here}/input_gate.log"
