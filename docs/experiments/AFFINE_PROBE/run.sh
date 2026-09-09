#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
build="${BUILD_DIR:-${repo}/build-portable}"
# Must be supplied from the intended kernel's occupancy result, not from a
# hardware maximum CTA count that ignores registers and shared memory.
resident="${RESIDENT_LIMIT:?set RESIDENT_LIMIT from measured kernel residency}"
workers="${WORKERS:-16}"
out="${OUT_DIR:-${here}/raw_mappings}"
mkdir -p "${out}"
cmake --build "${build}" --target tilemega-affine-probe -j "${JOBS:-4}"
for seq in ${SEQS:-4 128 512}; do
  "${build}/tools/tilemega-affine-probe" "${seq}" "${workers}" "${resident}" \
    > "${out}/s${seq}_w${workers}.txt" 2>&1
  rg '^(BAND|DIMENSIONS|SPAN|RESULT)' "${out}/s${seq}_w${workers}.txt"
done
