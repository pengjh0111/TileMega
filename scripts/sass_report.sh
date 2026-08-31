#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cubin-or-executable>" >&2
  exit 2
fi

dump=$(mktemp)
trap 'rm -f "$dump"' EXIT
nvdisasm_bin=${NVDISASM:-$(command -v nvdisasm || true)}
cuobjdump_bin=${CUOBJDUMP:-$(command -v cuobjdump || true)}
[[ -n "$nvdisasm_bin" ]] || nvdisasm_bin=/usr/local/cuda-12.8/bin/nvdisasm
[[ -n "$cuobjdump_bin" ]] || cuobjdump_bin=/usr/local/cuda-12.8/bin/cuobjdump
"$nvdisasm_bin" -g "$1" >"$dump" 2>/dev/null || "$cuobjdump_bin" --dump-sass "$1" >"$dump"

echo "SASS_REPORT file=$1"
echo "BRANCHES"
# Include suffixed branch forms such as BRA.U; do not impose a distance limit.
rg -n '\bBRA(?:\.[A-Z]+)*\b' "$dump" || true
echo "BARRIERS"
rg -n '\b(?:BAR|DEPBAR|MEMBAR)(?:\.[A-Z]+)*\b' "$dump" || true
echo "NANOSLEEP"
rg -n '\bNANOSLEEP(?:\.[A-Z]+)*\b' "$dump" || true
echo "ATOMICS_AND_STRONG_GLOBALS"
# ptxas can strength-reduce atomicAdd(ptr, 0) to an ordered LDG and an
# uncontended atomicExch to an ordered STG; retain both in the evidence.
rg -n '\b(?:(?:ATOM|RED)[A-Z]*(?:\.[A-Z0-9]+)*|(?:LDG|STG)(?:\.[A-Z0-9]+)*\.STRONG(?:\.[A-Z0-9]+)*)\b' "$dump" || true
