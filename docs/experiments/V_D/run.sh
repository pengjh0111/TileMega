#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
mkdir -p "$here/raw"

start_ns=$(date +%s%N)
"$nvcc" -std=c++17 -O0 -arch=sm_89 \
  -I"$root/third_party/cutlass/include" -I"$root/third_party/cutlass/test" \
  "$here/traits_probe.cu" -o "$here/traits_probe" \
  2>"$here/raw/traits_build.txt"
end_ns=$(date +%s%N)
awk -v a="$start_ns" -v b="$end_ns" 'BEGIN {printf "%.6f\n", (b-a)/1e9}' \
  >"$here/raw/traits_compile_seconds.txt"
"$here/traits_probe" >"$here/raw/traits.tsv"

"$nvcc" -std=c++17 -O2 -arch=sm_89 -cubin -Xptxas=-v \
  -I"$root/third_party/cutlass/include" -I"$root/third_party/cutlass/test" \
  "$here/ptxas_compare.cu" -o "$here/raw/ptxas_compare.cubin" \
  2>"$here/raw/ptxas_compare.txt"

echo "traits_compile_seconds=$(cat "$here/raw/traits_compile_seconds.txt")"
wc -l "$here/raw/traits.tsv"
grep -E 'Function properties|Used [0-9]+ registers' "$here/raw/ptxas_compare.txt"
