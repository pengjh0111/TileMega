#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
mkdir -p "$here/raw"
"$nvcc" -std=c++17 -O3 -arch=sm_89 -lineinfo -Xptxas=-v \
  -I"$root/third_party/cutlass/include" -I"$root/third_party/cutlass/test" \
  -I"$root/third_party/cutlass/tools/util/include" \
  "$here/collective_persistent.cu" -o "$here/collective_persistent" \
  2>"$here/raw/build.ptxas.txt"
"$here/collective_persistent" 2048 20 | tee "$here/raw/performance.txt"

for arch in 90 120; do
  "$nvcc" -std=c++17 -O3 -arch="sm_${arch}" -cubin -Xptxas=-v \
    -I"$root/include" -I"$root/third_party/cutlass/include" \
    "$here/builder_probe.cu" -o "$here/raw/builder_sm${arch}.cubin" \
    2>"$here/raw/builder_sm${arch}.ptxas.txt"
done
