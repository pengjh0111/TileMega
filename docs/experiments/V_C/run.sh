#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
nvcc=${CUDACXX:-/usr/local/cuda/bin/nvcc}
mkdir -p "$here/raw"
: >"$here/raw/crosscompile.txt"
for arch in 90 120; do
  if "$nvcc" -std=c++17 -arch="sm_${arch}" -cubin -Xptxas=-v \
      -I"$root/include" -I"$root/third_party/cutlass/include" \
      "$here/cluster_dsmem.cu" -o "$here/raw/sm_${arch}.cubin" \
      2>"$here/raw/sm_${arch}.ptxas.txt"; then
    echo "sm_${arch} PASS" | tee -a "$here/raw/crosscompile.txt"
  else
    echo "sm_${arch} FAIL" | tee -a "$here/raw/crosscompile.txt"
    exit 1
  fi
done
