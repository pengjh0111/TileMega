#!/bin/bash
# R2-G reproduction script: N-way branch dispatch resource-usage scan (N=3,5,8).
set -e
cd "$(dirname "$0")"

CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras

for n in 3 5 8; do
  echo "=== N=$n ==="
  python3 gen_branches.py "$n" "branch_dispatch_n${n}" > "branch_dispatch_n${n}.mlir"
  "$CUDA_TILE_OPT" "branch_dispatch_n${n}.mlir" > "verify_n${n}.log" 2>&1
  echo "verify exit=$?"
  "$CUDA_TILE_TRANSLATE" "branch_dispatch_n${n}.mlir" \
    --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module \
    -o "branch_dispatch_n${n}.tilebc" > "translate_n${n}.log" 2>&1
  echo "translate exit=$?"
  timeout 600 "$TILEIRAS" --gpu-name sm_120 "branch_dispatch_n${n}.tilebc" \
    -o "branch_dispatch_n${n}.cubin" > "tileiras_n${n}.log" 2>&1
  echo "tileiras exit=$? (N=5/8 can take well over 2 minutes)"
  cuobjdump --dump-resource-usage "branch_dispatch_n${n}.cubin"
done

echo "=== N=2 baseline (from round 1 E2, for comparison) ==="
cuobjdump --dump-resource-usage ../E2_heterogeneous_tiles/heterogeneous_dispatch.cubin
