#!/bin/bash
# E2: heterogeneous tile dispatch experiment — exact reproduction commands.
set -ex
cd "$(dirname "$0")"

CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras
CUOBJDUMP=/usr/local/cuda-13.1/bin/cuobjdump

for name in heterogeneous_dispatch branch_a_only branch_b_only; do
  $CUDA_TILE_OPT ${name}.mlir > /dev/null
  $CUDA_TILE_TRANSLATE ${name}.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o ${name}.tilebc
  $TILEIRAS --gpu-name sm_120 ${name}.tilebc -o ${name}.cubin
  $CUOBJDUMP --dump-resource-usage ${name}.cubin
done

g++ host_test.cpp -o host_test -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
./host_test
