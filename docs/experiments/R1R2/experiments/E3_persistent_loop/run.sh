#!/bin/bash
set -ex
cd "$(dirname "$0")"
CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras

for f in persistent_loop non_persistent; do
  $CUDA_TILE_OPT ${f}.mlir > /dev/null
  $CUDA_TILE_TRANSLATE ${f}.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o ${f}.tilebc
  $TILEIRAS --gpu-name sm_120 ${f}.tilebc -o ${f}.cubin
done

g++ host_test.cpp -o host_test -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
./host_test
