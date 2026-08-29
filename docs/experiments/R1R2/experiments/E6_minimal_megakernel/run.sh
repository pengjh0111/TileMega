#!/bin/bash
set -ex
cd "$(dirname "$0")"
CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras

$CUDA_TILE_OPT minimal_megakernel.mlir > /dev/null
$CUDA_TILE_TRANSLATE minimal_megakernel.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o minimal_megakernel.tilebc
$TILEIRAS --gpu-name sm_120 minimal_megakernel.tilebc -o minimal_megakernel.cubin
cuobjdump --dump-resource-usage minimal_megakernel.cubin

g++ host_test.cpp -o host_test -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
./host_test 200
