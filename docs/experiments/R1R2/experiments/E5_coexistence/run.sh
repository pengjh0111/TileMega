#!/bin/bash
set -ex
cd "$(dirname "$0")"
CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras

# Build the two optimization_hints variants derived from E4's working spin_wait_tokenchain.mlir
for f in spin_wait_occ1 spin_wait_cga1; do
  $CUDA_TILE_OPT ${f}.mlir > /dev/null
  $CUDA_TILE_TRANSLATE ${f}.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o ${f}.tilebc
  $TILEIRAS --gpu-name sm_120 ${f}.tilebc -o ${f}.cubin
  cuobjdump --dump-resource-usage ${f}.cubin
done

g++ host_test_occ.cpp    -o host_test_occ    -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
g++ host_test_occ_v2.cpp -o host_test_occ_v2 -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
g++ host_test_occ_v3.cpp -o host_test_occ_v3 -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
g++ host_test_occ_v4.cpp -o host_test_occ_v4 -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart

echo "=== occupancy=1 hint, grid=170 (still expected to hang, wrap with timeout) ==="
timeout 20 ./host_test_occ spin_wait_occ1.cubin spin_wait_occ1 170 || true
echo "=== num_cta_in_cga=1 hint, grid=170 (still expected to hang) ==="
timeout 20 ./host_test_occ spin_wait_cga1.cubin spin_wait_cga1 170 || true

echo "=== race-condition demonstration: same cubin/grid, with vs without GPU warm-up memset ==="
CUBIN=../E4_spin_wait/variants/spin_wait_tokenchain.cubin
timeout 12 ./host_test_occ_v3 $CUBIN spin_wait_tokenchain 80 || true   # with 1MB warm-up memset: succeeds
timeout 12 ./host_test_occ_v2 $CUBIN spin_wait_tokenchain 30 || true   # no warm-up: flaky/hangs
timeout 12 ./host_test_occ_v4 $CUBIN spin_wait_tokenchain 30 || true   # 50ms host sleep only: still hangs
