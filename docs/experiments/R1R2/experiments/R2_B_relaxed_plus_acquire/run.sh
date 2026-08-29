#!/bin/bash
# R2-B reproduction script: relaxed test-and-test-and-set spin-wait fix, SASS + correctness
# + grid-scan hang-rate evaluation.
set -e
cd "$(dirname "$0")"
CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras

echo "=== Step 1: compile relaxed_acquire variant (1MB) ==="
$CUDA_TILE_OPT spin_wait_relaxed_acquire.mlir > /dev/null
$CUDA_TILE_TRANSLATE spin_wait_relaxed_acquire.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o spin_wait_relaxed_acquire.tilebc
$TILEIRAS --gpu-name sm_120 spin_wait_relaxed_acquire.tilebc -o spin_wait_relaxed_acquire.cubin
cuobjdump --dump-resource-usage spin_wait_relaxed_acquire.cubin

echo "=== Step 2: SASS check -- CCTL.IVALL removed from poll loop, BAR.SYNC still present ==="
cuobjdump -sass spin_wait_relaxed_acquire.cubin > spin_wait_relaxed_acquire.sass
grep -n -A6 "/\*01f0\*/" spin_wait_relaxed_acquire.sass | head -8

echo "=== Step 3: correctness (small grid=2, many iterations, fresh state per iteration) ==="
g++ -O2 -I/usr/local/cuda-13.1/include host_test_correctness.cpp -o host_test_correctness -lcuda
./host_test_correctness spin_wait_relaxed_acquire.cubin spin_wait_relaxed_acquire 1000
g++ -O2 -I/usr/local/cuda-13.1/include host_test_correctness_4mb.cpp -o host_test_correctness_4mb -lcuda
./host_test_correctness_4mb spin_wait_relaxed_acquire_4mb.cubin spin_wait_relaxed_acquire_4mb 200

echo "=== Step 4: grid-scan hang-rate statistics (relaxed_acquire vs baseline_acquire_in_loop) ==="
g++ -O2 -I/usr/local/cuda-13.1/include host_test_single.cpp -o host_test_single -lcuda
./grid_scan.sh /tmp/r2b_gridscan.log 6 50
grep "^SUMMARY" /tmp/r2b_gridscan.log
