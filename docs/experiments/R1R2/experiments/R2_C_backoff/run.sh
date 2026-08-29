#!/bin/bash
# R2-C reproduction script: spin-wait backoff (LCG delay tiers 64/256/1024) + combined
# relaxed+backoff256 variant. SASS verification + grid-scan hang-rate evaluation.
set -e
cd "$(dirname "$0")"
CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras

echo "=== Step 1: compile all 4 backoff variants ==="
for v in spin_wait_backoff64 spin_wait_backoff256 spin_wait_backoff1024 spin_wait_relaxed_backoff256; do
  echo "--- $v ---"
  $CUDA_TILE_OPT $v.mlir > verify_$v.log 2>&1
  $CUDA_TILE_TRANSLATE $v.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o $v.tilebc > translate_$v.log 2>&1
  $TILEIRAS --gpu-name sm_120 $v.tilebc -o $v.cubin > tileiras_$v.log 2>&1
  cuobjdump --dump-resource-usage $v.cubin
done

echo "=== Step 2: SASS -- confirm LCG delay loop is NOT constant-folded, poll frequency unchanged ==="
for v in backoff64 backoff256 backoff1024; do
  cuobjdump -sass spin_wait_$v.cubin > $v.sass
  echo "--- $v: LCG constants present ---"
  grep -n "41c64e6d\|0x3039" $v.sass | head -3
  echo "--- $v: BAR.SYNC / CCTL.IVALL static counts (should be 85 / 226 across all tiers) ---"
  grep -c "BAR.SYNC" $v.sass
  grep -c "CCTL.IVALL" $v.sass
done

echo "=== Step 3: grid-scan hang-rate statistics (4 variants x grid in {30,170} x 30 runs x 5s timeout) ==="
g++ -O2 -I/usr/local/cuda-13.1/include ../R2_B_relaxed_plus_acquire/host_test_single.cpp -o host_test_single -lcuda
./grid_scan.sh /tmp/r2c_gridscan.log 5 30
grep "^SUMMARY" /tmp/r2c_gridscan.log
