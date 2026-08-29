#!/bin/bash
# E4: cross-tile-block producer/consumer spin-wait — exact reproduction commands.
set -ex
cd "$(dirname "$0")"

CUDA_TILE_OPT=/data/cuda-tile/build/bin/cuda-tile-opt
CUDA_TILE_TRANSLATE=/data/cuda-tile/build/bin/cuda-tile-translate
TILEIRAS=/usr/local/cuda-13.1/bin/tileiras
CUOBJDUMP=/usr/local/cuda-13.1/bin/cuobjdump

# --- main experiment ---
$CUDA_TILE_OPT spin_wait_test.mlir > /dev/null
$CUDA_TILE_TRANSLATE spin_wait_test.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o spin_wait_test.tilebc
$TILEIRAS --gpu-name sm_120 spin_wait_test.tilebc -o spin_wait_test.cubin
$CUOBJDUMP --dump-resource-usage spin_wait_test.cubin
$CUOBJDUMP -sass spin_wait_test.cubin > spin_wait_test.sass

g++ host_test.cpp -o host_test -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
./host_test 1000

# --- ordering/scope variants (relaxed instead of acquire; tl_blk instead of device) ---
cd variants
for name in spin_wait_relaxed spin_wait_tlblk; do
  $CUDA_TILE_OPT ${name}.mlir > /dev/null
  $CUDA_TILE_TRANSLATE ${name}.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o ${name}.tilebc
  $TILEIRAS --gpu-name sm_120 ${name}.tilebc -o ${name}.cubin
done
g++ host_test_variant.cpp -o host_test_variant -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
./host_test_variant spin_wait_relaxed.cubin spin_wait_relaxed 200
./host_test_variant spin_wait_tlblk.cubin spin_wait_tlblk 200

# --- token-chained fix (the version that actually works) ---
cd variants
$CUDA_TILE_OPT spin_wait_tokenchain.mlir > /dev/null
$CUDA_TILE_TRANSLATE spin_wait_tokenchain.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o spin_wait_tokenchain.tilebc
$TILEIRAS --gpu-name sm_120 spin_wait_tokenchain.tilebc -o spin_wait_tokenchain.cubin
$CUOBJDUMP -sass spin_wait_tokenchain.cubin > spin_wait_tokenchain.sass
g++ host_test_tokenchain.cpp -o host_test_tokenchain -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda -lcudart
./host_test_tokenchain spin_wait_tokenchain.cubin spin_wait_tokenchain 1000

# --- large-grid deadlock test (WARNING: this deliberately hangs for grid>=~120; always wrap in timeout) ---
g++ host_test_biggrid_tokenchain.cpp -o host_test_biggrid_tokenchain -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda
for g in 4 10 32 50 80 120 150 170; do
  timeout 20 ./host_test_biggrid_tokenchain $g || echo "grid=$g: TIMED OUT (deadlock)"
done

# --- cooperative launch does not save us either ---
g++ host_test_coop.cpp -o host_test_coop -I/usr/local/cuda-13.1/include -L/usr/local/cuda-13.1/lib64 -lcuda
timeout 20 ./host_test_coop 170 || echo "cooperative launch grid=170: TIMED OUT (deadlock)"
