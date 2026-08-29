#!/bin/bash
# R2-D reproduction script: occupancy audit (REG=228 vs driver's occupancy claims).
set -e
cd "$(dirname "$0")"

echo "=== Step 1: confirm EIATTR_REQNTID / EIATTR_REGCOUNT on the tileiras cubin ==="
cuobjdump -elf ../E4_spin_wait/variants/spin_wait_tokenchain.cubin | grep -A2 "EIATTR_REQNTID\|EIATTR_REGCOUNT"

echo "=== Step 2: occupancy scan across blockSize {1,32,128,256} on the tileiras cubin ==="
g++ -O2 -I/usr/local/cuda-13.1/include occ_probe.cpp -o occ_probe -lcuda
./occ_probe ../E4_spin_wait/variants/spin_wait_tokenchain.cubin spin_wait_tokenchain

echo "=== Step 3: build nvcc control-group kernel with comparable REG count ==="
nvcc -arch=sm_120 -maxrregcount=255 -cubin reg_pressure_kernel_v4.cu -o reg_pressure_kernel_v4.cubin
cuobjdump -elf reg_pressure_kernel_v4.cubin | grep -A2 EIATTR_REGCOUNT
./occ_probe reg_pressure_kernel_v4.cubin reg_pressure_kernel

echo "=== Step 4: confirm host_test_coop.cpp uses blockSize=1 / blockDim=(1,1,1) mismatch ==="
grep -n "cuOccupancyMaxActiveBlocksPerMultiprocessor\|cuLaunchCooperativeKernel" ../E4_spin_wait/variants/host_test_coop.cpp

echo "=== Step 5: re-test cooperative launch admission with CORRECT blockDim=256 ==="
g++ -O2 -I/usr/local/cuda-13.1/include host_test_coop_correct.cpp -o host_test_coop_correct -lcuda
for g in 170 171 340; do
  echo "--- grid=$g ---"
  timeout 8 ./host_test_coop_correct $g ../E4_spin_wait/variants/spin_wait_tokenchain.cubin spin_wait_tokenchain
  echo "rc=$?"
done
