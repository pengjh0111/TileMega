// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_occupancy.h>
#include <cassert>
#include <iostream>
#include <tuple>
#include <initializer_list>

int main() {
  cudaOccDeviceProp device{};
  device.computeMajor=9;device.computeMinor=0;device.warpSize=32;
  device.regsPerBlock=65536;device.regsPerMultiprocessor=65536;
  for(auto [threads,registers,expected]:{
      std::tuple{288,168,1},std::tuple{288,169,0},std::tuple{288,179,0},
      std::tuple{288,224,0},std::tuple{256,224,1}}) {
    cudaOccFuncAttributes function{};function.numRegs=registers;
    cudaOccResult result{};auto mode=PARTITIONED_GC_OFF;int blocks=-1;
    assert(cudaOccMaxBlocksPerSMRegsLimit(&blocks,&mode,&result,&device,
        &function,threads)==CUDA_OCC_SUCCESS);
    assert(blocks==expected);
  }
  std::cout<<"F3 Hopper register occupancy: five launch limits verified\n";
}
