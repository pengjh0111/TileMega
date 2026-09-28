// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/EventSync.cuh>
#include <cassert>
#include <cstdio>

using namespace tilemega::codegen;
__global__ void WaitForever(unsigned long long const* counter,WatchdogRecord* record) {
  Watch watch{record,50000000ull,3,7,8,9,0,10,11};
  WaitAtLeast(counter,2,watch);
}
int main() {
  WatchdogRecord* host=nullptr;WatchdogRecord* device=nullptr;
  assert(cudaHostAlloc(&host,sizeof(*host),cudaHostAllocMapped)==cudaSuccess);
  *host={};
  assert(cudaHostGetDevicePointer(&device,host,0)==cudaSuccess);
  unsigned long long* counter=nullptr;
  assert(cudaMalloc(&counter,sizeof(*counter))==cudaSuccess);
  assert(cudaMemset(counter,0,sizeof(*counter))==cudaSuccess);
  WaitForever<<<1,1>>>(counter,device);
  auto result=cudaDeviceSynchronize();
  assert(result!=cudaSuccess);
  assert(host->fired==1 && host->site==3 && host->need==2 && host->value==0);
  std::puts("serving watchdog PASS");
  return 0;
}
