// SPDX-License-Identifier: BSD-3-Clause
// Isolate the full/empty page handshake from GEMM compute and DRAM traffic.
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#ifndef BENCH_PAGE_BYTES
#error BENCH_PAGE_BYTES is required
#endif
#ifndef BENCH_PAGE_COUNT
#error BENCH_PAGE_COUNT is required
#endif
#ifndef BENCH_POOL_OFFSET
#error BENCH_POOL_OFFSET is required
#endif
#ifndef BENCH_SHARED_BYTES
#error BENCH_SHARED_BYTES is required
#endif

using Ring=tilemega::codegen::executor::PageRing<BENCH_PAGE_BYTES,
    BENCH_PAGE_COUNT,tilemega::arch::Sm80>;

__global__ __launch_bounds__(160,1)
void Handshake(unsigned char const* source,int rounds) {
  extern __shared__ __align__(1024) char shared[];
  Ring ring{reinterpret_cast<Ring::Slot*>(shared),shared+BENCH_POOL_OFFSET};
  ring.Initialize();
  for(std::uint64_t sequence=0;sequence<std::uint64_t(rounds);++sequence) {
    if(tilemega::codegen::executor::IsCompute()) {
      ring.AwaitFull(sequence);
      ring.Release(sequence);
    }else {
      ring.AcquireEmpty(sequence);
      auto* dest=ring.Page(sequence)+tilemega::codegen::executor::LoaderLane()*16;
      Ring::Copy::Copy16(dest,source+tilemega::codegen::executor::LoaderLane()*16);
      ring.PublishCopies(sequence);
    }
  }
}

static void Check(cudaError_t result) {
  if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
int main() try {
  int sms=0;Check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
  unsigned char* source=nullptr;Check(cudaMalloc(&source,512));Check(cudaMemset(source,0,512));
  Check(cudaFuncSetAttribute(Handshake,cudaFuncAttributeMaxDynamicSharedMemorySize,
                             BENCH_SHARED_BYTES));
  cudaEvent_t begin,end;Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
  int constexpr rounds=1024;
  std::vector<float> samples;
  for(int trial=0;trial<7;++trial) {
    Check(cudaEventRecord(begin));
    Handshake<<<sms,160,BENCH_SHARED_BYTES>>>(source,rounds);
    Check(cudaGetLastError());Check(cudaEventRecord(end));Check(cudaEventSynchronize(end));
    float ms=0;Check(cudaEventElapsedTime(&ms,begin,end));
    if(trial)samples.push_back(ms);
  }
  std::sort(samples.begin(),samples.end());
  double median=(samples[2]+samples[3])/2;
  std::printf("page_bytes=%d pages=%d shared_bytes=%d sms=%d rounds=%d median_ms=%.6f ns_per_page=%.3f\n",
      BENCH_PAGE_BYTES,BENCH_PAGE_COUNT,BENCH_SHARED_BYTES,sms,rounds,
      median,median*1e6/rounds);
  Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));Check(cudaFree(source));
  return 0;
}catch(std::exception const& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
