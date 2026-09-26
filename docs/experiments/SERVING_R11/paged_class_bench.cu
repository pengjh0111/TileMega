// SPDX-License-Identifier: BSD-3-Clause
// Isolated full-grid paged GEMM stage. Shape and page layout come from a plan.
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifndef BENCH_TILE_N
#error BENCH_TILE_N is required
#endif
#ifndef BENCH_TILE_K
#error BENCH_TILE_K is required
#endif
#ifndef BENCH_PAGE_BYTES
#error BENCH_PAGE_BYTES is required
#endif
#ifndef BENCH_PAGE_COUNT
#error BENCH_PAGE_COUNT is required
#endif
#ifndef BENCH_POOL_OFFSET
#error BENCH_POOL_OFFSET is required
#endif
#ifndef BENCH_WORKSPACE_OFFSET
#error BENCH_WORKSPACE_OFFSET is required
#endif
#ifndef BENCH_SHARED_BYTES
#error BENCH_SHARED_BYTES is required
#endif
#ifndef BENCH_STANDARD_STAGES
#error BENCH_STANDARD_STAGES is required
#endif

using Body = tilemega::codegen::PagedGemmTaskBody<tilemega::arch::Sm80,16,
    BENCH_TILE_N,BENCH_TILE_K,BENCH_PAGE_BYTES,BENCH_PAGE_COUNT>;
using Ring = Body::Ring;
using StandardBody = tilemega::codegen::ServingGemmTaskBody<tilemega::arch::Sm80,
    16,BENCH_TILE_N,BENCH_TILE_K,BENCH_STANDARD_STAGES>;
using Element = cutlass::bfloat16_t;
static_assert(BENCH_WORKSPACE_OFFSET >= sizeof(Ring::Slot)*BENCH_PAGE_COUNT);
static_assert(BENCH_POOL_OFFSET >= BENCH_WORKSPACE_OFFSET +
              std::max(Body::kActivationBytes,Body::kScratchBytes));

__global__ __launch_bounds__(160,1)
void Stage(tilemega::codegen::ServingGemmOperands operands,int tiles) {
  extern __shared__ __align__(1024) char shared[];
  Ring ring{reinterpret_cast<Ring::Slot*>(shared),shared+BENCH_POOL_OFFSET};
  ring.Initialize();
  std::uint64_t sequence=0;
  for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    if(tilemega::codegen::executor::IsCompute())
      Body::Run(operands,0,tile,ring,sequence,shared+BENCH_WORKSPACE_OFFSET);
    else Body::Load(operands,tile,ring,sequence);
  }
}

__global__ __launch_bounds__(128,1)
void StageStandard(tilemega::codegen::ServingGemmOperands operands,int tiles) {
  extern __shared__ __align__(1024) char shared[];
  for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x)
    StandardBody::Run(operands,0,tile,shared);
}

static void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}

int main(int argc,char** argv) try {
  if(argc!=3)throw std::invalid_argument("usage: paged_class_bench OUTPUT_N REDUCTION_K");
  int const n=std::atoi(argv[1]),k=std::atoi(argv[2]);
  if(n<1 || k<1 || k%8)throw std::invalid_argument("invalid GEMM shape");
  int sms=0,l2=0;
  Check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
  Check(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0));
  Element *a=nullptr,*b=nullptr,*output=nullptr;
  char* flush=nullptr;
  Check(cudaMalloc(&a,std::size_t(k)*sizeof(Element)));
  Check(cudaMalloc(&b,std::size_t(n)*k*sizeof(Element)));
  Check(cudaMalloc(&output,std::size_t(n)*sizeof(Element)));
  Check(cudaMalloc(&flush,std::size_t(l2)*3));
  Check(cudaMemset(a,0,std::size_t(k)*sizeof(Element)));
  Check(cudaMemset(b,0,std::size_t(n)*k*sizeof(Element)));
  tilemega::codegen::ServingGemmOperands operands;
  operands.a=a;operands.b=b;operands.output=output;
  operands.m=1;operands.n=n;operands.k_total=k;operands.k_count=k;
  operands.a_row_stride=operands.b_row_stride=k;operands.output_stride=n;
  int tiles=(n+BENCH_TILE_N-1)/BENCH_TILE_N;
  Check(cudaFuncSetAttribute(Stage,cudaFuncAttributeMaxDynamicSharedMemorySize,
                             BENCH_SHARED_BYTES));
  Check(cudaFuncSetAttribute(StageStandard,cudaFuncAttributeMaxDynamicSharedMemorySize,
                             StandardBody::kSharedBytes));
  cudaEvent_t begin,end;Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
  auto measure=[&](bool paged) {
    std::vector<float> times;
    for(int sample=0;sample<7;++sample) {
      Check(cudaMemsetAsync(flush,sample,std::size_t(l2)*3));
      Check(cudaEventRecord(begin));
      if(paged)Stage<<<sms,160,BENCH_SHARED_BYTES>>>(operands,tiles);
      else StageStandard<<<sms,128,StandardBody::kSharedBytes>>>(operands,tiles);
      Check(cudaGetLastError());Check(cudaEventRecord(end));
      Check(cudaEventSynchronize(end));
      float ms=0;Check(cudaEventElapsedTime(&ms,begin,end));
      if(sample)times.push_back(ms);
    }
    std::sort(times.begin(),times.end());
    return double(times[2]+times[3])/2;
  };
  double const standard_ms=measure(false),median_ms=measure(true);
  double const bytes=double(n)*k*sizeof(Element);
  std::printf("N=%d K=%d tile_n=%d tile_k=%d page_bytes=%d pages=%d sms=%d standard_ms=%.6f standard_gbps=%.3f paged_ms=%.6f paged_gbps=%.3f\n",
      n,k,BENCH_TILE_N,BENCH_TILE_K,BENCH_PAGE_BYTES,BENCH_PAGE_COUNT,sms,
      standard_ms,bytes/(standard_ms*1e6),median_ms,bytes/(median_ms*1e6));
  Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));
  Check(cudaFree(a));Check(cudaFree(b));Check(cudaFree(output));Check(cudaFree(flush));
  return 0;
}catch(std::exception const& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
