// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/MoeBinding.h>
#include <tilemega/Codegen/executor/BindingGate.cuh>
#include <tilemega/Codegen/executor/CountedDependency.cuh>
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <cassert>
#include <cstdio>

using namespace tilemega::codegen;
using Ring=executor::PageRing<8192,2>;
constexpr unsigned kBlocks=16,kExperts=8,kPageElements=4096;
__host__ __device__ unsigned short Pattern(unsigned expert,unsigned element) {
  return static_cast<unsigned short>((expert*137+element*11)%65521);
}
__global__ __launch_bounds__(160,1) void BindingPages(MoeBindingRecord* bindings,
    MoeBindingRow* rows,unsigned short const* weights,unsigned long long* epoch,
    unsigned* notified,unsigned* errors,unsigned iteration,WatchdogRecord* watchdog) {
  __shared__ Ring::Slot slots[2];
  __shared__ __align__(1024) char pages[2*8192];
  __shared__ unsigned attempted;
  if(threadIdx.x==0)attempted=0;
  __syncthreads();
  Watch watch{watchdog,10'000'000'000ull};
  Ring ring{slots,pages,nullptr,&watch};ring.Initialize();
  MoeBindingView view{bindings,rows,kBlocks,kBlocks*16,kExperts,16};
  std::uint64_t sequence=0;
  if(executor::IsCompute()) {
    if(executor::ComputeThread()==0)while(atomicAdd(&attempted,0)==0) {}
    executor::ComputeSync();
    if(threadIdx.x<kBlocks) {
      unsigned v=threadIdx.x;
      if((v+iteration)%3)bindings[v]={(v*3+iteration)%kExperts,v*16,1+(v+iteration)%16,1};
      else bindings[v].valid=0;
    }
    executor::CountedDependency::Publish(epoch,1);
    for(unsigned v=0;v<kBlocks;++v) {
      MoeBindingRecord block;
      auto status=view.Lookup(v,&block);
      if(status==MoeBindingStatus::kInvalid) {if(threadIdx.x==0)atomicAdd(errors,1);continue;}
      if(status==MoeBindingStatus::kActive) {
        ring.AwaitFull(sequence);
        auto* page=reinterpret_cast<unsigned short const*>(ring.Page(sequence));
        for(unsigned i=threadIdx.x;i<kPageElements;i+=128)
          if(page[i]!=Pattern((v*3+iteration)%kExperts,i))atomicAdd(errors,1);
        executor::ComputeSync();ring.Release(sequence++);
      }
      if(threadIdx.x==0)atomicAdd(notified+v,1);
      executor::ComputeSync();
    }
  } else {
    executor::BindingGate gate{epoch,iteration+1,&watch};
    if(gate.LoaderReady() && executor::LoaderLane()==0)atomicAdd(errors,1);
    if(executor::LoaderLane()==0)atomicExch(&attempted,1);
    __syncwarp();gate.WaitLoader();
    for(unsigned v=0;v<kBlocks;++v) {
      MoeBindingRecord block;
      auto status=view.Lookup(v,&block);
      if(status!=MoeBindingStatus::kActive)continue;
      std::uint64_t offset;
      if(!MoeExpertOffset(block,kPageElements,&offset)) {asm volatile("trap;");return;}
      ring.AcquireEmpty(sequence);
      for(unsigned i=executor::LoaderLane()*8;i<kPageElements;i+=32*8)
        Ring::Copy::Copy16Bytes(ring.Page(sequence)+i*2,weights+offset+i,16);
      ring.PublishCopies(sequence++);
    }
  }
}
int main() {
  MoeBindingRecord* bindings;MoeBindingRow* rows;unsigned short* weights;
  unsigned long long* epoch;unsigned* notified;unsigned* errors;WatchdogRecord* watch;
  assert(cudaMallocManaged(&bindings,kBlocks*sizeof(*bindings))==cudaSuccess);
  assert(cudaMallocManaged(&rows,kBlocks*16*sizeof(*rows))==cudaSuccess);
  assert(cudaMallocManaged(&weights,kExperts*kPageElements*sizeof(*weights))==cudaSuccess);
  assert(cudaMallocManaged(&epoch,sizeof(*epoch))==cudaSuccess);
  assert(cudaMallocManaged(&notified,kBlocks*sizeof(*notified))==cudaSuccess);
  assert(cudaMallocManaged(&errors,sizeof(*errors))==cudaSuccess);
  assert(cudaMallocManaged(&watch,sizeof(*watch))==cudaSuccess);
  *epoch=0;*errors=0;*watch={};
  for(unsigned v=0;v<kBlocks;++v)notified[v]=0;
  for(unsigned e=0;e<kExperts;++e)for(unsigned i=0;i<kPageElements;++i)weights[e*kPageElements+i]=Pattern(e,i);
  for(unsigned iteration=0;iteration<32;++iteration) {
    assert(cudaMemset(bindings,0xff,kBlocks*sizeof(*bindings))==cudaSuccess);
    BindingPages<<<1,160>>>(bindings,rows,weights,epoch,notified,errors,iteration,watch);
    assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
    assert(*errors==0 && *epoch==iteration+1 && !watch->fired);
    for(unsigned v=0;v<kBlocks;++v)assert(notified[v]==iteration+1);
  }
  cudaFree(bindings);cudaFree(rows);cudaFree(weights);cudaFree(epoch);
  cudaFree(notified);cudaFree(errors);cudaFree(watch);
  std::puts("binding gate: nonblocking probe, release/acquire, indirect pages, empty notifications and ring wraps passed");
}
