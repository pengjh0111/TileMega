// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/executor/EpochLastArriver.cuh>
#include <cassert>
#include <cstdio>
#include <initializer_list>
using namespace tilemega::codegen;
constexpr unsigned kGroups=8,kProducers=4,kElements=256;
__host__ __device__ unsigned Weight(unsigned producer,bool weighted) {
  return weighted ? (producer ? 2*producer+1 : 0) : 1;
}
__global__ void EpochReduce(unsigned long long* tickets,unsigned long long* partials,
    unsigned long long* output,unsigned* callbacks,unsigned* errors,
    unsigned epoch,bool weighted) {
  __shared__ unsigned last;
  unsigned producer=blockIdx.x%kProducers,group=blockIdx.x/kProducers;
  unsigned contribution=Weight(producer,weighted),total=weighted?15:4;
  for(unsigned i=threadIdx.x;i<kElements;i+=blockDim.x)
    if(contribution)partials[blockIdx.x*kElements+i]=
        std::uint64_t(epoch+1)*1048576+std::uint64_t(blockIdx.x)*4096+i;
  auto reduce=[&] {
    for(unsigned i=threadIdx.x;i<kElements;i+=blockDim.x) {
      unsigned long long sum=0;
      for(unsigned p=0;p<kProducers;++p)
        if(Weight(p,weighted))sum+=partials[(group*kProducers+p)*kElements+i];
      output[group*kElements+i]=sum;
    }
    if(threadIdx.x==0)atomicAdd(callbacks+group,1);
  };
  bool won=weighted ? executor::EpochLastArriver::RunWeighted(
      tickets+group,total,epoch,contribution,&last,reduce) :
      executor::EpochLastArriver::Run(tickets+group,total,epoch,&last,reduce);
  if(!contribution && won && threadIdx.x==0)atomicAdd(errors,1);
  if(executor::EpochLastArriver::RunWeighted(tickets+group,total,epoch,0,&last,
       [&]{if(threadIdx.x==0)atomicAdd(errors,1);}) && threadIdx.x==0)atomicAdd(errors,1);
}
template<class T>T* Allocate(unsigned count) {
  T* pointer;assert(cudaMallocManaged(&pointer,count*sizeof(T))==cudaSuccess);return pointer;
}
int main() {
  cudaDeviceProp properties;assert(cudaGetDeviceProperties(&properties,0)==cudaSuccess);
  int resident=0;assert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,EpochReduce,128,0)==cudaSuccess);
  assert(kGroups*kProducers<=unsigned(resident*properties.multiProcessorCount));
  auto* tickets=Allocate<unsigned long long>(2*2*kGroups);
  auto* partials=Allocate<unsigned long long>(kGroups*kProducers*kElements);
  auto* output=Allocate<unsigned long long>(kGroups*kElements);
  auto* callbacks=Allocate<unsigned>(2*2*kGroups);auto* errors=Allocate<unsigned>(1);
  assert(cudaMemset(tickets,0,2*2*kGroups*sizeof(*tickets))==cudaSuccess);
  assert(cudaMemset(callbacks,0,2*2*kGroups*sizeof(*callbacks))==cudaSuccess);*errors=0;
  for(bool weighted:{false,true})for(unsigned epoch=0;epoch<32;++epoch)
    for(unsigned mode=0;mode<2;++mode) {
      unsigned bank=(unsigned(weighted)*2+mode)*kGroups;
      assert(cudaMemset(partials,0xff,kGroups*kProducers*kElements*sizeof(*partials))==cudaSuccess);
      assert(cudaMemset(output,0xff,kGroups*kElements*sizeof(*output))==cudaSuccess);
      EpochReduce<<<kGroups*kProducers,128>>>(tickets+bank,partials,output,callbacks+bank,errors,epoch,weighted);
      assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
      assert(!*errors);
      for(unsigned group=0;group<kGroups;++group) {
        assert(tickets[bank+group]==std::uint64_t(weighted?15:4)*(epoch+1));
        assert(callbacks[bank+group]==epoch+1);
        for(unsigned element=0;element<kElements;++element) {
          unsigned long long expected=0;
          for(unsigned producer=weighted?1:0;producer<kProducers;++producer)
            expected+=std::uint64_t(epoch+1)*1048576+
                std::uint64_t(group*kProducers+producer)*4096+element;
          assert(output[group*kElements+element]==expected);
        }
      }
    }
  cudaFree(tickets);cudaFree(partials);cudaFree(output);cudaFree(callbacks);cudaFree(errors);
  std::puts("epoch LA: monotonic weighted/unweighted tickets, poisoned all-warp partials, zero arrivals and separate banks passed");
}
