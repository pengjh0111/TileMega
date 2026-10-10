// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeCountedThresholds.h>
#include <tilemega/Codegen/executor/CountedDependency.cuh>
#include <tilemega/Codegen/executor/EpochLastArriver.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {
using namespace tilemega::codegen;
void Check(cudaError_t error) {
  if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::abort();}
}
__host__ __device__ int Value(int slot,int lane,int epoch) {
  return (slot+1)*101+lane*3+epoch*17;
}
template<int Tokens>struct Geometry {
  static constexpr int kSlots=Tokens*2,kRows=4,kBlock=8;
  static constexpr int kTargets=(Tokens+kRows-1)/kRows;
  static constexpr int kVirtual=(kSlots+kBlock-1)/kBlock+(kSlots<128?kSlots:128);
};
template<int Tokens,bool LastArriver>
__global__ void Tasks(int const* rows,RuntimeCountedThresholdView thresholds,int* values,
    unsigned long long* counters,unsigned long long* completions,int* output,
    unsigned* callbacks,unsigned epoch) {
  using G=Geometry<Tokens>;
  int task=blockIdx.x,lane=threadIdx.x;
  __shared__ unsigned weights[G::kTargets];
  __shared__ unsigned last;
  if(task<G::kVirtual) {
    if(lane==0) {
      for(int target=0;target<G::kTargets;++target)weights[target]=0;
      for(int row=task*G::kBlock;row<((task+1)*G::kBlock<G::kSlots?(task+1)*G::kBlock:G::kSlots);++row)
        ++weights[(rows[row]/2)/G::kRows];
    }
    for(int row=task*G::kBlock;row<((task+1)*G::kBlock<G::kSlots?(task+1)*G::kBlock:G::kSlots);++row) {
      int slot=rows[row];values[slot*128+lane]=Value(slot,lane,epoch);
    }
    __syncthreads();
    for(int target=0;target<G::kTargets;++target) {
      if constexpr(LastArriver) {
        std::uint32_t total;
        if(!ReadCountedThreshold(thresholds,1,G::kTargets,target,0,&total))asm volatile("trap;");
        executor::EpochLastArriver::RunWeighted(counters+target,total,epoch,weights[target],&last,[&] {
          int sum=0;
          for(int slot=target*G::kRows*2;slot<((target+1)*G::kRows*2<G::kSlots?(target+1)*G::kRows*2:G::kSlots);++slot)
            sum+=values[slot*128+lane];
          output[target*128+lane]=sum;
          if(lane==0)atomicAdd(callbacks+target,1);
        });
      } else if(weights[target])executor::CountedDependency::Publish(counters+target,weights[target]);
    }
    // Capacity includes unused virtual tiles; they still publish completion.
    executor::CountedDependency::Publish(completions+task,1);
  } else if constexpr(!LastArriver) {
    int target=task-G::kVirtual;
    std::uint64_t expected;
    if(!CountedThresholdTarget(thresholds,1,G::kTargets,target,0,epoch,&expected))asm volatile("trap;");
    executor::CountedDependency::Wait(counters+target,expected);
    int sum=0;
    for(int slot=target*G::kRows*2;slot<((target+1)*G::kRows*2<G::kSlots?(target+1)*G::kRows*2:G::kSlots);++slot)
      sum+=values[slot*128+lane];
    output[target*128+lane]=sum;
  }
}
template<int Tokens>void Run(cudaDeviceProp const& device) {
  using G=Geometry<Tokens>;
  assert(G::kVirtual+G::kTargets<=device.multiProcessorCount);
  int* rows;int* values;int* output;std::uint32_t* thresholds;
  unsigned long long *counters,*tickets,*completions;unsigned* callbacks;
  Check(cudaMalloc(&rows,G::kSlots*sizeof(int)));
  Check(cudaMalloc(&values,G::kSlots*128*sizeof(int)));
  Check(cudaMalloc(&output,G::kTargets*128*sizeof(int)));
  Check(cudaMalloc(&thresholds,(G::kTargets+2)*sizeof(std::uint32_t)));
  Check(cudaMalloc(&counters,2*G::kTargets*sizeof(unsigned long long)));
  Check(cudaMalloc(&tickets,2*G::kTargets*sizeof(unsigned long long)));
  Check(cudaMalloc(&completions,4*G::kVirtual*sizeof(unsigned long long)));
  Check(cudaMalloc(&callbacks,G::kTargets*sizeof(unsigned)));
  Check(cudaMemset(counters,0,2*G::kTargets*sizeof(unsigned long long)));
  Check(cudaMemset(tickets,0,2*G::kTargets*sizeof(unsigned long long)));
  Check(cudaMemset(completions,0,4*G::kVirtual*sizeof(unsigned long long)));
  std::vector<std::uint32_t> counts(G::kTargets+2,999);
  for(int target=0;target<G::kTargets;++target)counts[target+1]=2*std::min(G::kRows,Tokens-target*G::kRows);
  Check(cudaMemcpy(thresholds,counts.data(),counts.size()*sizeof(counts[0]),cudaMemcpyHostToDevice));
  RuntimeCountedThresholdView table{thresholds,unsigned(counts.size())};
  std::vector<int> bindings(G::kSlots),observed(G::kTargets*128);
  std::vector<unsigned long long> notifications(G::kVirtual),arrivals(G::kTargets);
  std::vector<unsigned> reductions(G::kTargets);
  std::mt19937 random(20261008+Tokens);
  for(unsigned epoch=0;epoch<32;++epoch)for(unsigned mode=0;mode<2;++mode) {
    std::iota(bindings.begin(),bindings.end(),0);std::shuffle(bindings.begin(),bindings.end(),random);
    Check(cudaMemcpy(rows,bindings.data(),bindings.size()*sizeof(int),cudaMemcpyHostToDevice));
    for(bool la:{false,true}) {
      Check(cudaMemset(values,0xa5,G::kSlots*128*sizeof(int)));
      Check(cudaMemset(output,0xa5,G::kTargets*128*sizeof(int)));
      Check(cudaMemset(callbacks,0,G::kTargets*sizeof(unsigned)));
      auto* bank=(la?tickets:counters)+mode*G::kTargets;
      auto* events=completions+(mode*2+int(la))*G::kVirtual;
      if(la)Tasks<Tokens,true><<<G::kVirtual,128>>>(rows,table,values,bank,events,output,callbacks,epoch);
      else Tasks<Tokens,false><<<G::kVirtual+G::kTargets,128>>>(rows,table,values,bank,events,output,callbacks,epoch);
      Check(cudaGetLastError());Check(cudaDeviceSynchronize());
      Check(cudaMemcpy(observed.data(),output,observed.size()*sizeof(int),cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(notifications.data(),events,notifications.size()*sizeof(notifications[0]),cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(arrivals.data(),bank,arrivals.size()*sizeof(arrivals[0]),cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(reductions.data(),callbacks,reductions.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
      for(int v=0;v<G::kVirtual;++v)assert(notifications[v]==epoch+1);
      for(int target=0;target<G::kTargets;++target) {
        assert(arrivals[target]==std::uint64_t(counts[target+1])*(epoch+1));
        assert(reductions[target]==unsigned(la));
        for(int lane=0;lane<128;++lane) {
          int reference=0;
          for(int slot=target*G::kRows*2;slot<((target+1)*G::kRows*2<G::kSlots?(target+1)*G::kRows*2:G::kSlots);++slot)
            reference+=Value(slot,lane,epoch);
          assert(observed[target*128+lane]==reference);
        }
      }
    }
  }
  Check(cudaFree(rows));Check(cudaFree(values));Check(cudaFree(output));Check(cudaFree(thresholds));
  Check(cudaFree(counters));Check(cudaFree(tickets));Check(cudaFree(completions));Check(cudaFree(callbacks));
}
}
int main() {
  int device;cudaDeviceProp properties;Check(cudaGetDevice(&device));Check(cudaGetDeviceProperties(&properties,device));
  assert(properties.major==8 && properties.minor==9);
  Run<1>(properties);Run<2>(properties);Run<17>(properties);Run<33>(properties);
  std::puts("Counted threshold synchronization: tail counts, weighted LA, shuffled rows, empty tiles, 32 epochs and separate banks PASS");
}
