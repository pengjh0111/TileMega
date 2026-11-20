// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace tilemega::backend;
using Element=cutlass::bfloat16_t;
using Async=tilemega::codegen::executor::Async<tilemega::arch::Sm80>;

static void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}

template<int Slots,int Residency>
using Gate=Sm90DenseGemm<16,16384,2048,64,64,Slots,4,1,ServingEpilogueOp::kSwiGLU,
                         16,true,1,false,0,true,true>;

template<int Slots,int Residency,bool Producer>
__global__ __launch_bounds__(160,Residency)
void Role(Element const* input,Element const* weights,Element* output,Sm80DenseService* samples) {
  using G=Gate<Slots,Residency>;
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<Sm90DenseRingState*>(shared);
  ring.Initialize<true>();
  if(threadIdx.x==0)
    *reinterpret_cast<Sm80DenseService**>(shared+Sm90DenseOperandHeader::kTraceSlot)=samples+blockIdx.x;
  auto* a=reinterpret_cast<Element*>(shared+1024);
  auto* b=a+Slots*G::kRows*64;
  auto* scratch=reinterpret_cast<float*>(shared+G::kLayout.scratch);
  Sm90DenseOperands p{};
  p.activation_address=input;p.weight=weights;p.output=output;p.epsilon=1e-6f;
  int tasks=256/int(gridDim.x)+(int(blockIdx.x)<256%int(gridDim.x));

  // The non-measured role still performs every arrival. Resident data
  // substitutes for copies only in the consumer-only arm.
  if constexpr(!Producer) {
    for(int i=threadIdx.x;i<G::kOperandBytes/int(sizeof(Element));i+=160)a[i]=Element(0.125f);
  }
  __syncthreads();
  if(threadIdx.x<128) {
    if constexpr(Producer) {
      unsigned phase=0;
      for(int sequence=0;sequence<tasks*G::kIterations;++sequence) {
        int slot=sequence%Slots;
        Async::Wait(ring.full+slot,(phase>>slot)&1);
        Async::Arrive(ring.empty+slot);
        phase^=1u<<slot;
      }
    } else G::Consume(p,ring,a,b,scratch,nullptr,tasks,0,Sm90DenseNoFinish{});
  } else {
    if constexpr(Producer)G::template ProduceCpAsync<0>(p,ring,a,b,nullptr,tasks,0,false);
    else {
      unsigned phase=0;
      for(int sequence=0;sequence<tasks*G::kIterations;++sequence) {
        int slot=sequence%Slots;
        Async::Wait(ring.empty+slot,(phase>>slot)&1);
        Async::CompleteCopies(ring.full+slot);
        phase^=1u<<slot;
      }
    }
  }
  ring.Invalidate<true>();
}

template<int Slots,int Residency>
int Profile(cudaDeviceProp const& prop) {
  constexpr int shared=Gate<Slots,Residency>::kSharedBytes;
  auto producer=Role<Slots,Residency,true>,consumer=Role<Slots,Residency,false>;
  if(shared>prop.sharedMemPerBlockOptin) {
    std::puts("{\"status\":\"resource_rejected\"}");return 0;
  }
  Check(cudaFuncSetAttribute(producer,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Check(cudaFuncSetAttribute(consumer,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  int prod_resident,cons_resident;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&prod_resident,producer,160,shared));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cons_resident,consumer,160,shared));
  if(std::min(prod_resident,cons_resident)<Residency) {
    std::puts("{\"status\":\"resource_rejected\"}");return 0;
  }
  int grid=prop.multiProcessorCount*Residency;
  Element *input,*weights,*output;
  Sm80DenseService* samples;
  Check(cudaMalloc(&input,16*2048*sizeof(Element)));
  Check(cudaMalloc(&weights,16384*2048*sizeof(Element)));
  Check(cudaMalloc(&output,16*8192*sizeof(Element)));
  Check(cudaMalloc(&samples,grid*sizeof(Sm80DenseService)));
  Check(cudaMemset(input,0,16*2048*sizeof(Element)));
  Check(cudaMemset(weights,0,16384*2048*sizeof(Element)));

  std::vector<Sm80DenseService> host(grid);
  std::vector<double> issue,prod_wait,consume,cons_wait,epi;
  for(int i=0;i<=7;++i) {
    Role<Slots,Residency,true><<<grid,160,shared>>>(input,weights,output,samples);
    Check(cudaMemcpy(host.data(),samples,grid*sizeof(Sm80DenseService),cudaMemcpyDeviceToHost));
    if(i>0)for(int block=0;block<grid;++block) {
      int tasks=256/grid+(block<256%grid);
      issue.push_back(double(host[block].producer_issue)/(tasks*32));
      prod_wait.push_back(double(host[block].producer_wait)/(tasks*32));
    }
    Role<Slots,Residency,false><<<grid,160,shared>>>(input,weights,output,samples);
    Check(cudaMemcpy(host.data(),samples,grid*sizeof(Sm80DenseService),cudaMemcpyDeviceToHost));
    if(i>0)for(int block=0;block<grid;++block) {
      int tasks=256/grid+(block<256%grid);
      consume.push_back(double(host[block].consumer_mma)/(tasks*32));
      cons_wait.push_back(double(host[block].consumer_wait)/(tasks*32));
      epi.push_back(double(host[block].consumer_epilogue)/tasks);
    }
  }
  auto median=[](auto& values) {
    std::sort(values.begin(),values.end());return values[values.size()/2];
  };
  std::printf("{\"status\":\"diagnostic\",\"grid\":%d,\"slots\":%d,"
              "\"dynamic_shared\":%d,\"producer_capacity\":%d,\"consumer_capacity\":%d,"
              "\"isolated_producer_issue_ns\":%.3f,\"producer_wait_ns\":%.3f,"
              "\"isolated_consumer_mma_ns\":%.3f,\"consumer_wait_ns\":%.3f,"
              "\"isolated_epilogue_ns\":%.3f}\n",grid,Slots,shared,prod_resident,cons_resident,
              median(issue),median(prod_wait),median(consume),median(cons_wait),median(epi));
  Check(cudaFree(input));Check(cudaFree(weights));Check(cudaFree(output));Check(cudaFree(samples));
  return 0;
}

int main(int argc,char **argv) {
  try {
    if(argc!=2)throw std::invalid_argument("specify one isolated role case");
    cudaDeviceProp p;
    Check(cudaGetDeviceProperties(&p,0));
    if(p.major!=8 || p.minor || p.multiProcessorCount!=108)
      throw std::invalid_argument("requires the calibrated A800 hardware class");
    int result;
    if(!std::strcmp(argv[1],"slots2"))result=Profile<2,1>(p);
    else if(!std::strcmp(argv[1],"slots3"))result=Profile<3,1>(p);
    else if(!std::strcmp(argv[1],"slots8"))result=Profile<8,1>(p);
    else if(!std::strcmp(argv[1],"slots2-res2"))result=Profile<2,2>(p);
    else throw std::invalid_argument("unknown isolated role case");
    Check(cudaDeviceReset());
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return result;
  }catch(std::exception const& e) {
    std::fprintf(stderr,"%s\n",e.what());return 1;
  }
}
