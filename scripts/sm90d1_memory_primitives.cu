// SPDX-License-Identifier: BSD-3-Clause
#define main BasicPrimitivesMain
#define TILEMEGA_PRIMITIVE_COMPOSITE 1
#include "sm90d1_primitives.cu"
#undef main
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
using Copy=tilemega::codegen::executor::Async<>;

__global__ void Visibility(U64* counter,U64* ack,U64* stamps,unsigned* sms,int rounds) {
  if(threadIdx.x)return;
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));sms[blockIdx.x]=sm;
  for(U64 i=1;i<=unsigned(rounds);++i) {
    U64 seen=0,begin=Timer();
    if(blockIdx.x==0) {
      stamps[(i-1)*2]=Timer();
      asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(ack):"memory");
        if(Timer()-begin>2000000000ull)asm volatile("trap;");
      }while(seen<i);
    }else {
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");
        if(Timer()-begin>2000000000ull)asm volatile("trap;");
      }while(seen<i);
      stamps[(i-1)*2+1]=Timer();
      asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(ack):"memory");
    }
  }
}
constexpr unsigned kBufferBytes=512u<<20;
__device__ unsigned Node(unsigned sample,unsigned hop) {
  return ((sample*8+hop)*104729u% (kBufferBytes/128))*16;
}
__global__ void Chains(U64* memory,int samples) {
  unsigned s=blockIdx.x*blockDim.x+threadIdx.x;
  if(s<unsigned(samples))for(unsigned hop=0;hop<8;++hop)
    memory[Node(s,hop)]=reinterpret_cast<U64>(memory+Node(s,(hop+1)%8));
}
__global__ void Evict(uint4 const* memory,U64* checksum) {
  U64 sum=0;
  for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<kBufferBytes/16;i+=gridDim.x*blockDim.x)
    {auto v=memory[i];sum+=v.x+v.y+v.z+v.w;}
  atomicAdd(checksum,sum);
}
template<int Level>
__global__ void ChainSample(U64 const* memory,U64* output,int sample) {
  if(threadIdx.x)return;
  U64 address=reinterpret_cast<U64>(memory+Node(sample,0));
  if constexpr(Level<2)for(int i=0;i<8;++i) {
    if constexpr(Level==0)asm volatile("ld.global.ca.u64 %0, [%0];":"+l"(address)::"memory");
    else asm volatile("ld.global.cg.u64 %0, [%0];":"+l"(address)::"memory");
  }
  auto begin=clock64();
  for(int i=0;i<8;++i) {
    if constexpr(Level==0)asm volatile("ld.global.ca.u64 %0, [%0];":"+l"(address)::"memory");
    else asm volatile("ld.global.cg.u64 %0, [%0];":"+l"(address)::"memory");
  }
  output[sample*2]=clock64()-begin;output[sample*2+1]=address;
}
template<bool Tma>
__global__ void CopyLatency(char const* input,void const* map,U64* output,int samples) {
  __shared__ __align__(1024) char data[32768];
  __shared__ std::uint64_t full;
  if(threadIdx.x==0){Copy::Init(&full,1);Copy::InitFence();}
  __syncthreads();
  if(threadIdx.x==0)for(int i=0;i<samples;++i) {
    auto begin=clock64();Copy::ExpectTx(&full,32768);
    if constexpr(Tma)Copy::Tensor2D(data,map,0,0,&full);
    else Copy::Bulk(data,input,32768,&full);
    Copy::Wait(&full,i&1);output[i]=clock64()-begin;
    // Only complete_tx makes the destination observable to generic loads.
    if(*reinterpret_cast<unsigned*>(data)!=0x01010101u)asm volatile("trap;");
  }
}
template<int Mode>
__global__ void Contention(char const* input,U64* done,U64* output,unsigned* sms,int samples) {
  __shared__ __align__(1024) char data[32768];
  __shared__ std::uint64_t full;
  if(threadIdx.x==0){Copy::Init(&full,1);Copy::InitFence();
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));sms[blockIdx.x]=sm;}
  __syncthreads();
  bool producer=(Mode==1 && blockIdx.x==0 && threadIdx.x==0) ||
                (Mode==2 && threadIdx.x==0);
  bool consumer=(Mode==0 && threadIdx.x==0) ||
                (Mode==1 && blockIdx.x==1 && threadIdx.x==0) ||
                (Mode==2 && threadIdx.x==32);
  if(producer) {
    cuda::atomic_ref<U64,cuda::thread_scope_device> signal(*done);
    unsigned page=0;
    while(!signal.load(cuda::memory_order_acquire)) {
      Copy::Prefetch(input+((page++%4096)*32768),32768);
    }
  }
  if(consumer) {
    for(int i=0;i<samples;++i) {
      auto begin=clock64();Copy::ExpectTx(&full,32768);
      Copy::Bulk(data,input+((i%4096)*32768),32768,&full);
      Copy::Wait(&full,i&1);output[i]=clock64()-begin;
      if(*reinterpret_cast<unsigned*>(data)!=0x01010101u)asm volatile("trap;");
    }
    cuda::atomic_ref<U64,cuda::thread_scope_device> signal(*done);
    signal.store(1,cuda::memory_order_release);
  }
}
int main() {
  BasicPrimitivesMain();
  constexpr int samples=128;
  U64 *counter,*ack,*output,*memory;unsigned* sms;
  Check(cudaMalloc(&counter,8));Check(cudaMalloc(&ack,8));Check(cudaMalloc(&output,65538*8));
  Check(cudaMalloc(&sms,8));Check(cudaMalloc(&memory,kBufferBytes));
  Timers<<<1,1>>>(output,65536);Check(cudaDeviceSynchronize());
  std::vector<U64> host(65538);Check(cudaMemcpy(host.data(),output,host.size()*8,cudaMemcpyDeviceToHost));
  double frequency=double(host[65536])/host[65537];
  Check(cudaMemset(counter,0,8));Check(cudaMemset(ack,0,8));
  Visibility<<<2,128>>>(counter,ack,output,sms,1024);Check(cudaDeviceSynchronize());
  unsigned ids[2];Check(cudaMemcpy(ids,sms,8,cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(host.data(),output,2048*8,cudaMemcpyDeviceToHost));
  std::vector<double> values;
  if(ids[0]!=ids[1]) {
    for(int i=0;i<1024;++i)values.push_back(double(host[2*i+1])-double(host[2*i]));
    Print("release_to_acquire_other_sm",values,"ns",2);
  }
  Check(cudaMemset(memory,1,kBufferBytes));Chains<<<1,256>>>(memory,samples);
  for(int level=0;level<3;++level) {
    for(int sample=0;sample<samples;++sample) {
      if(level==2)Evict<<<528,256>>>(reinterpret_cast<uint4*>(memory),counter);
      if(level==0)ChainSample<0><<<1,1>>>(memory,output,sample);
      if(level==1)ChainSample<1><<<1,1>>>(memory,output,sample);
      if(level==2)ChainSample<2><<<1,1>>>(memory,output,sample);
    }
    Check(cudaDeviceSynchronize());Check(cudaMemcpy(host.data(),output,samples*16,cudaMemcpyDeviceToHost));
    values.clear();for(int i=0;i<samples;++i)values.push_back(host[i*2]/frequency);
    Print(level==0?"dependent_8_hops_l1":level==1?"dependent_8_hops_l2":"dependent_8_hops_dram",values,"ns");
  }
  Check(cudaMemset(memory,1,kBufferBytes));
  tilemega::codegen::executor::TensorMap map;
  auto status=tilemega::codegen::executor::EncodeTensorMap(map,memory,{64,256,128,64,256});
  if(status!=CUDA_SUCCESS)return 2;
  void* device_map;Check(cudaMalloc(&device_map,sizeof(map)));Check(cudaMemcpy(device_map,&map,sizeof(map),cudaMemcpyHostToDevice));
  for(int tma=0;tma<2;++tma)for(int dram=0;dram<2;++dram) {
    values.clear();
    for(int i=0;i<samples;++i) {
      if(dram)Evict<<<528,256>>>(reinterpret_cast<uint4*>(memory),counter);
      if(tma)CopyLatency<true><<<1,128>>>(reinterpret_cast<char*>(memory),device_map,output,1);
      else CopyLatency<false><<<1,128>>>(reinterpret_cast<char*>(memory),device_map,output,1);
      Check(cudaGetLastError());Check(cudaMemcpy(host.data(),output,8,cudaMemcpyDeviceToHost));
      values.push_back(host[0]/frequency);
    }
    Print(tma?(dram?"tma_32k_completion_dram":"tma_32k_completion_l2"):
              (dram?"bulk_32k_completion_dram":"bulk_32k_completion_l2"),values,"ns",1,32768);
  }
  for(int mode=0;mode<3;++mode) {
    Check(cudaMemset(counter,0,8));
    if(mode==0)Contention<0><<<1,64>>>(reinterpret_cast<char*>(memory),counter,output,sms,4096);
    if(mode==1)Contention<1><<<2,64>>>(reinterpret_cast<char*>(memory),counter,output,sms,4096);
    if(mode==2)Contention<2><<<1,64>>>(reinterpret_cast<char*>(memory),counter,output,sms,4096);
    Check(cudaDeviceSynchronize());Check(cudaMemcpy(host.data(),output,4096*8,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(ids,sms,8,cudaMemcpyDeviceToHost));
    if(mode==1 && ids[0]==ids[1])continue;
    values.clear();for(int i=0;i<4096;++i)values.push_back(host[i]/frequency);
    Print(mode==0?"bulk_stream_no_prefetch":mode==1?"bulk_stream_other_sm_prefetch":"bulk_stream_same_sm_prefetch",values,"ns",mode==1?2:1,32768);
  }
  Check(cudaFree(device_map));Check(cudaFree(memory));Check(cudaFree(sms));
  Check(cudaFree(output));Check(cudaFree(ack));Check(cudaFree(counter));
  DumpRaw();
}
