// SPDX-License-Identifier: BSD-3-Clause
#ifndef TILEMEGA_DN_VECTOR_SUMS
#define TILEMEGA_DN_VECTOR_SUMS 1
#endif
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseEpilogue.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using U64=unsigned long long;
using Async=tilemega::codegen::executor::Async<tilemega::arch::Sm90>;
void Check(cudaError_t error) {
  if(error!=cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
__device__ U64 Now() {U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__host__ __device__ unsigned Mix(unsigned x) {
  x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);
}
__global__ void Initialize(uint4* p,std::size_t count) {
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
    p[i]=make_uint4(Mix(i*4),Mix(i*4+1),Mix(i*4+2),Mix(i*4+3));
}
__global__ void Warm(uint4 const* p,std::size_t count,unsigned* checksum) {
  unsigned sum=0;
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
    auto v=p[i];sum^=v.x^v.y^v.z^v.w;
  }
  atomicXor(checksum,sum);
}
__global__ void Frequency(U64* output) {
  if(threadIdx.x)return;
  U64 t=Now(),c=clock64();while(Now()-t<2000000) {}
  output[0]=clock64()-c;output[1]=Now()-t;
}
__global__ void Barrier(U64* counter,U64* stamps,int samples) {
  for(int i=0;i<samples;++i) {
    __syncthreads();
    if(threadIdx.x==0) {
      stamps[(i*gridDim.x+blockIdx.x)*2]=Now();
      asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
      U64 value;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(value):"l"(counter):"memory");}
      while(value<U64(i+1)*gridDim.x);
    }
    __syncthreads();
    if(threadIdx.x==0)stamps[(i*gridDim.x+blockIdx.x)*2+1]=Now();
  }
}
template<bool Split>
__global__ void Latency(char const* input,int bytes,U64* output,unsigned* errors) {
  extern __shared__ __align__(1024) char shared[];
  auto* full=reinterpret_cast<std::uint64_t*>(shared);
  if(threadIdx.x==0) {
    Async::Init(full,1);Async::Init(full+1,1);Async::InitFence();
  }
  __syncthreads();
  if(threadIdx.x==0) {
    U64 begin=clock64();
    if constexpr(Split) {
      Async::ExpectTx(full,16);Async::Bulk(shared+1024,input,16,full);
      Async::ExpectTx(full+1,bytes-16);Async::Bulk(shared+1040,input+16,bytes-16,full+1);
      Async::Wait(full,0);output[0]=clock64()-begin;
      Async::Wait(full+1,0);output[1]=clock64()-begin;
    }else {
      Async::ExpectTx(full,bytes);Async::Bulk(shared+1024,input,bytes,full);
      Async::Wait(full,0);output[0]=clock64()-begin;output[1]=output[0];
    }
    auto v=*reinterpret_cast<uint4*>(shared+1024);
    if(v.x!=Mix(0) || v.y!=Mix(1) || v.z!=Mix(2) || v.w!=Mix(3))atomicAdd(errors,1);
  }
  __syncthreads();
  if(threadIdx.x==0) {Async::Invalidate(full);Async::Invalidate(full+1);}
}
template<int Bytes,int Slots>
__global__ void Stream(char const* input,std::size_t dataset,int samples,bool hot,
    U64* output,unsigned* checksum) {
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<tilemega::backend::Sm90DenseRingState*>(shared);
  ring.Initialize();unsigned sum=0;
  if(threadIdx.x>=128) {
    int lane=threadIdx.x-128;
    for(int i=0;i<samples;++i) {
      int slot=i%Slots;unsigned phase=(i/Slots)&1;
      Async::Wait(ring.empty+slot,phase);
      if(lane==0) {
        std::size_t index=std::size_t(hot?slot:i)*gridDim.x+blockIdx.x;
        auto* source=input+(index%(dataset/Bytes))*Bytes;
        Async::ExpectTx(ring.full+slot,Bytes);
        Async::Bulk(shared+1024+slot*Bytes,source,Bytes,ring.full+slot);
      }
      Async::CompleteCopies(ring.full+slot);
    }
  }else {
    for(int i=0;i<samples;++i) {
      int slot=i%Slots;U64 begin=clock64();
      Async::Wait(ring.full+slot,(i/Slots)&1);
      for(int offset=threadIdx.x*16;offset<Bytes;offset+=128*16) {
        auto v=*reinterpret_cast<uint4*>(shared+1024+slot*Bytes+offset);
        sum^=v.x^v.y^v.z^v.w;
      }
      Async::Arrive(ring.empty+slot);
      if(threadIdx.x==0)output[blockIdx.x*samples+i]=clock64()-begin;
    }
    checksum[blockIdx.x*128+threadIdx.x]=sum;
  }
  ring.Invalidate();
}
void Print(char const* kind,int grid,int bytes,int slots,char const* cache,
    std::vector<double> data,char const* unit,int batch=0,int tn=0,int norm_k=0,int tk=0) {
  std::sort(data.begin(),data.end());
  std::printf("%s,%d,%d,%d,%s,%zu,%.12g,%.12g,%s,%d,%d,%d,%d,%d\n",kind,grid,bytes,slots,cache,
      data.size(),data[data.size()/2],data[std::min(data.size()-1,data.size()*99/100)],unit,
      batch,tn,norm_k,tk,TILEMEGA_DN_VECTOR_SUMS);
}
struct Context {
  cudaDeviceProp prop;char *input,*eviction;std::size_t dataset,evict_bytes;
  U64 *output,*counter;unsigned *checksum,*errors;
  double frequency;
};
template<int Batch,int TK,int K=4096>
__global__ void Activation(char const* input,int samples,U64* output,unsigned* checksum,
    unsigned* errors) {
  using namespace cute;
  using Atom=decltype(composition(Swizzle<3,3,3>{},
      Layout<Shape<_8,_64>,Stride<_64,_1>>{}));
  using A=decltype(tile_to_shape(Atom{},Shape<_16,Int<TK>>{}));
  extern __shared__ __align__(1024) char shared[];
  auto& ring=*reinterpret_cast<tilemega::backend::Sm90DenseRingState*>(shared);
  auto* data=shared+1024;ring.Initialize();
  for(int i=threadIdx.x*16;i<32*TK;i+=160*16)
    *reinterpret_cast<uint4*>(data+i)=make_uint4(0,0,0,0);
  __syncthreads();unsigned sum=0;
  if(threadIdx.x>=128) {
    int lane=threadIdx.x-128;
    for(int sample=0;sample<samples;++sample) {
      Async::Wait(ring.empty,sample&1);
      if(lane==0)Async::Arrive(ring.full);
      for(int i=lane*8;i<Batch*TK;i+=32*8) {
        int row=i/TK,col=i%TK,tile=sample*TK%K;
        Async::Copy16Bytes(data+2*A{}(row,col),input+2*(row*K+tile+col),16);
      }
      Async::CompleteCopies(ring.full);
    }
  }else {
    for(int sample=0;sample<samples;++sample) {
      U64 begin=clock64();Async::Wait(ring.full,sample&1);
      for(int i=threadIdx.x*16;i<32*TK;i+=128*16) {
        auto v=*reinterpret_cast<uint4*>(data+i);sum^=v.x^v.y^v.z^v.w;
      }
      Async::Arrive(ring.empty);
      if(threadIdx.x==0)output[blockIdx.x*samples+sample]=clock64()-begin;
    }
    checksum[blockIdx.x*128+threadIdx.x]=sum;
  }
  __syncthreads();
  if(threadIdx.x==0 && *reinterpret_cast<unsigned*>(data)!=Mix(((samples-1)*TK%K)/2))
    atomicAdd(errors,1);
  ring.Invalidate();
}
template<int Batch,int TK>
void Activations(Context& c) {
  constexpr int samples=256,k=4096;int grid=c.prop.multiProcessorCount;
  int shared=int(c.prop.sharedMemPerBlockOptin/256)*256;
  Check(cudaFuncSetAttribute(Activation<Batch,TK>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Warm<<<grid*4,256>>>(reinterpret_cast<uint4 const*>(c.input),Batch*k*2/16,c.checksum);
  Activation<Batch,TK><<<grid,160,shared>>>(c.input,samples,c.output,c.checksum,c.errors);
  std::vector<U64> host(grid*samples);
  Check(cudaMemcpy(host.data(),c.output,host.size()*8,cudaMemcpyDeviceToHost));
  std::vector<double> values;for(U64 t:host)values.push_back(t/c.frequency);
  Print("activation_wait_and_shared_read",grid,Batch*TK*2,1,"l2",values,"ns",Batch,0,k,TK);
}
using Op=tilemega::backend::ServingEpilogueOp;
template<int Batch,int TN,Op Operation,bool DN>
__global__ void Epilogue(cutlass::bfloat16_t* output,cutlass::bfloat16_t const* residual,
    float const* norm,float* sums,int* indices,U64* times,int samples) {
  extern __shared__ __align__(1024) char shared[];
  auto* tile=reinterpret_cast<float*>(shared+1024);
  constexpr int U=[] {int value=16;while(TN%(2*value))value/=2;return value;}();
  int n=TN*gridDim.x;
  int stride=Operation==Op::kSwiGLU?n/2:Operation==Op::kArgmaxPartial?int(gridDim.x):n;
  for(int sample=0;sample<samples;++sample) {
    if(threadIdx.x<128)for(int i=threadIdx.x;i<16*TN;i+=128)tile[i]=(i%31-15)/32.f;
    __syncthreads();U64 begin=clock64();
    if(threadIdx.x<128) {
      tilemega::backend::Sm90DenseEpilogue<Operation,TN,U>::Run(tile,blockIdx.x,Batch,n,
          stride,output,residual,indices,DN?norm:nullptr,
          Operation==Op::kResidual?sums:nullptr,4096,1e-6f);
      tilemega::codegen::executor::ComputeSync();
    }
    __syncthreads();
    if(threadIdx.x==0)times[blockIdx.x*samples+sample]=clock64()-begin;
  }
}
template<int Batch,int TN,Op Operation,bool DN>
void Epilogues(Context& c) {
  int grid=c.prop.multiProcessorCount,samples=128,n=TN*grid;
  cutlass::bfloat16_t *out,*residual;float *norm,*sums;int* index;
  std::size_t count=std::size_t(Batch)*n;
  Check(cudaMalloc(&out,count*2));Check(cudaMalloc(&residual,count*2));
  Check(cudaMalloc(&norm,16*4096/8*4));Check(cudaMalloc(&sums,count*4));
  Check(cudaMalloc(&index,Batch*grid*4));Check(cudaMemset(index,0xff,Batch*grid*4));
  std::vector<cutlass::bfloat16_t> residuals(count,cutlass::bfloat16_t(.25f));
  std::vector<float> norms(16*4096/8,.25f);
  Check(cudaMemcpy(residual,residuals.data(),count*2,cudaMemcpyHostToDevice));
  Check(cudaMemcpy(norm,norms.data(),norms.size()*4,cudaMemcpyHostToDevice));
  int shared=int(c.prop.sharedMemPerBlockOptin/256)*256;
  Check(cudaFuncSetAttribute(Epilogue<Batch,TN,Operation,DN>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Epilogue<Batch,TN,Operation,DN><<<grid,160,shared>>>(out,residual,norm,sums,index,c.output,samples);
  std::vector<U64> host(grid*samples);
  Check(cudaMemcpy(host.data(),c.output,host.size()*8,cudaMemcpyDeviceToHost));
  std::vector<double> values;for(U64 t:host)values.push_back(t/c.frequency);
  constexpr auto name=Operation==Op::kStore?"epilogue_store_return":
      Operation==Op::kResidual?"epilogue_residual_return":
      Operation==Op::kSwiGLU?"epilogue_swiglu_return":"epilogue_argmax_return";
  Print(name,grid,0,0,DN?"dn":"plain",values,"ns",Batch,TN,DN?4096:0);
  if constexpr(Operation==Op::kArgmaxPartial) {
    std::vector<int> indices(Batch*grid);std::vector<float> result(Batch*grid);
    Check(cudaMemcpy(indices.data(),index,indices.size()*4,cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<result.size();++i)
      if(!std::isfinite(result[i]) || indices[i]<0 || indices[i]>=n)std::exit(5);
  }else {
    std::size_t written=Operation==Op::kSwiGLU?count/2:count;
    Check(cudaMemcpy(residuals.data(),out,written*2,cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<written;++i)if(!std::isfinite(float(residuals[i])))std::exit(5);
  }
  Check(cudaFree(index));Check(cudaFree(sums));Check(cudaFree(norm));
  Check(cudaFree(residual));Check(cudaFree(out));
}
template<int Batch,int TN>
void EpilogueCases(Context& c) {
  Epilogues<Batch,TN,Op::kStore,false>(c);Epilogues<Batch,TN,Op::kStore,true>(c);
  if constexpr(TILEMEGA_DN_VECTOR_SUMS || TN%32==0) {
    Epilogues<Batch,TN,Op::kResidual,false>(c);Epilogues<Batch,TN,Op::kResidual,true>(c);
  }
  Epilogues<Batch,TN,Op::kSwiGLU,false>(c);Epilogues<Batch,TN,Op::kSwiGLU,true>(c);
  Epilogues<Batch,TN,Op::kArgmaxPartial,false>(c);Epilogues<Batch,TN,Op::kArgmaxPartial,true>(c);
}
template<int Bytes,int Slots>
void Streaming(Context& c) {
  constexpr int payload=1024+Bytes*Slots,samples=256;
  for(int grid:{1,c.prop.multiProcessorCount,c.prop.multiProcessorCount*2}) {
    int requested=grid>c.prop.multiProcessorCount?2:1;
    int shared=int(std::min(c.prop.sharedMemPerBlockOptin,
        c.prop.sharedMemPerMultiprocessor/requested-c.prop.reservedSharedMemPerBlock)/256)*256;
    if(payload>shared)continue;
    Check(cudaFuncSetAttribute(Stream<Bytes,Slots>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    int residency=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residency,
        Stream<Bytes,Slots>,160,shared));
    if(residency!=requested)return;
    for(bool hot:{false,true}) {
      if(hot && std::size_t(Bytes)*Slots*grid>std::size_t(c.prop.l2CacheSize)/2)continue;
      int active=std::min(grid,c.prop.multiProcessorCount*residency);
      std::size_t dataset=hot?std::size_t(Bytes)*Slots*grid:c.dataset;
      Stream<Bytes,Slots><<<grid,160,shared>>>(c.input,dataset,32,hot,c.output,c.checksum);
      Check(cudaDeviceSynchronize());
      std::vector<double> rates,waits;
      for(int round=0;round<3;++round) {
        cudaEvent_t begin,end;Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
        Check(cudaEventRecord(begin));
        Stream<Bytes,Slots><<<grid,160,shared>>>(c.input,dataset,samples,hot,c.output,c.checksum);
        Check(cudaEventRecord(end));Check(cudaEventSynchronize(end));float ms;
        Check(cudaEventElapsedTime(&ms,begin,end));
        rates.push_back(double(Bytes)*samples*grid/(double(ms)*1e6*active));
        Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));
        std::vector<U64> host(grid*samples);
        Check(cudaMemcpy(host.data(),c.output,host.size()*8,cudaMemcpyDeviceToHost));
        for(U64 value:host)waits.push_back(value/c.frequency);
      }
      Print("bulk_stream_per_active_cta",grid,Bytes,Slots,hot?"l2":"dram",rates,"B/ns");
      Print("consumer_wait_and_read",grid,Bytes,Slots,hot?"l2":"dram",waits,"ns");
    }
  }
}
int main() {
  Context c{};Check(cudaSetDevice(0));Check(cudaGetDeviceProperties(&c.prop,0));
  if(c.prop.major!=9 || c.prop.minor!=0)return 3;
  c.dataset=std::size_t(c.prop.l2CacheSize)*32;c.evict_bytes=std::size_t(c.prop.l2CacheSize)*16;
  Check(cudaMalloc(&c.input,c.dataset));Check(cudaMalloc(&c.eviction,c.evict_bytes));
  Check(cudaMalloc(&c.output,c.prop.multiProcessorCount*2ull*512*16));Check(cudaMalloc(&c.counter,8));
  Check(cudaMalloc(&c.checksum,c.prop.multiProcessorCount*2ull*128*4));Check(cudaMalloc(&c.errors,4));
  Check(cudaMemset(c.errors,0,4));Check(cudaMemset(c.checksum,0,4));
  Initialize<<<c.prop.multiProcessorCount*4,256>>>(reinterpret_cast<uint4*>(c.input),c.dataset/16);
  Initialize<<<c.prop.multiProcessorCount*4,256>>>(reinterpret_cast<uint4*>(c.eviction),c.evict_bytes/16);
  Frequency<<<1,1>>>(c.output);U64 freq[2];
  Check(cudaMemcpy(freq,c.output,16,cudaMemcpyDeviceToHost));c.frequency=double(freq[0])/freq[1];
  std::printf("metric,grid,slot_bytes,slots,cache,samples,median,p99,unit,batch,tile_n,input_k,tile_k,vector_sums\n");
  Print("clock64_frequency",1,0,0,"na",{c.frequency*1000},"MHz");
#ifndef TILEMEGA_E_CALIBRATION_EPILOGUE_ONLY
  for(int grid:{c.prop.multiProcessorCount,c.prop.multiProcessorCount*2}) {
    int residency;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residency,Barrier,160,0));
    if(grid>residency*c.prop.multiProcessorCount)return 4;
    Check(cudaMemset(c.counter,0,8));Barrier<<<grid,160>>>(c.counter,c.output,256);
    std::vector<U64> stamps(grid*256*2);
    Check(cudaMemcpy(stamps.data(),c.output,stamps.size()*8,cudaMemcpyDeviceToHost));
    std::vector<double> tails;
    for(int sample=0;sample<256;++sample) {
      U64 last=0;for(int b=0;b<grid;++b)last=std::max(last,stamps[(sample*grid+b)*2]);
      for(int b=0;b<grid;++b)tails.push_back(double(stamps[(sample*grid+b)*2+1])-double(last));
    }
    Print("v3_last_arrival_to_departure",grid,0,0,"na",tails,"ns");
  }
  for(int bytes:{4096,16384,32768,65536})for(bool cold:{false,true}) {
    Check(cudaFuncSetAttribute(Latency<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes+1024));
    Check(cudaFuncSetAttribute(Latency<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes+1024));
    std::vector<double> first,complete,whole;
    for(int sample=0;sample<128;++sample)for(bool split:{false,true}) {
      auto* source=reinterpret_cast<uint4 const*>(cold?c.eviction:c.input);
      auto size=cold?c.evict_bytes:std::size_t(bytes);
      Warm<<<c.prop.multiProcessorCount*4,256>>>(source,size/16,c.checksum);
      if(split)Latency<true><<<1,160,bytes+1024>>>(c.input,bytes,c.output,c.errors);
      else Latency<false><<<1,160,bytes+1024>>>(c.input,bytes,c.output,c.errors);
      U64 values[2];Check(cudaMemcpy(values,c.output,16,cudaMemcpyDeviceToHost));
      if(split) {first.push_back(values[0]/c.frequency);complete.push_back(values[1]/c.frequency);}
      else whole.push_back(values[0]/c.frequency);
    }
    Print("bulk_first16_of_split_transfer",1,bytes,1,cold?"dram":"l2",first,"ns");
    Print("bulk_split16_remainder_completion",1,bytes,1,cold?"dram":"l2",complete,"ns");
    Print("bulk_single_completion",1,bytes,1,cold?"dram":"l2",whole,"ns");
  }
  Streaming<4096,2>(c);Streaming<4096,4>(c);Streaming<4096,8>(c);
  Streaming<8192,2>(c);Streaming<8192,4>(c);Streaming<8192,8>(c);
  Streaming<16384,2>(c);Streaming<16384,3>(c);Streaming<16384,4>(c);Streaming<16384,8>(c);
  Streaming<32768,2>(c);Streaming<32768,3>(c);Streaming<32768,4>(c);Streaming<32768,6>(c);
  Streaming<65536,2>(c);Streaming<65536,3>(c);
  Activations<1,64>(c);Activations<1,128>(c);Activations<1,256>(c);Activations<1,512>(c);
  Activations<2,64>(c);Activations<2,128>(c);Activations<2,256>(c);Activations<2,512>(c);
  Activations<4,64>(c);Activations<4,128>(c);Activations<4,256>(c);Activations<4,512>(c);
  Activations<8,64>(c);Activations<8,128>(c);Activations<8,256>(c);Activations<8,512>(c);
  Activations<16,64>(c);Activations<16,128>(c);Activations<16,256>(c);Activations<16,512>(c);
#endif
  EpilogueCases<1,8>(c);EpilogueCases<1,24>(c);EpilogueCases<1,48>(c);
  EpilogueCases<1,16>(c);EpilogueCases<1,32>(c);
  EpilogueCases<1,128>(c);EpilogueCases<1,224>(c);
  EpilogueCases<16,8>(c);EpilogueCases<16,24>(c);EpilogueCases<16,48>(c);
  EpilogueCases<16,16>(c);EpilogueCases<16,32>(c);
  EpilogueCases<16,128>(c);EpilogueCases<16,224>(c);
  unsigned errors;Check(cudaMemcpy(&errors,c.errors,4,cudaMemcpyDeviceToHost));
  Check(cudaFree(c.errors));Check(cudaFree(c.checksum));Check(cudaFree(c.counter));
  Check(cudaFree(c.output));Check(cudaFree(c.eviction));Check(cudaFree(c.input));
  return errors?1:0;
}
