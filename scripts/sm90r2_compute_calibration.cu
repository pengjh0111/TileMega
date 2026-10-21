// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseAttention.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using U64=unsigned long long;
void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* Device(std::size_t n) {T* p;Check(cudaMalloc(&p,n*sizeof(T)));return p;}
__device__ U64 Now() {U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__global__ void Frequency(U64* out) {
  if(threadIdx.x)return;U64 t=Now(),c=clock64();while(Now()-t<2000000) {}
  out[0]=clock64()-c;out[1]=Now()-t;
}
void Print(char const* metric,int grid,int b,int d,int q,int s,int tn,std::vector<double> values) {
  std::sort(values.begin(),values.end());
  std::printf("%s,%d,%d,%d,%d,%d,%d,%zu,%.12g,%.12g,ns\n",metric,grid,b,d,q,s,tn,
      values.size(),values[values.size()/2],values[std::min(values.size()-1,values.size()*99/100)]);
}
template<int Fragments>
__global__ __launch_bounds__(160,1)
void Mma(U64* out,unsigned* errors,int samples) {
  using namespace cute;
  if(threadIdx.x>=128)return;
  auto c=make_tensor<float>(Shape<_4,Int<Fragments>>{});clear(c);
  U64 begin=clock64();
  for(int repeat=0;repeat<samples;++repeat)
    for_each(make_int_sequence<Fragments>{},[&](auto f) {
      SM80_16x8x16_F32BF16BF16F32_TN::fma(c(0,f),c(1,f),c(2,f),c(3,f),
          0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u,
          c(0,f),c(1,f),c(2,f),c(3,f));
    });
  out[blockIdx.x*128+threadIdx.x]=clock64()-begin;
  for_each(make_int_sequence<Fragments>{},[&](auto f) {
    for_each(make_int_sequence<4>{},[&](auto r) {if(c(r,f)!=16.f*samples)atomicAdd(errors,1);});
  });
}
template<int F>
void MmaCase(cudaDeviceProp const& p,double frequency,U64* out,unsigned* errors) {
  int shared=int(p.sharedMemPerBlockOptin/256)*256,grid=p.multiProcessorCount;
  Check(cudaFuncSetAttribute(Mma<F>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  Mma<F><<<grid,160,shared>>>(out,errors,256);Check(cudaDeviceSynchronize());
  std::vector<double> values;
  for(int trial=0;trial<8;++trial) {
    Mma<F><<<grid,160,shared>>>(out,errors,256);
    std::vector<U64> host(grid*128);Check(cudaMemcpy(host.data(),out,host.size()*8,cudaMemcpyDeviceToHost));
    for(auto t:host)values.push_back(t/frequency/256/F);
  }
  Print("mma_sync_per_instruction",grid,16,0,0,0,F*8,values);
}
__device__ void Grid(U64* counter,int iteration) {
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
    U64 v;do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
    while(v<U64(iteration+1)*gridDim.x);
  }
  __syncthreads();
}
template<int B,int D,int Q,int S>
__global__ __launch_bounds__(160,1)
void LastArriver(Sm90DenseAttentionOperands p,U64* counter,U64* times,int repeats) {
  extern __shared__ __align__(1024) char shared[];
  struct Scratch {float weight[S*Q];unsigned last;};
  auto& scratch=*reinterpret_cast<Scratch*>(shared);
  for(int repeat=0;repeat<repeats;++repeat) {
    for(int task=blockIdx.x;task<B*8*S;task+=gridDim.x)if(threadIdx.x<128) {
      int bg=task/S;float value=(bg%17-8)/16.f+(task%S)/128.f;
      for(int i=threadIdx.x;i<Q*D;i+=128)p.partial[task*Q*D+i]=value;
      if(threadIdx.x<Q)p.lse[task*Q+threadIdx.x]=0;
      U64 begin=Now();bool last=Sm90DenseLastArriver::Publish<S>(p.arrivals+bg,&scratch.last);
      U64 published=Now();
      if(last)Sm90DenseAttention<1,D,Q,false,S>::Merge(p,scratch,bg);
      if(threadIdx.x==0) {
        auto* record=times+(std::size_t(repeat)*B*8*S+task)*3;
        record[0]=published-begin;record[1]=last?Now()-published:0;record[2]=last;
      }
      tilemega::codegen::executor::ComputeSync();
    }
    __syncthreads();Grid(counter,repeat);
  }
}
template<int B,int D,int Q,int S>
void LaCase(cudaDeviceProp const& prop) {
  constexpr int groups=B*8,tasks=groups*S,repeats=64;
  auto* partial=Device<float>(tasks*Q*D);auto* lse=Device<float>(tasks*Q);
  auto* output=Device<BF>(groups*Q*D);auto* arrivals=Device<std::uint64_t>(groups);
  auto* counter=Device<U64>(1);auto* times=Device<U64>(std::size_t(repeats)*tasks*3);
  for(int residency:{1,2}) {
    int grid=prop.multiProcessorCount*residency;
    int shared=int(std::min(prop.sharedMemPerBlockOptin,
        prop.sharedMemPerMultiprocessor/residency-prop.reservedSharedMemPerBlock)/256)*256;
    Check(cudaFuncSetAttribute(LastArriver<B,D,Q,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    int observed;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&observed,
        LastArriver<B,D,Q,S>,160,shared));
    if(observed!=residency)throw std::runtime_error("LA calibration occupancy mismatch");
    Check(cudaMemset(arrivals,0,groups*8));Check(cudaMemset(counter,0,8));
    Sm90DenseAttentionOperands p{};p.partial=partial;p.lse=lse;p.output=output;p.arrivals=arrivals;
    LastArriver<B,D,Q,S><<<grid,160,shared>>>(p,counter,times,repeats);
    std::vector<U64> host(std::size_t(repeats)*tasks*3);
    Check(cudaMemcpy(host.data(),times,host.size()*8,cudaMemcpyDeviceToHost));
    std::vector<double> producer,last,merge;
    for(std::size_t i=tasks;i<std::size_t(tasks)*repeats;++i) {
      if(host[i*3+2]) {last.push_back(host[i*3]);merge.push_back(host[i*3+1]);}
      else producer.push_back(host[i*3]);
    }
    Print("la_producer_protocol",grid,B,D,Q,S,0,producer);
    Print("la_last_protocol",grid,B,D,Q,S,0,last);Print("la_merge",grid,B,D,Q,S,0,merge);
    std::vector<BF> result(groups*Q*D);Check(cudaMemcpy(result.data(),output,result.size()*2,cudaMemcpyDeviceToHost));
    for(int bg=0;bg<groups;++bg)for(int i=0;i<Q*D;++i)
      if(result[bg*Q*D+i].raw()!=BF((bg%17-8)/16.f+(S-1)/256.f).raw())
        throw std::runtime_error("LA calibration did not preserve fixed-order known partials");
  }
  Check(cudaFree(times));Check(cudaFree(counter));Check(cudaFree(arrivals));
  Check(cudaFree(output));Check(cudaFree(lse));Check(cudaFree(partial));
}
template<int D,int Q>
void Shape(cudaDeviceProp const& p) {
  LaCase<1,D,Q,2>(p);LaCase<1,D,Q,8>(p);LaCase<1,D,Q,16>(p);LaCase<1,D,Q,32>(p);
  LaCase<16,D,Q,2>(p);LaCase<16,D,Q,8>(p);LaCase<16,D,Q,16>(p);LaCase<16,D,Q,32>(p);
}
int main() {try {
  Check(cudaSetDevice(0));cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0));
  if(p.major!=9 || p.minor!=0)throw std::runtime_error("native compute calibration requires Hopper");
  auto* times=Device<U64>(p.multiProcessorCount*128);auto* errors=Device<unsigned>(1);
  Check(cudaMemset(errors,0,4));Frequency<<<1,1>>>(times);
  U64 ticks[2];Check(cudaMemcpy(ticks,times,16,cudaMemcpyDeviceToHost));double frequency=double(ticks[0])/ticks[1];
  std::printf("metric,grid,batch,head_dim,queries,chunks,tile_n,samples,median,p99,unit\n");
  MmaCase<1>(p,frequency,times,errors);MmaCase<2>(p,frequency,times,errors);
  MmaCase<3>(p,frequency,times,errors);MmaCase<4>(p,frequency,times,errors);
  MmaCase<6>(p,frequency,times,errors);MmaCase<7>(p,frequency,times,errors);
  MmaCase<8>(p,frequency,times,errors);MmaCase<12>(p,frequency,times,errors);
  MmaCase<16>(p,frequency,times,errors);MmaCase<28>(p,frequency,times,errors);
  unsigned count;Check(cudaMemcpy(&count,errors,4,cudaMemcpyDeviceToHost));
  Check(cudaFree(errors));Check(cudaFree(times));
  if(count)throw std::runtime_error("MMA calibration arithmetic failed");
  Shape<128,2>(p);Shape<64,4>(p);Shape<128,3>(p);Shape<128,4>(p);return 0;
} catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
