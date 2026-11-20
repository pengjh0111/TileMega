// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
using U64=unsigned long long;
#ifdef TILEMEGA_A800_DUAL_MMA_SHARED
constexpr int kResidency=2;
#else
constexpr int kResidency=1;
#endif
struct RunSpan {U64 begin,end;unsigned sm;};

std::string selected_cases;
unsigned sampled_cases=0;
void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ U64 Now() {U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__global__ void Frequency(U64* out) {
  if(threadIdx.x)return;U64 t=Now(),c=clock64();while(Now()-t<2000000) {}
  out[0]=clock64()-c;out[1]=Now()-t;
}
template<int TN,int TK,int WN,int R=1>
__global__ __launch_bounds__(160,kResidency)
void Mainloop(U64* times,unsigned* errors,int repeats,RunSpan* spans) {
  using namespace cute;using namespace tilemega::backend;
  constexpr int WK=4/WN;
  using G=Sm90DenseGemm<16,TN,2048,TN,TK,2,WN,WK,ServingEpilogueOp::kStore,4,true,R>;
  extern __shared__ __align__(1024) char shared[];
  auto* a=reinterpret_cast<cutlass::bfloat16_t*>(shared);auto* b=a+16*TK;
  for(int i=threadIdx.x;i<(16+TN)*TK;i+=160)a[i]=cutlass::bfloat16_t(1);
  __syncthreads();if(threadIdx.x>=128)return;
  int lane=threadIdx.x%(32*WN),split=threadIdx.x/(32*WN);
  typename G::Mma mma;auto thread=mma.get_slice(lane);
  auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
  float squares[2]={0,0};
  U64 first=0;
  if constexpr(kResidency==2)if(threadIdx.x==0)first=Now();
  U64 begin=clock64();
  for(int repeat=0;repeat<repeats;++repeat) {
    // Consume waits on an async producer each iteration. Its memory clobber
    // prevents hoisting shared loads into a loop-invariant fragment table.
    asm volatile("" ::: "memory");
    if constexpr(WK==1)G::template MmaSlice<0>(a,b,accum,squares);
    else if constexpr(WK==2) {
      if(split==0)G::template MmaSlice<0>(a,b,accum,squares);
      else G::template MmaSlice<1>(a,b,accum,squares);
    }else {
      if(split==0)G::template MmaSlice<0>(a,b,accum,squares);
      else if(split==1)G::template MmaSlice<1>(a,b,accum,squares);
      else if(split==2)G::template MmaSlice<2>(a,b,accum,squares);
      else G::template MmaSlice<3>(a,b,accum,squares);
    }
  }
  times[blockIdx.x*128+threadIdx.x]=clock64()-begin;
  if constexpr(kResidency==2)if(threadIdx.x==0) {
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
    spans[blockIdx.x]={first,Now(),sm};
  }
  float expected=float(repeats*TK/(4*WK));
  if(squares[0]!=expected || squares[1]!=expected)atomicAdd(errors,1);
  for_each(make_int_sequence<size(accum)>{},[&](auto i) {
    if(accum(i)!=float(repeats*TK/WK))atomicAdd(errors,1);
  });
}
static int DualSms(std::vector<RunSpan>& samples) {
  std::sort(samples.begin(),samples.end(),[](auto const& a,auto const& b){return a.sm<b.sm;});
  int total=0;
  for(std::size_t i=0,j=0;i<samples.size();i=j) {
    j=i+1;
    while(j<samples.size() && samples[j].sm==samples[i].sm)++j;
    if(j-i==2)
      total+=std::max(samples[i].begin,samples[i+1].begin)<
          std::min(samples[i].end,samples[i+1].end);
  }
  return total;
}

template<int TN,int TK,int WN,int R=1>
void Case(cudaDeviceProp const& prop,double frequency,U64* times,unsigned* errors) {
  auto id=","+std::to_string(TN)+"x"+std::to_string(TK)+"x"+std::to_string(WN)+"x"+std::to_string(R)+",";
  if(selected_cases.find(id)==std::string::npos)return;
  constexpr int payload=2*(16+TN)*TK;
  int grid=prop.multiProcessorCount*kResidency;
#ifdef TILEMEGA_A800_DUAL_MMA_SHARED
  int shared=TILEMEGA_A800_DUAL_MMA_SHARED;
  if(shared>prop.sharedMemPerBlockOptin || shared<=prop.sharedMemPerMultiprocessor/3 ||
     shared>prop.sharedMemPerMultiprocessor/2)
    throw std::runtime_error("a dual-CTA probe requires the linked model's shared footprint");
#else
  int shared=int(prop.sharedMemPerBlockOptin/256)*256;
#endif
  if(payload>shared)return;
  ++sampled_cases;
  Check(cudaFuncSetAttribute(Mainloop<TN,TK,WN,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  int resident;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Mainloop<TN,TK,WN,R>,160,shared));
  if(resident!=kResidency)throw std::runtime_error("mainloop calibration occupancy mismatch");
  RunSpan* spans=nullptr;
  if constexpr(kResidency==2)Check(cudaMalloc(&spans,grid*sizeof(RunSpan)));
  int dual_sms=prop.multiProcessorCount;
  Mainloop<TN,TK,WN,R><<<grid,160,shared>>>(times,errors,16,spans);Check(cudaDeviceSynchronize());
  std::vector<double> values;
  for(int trial=0;trial<8;++trial) {
    Mainloop<TN,TK,WN,R><<<grid,160,shared>>>(times,errors,128,spans);
    std::vector<U64> host(grid*128);Check(cudaMemcpy(host.data(),times,host.size()*8,cudaMemcpyDeviceToHost));
    if constexpr(kResidency==2) {
      std::vector<RunSpan> timings(grid);
      Check(cudaMemcpy(timings.data(),spans,timings.size()*sizeof(RunSpan),cudaMemcpyDeviceToHost));
      dual_sms=std::min(dual_sms,DualSms(timings));
    }
    for(auto t:host)values.push_back(t/frequency/128/(TK/(16*(4/WN))));
  }
  std::sort(values.begin(),values.end());
  if constexpr(kResidency==2) {
    Check(cudaFree(spans));
    if(dual_sms!=prop.multiProcessorCount)throw std::runtime_error("CTA overlap was not observed on every SM");
    cudaFuncAttributes attr;
    Check(cudaFuncGetAttributes(&attr,Mainloop<TN,TK,WN,R>));
    std::printf("{\"status\":\"measured\",\"grid\":%d,\"tile_n\":%d,\"tile_k\":%d,"
                "\"warps_n\":%d,\"warps_k\":%d,\"register_stages\":%d,\"samples\":%zu,"
                "\"median_ns\":%.12g,\"p99_ns\":%.12g,\"dynamic_shared\":%d,"
                "\"registers\":%d,\"local_bytes\":%zu,\"overlap_sms\":%d,\"input_norm\":true}\n",
        grid,TN,TK,WN,4/WN,R,values.size(),values[values.size()/2],values[values.size()*99/100],
        shared,attr.numRegs,attr.localSizeBytes,dual_sms);
    return;
  }
  std::printf("native_ldmatrix_mma_k16,%d,%d,%d,%d,%d,%d,%zu,%.12g,%.12g,ns\n",grid,TN,TK,WN,4/WN,R,
      values.size(),values[values.size()/2],values[values.size()*99/100]);
}
template<int TK>
void Shapes(cudaDeviceProp const& p,double f,U64* times,unsigned* errors) {
#define C(N,W) Case<N,TK,W>(p,f,times,errors)
  C(8,1);C(16,1);C(16,2);C(24,1);C(32,1);C(32,2);C(32,4);C(40,1);
  C(48,1);C(48,2);C(64,1);C(64,2);C(64,4);C(96,2);C(96,4);
  C(128,2);C(128,4);C(224,4);C(256,4);C(512,4);
#undef C
}
int main(int argc,char** argv) {try {
#ifdef TILEMEGA_A800_DUAL_MMA_SHARED
  if(argc!=2 || !*argv[1])throw std::runtime_error("supply one linked SM80 geometry");
  selected_cases=std::string(",")+argv[1]+",";
#else
  if(argc!=3 || std::strcmp(argv[1],"--cases") || !*argv[2])
    throw std::runtime_error("supply --cases from the matching calibration resource audit");
  selected_cases=std::string(",")+argv[2]+",";
#endif
  Check(cudaSetDevice(0));cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0));
  if((p.major!=8 && p.major!=9) || p.minor!=0)
    throw std::runtime_error("native mainloop calibration requires SM80 or SM90");
  if(kResidency==2 && (p.major!=8 || p.multiProcessorCount!=108))
    throw std::runtime_error("dual-residency calibration requires the checked A800 target");
  U64* times;unsigned* errors;Check(cudaMalloc(&times,p.multiProcessorCount*kResidency*128*8));
  Check(cudaMalloc(&errors,4));Check(cudaMemset(errors,0,4));
  Frequency<<<1,1>>>(times);U64 ticks[2];Check(cudaMemcpy(ticks,times,16,cudaMemcpyDeviceToHost));
  double frequency=double(ticks[0])/ticks[1];
  if constexpr(kResidency==1)
    std::printf("metric,grid,tile_n,tile_k,warps_n,warps_k,register_stages,samples,median,p99,unit\n");
#ifdef TILEMEGA_A800_DUAL_MMA_SHARED
  Case<32,128,4>(p,frequency,times,errors);
  Case<128,64,2>(p,frequency,times,errors);Case<128,64,4>(p,frequency,times,errors);
  Case<32,256,4>(p,frequency,times,errors);
#elif defined(TILEMEGA_CALIBRATION_LARGE_ONLY)
  Case<8,512,1>(p,frequency,times,errors);
  Case<16,512,1>(p,frequency,times,errors);Case<16,512,2>(p,frequency,times,errors);
  Case<24,512,1>(p,frequency,times,errors);
  Case<32,512,1>(p,frequency,times,errors);Case<32,512,2>(p,frequency,times,errors);
  Case<32,512,4>(p,frequency,times,errors);Case<40,512,1>(p,frequency,times,errors);
  Case<48,512,1>(p,frequency,times,errors);Case<48,512,2>(p,frequency,times,errors);
  Case<64,512,1>(p,frequency,times,errors);Case<64,512,2>(p,frequency,times,errors);
  Case<64,512,4>(p,frequency,times,errors);
  Case<8,1024,1>(p,frequency,times,errors);
  Case<16,1024,1>(p,frequency,times,errors);Case<16,1024,2>(p,frequency,times,errors);
  Case<24,1024,1>(p,frequency,times,errors);
  Case<32,1024,1>(p,frequency,times,errors);Case<32,1024,2>(p,frequency,times,errors);
  Case<32,1024,4>(p,frequency,times,errors);Case<8,2048,1>(p,frequency,times,errors);
#else
  Shapes<64>(p,frequency,times,errors);Shapes<128>(p,frequency,times,errors);Shapes<256>(p,frequency,times,errors);
  Case<32,256,4,2>(p,frequency,times,errors);Case<32,256,4,4>(p,frequency,times,errors);
  Case<24,256,1,2>(p,frequency,times,errors);Case<24,256,1,4>(p,frequency,times,errors);
  Case<128,256,4,2>(p,frequency,times,errors);Case<128,256,4,4>(p,frequency,times,errors);
#endif
  unsigned count;Check(cudaMemcpy(&count,errors,4,cudaMemcpyDeviceToHost));
  if(!sampled_cases)throw std::runtime_error("no audited calibration case was selected");
  Check(cudaFree(errors));Check(cudaFree(times));
  if constexpr(kResidency==2) {
    Check(cudaDeviceReset());
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  return count?1:0;
} catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
