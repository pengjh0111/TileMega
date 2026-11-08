// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DNN4_TM
#define DNN4_TM 64
#endif
#ifndef DNN4_TN
#define DNN4_TN 32
#endif
#ifndef DNN4_SLOTS
#define DNN4_SLOTS 2
#endif

#ifndef DNN4_MULTICAST
#define DNN4_MULTICAST 0
#endif

using namespace tilemega::backend;
using Element=cutlass::bfloat16_t;
using Tick=unsigned long long;
constexpr int kTM=DNN4_TM,kTN=DNN4_TN,kSlots=DNN4_SLOTS,kGrid=132;
constexpr int kDramK=kTN==32?4096:2048;
using Chain=Sm90F3Chain<Sm90F3SideLayout<kTM,kTN,false,0,false>>;
template<int K>
using Gemm=Sm90F3Gemm<kTM,kTN,kSlots,1,Sm90F3Mode::kNormal,
    2*kTM,264*kTN,K,264*kTN,Chain,Sm90F3NoConv,Sm90F3NoPool,64,false,DNN4_MULTICAST>;

void Check(cudaError_t code) {
  if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> struct Device {
  T* data=nullptr;
  explicit Device(std::size_t count) {Check(cudaMalloc(&data,count*sizeof(T)));}
  ~Device() {cudaFree(data);}
};
__device__ __forceinline__ Tick Clock() {
  Tick value;asm volatile("mov.u64 %0, %%clock64;":"=l"(value)::"memory");return value;
}
__device__ __forceinline__ Tick Timer() {
  Tick value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;
}
__global__ void Calibrate(Tick* output) {
  Tick start=Timer(),ticks=Clock();
  while(Timer()-start<20000000ull){}
  output[0]=Clock()-ticks;output[1]=Timer()-start;
}
__global__ void Fill(Element* output,std::size_t size) {
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x)
    output[i]=Element(.125f);
}
__global__ void Flush(uint4* output,std::size_t size) {
  for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x)
    output[i]=make_uint4(unsigned(i),unsigned(i)^0x5a5a5a5a,17,29);
}

template<int K>
__global__ __launch_bounds__(384,1) void Pipe(
    const __grid_constant__ CUtensorMap map,Element const* weight,
    int source_tiles,int active,Tick* stamps,float* sink) {
  if(int(blockIdx.x)>=active)return;
  using G=Gemm<K>;
  extern __shared__ __align__(1024) char shared[];
  auto* header=reinterpret_cast<Sm90F3Header*>(shared);
  auto& ring=header->rings[0];
  auto* arena=reinterpret_cast<Element*>(shared+Sm90F3Budget::kArena0);
  if(threadIdx.x==0) {ring.Initialize();Sm90F3Async::Base::InitFence();}
  __syncthreads();
  if constexpr(DNN4_MULTICAST)Sm90F3Async::Base::ClusterSync();
  if(threadIdx.x<256 && (threadIdx.x&31)==0) {
    ring.Prime();if constexpr(DNN4_MULTICAST)ring.PrimeCluster();
  }
  __syncthreads();
  if constexpr(DNN4_MULTICAST)Sm90F3Async::Base::ClusterSync();
  Sm90F3Phase phase{};
  if(threadIdx.x>=256) {
    Sm90F3Async::RegDealloc<56>();
    if(threadIdx.x==256) {
      Sm90F3GemmOperands operands{};operands.activation=&map;operands.weight=weight;
      Sm90F3Async::Base::ProxyAsyncGlobalFence();
      for(int local=0;local<2;++local) {
        int n=(int(blockIdx.x)/2+local*kGrid/2)%source_tiles;
        int task=DNN4_MULTICAST?2*n+(int(blockIdx.x)&1):n+(int(blockIdx.x)&1)*264;
        for(int k=0;k<G::kIterations;++k)
          G::ProduceTaskK(operands,ring,arena,phase,task,k,
              (local*G::kIterations+k)%kSlots,false);
      }
    }
    __syncwarp();
  }else {
    Sm90F3Async::RegAlloc<224>();
    typename G::Accumulators accum;
    for(int local=0;local<2;++local) {
      Tick start=Clock();
      G::MainLoop(ring,arena,phase,accum,int(threadIdx.x)/128,
          (local*G::kIterations)%kSlots,G::kIterations,DNN4_MULTICAST);
      Tick ticks=Clock()-start;
      if(local==1) {
        if(threadIdx.x==0)stamps[blockIdx.x]=ticks;
        sink[blockIdx.x*256+threadIdx.x]=accum.tile[0](0);
      }
    }
  }
  __syncthreads();
  if constexpr(DNN4_MULTICAST)Sm90F3Async::Base::ClusterSync();
  if(threadIdx.x==0)ring.Invalidate();
}

template<int K>
void Run(bool dram,int active) {
  using G=Gemm<K>;
  int source_tiles=dram?132:8;
  std::size_t a_count=std::size_t(2*kTM)*K,b_count=std::size_t(source_tiles)*kTN*K;
  if(dram && b_count*2<(64u<<20))throw std::runtime_error("DRAM footprint below 64 MiB");
  // The largest requested L2 tile occupies 9 MiB with two A tiles; retain eight B tiles and K2048.
  std::size_t l2_limit=kTM==128 && kTN==256?10u<<20:8u<<20;
  if(!dram && (a_count+b_count)*2>=l2_limit)throw std::runtime_error("L2 footprint bound exceeded");
  Device<Element> a(a_count),b(b_count);
  Device<uint4> flush((128u<<20)/sizeof(uint4));
  Device<Tick> stamps(kGrid),calibration(2);Device<float> sink(kGrid*256);
  Fill<<<kGrid,256>>>(a.data,a_count);Fill<<<kGrid,256>>>(b.data,b_count);
  Check(cudaGetLastError());
  CUtensorMap map{};
  tilemega::codegen::executor::EncodeTiledMap<2>(map,a.data,{{K,2*kTM},{K*2},{64,kTM}});
  Calibrate<<<1,1>>>(calibration.data);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  Tick ticks[2];Check(cudaMemcpy(ticks,calibration.data,sizeof(ticks),cudaMemcpyDeviceToHost));
  double cycles_per_ns=double(ticks[0])/ticks[1];
  Check(cudaFuncSetAttribute(Pipe<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sm90F3Budget::kBudget));
  int occupancy=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,Pipe<K>,384,Sm90F3Budget::kBudget));
  if(occupancy!=1)throw std::runtime_error("probe must have one CTA per SM");
  auto launch=[&]() {
    if constexpr(DNN4_MULTICAST) {
      cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={2,1,1};
      cudaLaunchConfig_t config{};config.gridDim=dim3(kGrid);config.blockDim=dim3(384);
      config.dynamicSmemBytes=Sm90F3Budget::kBudget;config.attrs=&attr;config.numAttrs=1;
      int clusters=0;Check(cudaOccupancyMaxActiveClusters(&clusters,Pipe<K>,&config));
      if(clusters<66)throw std::runtime_error("multicast probe cannot keep66 clusters resident");
      Check(cudaLaunchKernelEx(&config,Pipe<K>,map,b.data,source_tiles,active,stamps.data,sink.data));
    }else Pipe<K><<<kGrid,384,Sm90F3Budget::kBudget>>>(map,b.data,source_tiles,active,stamps.data,sink.data);
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  };
  launch();
  if(dram) {
    Flush<<<kGrid,256>>>(flush.data,(128u<<20)/sizeof(uint4));
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());
  }
  launch();
  std::vector<Tick> raw(active);std::vector<float> values(active*256);
  Check(cudaMemcpy(raw.data(),stamps.data,raw.size()*sizeof(Tick),cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(values.data(),sink.data,values.size()*sizeof(float),cudaMemcpyDeviceToHost));
  for(float v:values)if(v!=float(K)/64)throw std::runtime_error("production main-loop accumulator mismatch");
  std::vector<double> ns;for(auto tick:raw)ns.push_back(tick/cycles_per_ns/G::kIterations);
  std::sort(ns.begin(),ns.end());
  double median=(ns[(ns.size()-1)/2]+ns[ns.size()/2])/2;
  std::cout<<std::setprecision(17)<<"{\"metric\":\"pipe\",\"tile_m\":"<<kTM
      <<",\"mc\":"<<DNN4_MULTICAST<<",\"tile_n\":"<<kTN<<",\"slots\":"<<kSlots<<",\"source\":\""<<(dram?"dram":"l2")
      <<"\",\"active_ctas\":"<<active<<",\"grid_ctas\":132,\"tasks_per_cta\":2,\"K\":"<<K
      <<",\"k_iterations\":"<<G::kIterations<<",\"slot_bytes\":"<<G::kSlotBytes
      <<",\"layout_bytes\":"<<G::kLayoutBytes<<",\"spanning\":"<<(G::kLayoutBytes>Sm90F3Budget::kArena?"true":"false")
      <<",\"footprint_bytes\":"<<(a_count+b_count)*2<<",\"ns_per_k\":"<<median
      <<",\"per_cta_gbps\":"<<G::kSlotBytes/median<<",\"total_gbps\":"<<active*G::kSlotBytes/median
      <<",\"cycles_per_ns\":"<<cycles_per_ns<<",\"cta_ns_per_k\":[";
  for(std::size_t i=0;i<ns.size();++i)std::cout<<(i?",":"")<<ns[i];
  std::cout<<"]}\n";
}

int main(int argc,char** argv)try {
  if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))
    throw std::runtime_error("probes require the scheduler and guard");
  if(argc!=3)throw std::runtime_error("usage: probe {l2|dram} {33|66|99|132}");
  std::string source=argv[1];bool dram=source=="dram";
  if(source!="l2" && !dram)throw std::runtime_error("invalid source");
  int active=std::stoi(argv[2]);
  if(active!=132)throw std::runtime_error("multicast comparison uses132 CTAs");
  if(active!=33 && active!=66 && active!=99 && active!=132)throw std::runtime_error("invalid CTA count");
  if(dram && active!=132)throw std::runtime_error("DRAM probe uses 132 active CTAs");
  if(dram)Run<kDramK>(true,active);else Run<2048>(false,active);
  return 0;
}catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
