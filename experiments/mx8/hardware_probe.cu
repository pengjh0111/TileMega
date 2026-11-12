// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using F4 = __NATIVE_VECTOR__(4, float);
using U4 = __NATIVE_VECTOR__(4, unsigned int);
using H4 = __fp16 __attribute__((__vector_size__(4*sizeof(__fp16))));
static void Check(cudaError_t status,const char* site) {
  if(status!=cudaSuccess) {std::fprintf(stderr,"%s: %s\n",site,cudaGetErrorString(status));std::exit(2);}
}

template<int Loads>
__global__ void ReadVector(uint4 const* data,std::size_t n,unsigned* sink) {
  using Global = __attribute__((address_space(1))) U4 const;
  std::size_t stride=std::size_t(gridDim.x)*blockDim.x;
  unsigned result=0;
  for(std::size_t base=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
      base+(Loads-1)*stride<n;base+=Loads*stride) {
    U4 values[Loads];
#pragma unroll
    for(int i=0;i<Loads;++i)values[i]=*((Global*)data+base+i*stride);
#pragma unroll
    for(int i=0;i<Loads;++i)result^=values[i][0]^values[i][1]^values[i][2]^values[i][3];
  }
  sink[blockIdx.x*blockDim.x+threadIdx.x]=result;
}

template<int Slots>
__global__ void ReadLds(uint4 const* data,std::size_t blocks,unsigned* sink) {
  extern __shared__ uint4 ring[];
  constexpr int kItems=32768/16,kLoads=kItems/256;
  unsigned result=0;
  auto issue=[&](std::size_t block,int slot) {
#pragma unroll
    for(int i=0;i<kLoads;++i) {
      int item=threadIdx.x+i*256;
      __builtin_mxc_ldg_b128_bsm(ring+slot*kItems+item,
          (void*)(data+block*kItems+item),0,-1,false,true,false,false);
    }
  };
  std::size_t rounds=blocks/gridDim.x;
#pragma unroll
  for(int i=0;i<Slots-1;++i)if(std::size_t(i)<rounds)issue(blockIdx.x+i*gridDim.x,i);
  for(std::size_t i=0;i<rounds;++i) {
    auto next=i+Slots-1;
    if(next<rounds) {issue(blockIdx.x+next*gridDim.x,int(next%Slots));
      __builtin_mxc_arrive_gvmcnt((Slots-1)*kLoads);}
    else __builtin_mxc_arrive_gvmcnt(0);
#pragma unroll
    for(int j=0;j<kLoads;++j) {
      uint4 value=ring[(i%Slots)*kItems+threadIdx.x+j*256];
      result^=value.x^value.y^value.z^value.w;
    }
    __builtin_mxc_arrive_bsmcnt(0);
  }
  sink[blockIdx.x*blockDim.x+threadIdx.x]=result;
}

template<int Chains>
__global__ void Mma(float* sink,int iterations) {
  std::uint64_t bits=0x3b803b803b803b80ull;H4 a,b;
  __builtin_memcpy(&a,&bits,8);__builtin_memcpy(&b,&bits,8);
  F4 c[Chains];
#pragma unroll
  for(int j=0;j<Chains;++j)c[j]=F4{0,0,0,0};
  for(int i=0;i<iterations;++i) {
#pragma unroll
    for(int j=0;j<Chains;++j)c[j]=__builtin_mxc_mma_16x16x16bf16(a,b,c[j]);
  }
  float result=0;
#pragma unroll
  for(int j=0;j<Chains;++j)result+=c[j][0]+c[j][1]+c[j][2]+c[j][3];
  sink[blockIdx.x*blockDim.x+threadIdx.x]=result;
}

template<class Launch>
double Time(Launch launch,int repeats) {
  cudaEvent_t begin,end;Check(cudaEventCreate(&begin),"event");Check(cudaEventCreate(&end),"event");
  launch();Check(cudaDeviceSynchronize(),"warmup");Check(cudaEventRecord(begin),"record");
  for(int i=0;i<repeats;++i)launch();
  Check(cudaEventRecord(end),"record");Check(cudaEventSynchronize(end),"sync");
  float ms=0;Check(cudaEventElapsedTime(&ms,begin,end),"elapsed");cudaEventDestroy(begin);cudaEventDestroy(end);
  return ms/repeats;
}

int main(int argc,char** argv) {
  if(argc!=2)return 2;std::string item=argv[1];
  cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0),"properties");
  if(p.multiProcessorCount!=104)return 2;
  constexpr std::size_t bytes=std::size_t(8)<<30;
  uint4* data=nullptr;unsigned* sink=nullptr;
  Check(cudaMalloc(&sink,208*512*4),"sink");
  if(item!="bf16_mma") {Check(cudaMalloc(&data,bytes),"8GiB allocation");Check(cudaMemset(data,1,bytes),"fill");}
  std::printf("{\"item\":\"%s\",\"sms\":%d,\"memory_clock_khz\":%d,\"memory_bus_bits\":%d,\"samples\":[",item.c_str(),p.multiProcessorCount,p.memoryClockRate,p.memoryBusWidth);
  bool first=true;
  auto emit=[&](const char* config,double ms,double value,int grid,int threads,int smem,int resident,std::size_t actual_bytes) {
    std::printf("%s{\"config\":\"%s\",\"ms\":%.9g,\"value\":%.12g,\"grid\":%d,\"threads\":%d,\"smem\":%d,\"resident\":%d,\"bytes\":%zu}",first?"":",",config,ms,value,grid,threads,smem,resident,actual_bytes);first=false;
  };
  if(item=="hbm_vector")for(int grid:{104,208}) {
    int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ReadVector<8>,256,0),"occupancy");
    double ms=Time([&]{ReadVector<8><<<grid,256>>>(data,bytes/16,sink);},5);
    char name[80];std::snprintf(name,sizeof(name),"vector128_x8_grid%d",grid);
    emit(name,ms,bytes/(ms*1e6),grid,256,0,resident,bytes);
  }else if(item=="hbm_lds")for(int grid:{104,208}) {
    Check(cudaFuncSetAttribute(ReadLds<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536),"LDS attribute");
    int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ReadLds<2>,256,65536),"occupancy");
    std::size_t blocks=bytes/32768,actual=blocks/grid*grid*32768;
    double ms=Time([&]{ReadLds<2><<<grid,256,65536>>>(data,blocks,sink);},5);
    char name[80];std::snprintf(name,sizeof(name),"lds32KiB_x2_grid%d",grid);
    emit(name,ms,actual/(ms*1e6),grid,256,65536,resident,actual);
  }else if(item=="bf16_mma")for(int threads:{256,512}) {
    constexpr int iterations=100000;
#define MX8_MMA(C) {int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Mma<C>,threads,0),"occupancy"); \
    double ms=Time([&]{Mma<C><<<104,threads>>>((float*)sink,iterations);},3); \
    double flops=8192.*iterations*C*(threads/64)*104;char name[80];std::snprintf(name,sizeof(name),"mma16x16x16_chains%d_threads%d",C,threads); \
    emit(name,ms,flops/(ms*1e9),104,threads,0,resident,0);}
    MX8_MMA(1);MX8_MMA(4);MX8_MMA(8);
#undef MX8_MMA
  }else return 2;
  Check(cudaGetLastError(),"launch");Check(cudaDeviceSynchronize(),"completion");
  std::printf("],\"units\":\"%s\",\"unsupported\":[\"lds32KiB_x4:128KiB exceeds64KiB\",\"lds32KiB_x8:256KiB exceeds64KiB\"]}\n",item=="bf16_mma"?"TFLOPS":"GB/s");
  cudaFree(data);cudaFree(sink);return 0;
}
