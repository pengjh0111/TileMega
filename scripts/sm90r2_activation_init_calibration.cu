// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

using U64=unsigned long long;
void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__device__ U64 Now() {U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}

template<int Batch,int TK,int Slots>
__global__ __launch_bounds__(160,1)
void Initialization(U64* times,unsigned* errors,int samples) {
  using namespace tilemega::backend;
  using G=Sm90DenseGemm<Batch,8,4096,8,TK,Slots,1,4,ServingEpilogueOp::kStore,4>;
  using A=typename G::LayoutA;
  extern __shared__ __align__(1024) char shared[];
  auto* a=reinterpret_cast<cutlass::bfloat16_t*>(shared+1024);
  U64 global_begin=Now(),clock_begin=clock64();
  for(int sample=0;sample<samples;++sample) {
    for(int i=threadIdx.x*8;i<Slots*16*TK;i+=160*8)
      *reinterpret_cast<uint4*>(a+i)=make_uint4(~0u,~0u,~0u,~0u);
    __syncthreads();
    U64 begin=clock64();
    // Match Run's inactive-row stores and CTA join. Reset and validation are
    // outside timing; this isolated cost is a pricing input, not an E2E result.
    if constexpr(Batch<16) {
      for(int i=threadIdx.x*8;i<Slots*(16-Batch)*TK;i+=160*8) {
        int slot=i/((16-Batch)*TK),within=i%((16-Batch)*TK);
        *reinterpret_cast<uint4*>(a+slot*16*TK+A{}(Batch+within/TK,within%TK))=
            make_uint4(0,0,0,0);
      }
      __syncthreads();
    }
    U64 elapsed=clock64()-begin;
    if(threadIdx.x==0)times[blockIdx.x*(samples+2)+sample]=elapsed;
    asm volatile("" ::: "memory");
    for(int i=threadIdx.x*8;i<Slots*16*TK;i+=160*8) {
      int slot=i/(16*TK),within=i%(16*TK),row=within/TK,col=within%TK;
      auto v=*reinterpret_cast<uint4*>(a+slot*16*TK+A{}(row,col));
      unsigned expected=row<Batch?~0u:0;
      if(v.x!=expected || v.y!=expected || v.z!=expected || v.w!=expected)atomicAdd(errors,1);
    }
    __syncthreads();
  }
  if(threadIdx.x==0) {
    times[blockIdx.x*(samples+2)+samples]=clock64()-clock_begin;
    times[blockIdx.x*(samples+2)+samples+1]=Now()-global_begin;
  }
}

template<int Batch,int TK,int Slots>
void Case(cudaDeviceProp const& p,U64* times,unsigned* errors) {
  constexpr int samples=256,payload=1024+Slots*32*TK;
  for(int resident:{1,2}) {
    int shared=int(std::min(p.sharedMemPerBlockOptin,
        p.sharedMemPerMultiprocessor/resident-p.reservedSharedMemPerBlock)/256)*256;
    if(payload>shared)continue;
    Check(cudaFuncSetAttribute(Initialization<Batch,TK,Slots>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    int active=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        Initialization<Batch,TK,Slots>,160,shared));
    if(active!=resident)throw std::runtime_error("initialization occupancy mismatch");
    int grid=p.multiProcessorCount*resident;
    Initialization<Batch,TK,Slots><<<grid,160,shared>>>(times,errors,4);
    Check(cudaDeviceSynchronize());
    std::vector<double> values,frequencies;
    for(int trial=0;trial<3;++trial) {
      Initialization<Batch,TK,Slots><<<grid,160,shared>>>(times,errors,samples);
      std::vector<U64> host(grid*(samples+2));
      Check(cudaMemcpy(host.data(),times,host.size()*8,cudaMemcpyDeviceToHost));
      // Match the conversion to this loaded kernel, not a separate one-CTA
      // warmup which can observe a different dynamic clock.
      for(int block=0;block<grid;++block) {
        auto* row=host.data()+block*(samples+2);
        if(!row[samples] || row[samples+1]<32000)
          throw std::runtime_error("active clock observation is too short");
        double frequency=double(row[samples])/row[samples+1];
        frequencies.push_back(frequency*1000);
        for(int i=0;i<samples;++i)values.push_back(row[i]/frequency);
      }
    }
    std::sort(values.begin(),values.end());
    std::printf("%s,%d,%d,%d,%d,%d,%zu,%.12g,%.12g,ns\n",
        Batch==16?"activation_init_clock_bracket":"activation_inactive_rows_initialize",
        grid,Batch,TK,Slots,2*Slots*(16-Batch)*TK,values.size(),
        values[values.size()/2],values[values.size()*99/100]);
    std::sort(frequencies.begin(),frequencies.end());
    std::printf("activation_init_active_clock,%d,%d,%d,%d,%d,%zu,%.12g,%.12g,MHz\n",
        grid,Batch,TK,Slots,2*Slots*(16-Batch)*TK,frequencies.size(),
        frequencies[frequencies.size()/2],frequencies[frequencies.size()*99/100]);
  }
}
template<int Batch,int TK>
void Slots(cudaDeviceProp const& p,U64* t,unsigned* e) {
  Case<Batch,TK,2>(p,t,e);Case<Batch,TK,3>(p,t,e);
  Case<Batch,TK,4>(p,t,e);Case<Batch,TK,6>(p,t,e);Case<Batch,TK,8>(p,t,e);
}
template<int Batch>
void Shapes(cudaDeviceProp const& p,U64* t,unsigned* e) {
#ifdef TILEMEGA_CALIBRATION_LARGE_ONLY
  Slots<Batch,2048>(p,t,e);
#else
  Slots<Batch,64>(p,t,e);Slots<Batch,128>(p,t,e);Slots<Batch,256>(p,t,e);
  Slots<Batch,512>(p,t,e);Slots<Batch,1024>(p,t,e);
#endif
}
int main() {try {
  Check(cudaSetDevice(0));cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0));
  if(p.major!=9 || p.minor!=0)throw std::runtime_error("requires Hopper");
  U64* times;unsigned* errors;
  Check(cudaMalloc(&times,p.multiProcessorCount*2ull*258*8));Check(cudaMalloc(&errors,4));
  Check(cudaMemset(errors,0,4));
  std::printf("metric,grid,batch,tile_k,slots,initialized_bytes,samples,median,p99,unit\n");
  Shapes<1>(p,times,errors);Shapes<2>(p,times,errors);
  Shapes<4>(p,times,errors);Shapes<8>(p,times,errors);
#ifndef TILEMEGA_CALIBRATION_LARGE_ONLY
  Case<16,256,2>(p,times,errors);
#endif
  unsigned count;Check(cudaMemcpy(&count,errors,4,cudaMemcpyDeviceToHost));
  Check(cudaFree(errors));Check(cudaFree(times));return count?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
