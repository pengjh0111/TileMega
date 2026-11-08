// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90F3Mark.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using U64=unsigned long long;
__device__ U64* main_stamps;
__device__ U64* epi_stamps;
__device__ __forceinline__ U64 Clock(){U64 x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}
#undef TM_F3_TRACE_BEGIN
#undef TM_F3_TRACE_END
#define TM_F3_TRACE_BEGIN(kind) SWAP_BEGIN_##kind
#define TM_F3_TRACE_END(kind) SWAP_END_##kind
#define SWAP_BEGIN_kSlotWait
#define SWAP_END_kSlotWait
#define SWAP_BEGIN_kSideWait
#define SWAP_END_kSideWait
#define SWAP_BEGIN_kStore
#define SWAP_END_kStore
#define SWAP_BEGIN_kSplitK
#define SWAP_END_kSplitK
#define SWAP_BEGIN_kMainLoop U64 main_start=Clock()
#define SWAP_END_kMainLoop if(threadIdx.x==0)main_stamps[2*blockIdx.x+local]=Clock()-main_start
#define SWAP_BEGIN_kEpilogue U64 epi_start=Clock()
#define SWAP_END_kEpilogue if(threadIdx.x==0)epi_stamps[2*blockIdx.x+n_tile/132]=Clock()-epi_start
#include <tilemega/Backend/Sm90F3SwapAB.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#ifndef DNN4_SWAP_M
#define DNN4_SWAP_M 8
#endif
#ifndef DNN4_SWAP_TK
#define DNN4_SWAP_TK 64
#endif
#ifndef DNN4_SWAP_RESIDENT
#define DNN4_SWAP_RESIDENT 0
#endif
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
constexpr int kM=DNN4_SWAP_M,kTK=DNN4_SWAP_TK,kResident=DNN4_SWAP_RESIDENT;
constexpr int kSlots=kResident?8:2,kK=2048,kIter=kK/kTK,kN=264*128;
using Side=Sm90F3SideLayout<kM,128,false,1,false>;
using Chain=Sm90F3Chain<Side,Sm90F3Step<Sm90F3Op::kBias>,Sm90F3Step<Sm90F3Op::kActivation>>;
using B=Sm90F3GemmSwapAB<2,1,kM,kN,kK,kN,Chain,kTK>;
constexpr bool kAlias=kSlots*B::kSlotBytes+512>Sm90F3Budget::kSpanBudget;
constexpr int kStorage=kAlias?1:kSlots;
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device{T* data;std::size_t count;explicit Device(std::size_t n):count(n){Check(cudaMalloc(&data,n*sizeof(T)));}~Device(){cudaFree(data);}std::vector<T> Get(){std::vector<T> v(count);Check(cudaMemcpy(v.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
__device__ U64 Timer(){U64 x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x)::"memory");return x;}
__global__ void Calibrate(U64* p){U64 start=Timer(),clock=Clock();while(Timer()-start<20000000ull){}p[0]=Clock()-clock;p[1]=Timer()-start;}
__global__ void Fill(E* p,std::size_t size){for(std::size_t i=blockIdx.x*256+threadIdx.x;i<size;i+=gridDim.x*256)p[i]=E(.125f);}
__global__ void Flush(uint4* p,std::size_t size){for(std::size_t i=blockIdx.x*256+threadIdx.x;i<size;i+=gridDim.x*256)p[i]=make_uint4(unsigned(i),13,17,19);}
__device__ void ResidentMath(Sm90F3Ring& ring,E* arena,Sm90F3Phase& phase,float* sink) {
 using namespace cute;typename B::Mma mma;auto thread=mma.get_slice(int(threadIdx.x)%128);
 typename B::Accumulators accum;cute::clear(accum.tile[0]);int previous=-1;
 U64 start=Clock();
 for(int k=0;k<kIter;++k){int slot=k%kSlots;
  Sm90F3Async::Base::Wait(ring.full+slot,phase.Bit(phase.full,slot));phase.full^=1u<<slot;
  auto* panel=arena+(kAlias?0:slot)*(B::kSlotBytes/2);
  cute::warpgroup_fence_operand(accum.tile[0]);cute::warpgroup_arrive();
  #pragma unroll
  for(int kb=0;kb<kTK/64;++kb){
   auto weight=make_tensor(make_smem_ptr(panel+std::size_t(kb)*128*64+(threadIdx.x/128)*64*64),Sm90KLayout<64,64>{});
   auto activation=make_tensor(make_smem_ptr(panel+128*kTK+std::size_t(kb)*kM*64),Sm90KLayout<kM,64>{});
   auto ra=thread.partition_fragment_A(weight);auto rb=thread.partition_fragment_B(activation);
   CUTE_UNROLL
   for(int step=0;step<size<2>(ra);++step)cute::gemm(mma,ra(cute::_,cute::_,step),rb(cute::_,cute::_,step),accum.tile[0]);
  }
  cute::warpgroup_commit_batch();cute::warpgroup_wait<1>();cute::warpgroup_fence_operand(accum.tile[0]);
  if(previous>=0 && (threadIdx.x&31)==0)Sm90F3Async::Base::Arrive(ring.empty+previous);previous=slot;
 }
 cute::warpgroup_wait<0>();cute::warpgroup_fence_operand(accum.tile[0]);
 if((threadIdx.x&31)==0)Sm90F3Async::Base::Arrive(ring.empty+previous);
 if(threadIdx.x==0)main_stamps[2*blockIdx.x+1]=Clock()-start;
 sink[blockIdx.x*256+threadIdx.x]=accum.tile[0](0);
}
__global__ __launch_bounds__(384,1) void Probe(const __grid_constant__ CUtensorMap map,
 E const* weight,float const* bias,E* output,float* sink,int source_tiles) {
 extern __shared__ __align__(1024) char shared[];auto* h=reinterpret_cast<Sm90F3Header*>(shared);auto& ring=h->rings[0];
 auto* arena=reinterpret_cast<E*>(shared+Sm90F3Budget::kArena0);char* side=reinterpret_cast<char*>(arena)+kStorage*B::kSlotBytes;
 if(threadIdx.x==0){ring.Initialize();Sm90F3Async::Base::InitFence();}
 if constexpr(kResident)for(int i=threadIdx.x;i<kStorage*B::kSlotBytes/2;i+=384)arena[i]=E(.125f);
 Sm90F3Async::Base::FenceProxyAsyncShared();__syncthreads();
 if(threadIdx.x<256 && (threadIdx.x&31)==0)ring.Prime();__syncthreads();Sm90F3Phase phase{};
 if(threadIdx.x>=256){
  Sm90F3Async::RegDealloc<56>();Sm90F3Async::Base::ProxyAsyncGlobalFence();
  if(threadIdx.x==256){
   for(int local=0;local<(kResident?1:2);++local)for(int k=0;k<kIter;++k){int slot=(local*kIter+k)%kSlots;
    Sm90F3Async::Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));phase.empty^=1u<<slot;
    if constexpr(kResident)Sm90F3Async::Base::Arrive(ring.full+slot);
    else {
     int n=(int(blockIdx.x)+local*132)%source_tiles;
     Sm90F3Async::Base::ExpectTx(ring.full+slot,B::kSlotBytes);
     Sm90F3Async::Base::Bulk(B::SlotWeight(arena,slot),weight+(std::size_t(n)*(kK/64)+k*(kTK/64))*128*64,2*128*kTK,ring.full+slot);
     #pragma unroll
     for(int kb=0;kb<kTK/64;++kb)Sm90F3Async::Base::Tensor2D(B::SlotActivation(arena,slot)+kb*kM*64,&map,(k*(kTK/64)+kb)*64,0,ring.full+slot);
    }
   }
  }
  if constexpr(!kResident)if(threadIdx.x==288){Sm90F3GemmOperands p{};p.epilogue.vectors[0]=bias;for(int local=0;local<2;++local)B::ProduceSide(p,ring,side,phase,int(blockIdx.x)+local*132);}
 }else {
  Sm90F3Async::RegAlloc<224>();
  if constexpr(kResident)ResidentMath(ring,arena,phase,sink);
  else {Sm90F3GemmOperands p{};p.epilogue.output=output;p.epilogue.output_stride=kN;B::Consume(p,ring,arena,side,phase,h,int(threadIdx.x)/128);}
 }
 __syncthreads();if(threadIdx.x==0)ring.Invalidate();
}
int main(int argc,char** argv)try {
 if(!std::getenv("TILEMEGA_RUN_NONCE") || !std::getenv("TILEMEGA_GPU_LOCK_HELD"))throw std::runtime_error("swap calibration requires scheduler/guard");
 bool dram=argc>1 && std::string(argv[1])=="dram";int tiles=dram?264:8;
 Device<E> a(kM*kK),b(tiles*128*kK),output(kM*kN);Device<float> bias(kN),sink(132*256);Device<U64> stamps(264),epi(264),cal(2);Device<uint4> flush((128u<<20)/16);
 Fill<<<132,256>>>(a.data,a.count);Fill<<<132,256>>>(b.data,b.count);Check(cudaMemset(bias.data,0,bias.count*sizeof(float)));
 CUtensorMap map{};tilemega::codegen::executor::EncodeTiledMap<2>(map,a.data,{{kK,kM},{kK*2},{64,kM}});
 Calibrate<<<1,1>>>(cal.data);Check(cudaDeviceSynchronize());auto calibration=cal.Get();double cycles=double(calibration[0])/calibration[1];
 Check(cudaMemcpyToSymbol(main_stamps,&stamps.data,sizeof(void*)));Check(cudaMemcpyToSymbol(epi_stamps,&epi.data,sizeof(void*)));
 constexpr int bytes=Sm90F3Budget::kBudget;
 Check(cudaFuncSetAttribute(Probe,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 Probe<<<132,384,bytes>>>(map,b.data,bias.data,output.data,sink.data,tiles);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
 if(dram){Flush<<<132,256>>>(flush.data,flush.count);Check(cudaDeviceSynchronize());}
 Probe<<<132,384,bytes>>>(map,b.data,bias.data,output.data,sink.data,tiles);Check(cudaGetLastError());Check(cudaDeviceSynchronize());
 if constexpr(kResident){for(float v:sink.Get())if(v!=32.f)throw std::runtime_error("resident swap mismatch");}
 else for(auto v:output.Get())if(float(v)!=32.f)throw std::runtime_error("swap result mismatch");
 auto raw=stamps.Get(),ep=epi.Get();std::vector<double> main,epilogue;
 for(int c=0;c<132;++c){main.push_back(raw[2*c+1]/cycles/kIter);if constexpr(!kResident)for(int local=0;local<2;++local)epilogue.push_back(ep[2*c+local]/cycles);}
 std::sort(main.begin(),main.end());std::sort(epilogue.begin(),epilogue.end());
 std::cout<<std::setprecision(17)<<"{\"tile_m\":"<<kM<<",\"tile_n\":128,\"tk\":"<<kTK<<",\"resident\":"<<kResident<<",\"slots\":"<<kSlots<<",\"aliased_operands\":"<<kAlias<<",\"ns\":"<<main[66]<<",\"epilogue_ns\":"<<(kResident?0:epilogue[132])<<",\"grid\":132}\n";
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
