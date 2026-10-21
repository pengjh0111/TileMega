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
__global__ void Frequency(U64* out) {
  if(threadIdx.x)return;U64 t=Now(),c=clock64();while(Now()-t<2000000) {}
  out[0]=clock64()-c;out[1]=Now()-t;
}
template<int TN,int TK,int WN,int R=1>
__global__ __launch_bounds__(160,1)
void Mainloop(U64* times,unsigned* errors,int repeats) {
  using namespace cute;using namespace tilemega::backend;
  constexpr int WK=4/WN;
  using G=Sm90DenseGemm<16,TN,2048,TN,TK,2,WN,WK,ServingEpilogueOp::kStore,4,R>;
  extern __shared__ __align__(1024) char shared[];
  auto* a=reinterpret_cast<cutlass::bfloat16_t*>(shared);auto* b=a+16*TK;
  for(int i=threadIdx.x;i<(16+TN)*TK;i+=160)a[i]=cutlass::bfloat16_t(1);
  __syncthreads();if(threadIdx.x>=128)return;
  int lane=threadIdx.x%(32*WN),split=threadIdx.x/(32*WN);
  typename G::Mma mma;auto thread=mma.get_slice(lane);
  auto copy_a=make_tiled_copy_A(typename G::CopyA{},mma);
  auto copy_b=make_tiled_copy_B(typename G::CopyB{},mma);
  auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
  auto sa=make_tensor(make_smem_ptr(a),typename G::LayoutA{});
  auto sb=make_tensor(make_smem_ptr(b),typename G::LayoutB{});
  auto ra=thread.partition_fragment_A(sa);auto rb=thread.partition_fragment_B(sb);
  auto as=copy_a.get_slice(lane).partition_S(sa);auto bs=copy_b.get_slice(lane).partition_S(sb);
  auto ad=copy_a.get_slice(lane).retile_D(ra);auto bd=copy_b.get_slice(lane).retile_D(rb);
  U64 begin=clock64();
  for(int repeat=0;repeat<repeats;++repeat) {
    for_each(make_int_sequence<R*WK>{},[&](auto k) {
      if(int(k)%WK==split) {
        copy(typename G::CopyA{},as(_,_,k),ad(_,_,k));
        copy(typename G::CopyB{},bs(_,_,k),bd(_,_,k));
      }
    });
    for_each(make_int_sequence<size<2>(ra)>{},[&](auto k) {
      if(int(k)%WK==split) {
        gemm(mma,ra(_,_,k),rb(_,_,k),accum);
        constexpr int next=int(k)+R*WK;
        if constexpr(next<size<2>(ra)) {
          copy(typename G::CopyA{},as(_,_,Int<next>{}),ad(_,_,Int<next>{}));
          copy(typename G::CopyB{},bs(_,_,Int<next>{}),bd(_,_,Int<next>{}));
        }
      }
    });
  }
  times[blockIdx.x*128+threadIdx.x]=clock64()-begin;
  for_each(make_int_sequence<size(accum)>{},[&](auto i) {
    if(accum(i)!=float(repeats*TK/WK))atomicAdd(errors,1);
  });
}
template<int TN,int TK,int WN,int R=1>
void Case(cudaDeviceProp const& prop,double frequency,U64* times,unsigned* errors) {
  constexpr int payload=2*(16+TN)*TK;
  int shared=int(prop.sharedMemPerBlockOptin/256)*256,grid=prop.multiProcessorCount;
  if(payload>shared)return;
  Check(cudaFuncSetAttribute(Mainloop<TN,TK,WN,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  int resident;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Mainloop<TN,TK,WN,R>,160,shared));
  if(resident!=1)throw std::runtime_error("mainloop calibration occupancy mismatch");
  Mainloop<TN,TK,WN,R><<<grid,160,shared>>>(times,errors,16);Check(cudaDeviceSynchronize());
  std::vector<double> values;
  for(int trial=0;trial<8;++trial) {
    Mainloop<TN,TK,WN,R><<<grid,160,shared>>>(times,errors,128);
    std::vector<U64> host(grid*128);Check(cudaMemcpy(host.data(),times,host.size()*8,cudaMemcpyDeviceToHost));
    for(auto t:host)values.push_back(t/frequency/128/(TK/(16*(4/WN))));
  }
  std::sort(values.begin(),values.end());
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
int main() {try {
  Check(cudaSetDevice(0));cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0));
  if(p.major!=9 || p.minor!=0)throw std::runtime_error("native mainloop calibration requires Hopper");
  U64* times;unsigned* errors;Check(cudaMalloc(&times,p.multiProcessorCount*128*8));
  Check(cudaMalloc(&errors,4));Check(cudaMemset(errors,0,4));
  Frequency<<<1,1>>>(times);U64 ticks[2];Check(cudaMemcpy(ticks,times,16,cudaMemcpyDeviceToHost));
  double frequency=double(ticks[0])/ticks[1];
  std::printf("metric,grid,tile_n,tile_k,warps_n,warps_k,register_stages,samples,median,p99,unit\n");
#ifdef TILEMEGA_CALIBRATION_LARGE_ONLY
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
  Check(cudaFree(errors));Check(cudaFree(times));return count?1:0;
} catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
