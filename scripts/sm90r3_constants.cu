// SPDX-License-Identifier: BSD-3-Clause
// SM90R3 native cost-model constants (prompt §4 L0-1). One process prints one
// CSV row per metric; scripts/collect_sm90r3_constants.py takes the median of
// five fresh processes. Diagnostic only: no serving default depends on a run.
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>
#include <tilemega/Backend/Sm90DenseToken.cuh>
#include <tilemega/Backend/Sm90DenseHead.cuh>
#include <tilemega/Backend/Sm90DenseMoe.cuh>
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tilemega;
using BF=cutlass::bfloat16_t;
using U64=unsigned long long;
using Async=codegen::executor::Async<arch::Sm90>;
using backend::ServingEpilogueOp;

void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* Device(std::size_t n) {T* p;Check(cudaMalloc(&p,n*sizeof(T)));Check(cudaMemset(p,0,n*sizeof(T)));return p;}
__device__ U64 Now() {U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ void GridBarrier(U64* counter,U64 round) {
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
    U64 v;do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
    while(v<(round+1)*gridDim.x);
  }
  __syncthreads();
}

double cycles_per_ns=0;
void Print(std::string const& name,std::string const& key,std::vector<double> x,char const* unit) {
  std::sort(x.begin(),x.end());
  std::printf("%s,%s,%zu,%.6g,%.6g,%s\n",name.c_str(),key.c_str(),x.size(),x[x.size()/2],
      x[std::min(x.size()-1,x.size()*99/100)],unit);
  std::fflush(stdout);
}

__global__ void Clock(U64* out) {
  U64 begin=Now(),cycles=clock64();
  while(Now()-begin<20000000ull) {}
  out[0]=clock64()-cycles;out[1]=Now()-begin;
}

// c_bar: V3 barrier from the last CTA arrival to each CTA's departure.
__global__ void V3(U64* counter,U64* out,int samples) {
  for(int i=0;i<samples;++i) {
    __syncthreads();
    if(threadIdx.x==0) {
      out[(i*gridDim.x+blockIdx.x)*2]=Now();
      asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
      U64 v;do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
      while(v<U64(i+1)*gridDim.x);
      out[(i*gridDim.x+blockIdx.x)*2+1]=Now();
    }
    __syncthreads();
  }
}

// One 16 KB bulk from issue to completion. Cold samples walk distinct regions.
__global__ void Latency(char const* base,std::size_t stride,int samples,U64* out,U64* counter,bool loaded) {
  __shared__ alignas(128) char buffer[16384];
  __shared__ std::uint64_t barrier;
  if(threadIdx.x==0) {Async::Init(&barrier,1);Async::InitFence();}
  __syncthreads();
  unsigned phase=0;
  for(int i=0;i<samples;++i) {
    if(loaded)GridBarrier(counter,i);
    if(threadIdx.x==0) {
      // stride 0: a distinct L2-resident region per CTA; otherwise a fresh region per sample.
      auto* source=stride?base+(std::size_t(i)*gridDim.x+blockIdx.x)*stride:base+blockIdx.x*16384;
      U64 begin=clock64();
      Async::ExpectTx(&barrier,16384);Async::Bulk(buffer,source,16384,&barrier);
      Async::Wait(&barrier,phase);phase^=1;
      out[i*gridDim.x+blockIdx.x]=clock64()-begin;
    }
  }
}

// Streaming bandwidth with Slots bulk/TMA requests of Bytes each in flight.
template<int Slots,int Bytes,bool Tensor>
__global__ void Stream(const __grid_constant__ CUtensorMap map,char const* base,std::size_t per_cta,
    int requests,int rows,U64* counter,U64* stamps,int round) {
  extern __shared__ __align__(1024) char shared[];
  auto* full=reinterpret_cast<std::uint64_t*>(shared+Slots*Bytes);
  if(threadIdx.x==0) {for(int i=0;i<Slots;++i)Async::Init(full+i,1);Async::InitFence();}
  GridBarrier(counter,round*2);
  if(threadIdx.x==0) {
    stamps[blockIdx.x*2]=Now();
    unsigned phase=0;
    for(int i=0;i<requests;++i) {
      int slot=i%Slots;
      if(i>=Slots) {Async::Wait(full+slot,(phase>>slot)&1);phase^=1u<<slot;}
      Async::ExpectTx(full+slot,Bytes);
      if constexpr(Tensor) {
        // Staggered start rows keep all CTAs on different lines of one buffer.
        int row=int((std::size_t(blockIdx.x)*997*128+std::size_t(i)*128)%rows);
        Async::Tensor2D(shared+slot*Bytes,&map,0,row,full+slot);
      }else Async::Bulk(shared+slot*Bytes,base+blockIdx.x*per_cta+std::size_t(i)*Bytes,Bytes,full+slot);
    }
    for(int i=std::max(0,requests-Slots);i<requests;++i) {
      int slot=i%Slots;Async::Wait(full+slot,(phase>>slot)&1);phase^=1u<<slot;
    }
    stamps[blockIdx.x*2+1]=Now();
  }
  GridBarrier(counter,round*2+1);
}

template<int Slots,int Bytes,bool Tensor>
double StreamRun(CUtensorMap const& map,char const* base,std::size_t per_cta,int requests,int rows,int grid) {
  auto kernel=Stream<Slots,Bytes,Tensor>;
  int shared=Slots*Bytes+1024;
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  auto* counter=Device<U64>(1);auto* stamps=Device<U64>(2*grid);
  std::vector<double> rates;
  std::vector<U64> host(2*grid);
  for(int round=0;round<6;++round) {
    Check(cudaMemset(counter,0,8));
    kernel<<<grid,128,shared>>>(map,base,per_cta,requests,rows,counter,stamps,0);
    Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(host.data(),stamps,host.size()*8,cudaMemcpyDeviceToHost));
    U64 begin=~0ull,end=0;
    for(int c=0;c<grid;++c) {begin=std::min(begin,host[2*c]);end=std::max(end,host[2*c+1]);}
    if(round)rates.push_back(double(grid)*requests*Bytes/double(end-begin));
  }
  Check(cudaFree(counter));Check(cudaFree(stamps));
  std::sort(rates.begin(),rates.end());
  return rates[rates.size()/2];
}

// Epilogue of the native GEMM body (prompt §5 D4): the register path (WK 1,
// store/residual) or the shared single pass, one task per sample.
template<ServingEpilogueOp Op,int TN,int WN,int Batch>
__global__ void EpilogueKernel(BF* output,int* index,int copies,std::size_t out_copy,int samples,U64* out) {
  using namespace cute;
  constexpr int WK=4/WN,N=TN*8,K=2048,U=TN%32==0?16:TN%16==0?8:4;
  constexpr bool norm=Op!=ServingEpilogueOp::kResidual;
  using G=backend::Sm90DenseGemm<Batch,N,K,TN,64,2,WN,WK,Op,U,norm>;
  extern __shared__ __align__(16) float tile[];
  float* norms=tile+WK*16*TN;
  auto* residual=reinterpret_cast<BF*>(norms+128);
  int tid=threadIdx.x,lane=tid%(32*WN);
  typename G::Mma mma;auto thread=mma.get_slice(lane);
  auto owned=thread.partition_C(make_identity_tensor(Shape<_16,Int<TN>>{}));
  auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});
  for(int i=0;i<size(accum);++i)accum(i)=0.25f*float((i*7+tid)%13)-1.5f;
  for(int i=tid;i<WK*16*TN;i+=128)tile[i]=0.125f*float(i%11);
  for(int i=tid;i<4*16;i+=128)norms[i]=float(K);
  for(int i=tid;i<2*Batch*TN;i+=128)residual[i]=BF(0.5f);
  float inv[2]={0.5f,0.5f};
  for(int s=0;s<samples;++s) {
    codegen::executor::ComputeSync();
    U64 begin=clock64();
    int c=s%copies,task=s%8;
    if constexpr(G::kRegisterEpilogue)G::Epilogue::Registers(accum,owned,inv,residual,output+c*out_copy,task);
    else G::Epilogue::template Shared<WK>(tile,norms,residual,output+c*out_copy,index+c*out_copy,task,1e-6f);
    codegen::executor::ComputeSync();
    if(tid==0)out[s]=clock64()-begin;
  }
}
struct EpilogueBuffers {BF* output;int* index;std::size_t out_copy;int copies=16;};
EpilogueBuffers epilogue;

template<ServingEpilogueOp Op,int TN,int WN,int Batch>
void EpilogueCase(char const* family) {
  constexpr int samples=256,WK=4/WN;
  auto kernel=EpilogueKernel<Op,TN,WN,Batch>;
  int shared=4*WK*16*TN+512+4*Batch*TN;
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  auto* out=Device<U64>(samples);std::vector<U64> host(samples);
  kernel<<<1,128,shared>>>(epilogue.output,epilogue.index,epilogue.copies,epilogue.out_copy,samples,out);
  Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(host.data(),out,samples*8,cudaMemcpyDeviceToHost));
  std::vector<double> x;for(int i=8;i<samples;++i)x.push_back(host[i]/cycles_per_ns);
  Print("t_ep",std::string(family)+":tn="+std::to_string(TN)+":b="+std::to_string(Batch)+
      ":wk="+std::to_string(WK),x,"ns");
  Check(cudaFree(out));
}
template<ServingEpilogueOp Op,int TN>
void EpilogueTile(char const* family) {
  [&]<int... B>(std::integer_sequence<int,B...>) {
    if constexpr(TN%8==0)(EpilogueCase<Op,TN,1,B>(family),...);
    if constexpr(TN%16==0)(EpilogueCase<Op,TN,2,B>(family),...);
    if constexpr(TN%32==0 && TN/4<=256)(EpilogueCase<Op,TN,4,B>(family),...);
  }(std::integer_sequence<int,1,2,4,8,16>{});
}
template<ServingEpilogueOp Op,int... TN>
void EpilogueTiles(char const* family,std::integer_sequence<int,TN...>) {(EpilogueTile<Op,TN>(family),...);}

// MoE epilogues (prompt §8): router logits, expert gate_up SwiGLU into h,
// expert down P writes and ctr[j] arrival. A slot holds round(8B/n_active)
// rows; n_active is set beyond reach so no sample is a combine last arriver.
// Experts selects the geometry: 128 is Qwen3-30B-A3B (I = 768), 64 is OLMoE
// (I = 1024); both have H = 2048.
template<backend::Sm90DenseMoeMode Mode,int TN,int WN,int Batch,int Experts=128>
__global__ void MoeEpilogueKernel(backend::Sm90DenseMoeOperands p,int samples,U64* out) {
  constexpr int I=Experts==128?768:1024,WK=4/WN,N=Mode==backend::Sm90DenseMoeMode::kRouter?Experts:
      Mode==backend::Sm90DenseMoeMode::kGateUp?2*I:2048,K=Mode==backend::Sm90DenseMoeMode::kDown?I:2048;
  using G=backend::Sm90DenseMoeGemm<Batch,N,K,TN,64,2,WN,WK,Mode,16,Experts,8,false>;
  extern __shared__ __align__(16) float tile[];
  float* norms=tile+WK*16*TN;int* flag=reinterpret_cast<int*>(norms+64);
  int tid=threadIdx.x;
  for(int i=tid;i<WK*16*TN;i+=128)tile[i]=0.125f*float(i%11);
  for(int i=tid;i<4*16;i+=128)norms[i]=float(K);
  for(int s=0;s<samples;++s) {
    codegen::executor::ComputeSync();
    U64 begin=clock64();
    G::Epilogue(p,tile,norms,s%G::kTiles,flag);
    codegen::executor::ComputeSync();
    if(tid==0)out[s]=clock64()-begin;
  }
}
template<backend::Sm90DenseMoeMode Mode,int TN,int WN,int Batch,int Experts=128>
void MoeEpilogueCase(char const* family) {
  constexpr int samples=256,WK=4/WN;
  using Schedule=backend::Sm90DenseMoeSchedule<Experts>;
  constexpr double E=Experts;
  int rows=Mode==backend::Sm90DenseMoeMode::kRouter?Batch:
      std::max(1,int(std::lround(8.0*Batch/(E*(1-std::pow(1-8.0/E,Batch))))));
  std::vector<int> schedule(Schedule::kInts,0);
  schedule[Schedule::Active()]=1<<30;schedule[Schedule::Count(0)]=rows;
  for(int r=0;r<rows;++r)schedule[Schedule::Row(0,r)]=r*8;
  auto* device_schedule=Device<int>(schedule.size());
  Check(cudaMemcpy(device_schedule,schedule.data(),schedule.size()*4,cudaMemcpyHostToDevice));
  auto* output=Device<float>(16*8*2048);auto* weights=Device<BF>(16*8);auto* x=Device<BF>(16*2048);
  auto* counter=Device<std::uint64_t>(16);
  backend::Sm90DenseMoeOperands p{nullptr,nullptr,nullptr,output,device_schedule,weights,nullptr,x,counter,1e-6f};
  auto kernel=MoeEpilogueKernel<Mode,TN,WN,Batch,Experts>;
  int shared=4*WK*16*TN+512;
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  auto* out=Device<U64>(samples);std::vector<U64> host(samples);
  kernel<<<1,128,shared>>>(p,samples,out);
  Check(cudaDeviceSynchronize());
  Check(cudaMemcpy(host.data(),out,samples*8,cudaMemcpyDeviceToHost));
  std::vector<double> v;for(int i=8;i<samples;++i)v.push_back(host[i]/cycles_per_ns);
  Print("t_ep",std::string(family)+":tn="+std::to_string(TN)+":b="+std::to_string(Batch)+
      ":wk="+std::to_string(WK),v,"ns");
  for(void* q:{(void*)device_schedule,(void*)output,(void*)weights,(void*)x,(void*)counter,(void*)out})
    Check(cudaFree(q));
}
template<backend::Sm90DenseMoeMode Mode,int TN,int Experts=128>
void MoeEpilogueTile(char const* family) {
  [&]<int... B>(std::integer_sequence<int,B...>) {
    if constexpr(TN%8==0)(MoeEpilogueCase<Mode,TN,1,B,Experts>(family),...);
    if constexpr(TN%16==0)(MoeEpilogueCase<Mode,TN,2,B,Experts>(family),...);
    if constexpr(TN%32==0 && TN/4<=256)(MoeEpilogueCase<Mode,TN,4,B,Experts>(family),...);
  }(std::integer_sequence<int,1,2,4,8,16>{});
}
// t_route_la: the router's 16 tasks arrive (CTAs 0-15), the last runs top-8,
// routes and the schedule, then the V3 barrier (prompt §8 M4).
template<int Batch,int Experts=128,bool All=false>
__global__ void RouteLaStage(float* logits,int* schedule,BF* weights,int* index,U64* lm,
    U64* counter,U64* stamps,int iterations) {
  __shared__ alignas(128) char free[1024];
  backend::Sm90DenseMoeOperands p{nullptr,nullptr,nullptr,logits,schedule,weights,index,nullptr,
      reinterpret_cast<std::uint64_t*>(lm),0.0f};
  for(int it=0;it<iterations;++it) {
    if(threadIdx.x<128)
      backend::Sm90DenseRouterFinish<Batch,Experts,8,Experts/8,false,All>{}(p,blockIdx.x<Experts/8?1:0,free);
    GridBarrier(counter,it);
    if(blockIdx.x==0 && threadIdx.x==0)stamps[it]=Now();
  }
}

// MR6 t_task_expert: one CTA runs the expert body over A active slots that
// all name expert 0 (L2-resident weights), so its time is tasks x (bytes at
// the CTA's L2 rate + per-task cost). Two task counts give the per-task slope;
// the collector subtracts the slot bytes at BW_sm and the measured epilogue.
template<class G>
__global__ void MoeTaskKernel(backend::Sm90DenseMoeOperands p,U64* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  rings[0].Initialize();
  __syncthreads();
  U64 begin=Now();
  G::Run(p,shared,&rings[0]);
  if(threadIdx.x==0)stamps[0]=Now()-begin;
  rings[0].Invalidate();
}
template<backend::Sm90DenseMoeMode Mode,int Batch,int N,int K,int TN,int TK,int Slots,int WN,int Experts=128>
void MoeTaskCase(char const* family) {
  using G=backend::Sm90DenseMoeGemm<Batch,N,K,TN,TK,Slots,WN,4/WN,Mode,16,Experts,8,false,0>;
  using Schedule=backend::Sm90DenseMoeSchedule<Experts>;
  constexpr bool down=Mode==backend::Sm90DenseMoeMode::kDown;
  constexpr double E=Experts;
  int rows=std::max(1,int(std::lround(8.0*Batch/(E*(1-std::pow(1-8.0/E,Batch))))));
  auto* weight=Device<BF>(std::size_t(N)*K);auto* x=Device<BF>(16*std::max(N,K));
  auto* h=Device<BF>(std::size_t(Experts)*16*K);auto* output=Device<float>(std::size_t(16)*8*std::max(N,K)*8);
  auto* routes=Device<BF>(16*8);auto* counter=Device<std::uint64_t>(N/TN);auto* stamps=Device<U64>(1);
  CUtensorMap map{};CUtensorMap* device_map=Device<CUtensorMap>(1);
  if(down) {
    codegen::executor::EncodeTiledMap<4>(map,h,{{64,16,std::uint64_t(K/64),std::uint64_t(Experts)},
        {2ull*K,128,2ull*16*K},
        {64,std::uint32_t(backend::Sm90DenseLayout::ActivationRows(Batch)),TK/64,1}});
    Check(cudaMemcpy(device_map,&map,sizeof map,cudaMemcpyHostToDevice));
  }
  Check(cudaFuncSetAttribute(MoeTaskKernel<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::kSharedBytes));
  std::vector<double> per_task;
  for(int sample=0;sample<12;++sample) {
    double time[2];
    for(int i=0;i<2;++i) {
      int active=i?8:2;
      std::vector<int> schedule(Schedule::kInts,0);
      schedule[Schedule::Active()]=active;
      for(int a=0;a<active;++a) {
        schedule[Schedule::Count(a)]=rows;
        for(int r=0;r<rows;++r)schedule[Schedule::Row(a,r)]=r*8;
      }
      auto* device_schedule=Device<int>(schedule.size());
      Check(cudaMemcpy(device_schedule,schedule.data(),schedule.size()*4,cudaMemcpyHostToDevice));
      Check(cudaMemset(counter,0,8*(N/TN)));
      backend::Sm90DenseMoeOperands p{down?device_map:nullptr,weight,x,down?(void*)output:(void*)h,
          device_schedule,routes,nullptr,x,counter,1e-6f};
      MoeTaskKernel<G><<<1,160,G::kSharedBytes>>>(p,stamps);Check(cudaDeviceSynchronize());
      U64 host;Check(cudaMemcpy(&host,stamps,8,cudaMemcpyDeviceToHost));
      time[i]=double(host);Check(cudaFree(device_schedule));
    }
    if(sample>=2)per_task.push_back((time[1]-time[0])/(6.0*G::kTiles));
  }
  Print("t_task_slope",std::string(family)+":tn="+std::to_string(TN)+":tk="+std::to_string(TK)+
      ":b="+std::to_string(Batch)+":wk="+std::to_string(4/WN)+":bytes="+std::to_string(2*TN*K),per_task,"ns");
  for(void* q:{(void*)weight,(void*)x,(void*)h,(void*)output,(void*)routes,(void*)counter,(void*)stamps,
      (void*)device_map})Check(cudaFree(q));
}

// Attention pieces of the D2 body (prompt §4 L0-1): one 64-key block with data
// resident, the q preparation, and the S-chunk merge with this CTA last. Hkv
// matters only to the row norm (mode 2), whose preparation reduces the token's
// whole q and k rows across every KV group: OLMoE measures it at its own 16.
template<int D,int Q,int Norm,int S,int Hkv=8>
__global__ void AttentionKernel(backend::Sm90DenseAttentionOperands p,int samples,U64* out) {
  using A=backend::Sm90DenseAttentionBody<1,Hkv,D,Q,Norm,S,64,2,0,1088>;
  extern __shared__ __align__(1024) char shared[];
  auto& s=*reinterpret_cast<typename A::Storage*>(shared);
  int tid=threadIdx.x;
  for(int i=tid*8;i<int(sizeof(s.kv[0]))/2;i+=128*8)
    *reinterpret_cast<uint4*>(&s.kv[0][0][0]+i)=make_uint4(0x3c003c00u,0x3c003c00u,0x3c003c00u,0x3c003c00u);
  for(int i=tid*8;i<16*D;i+=128*8)*reinterpret_cast<uint4*>(s.query+i)=make_uint4(0x3c003c00u,0,0x3c003c00u,0);
  unsigned query=A::Async::Shared(s.query),key=A::Async::Shared(s.kv[0][0]),value=A::Async::Shared(s.kv[0][1]);
  typename A::State st;
  for(int n=0;n<D/8;++n)st.o[n][0]=st.o[n][1]=st.o[n][2]=st.o[n][3]=0;
  st.maximum[0]=st.maximum[1]=-INFINITY;st.sum[0]=st.sum[1]=0;
  int warp=tid>>5;
  for(int k=0;k<3;++k)for(int i=0;i<samples;++i) {
    codegen::executor::ComputeSync();
    U64 begin=clock64();
    if(k==0)A::template Block<64>(query,key,value,warp*16,64,st);
    else if(k==1)A::Prepare(p,s,0,backend::Sm90DenseKeyRange{0,p.past+1});
    else {
      // S-1 earlier arrivals make this CTA the last of S chunks.
      if(tid==0)atomicAdd(reinterpret_cast<unsigned long long*>(p.arrivals),(unsigned long long)(S-1));
      codegen::executor::ComputeSync();
      A::Finish(p,s,0,S-1);
      if(backend::Sm90DenseLastArriver::Publish<S>(p.arrivals,&s.last))A::Merge(p,s,0);
    }
    codegen::executor::ComputeSync();
    if(tid==0)out[k*samples+i]=clock64()-begin;
  }
  if(tid==0 && st.sum[0]<0)out[0]=0;
}
template<int D,int Q,int Norm,int S,int Hkv=8>
void AttentionCase() {
  constexpr int samples=256;
  using A=backend::Sm90DenseAttentionBody<1,Hkv,D,Q,Norm,S,64,2,0,1088>;
  auto kernel=AttentionKernel<D,Q,Norm,S,Hkv>;
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(sizeof(typename A::Storage))));
  auto* qkv=Device<BF>(Hkv*(Q+2)*D);auto* cache=Device<BF>(Hkv*1088*D);auto* table=Device<BF>(1088*D);
  auto* output=Device<BF>(Hkv*Q*D);auto* partial=Device<float>(Hkv*S*Q*D);auto* lse=Device<float>(Hkv*S*Q);
  auto* arrivals=Device<std::uint64_t>(Hkv);auto* out=Device<U64>(3*samples);
  backend::Sm90DenseAttentionOperands p{nullptr,nullptr,qkv,cache,cache,output,table,table,table,table,
      partial,lse,arrivals,575,1e-6f,0};
  kernel<<<1,128,sizeof(typename A::Storage)>>>(p,samples,out);
  Check(cudaDeviceSynchronize());
  std::vector<U64> host(3*samples);
  Check(cudaMemcpy(host.data(),out,host.size()*8,cudaMemcpyDeviceToHost));
  auto key="d="+std::to_string(D)+":q="+std::to_string(Q)+":norm="+std::to_string(Norm);
  char const* names[]={"t_blk","t_qprep","t_merge"};
  // Block and preparation do not depend on S: report them once (S = 8).
  for(int k=S==8?0:2;k<3;++k) {
    std::vector<double> x;for(int i=8;i<samples;++i)x.push_back(host[k*samples+i]/cycles_per_ns);
    Print(names[k],key+(k==2?":s="+std::to_string(S):""),x,"ns");
  }
  for(void* ptr:{(void*)qkv,(void*)cache,(void*)table,(void*)output,(void*)partial,(void*)lse,(void*)arrivals,(void*)out})
    Check(cudaFree(ptr));
}

// t_merge_dsmem (D10b): a cluster of S CTAs, one per chunk, in the program's
// shared-memory layout (rings, cluster state, storage at 1024). Each round
// starts at a cluster barrier; the leader's time runs from its partial to
// the merged context (all chunks write their partials concurrently).
template<int D,int Q,int Norm,int S>
__global__ void DsmemMergeKernel(backend::Sm90DenseAttentionOperands p,int samples,U64* out) {
  using A=backend::Sm90DenseAttentionBody<1,8,D,Q,Norm,S,64,2,0,1088,true>;
  using Async=codegen::executor::Async<arch::Sm90>;
  extern __shared__ __align__(1024) char shared[];
  auto* ring=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  auto& cluster=backend::Sm90DenseClusterState::Of(shared);
  auto& s=*reinterpret_cast<typename A::Storage*>(shared+1024);
  int tid=threadIdx.x;
  for(int i=tid;i<4*Q;i+=128) {s.maximum[i]=0;s.sum[i]=1;}
  for(int i=tid;i<4*Q*D;i+=128)s.output[i]=float(i%7);
  if(tid==0) {Async::Init(&cluster.inbox,S-1);cluster.inbox_generation=0;Async::InitFence();}
  __syncthreads();
  int chunk=int(Async::ClusterRank());
  for(int i=0;i<samples;++i) {
    Async::ClusterSync();
    U64 begin=clock64();
    A::Finish(p,s,0,chunk);
    A::DsmemMerge(p,*ring,s,0,chunk);
    codegen::executor::ComputeSync();
    if(tid==0 && chunk==0)out[i]=clock64()-begin;
  }
  Async::ClusterSync();
}
template<int D,int Q,int Norm,int S>
void DsmemMergeCase() {
  constexpr int samples=256;
  using A=backend::Sm90DenseAttentionBody<1,8,D,Q,Norm,S,64,2,0,1088,true>;
  auto kernel=DsmemMergeKernel<D,Q,Norm,S>;
  int bytes=1024+int(sizeof(typename A::Storage));
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  auto* output=Device<BF>(8*Q*D);auto* out=Device<U64>(samples);
  backend::Sm90DenseAttentionOperands p{nullptr,nullptr,nullptr,nullptr,nullptr,output,nullptr,nullptr,
      nullptr,nullptr,nullptr,nullptr,nullptr,575,1e-6f,0};
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(S);config.blockDim=dim3(128);config.dynamicSmemBytes=bytes;
  cudaLaunchAttribute attribute{};
  attribute.id=cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x=S;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
  config.attrs=&attribute;config.numAttrs=1;
  Check(cudaLaunchKernelEx(&config,kernel,p,samples,out));
  Check(cudaDeviceSynchronize());
  std::vector<U64> host(samples);
  Check(cudaMemcpy(host.data(),out,host.size()*8,cudaMemcpyDeviceToHost));
  std::vector<double> x;for(int i=8;i<samples;++i)x.push_back(host[i]/cycles_per_ns);
  Print("t_merge_dsmem","d="+std::to_string(D)+":q="+std::to_string(Q)+":norm="+std::to_string(Norm)+
      ":s="+std::to_string(S),x,"ns");
  Check(cudaFree(output));Check(cudaFree(out));
}

// N-1 forward constants (prompt §10 item 8): one SM's sustained WGMMA
// throughput at each forward tile width. eta(N) is that rate divided by the
// best rate over the measured widths; the cost model prices a stage with its
// own N, so no shape is extrapolated from another.
template<int N>
__global__ __launch_bounds__(128) void WgmmaRate(U64* out,int iterations) {
  using namespace cute;
  using Mma=decltype(cute::make_tiled_mma(typename backend::Sm90Gmma<arch::Sm90>::template SS<N>{}));
  extern __shared__ __align__(1024) char shared[];
  auto* a=reinterpret_cast<BF*>(shared);
  auto* b=a+64*64;
  for(int i=threadIdx.x;i<64*64+N*64;i+=128)
    (i<64*64?a:b)[i<64*64?i:i-64*64]=BF(1.0f/64.0f);
  __syncthreads();
  auto accum=partition_fragment_C(Mma{},Shape<_64,Int<N>>{});clear(accum);
  auto thread=Mma{}.get_slice(int(threadIdx.x));
  auto ta=make_tensor(make_smem_ptr(a),backend::Sm90KLayout<64,64>{});
  auto tb=make_tensor(make_smem_ptr(b),backend::Sm90KLayout<N,64>{});
  auto ra=thread.partition_fragment_A(ta);
  auto rb=thread.partition_fragment_B(tb);
  U64 begin=clock64();
  for(int it=0;it<iterations;++it) {
    backend::Sm90Gmma<arch::Sm90>::Arrive();
    CUTE_UNROLL
    for(int k=0;k<size<2>(ra);++k)gemm(Mma{},ra(_,_,k),rb(_,_,k),accum);
    backend::Sm90Gmma<arch::Sm90>::Commit();
    backend::Sm90Gmma<arch::Sm90>::template Wait<0>();
  }
  U64 elapsed=clock64()-begin;
  if(threadIdx.x==0)out[blockIdx.x]=elapsed;
  // Keep the accumulator live so the loop is not removed.
  if(threadIdx.x==1024)out[0]=U64(accum(0));
}
template<int N>
void WgmmaCase(int sms) {
  constexpr int iterations=512;
  int bytes=2*(64*64+N*64);
  auto kernel=WgmmaRate<N>;
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  auto* out=Device<U64>(sms);
  kernel<<<sms,128,bytes>>>(out,iterations);
  Check(cudaDeviceSynchronize());
  std::vector<U64> host(sms);
  Check(cudaMemcpy(host.data(),out,host.size()*8,cudaMemcpyDeviceToHost));
  Check(cudaFree(out));
  // 2 * 64 * N * 64 FLOP per iteration on one SM.
  std::vector<double> rates;
  for(int i=0;i<sms;++i) {
    double ns=double(host[i])/cycles_per_ns;
    rates.push_back(2.0*64.0*N*64.0*iterations/ns);
  }
  Print("t_wgmma","n="+std::to_string(N),rates,"flop_per_ns");
}

// Whole embedding/argmax stages, each followed by the program's V3 barrier.
template<int Width>
__global__ void EmbeddingStage(int const* tokens,BF const* table,BF* output,float* sums,int batch,
    U64* counter,U64* stamps,int iterations) {
  for(int it=0;it<iterations;++it) {
    if(threadIdx.x<128)for(int row=blockIdx.x;row<batch;row+=gridDim.x)
      backend::Sm90DenseEmbedding<Width,151936,1088>::Run(tokens,table,output,row,it%1086);
    GridBarrier(counter,it);
    if(blockIdx.x==0 && threadIdx.x==0)stamps[it]=Now();
  }
}
// t_argmax_la (D9): the lm_head finish without its GEMM: per-CTA partials,
// lm_ctr, the last arriver's B x G reduction, then the V3 barrier.
template<int Batch>
__global__ void ArgmaxLaStage(float* values,int* indices,int* tokens,U64* lm,
    U64* counter,U64* stamps,int iterations) {
  __shared__ float best[16];__shared__ int arg[16];
  __shared__ alignas(128) char free[256];
  __shared__ backend::Sm90DenseRingState unused;  // only the LA embedding uses a barrier
  backend::Sm90DenseOperands p{nullptr,nullptr,nullptr,reinterpret_cast<BF*>(values),indices,0.0f};
  backend::Sm90DenseHeadOperands head{tokens,reinterpret_cast<std::uint64_t*>(lm),nullptr,nullptr,0};
  for(int it=0;it<iterations;++it) {
    if(threadIdx.x<16) {
      best[threadIdx.x]=float((blockIdx.x*7+it*3+threadIdx.x)%101);
      arg[threadIdx.x]=int(blockIdx.x*16+threadIdx.x);
    }
    head.past=it%1086;
    if(threadIdx.x<128) {
      codegen::executor::ComputeSync();
      backend::Sm90DenseHeadFinish<Batch,1024,1088,false>{&head}(p,unused,best,arg,free,256);
    }
    GridBarrier(counter,it);
    if(blockIdx.x==0 && threadIdx.x==0)stamps[it]=Now();
  }
}

template<class Kernel,class... Args>
void StageCase(char const* name,std::string const& key,Kernel kernel,int grid,Args... args) {
  constexpr int iterations=512;
  auto* counter=Device<U64>(1);auto* stamps=Device<U64>(iterations);
  kernel<<<grid,160>>>(args...,counter,stamps,iterations);
  Check(cudaDeviceSynchronize());
  std::vector<U64> host(iterations);
  Check(cudaMemcpy(host.data(),stamps,iterations*8,cudaMemcpyDeviceToHost));
  std::vector<double> x;for(int i=17;i<iterations;++i)x.push_back(double(host[i]-host[i-1]));
  Print(name,key,x,"ns");
  Check(cudaFree(counter));Check(cudaFree(stamps));
}

// L3 gates (prompt §7). %globaltimer advances in 256 ns steps on H800, which
// cannot resolve a 0.15 us gate per sample. Each gate kernel therefore runs
// back-to-back rounds of one mechanism; CTA 0 stamps every round and a sample
// is the mean round over a 32-round window. V3 runs in the same form.
constexpr int kGateRounds=4096,kGateWindow=32;
__device__ void Stamp(U64* out,int round) {
  if(blockIdx.x==0 && threadIdx.x==0)out[round]=Now();
}
__device__ void Arrive(U64* counter) {
  asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
}
__device__ void Await(U64* counter,U64 target) {
  U64 v;do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
  while(v<target);
}
// V3 and the D8 group counter: CTAs [16g, 16g+16) count on their own word.
template<bool Group>
__global__ void BarrierRounds(U64* counters,U64* out) {
  U64* counter=counters+(Group?(blockIdx.x/16)*8:0);
  unsigned members=Group?min(16u,gridDim.x-blockIdx.x/16*16):gridDim.x;
  for(int i=0;i<kGateRounds;++i) {
    __syncthreads();
    if(threadIdx.x==0) {Arrive(counter);Await(counter,U64(i+1)*members);}
    __syncthreads();
    Stamp(out,i);
  }
}
// D10a: cluster barrier, cluster rank 0 joins the global counter (G/C
// arrivals), cluster barrier again.
__global__ void ClusterRounds(U64* counter,U64* out) {
  unsigned rank,size;
  asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(rank));
  asm volatile("mov.u32 %0, %%cluster_nctarank;":"=r"(size));
  for(int i=0;i<kGateRounds;++i) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;":::"memory");
    if(rank==0 && threadIdx.x==0) {Arrive(counter);Await(counter,U64(i+1)*(gridDim.x/size));}
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;":::"memory");
    Stamp(out,i);
  }
}
__global__ void ClusterProbe() {}
// DT-5: an all-to-all boundary, as between dense stages: CTA b writes the
// words k = b (mod G) of one activation and every CTA reads all of it. Bytes
// is the LL-encoded size per reader (the rule 2*B*K*2 <= X_LL); both variants
// carry the same Bytes/2 payload. Two banks: a writer reaches round i+2 only
// after every reader has finished round i (all-to-all transitivity).
// V3: generic stores, V3 barrier, proxy fence, one bulk read into smem.
template<int Bytes>
__global__ void HandoffV3(char* data,U64* counter,U64* out) {
  constexpr int kPayload=Bytes/2;
  __shared__ alignas(128) char buffer[kPayload];
  __shared__ alignas(8) std::uint64_t bar;
  if(threadIdx.x==0) {Async::Init(&bar,1);Async::InitFence();}
  __syncthreads();
  for(int i=0;i<kGateRounds;++i) {
    auto* bank=data+(i%2)*kPayload;auto* words=reinterpret_cast<unsigned*>(bank);
    for(int k=int(blockIdx.x)+int(gridDim.x)*int(threadIdx.x);k<kPayload/4;k+=int(gridDim.x*blockDim.x))
      words[k]=unsigned(i*31+k);
    __syncthreads();
    if(threadIdx.x==0) {
      Arrive(counter);Await(counter,U64(i+1)*gridDim.x);
      Async::ProxyAsyncGlobalFence();
      Async::ExpectTx(&bar,kPayload);Async::Bulk(buffer,bank,kPayload,&bar);
      Async::Wait(&bar,i&1);
    }
    __syncthreads();
    Stamp(out,i);
  }
}
// LL: 8-byte {payload, stamp} words polled by every reader (D10d).
template<int Bytes>
__global__ void HandoffLL(U64* data,U64* out) {
  constexpr int kWords=Bytes/8;
  for(int i=0;i<kGateRounds;++i) {
    U64* bank=data+(i%2)*kWords;U64 stamp=U64(i+1)<<32;
    for(int k=int(blockIdx.x)+int(gridDim.x)*int(threadIdx.x);k<kWords;k+=int(gridDim.x*blockDim.x))
      asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"::"l"(bank+k),"l"(stamp|unsigned(k)):"memory");
    for(int k=threadIdx.x;k<kWords;k+=blockDim.x) {
      U64 v;do {asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];":"=l"(v):"l"(bank+k):"memory");}
      while((v>>32)!=U64(i+1));
    }
    __syncthreads();
    Stamp(out,i);
  }
}
void Rounds(char const* name,std::string const& key,U64 const* stamps) {
  std::vector<U64> host(kGateRounds);
  Check(cudaMemcpy(host.data(),stamps,kGateRounds*8,cudaMemcpyDeviceToHost));
  std::vector<double> x;
  for(int i=256;i+kGateWindow<kGateRounds;i+=kGateWindow)
    x.push_back(double(host[i+kGateWindow]-host[i])/kGateWindow);
  Print(name,key,x,"ns");
}
void Gates(int sms) {
  auto* counter=Device<U64>(64*8);auto* stamps=Device<U64>(kGateRounds);
  for(int grid:{sms,2*sms}) {
    auto key="g="+std::to_string(grid);
    Check(cudaMemset(counter,0,64*64));BarrierRounds<false><<<grid,160>>>(counter,stamps);
    Check(cudaDeviceSynchronize());Rounds("round_v3",key,stamps);
    Check(cudaMemset(counter,0,64*64));BarrierRounds<true><<<grid,160>>>(counter,stamps);
    Check(cudaDeviceSynchronize());Rounds("round_group16",key,stamps);
  }
  // C*: the largest C whose maximum active clusters cover 128 CTAs, at the
  // native kernel's dynamic shared memory for each residency.
  cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
  for(int residency:{1,2}) {
    int reserved=std::max(0,int(prop.sharedMemPerMultiprocessor)-int(prop.sharedMemPerBlockOptin));
    int smem=std::min(int(prop.sharedMemPerBlockOptin),int(prop.sharedMemPerMultiprocessor)/residency-reserved);
    Check(cudaFuncSetAttribute(ClusterProbe,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    Check(cudaFuncSetAttribute(ClusterProbe,cudaFuncAttributeNonPortableClusterSizeAllowed,1));
    int best=1;
    for(int c:{2,4,8}) {
      cudaLaunchConfig_t config{};config.gridDim=dim3(c*64);config.blockDim=dim3(160);
      config.dynamicSmemBytes=smem;
      cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim.x=c;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
      config.attrs=&attribute;config.numAttrs=1;
      int clusters=0;
      Check(cudaOccupancyMaxActiveClusters(&clusters,ClusterProbe,&config));
      Print("max_active_clusters","c="+std::to_string(c)+":r="+std::to_string(residency),
          {double(clusters)},"clusters");
      if(clusters*c>=128)best=c;
    }
    Print("c_star","r="+std::to_string(residency),{double(best)},"ctas");
    if(best<2)continue;
    int grid=sms*residency/best*best;
    cudaLaunchConfig_t config{};config.gridDim=dim3(grid);config.blockDim=dim3(160);
    cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x=best;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
    config.attrs=&attribute;config.numAttrs=1;
    Check(cudaMemset(counter,0,8));
    Check(cudaLaunchKernelEx(&config,ClusterRounds,counter,stamps));Check(cudaDeviceSynchronize());
    Rounds("round_hier","c="+std::to_string(best)+":g="+std::to_string(grid),stamps);
  }
  auto* data=Device<char>(2*32768);
  auto handoff=[&]<int Bytes>(std::integral_constant<int,Bytes>) {
    for(int grid:{sms,2*sms}) {
      auto key="bytes="+std::to_string(Bytes)+":g="+std::to_string(grid);
      Check(cudaMemset(counter,0,8));HandoffV3<Bytes><<<grid,160>>>(data,counter,stamps);
      Check(cudaDeviceSynchronize());Rounds("round_handoff_v3",key,stamps);
      Check(cudaMemset(data,0,2*32768));HandoffLL<Bytes><<<grid,160>>>(reinterpret_cast<U64*>(data),stamps);
      Check(cudaDeviceSynchronize());Rounds("round_handoff_ll",key,stamps);
    }
  };
  handoff(std::integral_constant<int,2048>{});handoff(std::integral_constant<int,8192>{});
  handoff(std::integral_constant<int,32768>{});
  Check(cudaFree(data));Check(cudaFree(counter));Check(cudaFree(stamps));
}

int main(int argc,char** argv) {try {
  bool full=argc<2;
  Check(cudaSetDevice(0));cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=9 || prop.minor!=0)throw std::runtime_error("SM90R3 constants require sm90");
  int sms=prop.multiProcessorCount;
  {
    auto* clock=Device<U64>(2);Clock<<<1,1>>>(clock);Check(cudaDeviceSynchronize());
    U64 host[2];Check(cudaMemcpy(host,clock,16,cudaMemcpyDeviceToHost));Check(cudaFree(clock));
    cycles_per_ns=double(host[0])/double(host[1]);
  }
  std::printf("name,key,samples,median,p99,unit\n");
  Print("clock","",{cycles_per_ns*1e3},"MHz");
  if(full || !std::strcmp(argv[1],"--base-only")) {
    auto* counter=Device<U64>(1);auto* stamps=Device<U64>(264*256*2);
    std::vector<U64> host(264*256*2);
    for(int grid:{128,sms,2*sms}) {
      Check(cudaMemset(counter,0,8));V3<<<grid,160>>>(counter,stamps,256);Check(cudaDeviceSynchronize());
      Check(cudaMemcpy(host.data(),stamps,grid*256*16,cudaMemcpyDeviceToHost));
      std::vector<double> tails;
      for(int i=16;i<256;++i) {
        U64 last=0;for(int b=0;b<grid;++b)last=std::max(last,host[(i*grid+b)*2]);
        for(int b=0;b<grid;++b)tails.push_back(double(host[(i*grid+b)*2+1])-double(last));
      }
      Print("c_bar","g="+std::to_string(grid),tails,"ns");
    }
    // 4 GiB exceeds L2 by far; cold samples use each 16 KB region once.
    std::size_t big=std::size_t(4)<<30;
    auto* data=Device<char>(big);
    auto* latency=Device<U64>(sms*64);std::vector<U64> lat(sms*64);
    auto latency_case=[&](char const* name,std::size_t stride,int grid,int samples,bool loaded,int skip) {
      Check(cudaMemset(counter,0,8));
      Latency<<<grid,32>>>(data,stride,samples,latency,counter,loaded);Check(cudaDeviceSynchronize());
      Check(cudaMemcpy(lat.data(),latency,std::size_t(grid)*samples*8,cudaMemcpyDeviceToHost));
      std::vector<double> x;
      for(int i=skip*grid;i<grid*samples;++i)x.push_back(lat[i]/cycles_per_ns);
      Print(name,"bytes=16384:g="+std::to_string(grid),x,"ns");
    };
    latency_case("L_l2_single",0,1,64,false,4);
    latency_case("L_dram_single",std::size_t(1)<<20,1,64,false,0);
    latency_case("L_l2_loaded",16384*0,sms,64,true,4);
    latency_case("L_dram_loaded",std::size_t(1)<<20,sms,24,true,0);
    CUtensorMap none{};
    double dram=StreamRun<4,32768,false>(none,data,big/sms/32768*32768,int(big/sms/32768),0,sms);
    Print("BW_dram","slots=4:bytes=32768:g="+std::to_string(sms),{dram},"GB/s");
    // 8 MiB of 64-column BF16 rows, read as 128-row 16 KB SWIZZLE_128B boxes.
    codegen::executor::TensorMap resident{};
    int rows=8*1024*1024/128;
    if(codegen::executor::EncodeTensorMap(resident,data,{64,std::uint64_t(rows),128,64,128})!=CUDA_SUCCESS)
      throw std::runtime_error("tensor map encode failed");
    StreamRun<4,16384,true>(resident.descriptor,data,0,512,rows,sms);
    double l2=StreamRun<4,16384,true>(resident.descriptor,data,0,2048,rows,sms);
    Print("BW_l2","slots=4:bytes=16384:resident=8MiB:g="+std::to_string(sms),{l2},"GB/s");
    double one=StreamRun<8,16384,true>(resident.descriptor,data,0,4096,rows,1);
    Print("BW_sm","slots=8:bytes=16384:resident=8MiB:g=1",{one},"GB/s");
    // MR6: one CTA, 8 x 16 KB bulk requests in flight over cold memory.
    double cta=StreamRun<8,16384,false>(none,data,big,4096,0,1);
    Print("BW_cta_dram","slots=8:bytes=16384:g=1",{cta},"GB/s");
    Check(cudaFree(data));Check(cudaFree(latency));
    // Embedding/argmax stages for the five dense models' widths and vocabularies.
    std::mt19937 random(7);
    std::vector<int> tokens(16*1088);for(auto& t:tokens)t=int(random()%128000);
    auto* token_device=Device<int>(tokens.size());
    Check(cudaMemcpy(token_device,tokens.data(),tokens.size()*4,cudaMemcpyHostToDevice));
    auto* table=Device<BF>(std::size_t(151936)*4096);
    auto* output=Device<BF>(16*4096);auto* sums=Device<float>(16*4096/8);
    for(int grid:{sms,2*sms})for(int batch:{1,2,4,8,16}) {
      auto key=[&](int w) {return "w="+std::to_string(w)+":b="+std::to_string(batch)+":g="+std::to_string(grid);};
      StageCase("t_emb",key(1024),EmbeddingStage<1024>,grid,(int const*)token_device,(BF const*)table,output,sums,batch);
      StageCase("t_emb",key(2048),EmbeddingStage<2048>,grid,(int const*)token_device,(BF const*)table,output,sums,batch);
      StageCase("t_emb",key(3072),EmbeddingStage<3072>,grid,(int const*)token_device,(BF const*)table,output,sums,batch);
      StageCase("t_emb",key(4096),EmbeddingStage<4096>,grid,(int const*)token_device,(BF const*)table,output,sums,batch);
    }
    Check(cudaFree(table));Check(cudaFree(output));Check(cudaFree(sums));
    auto* values=Device<float>(16*2*sms);auto* indices=Device<int>(16*2*sms);auto* lm=Device<U64>(1);
    for(int grid:{sms,2*sms}) {
      auto key=[&](int b) {return "b="+std::to_string(b)+":g="+std::to_string(grid);};
      StageCase("t_argmax_la",key(1),ArgmaxLaStage<1>,grid,values,indices,token_device,lm);
      StageCase("t_argmax_la",key(2),ArgmaxLaStage<2>,grid,values,indices,token_device,lm);
      StageCase("t_argmax_la",key(4),ArgmaxLaStage<4>,grid,values,indices,token_device,lm);
      StageCase("t_argmax_la",key(8),ArgmaxLaStage<8>,grid,values,indices,token_device,lm);
      StageCase("t_argmax_la",key(16),ArgmaxLaStage<16>,grid,values,indices,token_device,lm);
    }
    Check(cudaFree(values));Check(cudaFree(indices));Check(cudaFree(lm));
    {
      std::vector<float> l(16*128);for(auto& x:l)x=float(random()%100000)/1000.f;
      auto* logits=Device<float>(l.size());
      Check(cudaMemcpy(logits,l.data(),l.size()*4,cudaMemcpyHostToDevice));
      auto* schedule=Device<int>(backend::Sm90DenseMoeSchedule<128>::kInts);
      auto* routes=Device<BF>(16*8);auto* route_index=Device<int>(16*8);auto* router=Device<U64>(1);
      for(int grid:{sms,2*sms}) {
        auto key=[&](int b) {return "b="+std::to_string(b)+":g="+std::to_string(grid);};
        StageCase("t_route_la",key(1),RouteLaStage<1>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",key(2),RouteLaStage<2>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",key(4),RouteLaStage<4>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",key(8),RouteLaStage<8>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",key(16),RouteLaStage<16>,grid,logits,schedule,routes,route_index,router);
        // OLMoE (A3): 64 experts, softmax over all of them.
        auto e64=[&](int b) {return "b="+std::to_string(b)+":g="+std::to_string(grid)+":e=64:all=1";};
        StageCase("t_route_la",e64(1),RouteLaStage<1,64,true>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",e64(2),RouteLaStage<2,64,true>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",e64(4),RouteLaStage<4,64,true>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",e64(8),RouteLaStage<8,64,true>,grid,logits,schedule,routes,route_index,router);
        StageCase("t_route_la",e64(16),RouteLaStage<16,64,true>,grid,logits,schedule,routes,route_index,router);
      }
      for(void* q:{(void*)logits,(void*)schedule,(void*)routes,(void*)route_index,(void*)router})Check(cudaFree(q));
    }
    Check(cudaFree(token_device));
    Check(cudaFree(counter));Check(cudaFree(stamps));
  }
  // Epilogue outputs rotate through copies so L1-resident lines do not persist.
  epilogue.out_copy=16*151936;
  epilogue.output=Device<BF>(epilogue.copies*epilogue.out_copy*2);
  epilogue.index=Device<int>(epilogue.copies*epilogue.out_copy);
  if(full || !std::strcmp(argv[1],"--gates-only"))Gates(sms);
  if(full || !std::strcmp(argv[1],"--epilogue-only")) {
    EpilogueTiles<ServingEpilogueOp::kStore>("qkv",std::integer_sequence<int,8,16,24,32,40,48,64,80,96,128>{});
    EpilogueTiles<ServingEpilogueOp::kResidual>("residual",std::integer_sequence<int,8,16,24,32,48,64>{});
    EpilogueTiles<ServingEpilogueOp::kSwiGLU>("gate_up",
        std::integer_sequence<int,16,24,32,48,56,64,96,112,128,192,224,256,448,512>{});
    EpilogueTiles<ServingEpilogueOp::kArgmaxPartial>("lm_head",std::integer_sequence<int,128,256>{});
    MoeEpilogueTile<backend::Sm90DenseMoeMode::kRouter,8>("router");
    MoeEpilogueTile<backend::Sm90DenseMoeMode::kGateUp,96>("expert_gate_up");
    MoeEpilogueTile<backend::Sm90DenseMoeMode::kGateUp,192>("expert_gate_up");
    MoeEpilogueTile<backend::Sm90DenseMoeMode::kDown,128>("expert_down");
    MoeEpilogueTile<backend::Sm90DenseMoeMode::kDown,256>("expert_down");
  }
  if(full || !std::strcmp(argv[1],"--attention-only")) {
    [&]<int... S>(std::integer_sequence<int,S...>) {
      (AttentionCase<128,2,1,S>(),...);(AttentionCase<64,4,0,S>(),...);
      (AttentionCase<128,3,0,S>(),...);(AttentionCase<128,4,0,S>(),...);
      (AttentionCase<128,8,1,S>(),...);
    }(std::integer_sequence<int,2,4,8,16,32>{});
  }
  if(full || !std::strcmp(argv[1],"--moe-task-only")) {
    using backend::Sm90DenseMoeMode;
    [&]<int... B>(std::integer_sequence<int,B...>) {
      (MoeTaskCase<Sm90DenseMoeMode::kGateUp,B,1536,2048,96,128,4,2>("expert_gate_up"),...);
      (MoeTaskCase<Sm90DenseMoeMode::kDown,B,2048,768,128,64,6,2>("expert_down"),...);
    }(std::integer_sequence<int,1,2,4,8,16>{});
  }
  // A3: OLMoE (E = 64, I = 1024) under its own families. A slot's rows follow
  // n_active(E), so Qwen3's E = 128 entries do not stand in for these tiles.
  if(full || !std::strcmp(argv[1],"--e64-only")) {
    using backend::Sm90DenseMoeMode;
    MoeEpilogueTile<Sm90DenseMoeMode::kRouter,8,64>("router_e64");
    MoeEpilogueTile<Sm90DenseMoeMode::kGateUp,64,64>("expert_gate_up_e64");
    MoeEpilogueTile<Sm90DenseMoeMode::kGateUp,128,64>("expert_gate_up_e64");
    MoeEpilogueTile<Sm90DenseMoeMode::kGateUp,256,64>("expert_gate_up_e64");
    MoeEpilogueTile<Sm90DenseMoeMode::kDown,64,64>("expert_down_e64");
    MoeEpilogueTile<Sm90DenseMoeMode::kDown,128,64>("expert_down_e64");
    MoeEpilogueTile<Sm90DenseMoeMode::kDown,256,64>("expert_down_e64");
    [&]<int... B>(std::integer_sequence<int,B...>) {
      (MoeTaskCase<Sm90DenseMoeMode::kGateUp,B,2048,2048,128,128,4,2,64>("expert_gate_up_e64"),...);
      (MoeTaskCase<Sm90DenseMoeMode::kDown,B,2048,1024,128,64,6,2,64>("expert_down_e64"),...);
    }(std::integer_sequence<int,1,2,4,8,16>{});
  }
  // A3: OLMoE's attention -- MHA (Q = 1), the whole-row q/k norm, Hkv = 16.
  // Its S0 is 4 (G = 132) or 8 (G = 264) at B1 and 1 at B16, so S <= 8.
  if(full || !std::strcmp(argv[1],"--q1-only")) {
    [&]<int... S>(std::integer_sequence<int,S...>) {(AttentionCase<128,1,2,S,16>(),...);}
        (std::integer_sequence<int,2,4,8>{});
    DsmemMergeCase<128,1,2,2>();
  }
  // MoE (Hkv = 4) at G = 264 reaches S0 = 64 chunks (prompt §8 M2).
  if(full || !std::strcmp(argv[1],"--moe-merge-only"))AttentionCase<128,8,1,64>();
  // D10b: clusters of C = 2 (DT-2's grids) merge S = 2 chunks through DSMEM.
  if(full || !std::strcmp(argv[1],"--forward-only")) {
    WgmmaCase<32>(sms);WgmmaCase<64>(sms);WgmmaCase<128>(sms);WgmmaCase<256>(sms);
  }
  if(full || !std::strcmp(argv[1],"--dsmem-merge-only")) {
    DsmemMergeCase<128,2,1,2>();DsmemMergeCase<64,4,0,2>();DsmemMergeCase<128,3,0,2>();
    DsmemMergeCase<128,4,0,2>();DsmemMergeCase<128,8,1,2>();
  }
  return 0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
