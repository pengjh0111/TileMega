// SPDX-License-Identifier: BSD-3-Clause
// SM90R5 tensor parallel constants (prompt B1, B9.1), one process over the
// T devices CUDA_VISIBLE_DEVICES exposes. Every section prints one JSON line
// per key: {"name","key","median_ns","samples","p10_ns","p90_ns"}.
//
//   --nvl   nvl_store_ns: one-way time of an 8-byte strong store between two
//           cards (ping-pong, half the median round trip); nvl_gbps: c CTAs
//           storing 8 MB with 16-byte strong stores to 1 or 3 peers;
//           t_fence_sys: one thread's fence.acq_rel.sys after n remote stores.
//   --ar    t_ar(B, TN, T): an o-shaped stage (N 2048, K 2048/T) on a persistent
//           grid of every SM, built with kResidualAllReduce and with
//           kResidual; 1000 iterations each, median of the per-iteration
//           differences. Each iteration owns a fresh receive region, so the
//           reuse argument of plan §3.5 C2 (which needs the head exchange)
//           is not exercised; the end-to-end litmus covers it.
//   --head  t_head_x(B, T): Sm90DenseHeadFinish at Tp = T against Tp = 1 on
//           the same grid and candidates, same method.
// Weights stay L2-resident across iterations; both builds read the same
// bytes, so the difference carries only the reduction (or exchange).
#define TILEMEGA_SWIGLU_U 8
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseHead.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using U64=unsigned long long;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}

__device__ U64 Now() {U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ void GridBarrier(U64* counter,U64 round) {
  __syncthreads();
  if(threadIdx.x==0) {
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;"::"l"(counter):"memory");
    U64 v;
    do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(counter):"memory");}
    while(v<(round+1)*gridDim.x);
  }
  __syncthreads();
}

void Print(char const* name,std::string const& key,std::vector<double> x) {
  if(x.empty())return;
  std::sort(x.begin(),x.end());
  auto at=[&](double q){return x[std::size_t(q*(x.size()-1))];};
  std::printf("{\"name\":\"%s\",\"key\":\"%s\",\"median_ns\":%.3f,\"samples\":%zu,\"p10_ns\":%.3f,\"p90_ns\":%.3f}\n",
              name,key.c_str(),at(.5),x.size(),at(.1),at(.9));
  std::fflush(stdout);
}

// ---------------------------------------------------------------- --nvl
__global__ void PingPong(U64* mine,U64* peer,int rounds,bool leader,U64* out) {
  if(threadIdx.x)return;
  for(int r=1;r<=rounds;++r) {
    U64 begin=Now();
    if(leader) {
      asm volatile("st.relaxed.sys.global.b64 [%0], %1;"::"l"(peer),"l"(U64(r)):"memory");
      U64 v;do {asm volatile("ld.relaxed.sys.global.b64 %0, [%1];":"=l"(v):"l"(mine):"memory");} while(v!=U64(r));
      out[r-1]=Now()-begin;
    }else {
      U64 v;do {asm volatile("ld.relaxed.sys.global.b64 %0, [%1];":"=l"(v):"l"(mine):"memory");} while(v!=U64(r));
      asm volatile("st.relaxed.sys.global.b64 [%0], %1;"::"l"(peer),"l"(U64(r)):"memory");
    }
  }
}
__global__ void Stores(U64* const* peers,int count,std::size_t words,U64* out) {
  // words 16-byte stores per peer, spread over the grid.
  __syncthreads();
  U64 begin=Now();
  for(int p=0;p<count;++p)
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=std::size_t(gridDim.x)*blockDim.x)
      asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};"::"l"(peers[p]+2*i),"l"(U64(i)),"l"(U64(p)):"memory");
  asm volatile("fence.acq_rel.sys;":::"memory");
  __syncthreads();
  if(threadIdx.x==0)atomicMax(out,Now()-begin);
}
__global__ void Fence(U64* peer,int stores,int rounds,U64* out) {
  if(threadIdx.x)return;
  for(int r=0;r<rounds;++r) {
    for(int i=0;i<stores;++i)asm volatile("st.relaxed.sys.global.b64 [%0], %1;"::"l"(peer+i*8),"l"(U64(r)):"memory");
    U64 begin=Now();
    asm volatile("fence.acq_rel.sys;":::"memory");
    out[r]=Now()-begin;
  }
}

void Nvl(int T) {
  std::vector<U64*> flag(T);
  for(int d=0;d<T;++d) {check(cudaSetDevice(d));check(cudaMalloc(&flag[d],64));check(cudaMemset(flag[d],0,64));}
  constexpr int kRounds=1000;
  for(int peer=1;peer<T;++peer) {
    std::vector<std::vector<double>> runs;
    for(int d:{0,peer}) {check(cudaSetDevice(d));check(cudaMemset(flag[d],0,64));}
    U64* out;check(cudaSetDevice(0));check(cudaMalloc(&out,kRounds*8));
    check(cudaSetDevice(peer));PingPong<<<1,32>>>(flag[peer],flag[0],kRounds,false,nullptr);
    check(cudaSetDevice(0));PingPong<<<1,32>>>(flag[0],flag[peer],kRounds,true,out);
    for(int d:{0,peer}) {check(cudaSetDevice(d));check(cudaDeviceSynchronize());}
    std::vector<U64> host(kRounds);check(cudaSetDevice(0));check(cudaMemcpy(host.data(),out,kRounds*8,cudaMemcpyDeviceToHost));
    std::vector<double> x;for(int i=16;i<kRounds;++i)x.push_back(host[i]/2.0);
    Print("nvl_store_ns","p=0:"+std::to_string(peer),x);check(cudaFree(out));
  }
  // Bandwidth: 8 MB per peer from device 0.
  constexpr std::size_t kBytes=8u<<20,kWords=kBytes/16;
  std::vector<U64*> sink(T);
  for(int d=1;d<T;++d) {check(cudaSetDevice(d));check(cudaMalloc(&sink[d],kBytes));}
  check(cudaSetDevice(0));
  U64** peers;check(cudaMalloc(&peers,8*8));U64* out;check(cudaMalloc(&out,8));
  for(int count:{1,3}) {
    if(count>=T)continue;
    check(cudaMemcpy(peers,sink.data()+1,count*8,cudaMemcpyHostToDevice));
    for(int ctas:{1,8,32,132}) {
      std::vector<double> x;
      for(int r=0;r<12;++r) {
        check(cudaMemset(out,0,8));Stores<<<ctas,512>>>(peers,count,kWords,out);check(cudaDeviceSynchronize());
        U64 ns;check(cudaMemcpy(&ns,out,8,cudaMemcpyDeviceToHost));
        if(r>=2)x.push_back(double(count*kBytes)/ns);  // bytes per ns = GB/s
      }
      Print("nvl_gbps","c="+std::to_string(ctas)+":peers="+std::to_string(count),x);
    }
  }
  U64* fence;check(cudaMalloc(&fence,kRounds*8));
  for(int stores:{0,16}) {
    Fence<<<1,32>>>(sink[1],stores,kRounds,fence);check(cudaDeviceSynchronize());
    std::vector<U64> host(kRounds);check(cudaMemcpy(host.data(),fence,kRounds*8,cudaMemcpyDeviceToHost));
    std::vector<double> x;for(int i=16;i<kRounds;++i)x.push_back(double(host[i]));
    Print("t_fence_sys","n="+std::to_string(stores),x);
  }
}

// ---------------------------------------------------------------- --ar
constexpr int kIterations=1000;
template<class G>
__global__ __launch_bounds__(160) void StageLoop(Sm90DenseOperands p,Sm90DenseTpOperands tp,unsigned per_region,
    U64* counter,U64* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  for(int it=0;it<kIterations;++it) {
    GridBarrier(counter,2*it);
    U64 begin=Now();
    if constexpr(G::kAllReduce) {
      tp.region=unsigned(it)*per_region;tp.epoch=unsigned(it+1);
      G::template Run<0>(p,shared,nullptr,Sm90DenseNoTail{},true,Sm90DenseNoFinish{},&tp);
    }else G::template Run<0>(p,shared);
    GridBarrier(counter,2*it+1);
    if(blockIdx.x==0 && threadIdx.x==0)stamps[it]=Now()-begin;
  }
  state.Invalidate();
}

template<int T,int B,int TN,int WN,ServingEpilogueOp Op>
std::vector<double> StageTimes(int grid) {
  constexpr int N=2048,K=2048/T,TK=K<512?K:512,WK=4/WN;
  using G=Sm90DenseGemm<B,N,K,TN,TK,2,WN,WK,Op,8,false>;
  constexpr int rows=Sm90DenseLayout::ActivationRows(B);
  unsigned per_region=unsigned(T*B*(N/2));
  std::vector<std::uint64_t*> ar(T);std::vector<BF*> act(T),weight(T),res(T),out(T);
  std::vector<U64*> counter(T),stamps(T);std::vector<CUtensorMap*> map(T);
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(q));
    std::size_t region_bytes=Op==ServingEpilogueOp::kResidualAllReduce?std::size_t(kIterations)*per_region*8:64;
    check(cudaMalloc(&ar[q],region_bytes));check(cudaMemset(ar[q],0,region_bytes));
    std::vector<BF> host_w(std::size_t(N)*K),host_a(B*K),host_r(B*N);
    for(std::size_t i=0;i<host_w.size();++i)host_w[i]=BF(float(int(i*2654435761u%17)-8)/1024.0f);
    for(int i=0;i<B*K;++i)host_a[i]=BF(float(i%13-6)/64.0f);
    for(int i=0;i<B*N;++i)host_r[i]=BF(float(i%7-3)/8.0f);
    check(cudaMalloc(&act[q],host_a.size()*2));check(cudaMalloc(&weight[q],host_w.size()*2));
    check(cudaMalloc(&res[q],host_r.size()*2));check(cudaMalloc(&out[q],host_r.size()*2));
    check(cudaMemcpy(act[q],host_a.data(),host_a.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(weight[q],host_w.data(),host_w.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(res[q],host_r.data(),host_r.size()*2,cudaMemcpyHostToDevice));
    check(cudaMalloc(&counter[q],8));check(cudaMemset(counter[q],0,8));
    check(cudaMalloc(&stamps[q],kIterations*8));check(cudaMalloc(&map[q],128));
    CUtensorMap host;
    tilemega::codegen::executor::EncodeTiledMap<3>(host,act[q],
        {{64,std::uint64_t(B),K/64},{2ull*K,128},{64,std::uint32_t(rows),TK/64}});
    check(cudaMemcpy(map[q],&host,sizeof host,cudaMemcpyHostToDevice));
    check(cudaFuncSetAttribute(StageLoop<G>,cudaFuncAttributeMaxDynamicSharedMemorySize,G::kSharedBytes));
  }
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(q));
    Sm90DenseOperands p{map[q],weight[q],res[q],out[q],nullptr,0.f};
    Sm90DenseTpOperands tp{};for(int r=0;r<T;++r)tp.recv[r]=ar[r];tp.rank=q;tp.world=T;
    StageLoop<G><<<grid,160,G::kSharedBytes>>>(p,tp,per_region,counter[q],stamps[q]);
    check(cudaGetLastError());
  }
  std::vector<U64> host(kIterations);
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(q));check(cudaDeviceSynchronize());
    if(q==0)check(cudaMemcpy(host.data(),stamps[q],kIterations*8,cudaMemcpyDeviceToHost));
    for(void* ptr:{(void*)ar[q],(void*)act[q],(void*)weight[q],(void*)res[q],(void*)out[q],
                   (void*)counter[q],(void*)stamps[q],(void*)map[q]})check(cudaFree(ptr));
  }
  return std::vector<double>(host.begin()+16,host.end());
}

template<int T,int B,int TN,int WN>
void ArCase(int grid) {
  auto with=StageTimes<T,B,TN,WN,ServingEpilogueOp::kResidualAllReduce>(grid);
  auto without=StageTimes<T,B,TN,WN,ServingEpilogueOp::kResidual>(grid);
  std::vector<double> difference;
  for(std::size_t i=0;i<with.size();++i)difference.push_back(with[i]-without[i]);
  std::string key="b="+std::to_string(B)+":tn="+std::to_string(TN)+":t="+std::to_string(T);
  Print("t_ar",key+":wk="+std::to_string(4/WN),difference);
  Print("t_ar_stage",key+":wk="+std::to_string(4/WN),with);
  Print("t_residual_stage",key+":wk="+std::to_string(4/WN),without);
}

template<int T,int TN,int WN>
void ArTile(int grid) {
  ArCase<T,1,TN,WN>(grid);ArCase<T,2,TN,WN>(grid);ArCase<T,4,TN,WN>(grid);
  ArCase<T,8,TN,WN>(grid);ArCase<T,16,TN,WN>(grid);
}

// ---------------------------------------------------------------- --head
template<class Finish,int B>
__global__ __launch_bounds__(128) void HeadLoop(Sm90DenseOperands p,Sm90DenseHeadOperands head,
    Sm90DenseTpHeadOperands tp,U64* counter,U64* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);state.Initialize();
  __shared__ float best[16];__shared__ int arg[16];
  for(int it=0;it<kIterations;++it) {
    if(threadIdx.x<B) {best[threadIdx.x]=float((it*7+blockIdx.x*3+threadIdx.x)%23);arg[threadIdx.x]=int(blockIdx.x)*8+threadIdx.x;}
    GridBarrier(counter,2*it);
    U64 begin=Now();
    head.past=it%1000;tp.epoch=std::uint64_t(it)+1;
    if constexpr(sizeof(Finish)>sizeof(void*))Finish{&head,&tp}(p,state,best,arg,shared+1024,4096);
    else Finish{&head}(p,state,best,arg,shared+1024,4096);
    GridBarrier(counter,2*it+1);
    if(blockIdx.x==0 && threadIdx.x==0)stamps[it]=Now()-begin;
  }
  state.Invalidate();
}

template<int T,int B,int Tp>
std::vector<double> HeadTimes(int grid) {
  constexpr int VocabLocal=Tp>1?151936/Tp:0;
  using Finish=Sm90DenseHeadFinish<B,64,1088,false,Tp,VocabLocal>;
  std::vector<Sm90DenseTpExchange*> x(T);std::vector<int*> tokens(T),arg(T);std::vector<float*> part(T);
  std::vector<std::uint64_t*> lm(T);std::vector<U64*> counter(T),stamps(T);
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(q));
    check(cudaMalloc(&x[q],sizeof(Sm90DenseTpExchange)));check(cudaMemset(x[q],0,sizeof(Sm90DenseTpExchange)));
    check(cudaMalloc(&tokens[q],B*1088*4));check(cudaMalloc(&part[q],B*grid*4));check(cudaMalloc(&arg[q],B*grid*4));
    check(cudaMalloc(&lm[q],8));check(cudaMemset(lm[q],0,8));check(cudaMalloc(&counter[q],8));check(cudaMemset(counter[q],0,8));
    check(cudaMalloc(&stamps[q],kIterations*8));
    check(cudaFuncSetAttribute(HeadLoop<Finish,B>,cudaFuncAttributeMaxDynamicSharedMemorySize,8192));
  }
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(q));
    Sm90DenseOperands p{nullptr,nullptr,nullptr,reinterpret_cast<BF*>(part[q]),arg[q],0.f};
    Sm90DenseHeadOperands head{tokens[q],lm[q],nullptr,nullptr,0,0};
    Sm90DenseTpHeadOperands tp{};for(int r=0;r<T;++r)tp.x[r]=x[r];
    tp.rank=q;tp.world=Tp>1?T:1;tp.vocab_local=VocabLocal;
    HeadLoop<Finish,B><<<grid,128,8192>>>(p,head,tp,counter[q],stamps[q]);
    check(cudaGetLastError());
  }
  std::vector<U64> host(kIterations);
  for(int q=0;q<T;++q) {
    check(cudaSetDevice(q));check(cudaDeviceSynchronize());
    if(q==0)check(cudaMemcpy(host.data(),stamps[q],kIterations*8,cudaMemcpyDeviceToHost));
    for(void* ptr:{(void*)x[q],(void*)tokens[q],(void*)part[q],(void*)arg[q],(void*)lm[q],
                   (void*)counter[q],(void*)stamps[q]})check(cudaFree(ptr));
  }
  return std::vector<double>(host.begin()+16,host.end());
}

template<int T,int B>
void HeadCase(int grid) {
  auto with=HeadTimes<T,B,T>(grid);
  auto without=HeadTimes<T,B,1>(grid);
  std::vector<double> difference;
  for(std::size_t i=0;i<with.size();++i)difference.push_back(with[i]-without[i]);
  std::string key="b="+std::to_string(B)+":t="+std::to_string(T);
  Print("t_head_x",key,difference);Print("t_head_stage",key,with);Print("t_head_tp1_stage",key,without);
}

template<int T>
void Run(std::string const& section,int grid) {
  if(section=="--ar") {
    ArTile<T,8,1>(grid);ArTile<T,16,2>(grid);ArTile<T,32,2>(grid);ArTile<T,64,4>(grid);
  }else if(section=="--head") {
    HeadCase<T,1>(grid);HeadCase<T,2>(grid);HeadCase<T,4>(grid);HeadCase<T,8>(grid);HeadCase<T,16>(grid);
  }else if(section=="--nvl")Nvl(T);
  else throw std::invalid_argument("section must be --nvl, --ar or --head");
}

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("usage: sm90r5_tp_probe --nvl|--ar|--head");
    int T=0;check(cudaGetDeviceCount(&T));
    if(T!=2 && T!=4)throw std::invalid_argument("expose exactly 2 or 4 devices");
    for(int d=0;d<T;++d) {
      check(cudaSetDevice(d));
      for(int e=0;e<T;++e)if(e!=d) {
        auto status=cudaDeviceEnablePeerAccess(e,0);
        if(status==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(status);
      }
    }
    cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0));
    int grid=properties.multiProcessorCount;
    if(T==2)Run<2>(argv[1],grid);else Run<4>(argv[1],grid);
    return 0;
  }catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}
}
