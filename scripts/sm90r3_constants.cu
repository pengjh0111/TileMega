// SPDX-License-Identifier: BSD-3-Clause
// SM90R3 native cost-model constants (prompt §4 L0-1). One process prints one
// CSV row per metric; scripts/collect_sm90r3_constants.py takes the median of
// five fresh processes. Diagnostic only: no serving default depends on a run.
#include <tilemega/Backend/Sm90DenseGemm.h>
#include <tilemega/Backend/Sm90DenseAttentionBody.cuh>
#include <tilemega/Backend/Sm90DenseToken.cuh>
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
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

// Attention pieces of the D2 body (prompt §4 L0-1): one 64-key block with data
// resident, the q preparation, and the S-chunk merge with this CTA last.
template<int D,int Q,bool Norm,int S>
__global__ void AttentionKernel(backend::Sm90DenseAttentionOperands p,int samples,U64* out) {
  using A=backend::Sm90DenseAttentionBody<1,8,D,Q,Norm,S,64,2,0,1088>;
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
template<int D,int Q,bool Norm,int S>
void AttentionCase() {
  constexpr int samples=256;
  using A=backend::Sm90DenseAttentionBody<1,8,D,Q,Norm,S,64,2,0,1088>;
  auto kernel=AttentionKernel<D,Q,Norm,S>;
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(sizeof(typename A::Storage))));
  auto* qkv=Device<BF>(8*(Q+2)*D);auto* cache=Device<BF>(8*1088*D);auto* table=Device<BF>(1088*D);
  auto* output=Device<BF>(8*Q*D);auto* partial=Device<float>(8*S*Q*D);auto* lse=Device<float>(8*S*Q);
  auto* arrivals=Device<std::uint64_t>(8);auto* out=Device<U64>(3*samples);
  backend::Sm90DenseAttentionOperands p{nullptr,nullptr,qkv,cache,cache,output,table,table,table,table,
      partial,lse,arrivals,575,1e-6f};
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
template<int Parts>
__global__ void ArgmaxStage(float const* values,int const* indices,int* tokens,int batch,
    U64* counter,U64* stamps,int iterations) {
  __shared__ typename backend::Sm90DenseArgmax<Parts,1088>::Storage storage;
  for(int it=0;it<iterations;++it) {
    if(threadIdx.x<128)for(int row=blockIdx.x;row<batch;row+=gridDim.x)
      backend::Sm90DenseArgmax<Parts,1088>::Run(values+std::size_t(it%8)*16*Parts,
          indices+std::size_t(it%8)*16*Parts,tokens,storage,row,it%1086);
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
  if(full) {
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
    auto* values=Device<float>(8*16*1187);auto* indices=Device<int>(8*16*1187);
    {
      std::vector<float> v(8*16*1187);for(auto& x:v)x=float(random()%100000)/1000.f;
      Check(cudaMemcpy(values,v.data(),v.size()*4,cudaMemcpyHostToDevice));
    }
    for(int grid:{sms,2*sms})for(int batch:{1,2,4,8,16}) {
      auto key=[&](int p) {return "parts="+std::to_string(p)+":b="+std::to_string(batch)+":g="+std::to_string(grid);};
      StageCase("t_argmax",key(1187),ArgmaxStage<1187>,grid,(float const*)values,(int const*)indices,token_device,batch);
      StageCase("t_argmax",key(1002),ArgmaxStage<1002>,grid,(float const*)values,(int const*)indices,token_device,batch);
      StageCase("t_argmax",key(501),ArgmaxStage<501>,grid,(float const*)values,(int const*)indices,token_device,batch);
    }
    Check(cudaFree(values));Check(cudaFree(indices));Check(cudaFree(token_device));
    Check(cudaFree(counter));Check(cudaFree(stamps));
  }
  // Epilogue outputs rotate through copies so L1-resident lines do not persist.
  epilogue.out_copy=16*151936;
  epilogue.output=Device<BF>(epilogue.copies*epilogue.out_copy*2);
  epilogue.index=Device<int>(epilogue.copies*epilogue.out_copy);
  if(full || !std::strcmp(argv[1],"--epilogue-only")) {
    EpilogueTiles<ServingEpilogueOp::kStore>("qkv",std::integer_sequence<int,8,16,24,32,40,48,64,80,96,128>{});
    EpilogueTiles<ServingEpilogueOp::kResidual>("residual",std::integer_sequence<int,8,16,24,32,48,64>{});
    EpilogueTiles<ServingEpilogueOp::kSwiGLU>("gate_up",
        std::integer_sequence<int,16,24,32,48,56,64,96,112,128,192,224,256,448,512>{});
    EpilogueTiles<ServingEpilogueOp::kArgmaxPartial>("lm_head",std::integer_sequence<int,128,256>{});
  }
  if(full || !std::strcmp(argv[1],"--attention-only")) {
    [&]<int... S>(std::integer_sequence<int,S...>) {
      (AttentionCase<128,2,true,S>(),...);(AttentionCase<64,4,false,S>(),...);
      (AttentionCase<128,3,false,S>(),...);(AttentionCase<128,4,false,S>(),...);
      (AttentionCase<128,8,true,S>(),...);
    }(std::integer_sequence<int,2,4,8,16,32>{});
  }
  return 0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
