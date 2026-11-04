// SPDX-License-Identifier: BSD-3-Clause
// DNN1 Phase X. This source is compiled twice: uninstrumented direct
// primitives and production-body segment probes with a build-local clock64
// Mark overlay. The required scalar-store reference is a separate probe.
#ifndef DNN1_SEGMENT_PROBES
#define DNN1_SEGMENT_PROBES 0
#endif
#ifndef DNN1_SHARED_LIMIT
#define DNN1_SHARED_LIMIT 232448
#endif
#ifndef DNN1_FIXED_FORWARD
#define DNN1_FIXED_FORWARD 0
#endif
#if DNN1_SEGMENT_PROBES
#define TILEMEGA_TRACE_STAGE 1
#define TILEMEGA_FORWARD_TRACE_STAGES 1
#define TILEMEGA_FORWARD_TRACE_GRID 256
#include <tilemega/Backend/Sm90ForwardGemm.cuh>
#include <tilemega/Backend/Sm90ForwardDepthwise.cuh>
#include <tilemega/Backend/Sm90ForwardEncoderAttention.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#else
#define TILEMEGA_TRACE_STAGE 0
#include <tilemega/Backend/Sm90ForwardSmall.cuh>
#if DNN1_FIXED_FORWARD
#include <tilemega/Backend/Sm90ForwardGemv.cuh>
#endif
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tilemega;
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using U64=unsigned long long;
constexpr int kDiscard=8;
double cycles_per_ns=0;
std::ofstream raw;
void Check(cudaError_t e) {
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
template<class T>struct Buffer {
  T* p=nullptr;std::size_t count;
  explicit Buffer(std::size_t n):count(n) {Check(cudaMalloc(&p,n*sizeof(T)));Check(cudaMemset(p,0,n*sizeof(T)));}
  ~Buffer(){if(p)cudaFree(p);}
  operator T*()const{return p;}
  Buffer(Buffer const&)=delete;
};
template<class T>void Fill(Buffer<T>& p,float value) {
  std::vector<T> host(p.count,T(value));
  Check(cudaMemcpy(p.p,host.data(),p.count*sizeof(T),cudaMemcpyHostToDevice));
}
__device__ __forceinline__ U64 Cycles() {
  U64 t;asm volatile("mov.u64 %0, %%clock64;":"=l"(t)::"memory");return t;
}
__device__ __forceinline__ U64 GlobalNs() {
  U64 t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;
}
__global__ void Calibrate(U64* out) {
  U64 ns=GlobalNs(),tick=Cycles();
  while(GlobalNs()-ns<20000000ull){}
  out[0]=Cycles()-tick;out[1]=GlobalNs()-ns;
  U64 old=GlobalNs(),minimum=~0ull;
  for(int i=0;i<4096;++i) {
    U64 next=GlobalNs();if(next>old)minimum=min(minimum,next-old);old=next;
  }
  out[2]=minimum;
}
__global__ void Observer(U64* out,int count) {
  for(int i=0;i<count;++i) {
    codegen::executor::ComputeSync();U64 begin=Cycles();
    codegen::executor::ComputeSync();
    if(threadIdx.x==0)out[i]=Cycles()-begin;
  }
}
void Emit(std::string const& metric,std::string const& key,std::vector<double> values,
          std::string const& unit,std::string const& method) {
  if(values.empty())throw std::runtime_error("no measured samples for "+metric);
  for(std::size_t i=0;i<values.size();++i) {
    if(!std::isfinite(values[i]) || values[i]<=0)
      throw std::runtime_error("nonpositive/nonfinite measured interval, not a usable constant: "+metric);
    if(raw)raw<<metric<<','<<key<<','<<i<<','<<std::setprecision(17)<<values[i]<<'\n';
  }
  std::sort(values.begin(),values.end());
  std::cout<<metric<<','<<key<<','<<values.size()<<','<<std::setprecision(17)
      <<values[values.size()/2]<<','<<values[std::min(values.size()-1,values.size()*99/100)]
      <<','<<unit<<','<<method<<'\n';
}
template<class Kernel>void Shared(Kernel kernel,int bytes,int threads=160) {
  int maximum=0;Check(cudaDeviceGetAttribute(&maximum,cudaDevAttrMaxSharedMemoryPerBlockOptin,0));
  if(bytes>maximum)throw std::runtime_error("resource_infeasible: shared bytes "+std::to_string(bytes)+
      " exceed device opt-in "+std::to_string(maximum));
  Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  int active=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,threads,bytes));
  if(active<1)throw std::runtime_error("resource_infeasible: zero resident CTA");
  std::cerr<<"resource shared_bytes="<<bytes<<" active_ctas_per_sm="<<active<<'\n';
}
void Samples(std::string const& metric,std::string const& key,Buffer<U64> const& stamps,
             int count,double elements,char const* method) {
  std::vector<U64> host(count);Check(cudaMemcpy(host.data(),stamps.p,count*8,cudaMemcpyDeviceToHost));
  std::vector<double> values;
  for(int i=kDiscard;i<count;++i)values.push_back(double(host[i])/cycles_per_ns/elements);
  Emit(metric,key,std::move(values),elements==1?"ns":"ns/element",method);
}
struct Case {
  std::string id,metric,key,status,reason;int shared;
  std::function<void(int)> run;
};
std::vector<Case> cases;
void Add(std::string id,std::string metric,std::string key,int shared,std::function<void(int)> fn) {
  if(shared>DNN1_SHARED_LIMIT)
    cases.push_back({std::move(id),std::move(metric),std::move(key),"infeasible",
      "production body exceeds target shared-memory opt-in budget",shared,{}});
  else cases.push_back({std::move(id),std::move(metric),std::move(key),"ready","",shared,std::move(fn)});
}
void Pending(std::string id,std::string metric,std::string key,std::string reason) {
  cases.push_back({std::move(id),std::move(metric),std::move(key),"pending",std::move(reason),0,{}});
}
std::string Geometry(int m,int n) {
  return "tm="+std::to_string(m)+";tn="+std::to_string(n)+";elements="+std::to_string(m*n);
}
#if !DNN1_SEGMENT_PROBES
// Input initialization and result drainage are outside the clock interval.
// There is no timing-overhead subtraction and no empirical fitting.
template<int M,int N,class Step,int Store=0,int Width=N>
__global__ void Epilogue(Sm90ForwardEpilogueOperands p,BF* output,int count,U64* stamps) {
  extern __shared__ __align__(16) float tile[];
  using Chain=Sm90ForwardEpilogueChain<M,N>;
  for(int sample=0;sample<count;++sample) {
    for(int i=threadIdx.x;i<M*N;i+=128)tile[i]=float((i%31)-15)*0.03125f;
    codegen::executor::ComputeSync();U64 begin=Cycles();
    if constexpr(Store==1) {
      // The scalar reference is required by Phase X even after FX-4 replaces
      // production dense stores. It is a probe, never a production fallback.
      for(int i=threadIdx.x;i<M*Width;i+=128)
        output[(sample%16)*M*N+(i/Width)*N+i%Width]=BF(tile[(i/Width)*N+i%Width]);
    }else if constexpr(Store==2)Chain::Finish(tile,M,0,Width,p,output+(sample%16)*M*N,N);
    else {int channel=0,width=N;Chain::template Apply<Step>(tile,M,channel,width,p);}
    codegen::executor::ComputeSync();
    if(threadIdx.x==0)stamps[sample]=Cycles()-begin;
    if constexpr(!Store) {
      for(int i=threadIdx.x;i<M*N;i+=128)output[(sample%16)*M*N+i]=BF(tile[i]);
      codegen::executor::ComputeSync();
    }
  }
}
template<int M,int N,class Step,int Store=0,int Width=N>
void EpilogueCase(char const* name) {
  std::string key=Geometry(M,Store?Width:N)+";op="+name;
  constexpr int SharedBytes=4*M*N+8*M+16;
  Add("epi_"+std::to_string(M)+"_"+std::to_string(N)+"_"+name,
      Store?"epilogue_store_ns":"epilogue_step_ns",key,SharedBytes,[key](int count) {
    Buffer<float> vectors(4*N),stats(2*M);Buffer<BF> residual(M*N),output(16*M*N);Buffer<U64> stamps(count);
    Fill(vectors,0.5f);Fill(residual,0.25f);
    std::vector<float> stat(2*M);
    for(int r=0;r<M;++r){stat[2*r]=0;stat[2*r+1]=float(N);}
    Check(cudaMemcpy(stats.p,stat.data(),stat.size()*4,cudaMemcpyHostToDevice));
    Sm90ForwardEpilogueOperands p{};
    for(int i=0;i<4;++i)p.vectors[i]=vectors.p+i*N;
    if(std::string(key).find(";op=residual")!=std::string::npos &&
        std::string(key).find(";op=residual_")==std::string::npos)p.vectors[0]=nullptr;
    p.residual=residual;p.stats=stats;p.epsilon=1e-5f;p.residual_stride=N;p.norm_width=N;
    auto kernel=Epilogue<M,N,Step,Store,Width>;Shared(kernel,SharedBytes,128);
    kernel<<<1,128,SharedBytes>>>(p,output,count,stamps);Check(cudaDeviceSynchronize());
    Samples(Store?"epilogue_store_ns":"epilogue_step_ns",key,stamps,count,1,
        Store==1?"reference_scalar_2byte_clock64":Store==2?"production_Finish_dense_clock64":"production_Apply_clock64");
  });
}
template<int M,int N>void Epilogues() {
  using Op=Sm90ForwardOp;
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kBias>>("bias");
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kScale>>("scale");
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kResidual>>("residual");
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kResidual>>("residual_scaled");
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kDeferredLayerNorm,0,
      codegen::DmActivation::kRelu,codegen::DmGatePair::kSimpleGate,16,1>>("deferred_ln");
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kResidualLN,0,
      codegen::DmActivation::kRelu,codegen::DmGatePair::kSimpleGate,16,1>>("residual_ln");
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kGatePair>>("simple_gate");
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kGatePair,0,
      codegen::DmActivation::kRelu,codegen::DmGatePair::kSwiGLU>>("swiglu");
#define ACT(name,kind) EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kActivation,0,codegen::DmActivation::kind>>(name)
  ACT("relu",kRelu);ACT("relu6",kRelu6);ACT("gelu_erf",kGeluErf);
  ACT("gelu_tanh",kGeluTanh);ACT("tanh",kTanh);ACT("silu",kSilu);
#undef ACT
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kBias>,1>("scalar_2byte");
#if DNN1_FIXED_FORWARD
  EpilogueCase<M,N,Sm90ForwardEpilogueStep<Op::kBias>,2>("vector_16byte");
#else
  Pending("epi_"+std::to_string(M)+"_"+std::to_string(N)+"_vector16",
      "epilogue_store_ns",Geometry(M,N)+";op=vector_16byte",
      "FX4 production vector store not yet available; integrate actual callable then rebuild");
  for(auto op:{"deferred_ln","residual_ln"})
    Pending("epi_"+std::to_string(M)+"_"+std::to_string(N)+"_"+op+"_fx2",
      "epilogue_step_ns",Geometry(M,N)+";op="+op+";implementation=FX2_row_cache",
      "FX2 row-cache scratch/API pending; current per-element global loads recorded separately by build variant");
#endif
}
// width=0 deliberately isolates Finish's real cross-tile row-stat protocol:
// volatile partial writes, atomics, fixed-order last-arriver reduction and
// publication. Local tile arithmetic and output stores execute zero iterations.
// It is explicitly not a measurement of the preceding per-column statistics.
template<int Rows,int Tiles>
__global__ void Rowstat(Sm90ForwardEpilogueOperands p,BF* sink,U64* stamps) {
  extern __shared__ float dummy[];
  p.row_tile=blockIdx.x;
  codegen::executor::ComputeSync();U64 begin=Cycles();
  Sm90ForwardEpilogueChain<Rows,32>::Finish(dummy,Rows,0,0,p,sink,32);
  codegen::executor::ComputeSync();
  if(threadIdx.x==0)stamps[blockIdx.x]=Cycles()-begin;
}
template<int Rows,int Tiles>void RowstatCase() {
  auto key="rows="+std::to_string(Rows)+";tiles="+std::to_string(Tiles);
  constexpr int SharedBytes=4*Rows*32+16;
  Add("rowstat_"+std::to_string(Rows)+"_"+std::to_string(Tiles),"rowstat_ns",key,SharedBytes,[key](int count) {
    Buffer<float> partial(2*Rows*Tiles),stats(2*Rows);Buffer<U64> counters(Rows),stamps(Tiles);
    Buffer<BF> sink(1);Sm90ForwardEpilogueOperands p{};
    p.row_stats=stats;p.row_partials=partial;p.row_counters=counters;p.row_tiles=Tiles;
    std::vector<double> values;std::vector<U64> host(Tiles),zeros(Rows);
    Shared(Rowstat<Rows,Tiles>,SharedBytes,128);
    for(int sample=0;sample<count;++sample) {
      Check(cudaMemset(counters.p,0,Rows*8));
      Rowstat<Rows,Tiles><<<Tiles,128,SharedBytes>>>(p,sink,stamps);Check(cudaDeviceSynchronize());
      Check(cudaMemcpy(host.data(),stamps.p,Tiles*8,cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(zeros.data(),counters.p,Rows*8,cudaMemcpyDeviceToHost));
      if(std::any_of(zeros.begin(),zeros.end(),[](U64 x){return x!=0;}))
        throw std::runtime_error("rowstat counter did not reset");
      if(sample>=kDiscard)values.push_back(double(*std::max_element(host.begin(),host.end()))/cycles_per_ns);
    }
    Emit("rowstat_ns",key,std::move(values),"ns","production_Finish_width0_max_CTA_clock64");
  });
#if !DNN1_FIXED_FORWARD
  Pending("rowstat_"+std::to_string(Rows)+"_"+std::to_string(Tiles)+"_fx3","rowstat_ns",
      key+";implementation=FX3_block_counter",
      "FX3 block-counter production implementation pending; measure actual Finish protocol after integration");
#endif
}
template<class Body>
__global__ void Small(Sm90ForwardSmallOperands p,int count,U64* stamps) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);
  state.Initialize();
  for(int sample=0;sample<count;++sample) {
    // GlobalPool consumes and clears its partials. Restore outside timing.
    if(p.accumulator)for(int i=threadIdx.x;i<p.width*p.channels;i+=160)p.accumulator[i]=0.25f;
    __syncthreads();U64 begin=Cycles();
    Body::Run(p,shared,state);
    __syncthreads();
    if(threadIdx.x==0)stamps[sample]=Cycles()-begin;
  }
  state.Invalidate();
}
template<class Body>void SmallCase(std::string id,std::string key,int width,int height,int channels,
                                   int out_channels,double elements) {
  Add(std::move(id),"small_body_ns",key,Body::kSharedBytes,[=](int count) {
    // Buffers intentionally contain nonzero values and remain observable.
    std::size_t pixels=std::size_t(width)*height;
    Buffer<BF> input(std::max<std::size_t>(pixels*std::max(channels,out_channels),4096)),
      weight(std::max<std::size_t>(std::size_t(channels)*((out_channels+255)/256*256),4096)),
      table(std::max<std::size_t>(pixels*channels,4096)),
      output(std::max<std::size_t>(pixels*std::max(channels,out_channels),4096));
    Buffer<float> bias(std::max(channels,out_channels)),accum(std::size_t(width)*channels),stats(2*std::max(width,height));
    Buffer<long long> ids(std::max<std::size_t>(pixels,128));Buffer<U64> stamps(count);
    Fill(input,0.25f);Fill(weight,0.5f);Fill(table,0.125f);Fill(bias,0.125f);
    Sm90ForwardSmallOperands p{};p.input=input;p.weight=weight;p.table=table;p.bias=bias;p.output=output;
    p.width=width;p.height=height;p.channels=channels;p.out_channels=out_channels;p.epsilon=1e-5f;
    p.index0=ids;p.index1=ids;p.stats=stats;
    if(key.find("kind=global_pool")!=std::string::npos)p.accumulator=accum;
    auto kernel=Small<Body>;Shared(kernel,Body::kSharedBytes);
    kernel<<<1,160,Body::kSharedBytes>>>(p,count,stamps);Check(cudaDeviceSynchronize());
    Samples("small_body_ns",key,stamps,count,elements,"production_Run_uninstrumented_one_CTA_clock64");
  });
}
void Catalog() {
#define EP(M) Epilogues<M,32>();Epilogues<M,64>();Epilogues<M,128>();Epilogues<M,256>()
  EP(64);EP(128);
#if DNN1_FIXED_FORWARD
  Epilogues<256,64>(); // The legal 16x16 depthwise spatial tile.
  // Exact store widths present in the five model plans; no interpolation.
  EpilogueCase<64,8,Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias>,1,3>("scalar_2byte");
  EpilogueCase<128,8,Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias>,1,3>("scalar_2byte");
#define TAIL(M,N) EpilogueCase<M,N,Sm90ForwardEpilogueStep<Sm90ForwardOp::kBias>,2>("vector_16byte")
  TAIL(64,16);TAIL(128,16);TAIL(64,24);TAIL(128,24);
  TAIL(256,32);
#undef TAIL
#endif
#undef EP
#define RS(R) RowstatCase<R,1>();RowstatCase<R,2>();RowstatCase<R,3>();RowstatCase<R,4>();RowstatCase<R,6>();RowstatCase<R,8>();RowstatCase<R,12>();RowstatCase<R,16>();RowstatCase<R,24>();RowstatCase<R,48>()
  RS(64);RS(128);
#undef RS
  SmallCase<Sm90ForwardLayout<64,128>>("small_layout3","kind=layout;pixels=1024;channels=3;padded=64",32,32,64,3,1024*3);
  SmallCase<Sm90ForwardMaxPool<3,2>>("small_pool64","kind=pool;window=3;stride=2;pixels=256;channels=64",32,32,64,64,256*64);
#if DNN1_FIXED_FORWARD
#define GAP(P,C) SmallCase<Sm90ForwardGlobalPool<1>>("small_global_pool" #P "_" #C,"kind=global_pool;parts=" #P ";channels=" #C,P,49,C,C,P*C)
  GAP(1,512);GAP(2,512);GAP(4,512);GAP(7,512);
  GAP(1,1024);GAP(2,1024);GAP(4,1024);GAP(7,1024);
  GAP(1,1280);GAP(2,1280);GAP(4,1280);GAP(7,1280);
  GAP(256,32);GAP(128,64);GAP(64,128);GAP(32,256);GAP(16,512);
#undef GAP
#define GEMV(B,K) SmallCase<Sm90ForwardGemv<B,K>>("small_gemv" #B "_k" #K,"kind=gemv;block=" #B ";rows=1;k=" #K ";n=1000",1,1,K,1000,K*1000)
  GEMV(64,512);GEMV(128,512);GEMV(256,512);
  GEMV(64,1024);GEMV(128,1024);GEMV(256,1024);
  GEMV(64,1280);GEMV(128,1280);GEMV(256,1280);
#undef GEMV
#define LN(C,S) SmallCase<Sm90ForwardLayerNorm<C,S>>("small_layernorm" #C "_" #S,"kind=layernorm;rows=128;channels=" #C ";stride=" #S,128,1,S,S,128*C)
  LN(32,64);LN(64,64);LN(128,128);LN(256,256);LN(768,768);
#undef LN
#else
  SmallCase<Sm90ForwardGlobalPool>("small_global_pool512","kind=global_pool;parts=24;channels=512;pixels=49",24,49,512,512,24*512);
  SmallCase<Sm90ForwardGemv<8>>("small_gemv8","kind=gemv;block=8;rows=1;k=1024;n=1024",1,1,1024,1024,1024*1024);
  SmallCase<Sm90ForwardLayerNorm<768>>("small_layernorm768","kind=layernorm;rows=128;channels=768;stride=768",128,1,768,768,128*768);
#endif
  SmallCase<Sm90ForwardEmbeddingSum<768>>("small_embedding768","kind=embedding;rows=128;channels=768",128,128,768,768,128*768);
#if !DNN1_FIXED_FORWARD
  for(int block:{64,128,256})Pending("small_gemv"+std::to_string(block)+"_bulk","small_body_ns",
      "kind=gemv;block="+std::to_string(block),"FX5 bulk GEMV implementation pending; current scalar body is not its substitute");
#endif
}
#else
// A build-local Mark overlay changes only TraceNow to clock64. Span totals
// below are cycles, not the original ABI's nanoseconds. No absolute timestamp
// from different CTAs is ever compared, and no trace overhead is subtracted.
using Segment=Sm90ForwardTraceSegment;
using Record=Sm90ForwardTraceRecord;
void ResetTrace() {
  void* records=nullptr;void* stages=nullptr;
  Check(cudaGetSymbolAddress(&records,sm90_forward_trace_records));
  Check(cudaGetSymbolAddress(&stages,sm90_forward_trace_stage));
  Check(cudaMemset(records,0,sizeof(sm90_forward_trace_records)));
  Check(cudaMemset(stages,0,sizeof(sm90_forward_trace_stage)));
}
std::vector<Record> ReadTrace(int grid) {
  std::vector<Record> records(grid);
  Check(cudaMemcpyFromSymbol(records.data(),sm90_forward_trace_records,grid*sizeof(Record)));
  return records;
}
template<class Body,class Operands>
__global__ void TiledBody(const __grid_constant__ CUtensorMap map,Operands p) {
  extern __shared__ __align__(1024) char shared[];
  auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);
  state.Initialize();Sm90ForwardTraceEnter(0);p.activation=&map;
  Body::Run(p,shared,state);state.Invalidate();
}
template<int M,int N,int S>void SplitCase() {
  // 31 output tiles are relatively prime to every allowed SplitK/grid size.
  // Thus different slices of one tile execute on different CTAs, while 31
  // tasks per CTA amortize the trace observer. A 32-tile grid=S arrangement
  // would accidentally put every slice of a tile on the same CTA.
  using Chain=Sm90ForwardEpilogueChain<M,N>;
  using G=Sm90ForwardGemm<M,N,64,2,S,Sm90ForwardAMode::kDense,31*M,N,S*128,N,Chain>;
  auto key=Geometry(M,N)+";slices="+std::to_string(S)+";k="+std::to_string(S*128)+";tiles=31";
  Add("split_"+std::to_string(M)+"_"+std::to_string(N)+"_"+std::to_string(S),
      "splitk_partial_ns",key,G::kSharedBytes,[key](int count) {
    Buffer<BF> input(31*M*S*128),weight(N*S*128),output(31*M*N);
    Buffer<float> partial(31*S*M*N);Buffer<std::uint64_t> counters(31);
    Fill(input,0.25f);Fill(weight,0.5f);
    CUtensorMap map{};codegen::executor::EncodeTiledMap<2>(map,input.p,
        {{std::uint64_t(S*128),std::uint64_t(31*M)},{std::uint64_t(2*S*128)},{64,M}});
    Sm90ForwardGemmOperands p{};p.weight=weight;p.output=output;p.partial=partial;p.counters=counters;p.output_stride=N;
    auto kernel=TiledBody<G,Sm90ForwardGemmOperands>;Shared(kernel,G::kSharedBytes);
    std::vector<double> total,write,reduce;
    for(int sample=0;sample<count;++sample) {
      Check(cudaMemset(counters.p,0,31*8));ResetTrace();
      kernel<<<S,160,G::kSharedBytes>>>(map,p);Check(cudaDeviceSynchronize());
      double best=0,bw=0,br=0;
      for(auto const& rec:ReadTrace(S)) {
        auto const& w=rec.segments[unsigned(Segment::kPartialWrite)];
        auto const& r=rec.segments[unsigned(Segment::kSplitKReduce)];
        if(!w.count || !r.count)continue;
        double a=double(w.total_ns)/w.count/cycles_per_ns,b=double(r.total_ns)/r.count/cycles_per_ns;
        if(a+b>best){best=a+b;bw=a;br=b;}
      }
      if(best<=0)throw std::runtime_error("missing real split-K last-arriver span");
      if(sample>=kDiscard){total.push_back(best);write.push_back(bw);reduce.push_back(br);}
    }
    std::vector<BF> result(output.count);std::vector<std::uint64_t> reset(31);
    Check(cudaMemcpy(result.data(),output.p,result.size()*sizeof(BF),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(reset.data(),counters.p,reset.size()*8,cudaMemcpyDeviceToHost));
    if(std::any_of(result.begin(),result.end(),[](BF x){return float(x)!=float(S*16);}) ||
       std::any_of(reset.begin(),reset.end(),[](auto x){return x!=0;}))
      throw std::runtime_error("split-K synthetic-input result or counter check failed");
    Emit("splitk_partial_ns",key,std::move(total),"ns","production_Run_last_arriver_CTA_mean_clock64_spans");
    Emit("splitk_write_diag_ns",key,std::move(write),"ns","same_selected_CTA_partial_write");
    Emit("splitk_reduce_diag_ns",key,std::move(reduce),"ns","same_selected_CTA_fixed_order_reduce");
  });
}
template<int W,int H,int C,int Live=C,int Stride=1>void DepthwiseCase() {
  using Chain=Sm90ForwardEpilogueChain<W*H,C>;
  using Body=Sm90ForwardDepthwise<W,H,C,Stride,2,Chain>;
  auto key="tw="+std::to_string(W)+";th="+std::to_string(H)+";channels="+std::to_string(Live)+
      ";ct="+std::to_string(C)+";stride="+std::to_string(Stride)+";pack=8";
  Add("dw_"+std::to_string(W)+"_"+std::to_string(H)+"_"+std::to_string(C)+"_"+
      std::to_string(Live)+"_s"+std::to_string(Stride),"dw_tap_ns",key,
      Body::kSharedBytes,[key](int count) {
    // 32 spatial tiles per CTA; real TMA halo and bulk filters feed Consume.
    constexpr int Width=W*32,Height=H,InputW=Width*Stride,InputH=Height*Stride;
    Buffer<BF> input(InputW*InputH*C),weight(C*9*8),output(Width*Height*Live);
    Fill(input,0.25f);Fill(weight,0.5f);CUtensorMap map{};
    codegen::executor::EncodeTiledMap<4>(map,input.p,
      {{C,InputW,InputH,1},{2*C,2*C*InputW,2*C*InputW*InputH},{C,Body::kHaloW,Body::kHaloH,1}},false);
    Sm90ForwardDepthwiseOperands p{};p.weight=weight;p.output=output;p.width=InputW;p.height=InputH;
    p.channels=Live;p.stride=Stride;p.output_width=Width;p.output_height=Height;p.padded_channels=C;p.output_stride=Live;
    auto kernel=TiledBody<Body,Sm90ForwardDepthwiseOperands>;Shared(kernel,Body::kSharedBytes);
    std::vector<double> values;
    for(int sample=0;sample<count;++sample) {
      ResetTrace();kernel<<<1,160,Body::kSharedBytes>>>(map,p);Check(cudaDeviceSynchronize());
      auto span=ReadTrace(1)[0].segments[unsigned(Segment::kDepthwiseCompute)];
      if(span.count!=32)throw std::runtime_error("depthwise trace task count differs");
      if(sample>=kDiscard)values.push_back(double(span.total_ns)/cycles_per_ns/span.count/(W*H*Live*9));
    }
    Emit("dw_tap_ns",key,std::move(values),"ns/element/tap","production_Run_compute_clock64_spans");
  });
}
template<class Body>
__global__ void Attention(const __grid_constant__ CUtensorMap key,const __grid_constant__ CUtensorMap value,
                         const __grid_constant__ CUtensorMap query,Sm90ForwardEncoderAttentionOperands p) {
  extern __shared__ __align__(1024) char shared[];auto& state=*reinterpret_cast<Sm90DenseRingState*>(shared);
  state.Initialize();Sm90ForwardTraceEnter(0);p.key=&key;p.value=&value;p.query=&query;
  Body::Run(p,shared,state);state.Invalidate();
}
#if DNN1_FIXED_FORWARD
template<int Rows>void AttentionCase() {
  constexpr int Seq=128,Heads=4,D=64;
  using Body=Sm90ForwardEncoderAttention<1,Seq,D,Heads,Rows,2>;
  std::string mode="tensor_core";
#else
void AttentionCase() {
  constexpr int Seq=128,Rows=16,Heads=4,D=64;
  using Body=Sm90ForwardEncoderAttention<Seq,D,Heads,Rows,2>;
  std::string mode="scalar";
#endif
  std::string key="seq=128;rows="+std::to_string(Rows)+";heads=4;batch=1;pv="+mode;
  Add("attention_128_"+std::to_string(Rows)+"_"+mode,"attention_ns",key,Body::kSharedBytes,[key](int count) {
    Buffer<BF> qkv(Seq*3*Heads*D),output(Seq*Heads*D);Fill(qkv,0.25f);
    CUtensorMap maps[3]{};
    for(int i=0;i<3;++i)codegen::executor::EncodeTiledMap<3>(maps[i],qkv.p+i*Heads*D,
        {{D,Heads,Seq},{2*D,2*3*Heads*D},{64,1,std::uint32_t(i==0?Rows:64)}});
    Sm90ForwardEncoderAttentionOperands p{};p.output=output;p.scale=0.125f;p.heads=Heads;p.sequence=Seq;
    auto kernel=Attention<Body>;Shared(kernel,Body::kSharedBytes);std::vector<double> values[3];
    Segment segments[]={Segment::kAttentionQk,Segment::kAttentionSoftmax,Segment::kAttentionPv};
    for(int sample=0;sample<count;++sample) {
      ResetTrace();kernel<<<1,160,Body::kSharedBytes>>>(maps[1],maps[2],maps[0],p);Check(cudaDeviceSynchronize());
      auto records=ReadTrace(1);
      for(int s=0;s<3;++s) {
        auto const& span=records[0].segments[unsigned(segments[s])];
        if(span.count!=Body::kTasks)throw std::runtime_error("attention trace task count differs");
        if(sample>=kDiscard)values[s].push_back(double(span.total_ns)/cycles_per_ns/span.count);
      }
    }
    char const* names[]={"qk","softmax","pv"};
    for(int s=0;s<3;++s)Emit("attention_ns",key+";segment="+names[s],std::move(values[s]),"ns",
        "production_Run_full_segment_clock64_spans");
  });
#if !DNN1_FIXED_FORWARD
  for(int rows:{16,32,64,128})Pending("attention_128_"+std::to_string(rows)+"_fx1","attention_ns",
      "seq=128;rows="+std::to_string(rows)+";pv=tensor_core",
      "FX1 batch-aware tensor-core PV body pending; scalar QK MMA is not a replacement");
#endif
}
void Catalog() {
#define SK(M,N) SplitCase<M,N,2>();SplitCase<M,N,4>();SplitCase<M,N,8>();SplitCase<M,N,16>()
  SK(64,32);SK(64,64);SK(64,128);SK(64,256);SK(128,32);SK(128,64);SK(128,128);
#undef SK
  for(int s:{2,4,8,16})cases.push_back({"split_128_256_"+std::to_string(s),"splitk_partial_ns",
      Geometry(128,256)+";slices="+std::to_string(s),"infeasible",
      "read-only Sm90ForwardGmma requires TM*TN<=16384",0,{}});
#define DW(W,H) DepthwiseCase<W,H,64,32>();DepthwiseCase<W,H,64>();DepthwiseCase<W,H,128>(); \
  DepthwiseCase<W,H,64,32,2>();DepthwiseCase<W,H,64,64,2>();DepthwiseCase<W,H,128,128,2>()
  DW(8,8);DW(8,16);DW(16,16);
#undef DW
#if DNN1_FIXED_FORWARD
  AttentionCase<16>();AttentionCase<32>();AttentionCase<64>();AttentionCase<128>();
#else
  AttentionCase();
#endif
}
#endif
std::string Json(std::string const& s) {
  std::string out="\"";for(char c:s){if(c=='\"'||c=='\\')out+='\\';out+=c;}return out+'\"';
}
void RunOne(Case const& item,int count,std::string const& raw_path) {
  Check(cudaSetDevice(0));Buffer<U64> clock(3);Calibrate<<<1,1>>>(clock);Check(cudaDeviceSynchronize());
  U64 calibration[3];Check(cudaMemcpy(calibration,clock.p,24,cudaMemcpyDeviceToHost));
  cycles_per_ns=double(calibration[0])/calibration[1];
  std::cerr<<"calibration cycles="<<calibration[0]<<" global_ns="<<calibration[1]
      <<" cycles_per_ns="<<std::setprecision(17)<<cycles_per_ns
      <<" clock_resolution_ns="<<1/cycles_per_ns<<" observed_globaltimer_step_ns="<<calibration[2]<<'\n';
  Buffer<U64> observer(256);Observer<<<1,128>>>(observer,256);Check(cudaDeviceSynchronize());
  std::vector<U64> overhead(256);Check(cudaMemcpy(overhead.data(),observer.p,256*8,cudaMemcpyDeviceToHost));
  std::sort(overhead.begin()+kDiscard,overhead.end());
  std::cerr<<"observer_empty_ComputeSync_median_ns="<<double(overhead[kDiscard+(256-kDiscard)/2])/cycles_per_ns
      <<" observer_subtracted=false\n";
  if(!raw_path.empty()){raw.open(raw_path);if(!raw)throw std::runtime_error("cannot open raw sample file");raw<<"metric,key,sample,value\n";}
  std::cout<<"metric,key,samples,median_ns,p99_ns,unit,method\n";item.run(count);
  if(raw.is_open())raw.close();
  raw.clear();
}
int main(int argc,char** argv) {try {
  std::string selected,raw_path,batch_cases,batch_root;bool list=false;int count=256;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    if(arg=="--list")list=true;
    else if(arg=="--case" && i+1<argc)selected=argv[++i];
    else if(arg=="--batch-cases" && i+1<argc)batch_cases=argv[++i];
    else if(arg=="--batch-root" && i+1<argc)batch_root=argv[++i];
    else if(arg=="--samples" && i+1<argc)count=std::stoi(argv[++i]);
    else if(arg=="--raw" && i+1<argc)raw_path=argv[++i];
    else throw std::runtime_error("usage: --list | --case ID [--samples 256] [--raw PATH]");
  }
  Catalog();
  if(list){for(auto const& c:cases)std::cout<<"{\"id\":"<<Json(c.id)<<",\"metric\":"<<Json(c.metric)
      <<",\"key\":"<<Json(c.key)<<",\"status\":"<<Json(c.status)<<",\"reason\":"<<Json(c.reason)
      <<",\"shared_bytes\":"<<c.shared<<",\"binary\":\""<<(DNN1_SEGMENT_PROBES?"segments":"direct")<<"\"}\n";return 0;}
  auto found=std::find_if(cases.begin(),cases.end(),[&](auto const& c){return c.id==selected;});
  if(batch_cases.empty() && (found==cases.end() || !found->run))throw std::runtime_error("unknown/pending case: "+selected);
  if(count<kDiscard+16)throw std::runtime_error("at least 24 samples are required");
  auto env=[](char const* k){auto* v=std::getenv(k);return v?std::string(v):std::string();};
  auto gpu=env("TILEMEGA_DEVICE_INDEX");
  if((gpu!="0" && gpu!="5" && gpu!="7") || env("CUDA_VISIBLE_DEVICES")!=gpu ||
      env("TILEMEGA_GPU_LOCK_HELD")!="1" || env("TILEMEGA_TIMING")!="1")
    throw std::runtime_error("GPU case requires explicit GPU0/5/7 timing guard; use collector run");
  if(batch_cases.empty())RunOne(*found,count,raw_path);
  else {
    if(batch_root.empty())throw std::runtime_error("--batch-root required");
    std::istringstream ids(batch_cases);std::string id;
    while(std::getline(ids,id,',')) {
      auto item=std::find_if(cases.begin(),cases.end(),[&](auto const& c){return c.id==id;});
      if(item==cases.end() || !item->run)throw std::runtime_error("unknown batch case: "+id);
      std::ofstream out(batch_root+"/"+id+"/stdout.csv"),err(batch_root+"/"+id+"/stderr.log");
      if(!out || !err)throw std::runtime_error("batch case output directory missing: "+id);
      auto* old_out=std::cout.rdbuf(out.rdbuf());auto* old_err=std::cerr.rdbuf(err.rdbuf());
      try {RunOne(*item,count,batch_root+"/"+id+"/samples.csv");}
      catch(std::exception const& e) {
        std::cerr<<e.what()<<'\n';std::cout.flush();std::cerr.flush();
        std::cout.rdbuf(old_out);std::cerr.rdbuf(old_err);throw;
      }
      std::cout.flush();std::cerr.flush();std::cout.rdbuf(old_out);std::cerr.rdbuf(old_err);
    }
  }
  return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 2;}}
