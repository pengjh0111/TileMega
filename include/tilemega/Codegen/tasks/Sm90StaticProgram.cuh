// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <utility>

namespace tilemega::codegen::static_dense {

using BF16 = cutlass::bfloat16_t;
inline constexpr int kBatch = TILEMEGA_SERVING_BATCH_LO;
static_assert(kBatch==TILEMEGA_SERVING_BATCH_HI && TILEMEGA_SERVING_SEQ==1);
static_assert(TILEMEGA_ARCH_ID==900 && !TILEMEGA_PAGED);

constexpr int Chunks(unsigned gemm) {
  auto g=kRuntimeGemms0[gemm];
  int tiles=(kGemms[gemm].k+g.tile_k-1)/g.tile_k;
  return g.split_k<tiles?g.split_k:tiles;
}
constexpr unsigned PartialIndex(unsigned gemm) {
  unsigned id=kModel.buffer_count;
  for(unsigned i=0;i<gemm;++i)if(Chunks(i)>1)++id;
  return id;
}
constexpr unsigned StageCount() {
  unsigned count=kModel.stage_count;
  for(auto const& s:kStages)if(s.kind==TaskKind::kGemm && Chunks(s.gemm)>1)++count;
  return count;
}
struct Entry { unsigned source; bool combine; };
constexpr auto MakeEntries() {
  std::array<Entry,StageCount()> result{};
  unsigned n=0;
  for(unsigned i=0;i<kModel.stage_count;++i) {
    result[n++]={i,false};
    if(kStages[i].kind==TaskKind::kGemm && Chunks(kStages[i].gemm)>1)
      result[n++]={i,true};
  }
  return result;
}
inline constexpr auto kEntries=MakeEntries();

// Keep shared stage inputs unique in the parameter working set. Each plan
// owns its table, so simultaneously created plans cannot alias addresses.
struct Bindings { BF16* buffers[kModel.buffer_count+kModel.gemm_count]; };
static_assert(sizeof(Bindings)+256<32764,"Hopper kernel parameter limit");
template<unsigned Id>
__device__ __forceinline__ BF16* Buffer(Bindings const& b) {
  if constexpr(Id==kNoOperand)return nullptr;
  else return b.buffers[Id];
}

template<int M,int N,int K,int TM,int TN,int TK,int S,int Split,
         bool Wgmma,backend::ServingEpilogueOp Op>
__device__ __noinline__ void Gemm(BF16 const* a,BF16 const* weight,
    BF16 const* residual,BF16* output,float* partial,int* index,
    float const* norm,float* sums,int task,char* shared) {
  using Body=ServingGemmTaskBody<arch::Sm90,TM,TN,TK,S,Wgmma>;
  constexpr int NT=(N+TN-1)/TN,KT=(K+TK-1)/TK;
  int chunk=task%Split,local=task/Split;
  int begin=chunk*KT/Split*TK,end=min(K,(chunk+1)*KT/Split*TK);
  if constexpr(Wgmma && Split>1)end=(chunk+1)*KT/Split*TK;
  static_assert(K%TK==0 && Split<=KT && TM>=M);
  __builtin_assume(a!=nullptr && weight!=nullptr);
  __builtin_assume(begin>=0 && end>begin && end<=K && begin%8==0);
  __builtin_assume((executor::Async<arch::Sm90>::Shared(shared)&1023)==0);
  if constexpr(Split>1)__builtin_assume(partial!=nullptr);
  ServingGemmOperands p;
  p.a=a;p.b=weight;p.weight_base=weight;p.residual=residual;p.output=output;
  p.partial=Split>1?partial+std::size_t(chunk)*M*N:nullptr;
  p.argmax_value=reinterpret_cast<float*>(output);p.argmax_index=index;
  p.m=M;p.n=N;p.k_total=K;p.k_total_full=K;
  p.k_begin=begin;p.k_count=end-begin;
  if constexpr(Wgmma && Split>1) {
    __builtin_assume(p.k_begin+p.k_count<=K);
    __builtin_assume(p.k_begin+((p.k_count+7)&~7)<=K);
  }
  p.output_stride=Op==backend::ServingEpilogueOp::kSwiGLU?N/2:
      Op==backend::ServingEpilogueOp::kArgmaxPartial?NT:N;
  p.partial_stride=N;p.a_row_stride=K;p.b_row_stride=K;
  p.epilogue=Split>1?backend::ServingEpilogueOp::kPartial:Op;
  p.norm_ss=norm;p.ss_out=sums;p.norm_k=K;p.norm_eps=TILEMEGA_NORM_EPSILON;
  constexpr auto finish=Split>1?backend::ServingEpilogueOp::kPartial:Op;
  if constexpr(Wgmma) {
    auto* tile=Body::ComputeWgmma(p,local/NT,local%NT,shared);
    backend::ServingEpilogue<finish,TM,TN>::template RunFromTile<false>(tile,
        local/NT,local%NT,M,N,p.output_stride,N,output,residual,p.partial,
        p.argmax_value,index,norm,sums,K,TILEMEGA_NORM_EPSILON);
    executor::ComputeSync();
  }else {
    using Config=typename Body::Config;
    typename Config::TiledMma mma;
    auto accum=cute::partition_fragment_C(mma,cute::Shape<cute::Int<TM>,cute::Int<TN>>{});
    cute::clear(accum);
    backend::ServingTiledMainloop<arch::Sm90,Config,TM,TN,TK,S>::Run(
        p,local/NT,local%NT,shared,accum);
    backend::ServingEpilogue<finish,TM,TN>::Run(accum,mma,shared,local/NT,local%NT,
        M,N,p.output_stride,N,output,residual,p.partial,p.argmax_value,index,
        norm,sums,K,TILEMEGA_NORM_EPSILON);
  }
}

template<int M,int N,int K,int TM,int TN,int Split,backend::ServingEpilogueOp Op>
__device__ __noinline__ void Combine(float const* partial,BF16* output,
    BF16 const* residual,int* index,float const* norm,float* sums,int task,char* shared) {
  constexpr int NT=(N+TN-1)/TN;
  constexpr int stride=Op==backend::ServingEpilogueOp::kSwiGLU?N/2:
      Op==backend::ServingEpilogueOp::kArgmaxPartial?NT:N;
  ServingGemmCombineTaskBody<TM,TN,Op>::Run(partial,Split,task/NT,task%NT,M,N,N,
      stride,output,residual,reinterpret_cast<float*>(output),index,
      reinterpret_cast<float*>(shared),norm,sums,K,TILEMEGA_NORM_EPSILON);
}

template<int Width,int Vocab,int Capacity>
__device__ __noinline__ void Embedding(int const* tokens,BF16 const* table,
    BF16* output,float* sums,int row,int past) {
  // set_steps validates positions; the prompt and the preceding argmax supply
  // vocabulary-bounded tokens. The static ABI has no unchecked external IDs.
  __builtin_assume(past>=TILEMEGA_SERVING_PAST_LO && past<=TILEMEGA_SERVING_PAST_HI);
  __builtin_assume(past>=0 && past<Capacity);
  int id=tokens[row*Capacity+past];
  __builtin_assume(id>=0 && id<Vocab);
  ServingEmbeddingTaskBody::RunRow(tokens,table,output,row,1,past,Capacity,Width,Vocab,sums);
}
template<int Width,int Stride,int Offset>
__device__ __noinline__ void Norm(BF16 const* input,BF16 const* weight,
    BF16* output,int row,char* shared) {
  ServingRMSNormTaskBody::RunRow(input,weight,output,row,Stride,Offset,
      Width,TILEMEGA_NORM_EPSILON,reinterpret_cast<float*>(shared));
}
template<int Parts,int Capacity>
__device__ __noinline__ void Argmax(float const* values,int const* indices,
    int* tokens,int row,int past,char* shared) {
  static_assert(TILEMEGA_SERVING_PAST_HI+1<Capacity);
  __builtin_assume(past>=TILEMEGA_SERVING_PAST_LO && past<=TILEMEGA_SERVING_PAST_HI);
  __builtin_assume(past+1>=0 && past+1<Capacity);
  ServingArgmaxReduceTaskBody::RunRow(values,indices,tokens,row,Parts,Capacity,
      past+1,reinterpret_cast<ServingArgmaxReduceTaskBody::SharedStorage*>(shared));
}
template<int Heads,int Extent,int Capacity>
__device__ __noinline__ void Attention(ServingAttentionOperands p,char* shared,int task) {
  constexpr int blocks=(Capacity+Extent-1)/Extent;
  auto coordinate=DecodeServingAttentionTaskGMajor(task,kBatch,blocks);
  p.batch=kBatch;p.heads_kv=Heads;p.capacity=Capacity;p.block_extent=Extent;
  p.epsilon=TILEMEGA_NORM_EPSILON;
  T_ServingAttention::Run(p,*reinterpret_cast<typename T_ServingAttention::SharedStorage*>(shared),
      coordinate.batch,coordinate.group,0,coordinate.cache_block);
}
template<int Heads,int Extent,int Capacity>
__device__ __noinline__ void Merge(float const* partial,float const* lse,
    BF16* output,int task,int past) {
  T_ServingMerge::Run(partial,lse,output,task%kBatch,task/kBatch,Heads,
      Capacity,Extent,past);
}

template<unsigned Stage>
constexpr int Tasks() {
  constexpr auto e=kEntries[Stage];constexpr auto s=kStages[e.source];
  if constexpr(s.kind==TaskKind::kGemm) {
    constexpr auto g=kGemms[s.gemm];constexpr auto v=kRuntimeGemms0[s.gemm];
    return ((kBatch+v.tile_m-1)/v.tile_m)*((g.n+v.tile_n-1)/v.tile_n)*
        (e.combine?1:Chunks(s.gemm));
  }else if constexpr(s.kind==TaskKind::kFusedAttention)
    return kBatch*s.extent*((kModel.dims.capacity+s.attention_kv_block-1)/s.attention_kv_block);
  else if constexpr(s.kind==TaskKind::kAttentionMerge)return kBatch*s.extent;
  else return kBatch;
}

template<unsigned Stage>
__device__ __forceinline__ void RunStage(Bindings const& b,int past,char* shared) {
  constexpr auto e=kEntries[Stage];constexpr auto s=kStages[e.source];
  for(int task=blockIdx.x;task<Tasks<Stage>();task+=gridDim.x) {
    if constexpr(s.kind==TaskKind::kGemm) {
      constexpr auto g=kGemms[s.gemm];constexpr auto v=kRuntimeGemms0[s.gemm];
      constexpr auto op=static_cast<backend::ServingEpilogueOp>(g.serving_epilogue);
      auto* index=reinterpret_cast<int*>(Buffer<g.serving_argmax_index>(b));
      auto* norm=reinterpret_cast<float const*>(Buffer<g.serving_norm_ss>(b));
      auto* sums=reinterpret_cast<float*>(Buffer<g.serving_ss_out>(b));
      auto* partial=Chunks(s.gemm)>1?reinterpret_cast<float*>(Buffer<PartialIndex(s.gemm)>(b)):nullptr;
      if constexpr(e.combine)
        Combine<kBatch,g.n,g.k,v.tile_m,v.tile_n,Chunks(s.gemm),op>(partial,
            Buffer<g.d>(b),Buffer<g.c>(b),index,norm,sums,task,shared);
      else
        Gemm<kBatch,g.n,g.k,v.tile_m,v.tile_n,v.tile_k,v.stages,Chunks(s.gemm),
            GemmVariant<v.compiled_variant>::kWgmma,op>(Buffer<g.a>(b),Buffer<g.b>(b),
            Buffer<g.c>(b),Buffer<g.d>(b),partial,index,norm,sums,task,shared);
    }else if constexpr(s.kind==TaskKind::kEmbedding)
      Embedding<s.width,s.extent,kModel.dims.capacity>(reinterpret_cast<int const*>(Buffer<s.operand[0]>(b)),
          Buffer<s.operand[1]>(b),Buffer<s.operand[2]>(b),reinterpret_cast<float*>(Buffer<s.operand[3]>(b)),task,past);
    else if constexpr(s.kind==TaskKind::kRMSNorm)
      Norm<s.width,s.row_stride,s.row_offset>(Buffer<s.operand[0]>(b),Buffer<s.operand[1]>(b),
          Buffer<s.operand[2]>(b),task,shared);
    else if constexpr(s.kind==TaskKind::kArgmaxReduce)
      Argmax<s.width,kModel.dims.capacity>(reinterpret_cast<float const*>(Buffer<s.operand[0]>(b)),
          reinterpret_cast<int const*>(Buffer<s.operand[1]>(b)),reinterpret_cast<int*>(Buffer<s.operand[2]>(b)),task,past,shared);
    else if constexpr(s.kind==TaskKind::kAttentionMerge)
      Merge<s.extent,s.attention_kv_block,kModel.dims.capacity>(reinterpret_cast<float const*>(Buffer<s.operand[0]>(b)),
          reinterpret_cast<float const*>(Buffer<s.operand[1]>(b)),Buffer<s.operand[2]>(b),task,past);
    else if constexpr(s.kind==TaskKind::kFusedAttention) {
      ServingAttentionOperands p{Buffer<s.operand[0]>(b),Buffer<s.operand[1]>(b),Buffer<s.operand[2]>(b),
          Buffer<s.operand[3]>(b),Buffer<s.operand[4]>(b),Buffer<s.operand[5]>(b),Buffer<s.operand[6]>(b),
          Buffer<s.operand[7]>(b),reinterpret_cast<float*>(Buffer<s.operand[8]>(b)),
          reinterpret_cast<float*>(Buffer<s.operand[9]>(b)),kBatch,int(s.extent),kModel.dims.capacity,
          past,s.attention_kv_block,TILEMEGA_NORM_EPSILON};
      Attention<s.extent,s.attention_kv_block,kModel.dims.capacity>(p,shared,task);
    }else static_assert(s.kind==TaskKind::kGemm,"unsupported static dense stage");
    __syncthreads();
  }
}

template<unsigned Stage>
__device__ __forceinline__ void RunBoundary(Bindings const& b,int past,char* shared,
    Params const& p,EventCounter* events,unsigned long long iteration) {
  constexpr int count=Tasks<Stage>();
  constexpr unsigned index=Stage;
  executor::StageBegin(p,index,iteration,count/int(gridDim.x)+(int(blockIdx.x)<count%int(gridDim.x)));
  RunStage<Stage>(b,past,shared);
  __syncthreads();
  if(threadIdx.x==0) {
    RedRelease(&events[index].arrivals,1ull);
    executor::StageTasksEnd(&p,index,iteration);
    WaitAtLeast(&events[index].arrivals,static_cast<unsigned long long>(gridDim.x)*(iteration+1ull));
  }
  __syncthreads();
  executor::StageRelease(p,index,iteration);
}
template<std::size_t... I>
__device__ __forceinline__ void Sequence(Bindings const& b,int past,Params const& p,
    char* shared,EventCounter* events,unsigned long long iteration,std::index_sequence<I...>) {
  (RunBoundary<I>(b,past,shared,p,events,iteration),...);
}
__device__ __forceinline__ void Step(Bindings const& b,Params const& p,char* shared,
    EventCounter* events,unsigned long long iteration) {
  int past=p.dims.past;
  executor::StepBegin(p,iteration);
  Sequence(b,past,p,shared,events,iteration,std::make_index_sequence<StageCount()>{});
  executor::StepEnd(p,iteration);
}
}  // namespace tilemega::codegen::static_dense

namespace tilemega::codegen {
__global__ __launch_bounds__(128,TILEMEGA_MIN_BLOCKS_PER_SM)
void sm90_dense_step_kernel(const __grid_constant__ static_dense::Bindings bindings,
    Params const* params,EventCounter* events,unsigned long long iteration) {
  extern __shared__ __align__(1024) unsigned char shared[];
  ServingPdlEnter(*params);
  static_dense::Step(bindings,*params,reinterpret_cast<char*>(shared),events,iteration);
  ServingPdlExit();
}
__global__ __launch_bounds__(128,TILEMEGA_MIN_BLOCKS_PER_SM)
void sm90_dense_loop_kernel(const __grid_constant__ static_dense::Bindings bindings,
    Params const* params,unsigned steps,EventCounter* events,
    unsigned long long iteration,unsigned long long* step_ns) {
  extern __shared__ __align__(1024) unsigned char shared[];
  ServingPdlEnter(params[0]);
  if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[0]=executor::ServingTraceNow();
  for(unsigned step=0;step<steps;++step) {
    static_dense::Step(bindings,params[step],reinterpret_cast<char*>(shared),events,iteration+step);
    if(blockIdx.x==0 && threadIdx.x==0 && step_ns)step_ns[step+1]=executor::ServingTraceNow();
  }
  ServingPdlExit();
}
}  // namespace tilemega::codegen
