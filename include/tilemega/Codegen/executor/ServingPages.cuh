// SPDX-License-Identifier: BSD-3-Clause
// Included inside tilemega::codegen after event and task descriptor helpers.
namespace paged {
using executor::ComputeThread;
using executor::ComputeSync;
using executor::kComputeThreads;
// The emitted geometry is portable; instructions follow the compilation target.
using PageArch=std::conditional_t<std::is_void_v<arch::CurrentArch>,GemmVariantArch,arch::CurrentArch>;
using Ring=executor::PageRing<TILEMEGA_PAGE_BYTES,TILEMEGA_PAGE_COUNT,PageArch,TILEMEGA_ARCH_PATH_SM80!=0>;
using Attention=PagedAttentionTaskBody<PageArch,TILEMEGA_SERVING_HEAD_DIM,
    TILEMEGA_SERVING_QPERKV,TILEMEGA_SERVING_QK_NORM!=0,TILEMEGA_PAGE_BYTES,TILEMEGA_PAGE_COUNT,TILEMEGA_ARCH_PATH_SM80!=0>;
static_assert(TILEMEGA_SERVING_SEQ==1,"page executor currently covers decode");
static_assert(TILEMEGA_PAGE_WORKSPACE_OFFSET>=sizeof(Ring::Slot)*TILEMEGA_PAGE_COUNT);
static_assert(TILEMEGA_PAGE_POOL_OFFSET%1024==0);
static_assert(TILEMEGA_PAGE_POOL_OFFSET-TILEMEGA_PAGE_WORKSPACE_OFFSET>=sizeof(Attention::SharedStorage));
#ifndef TILEMEGA_LOOKAHEAD_BYTES
#define TILEMEGA_LOOKAHEAD_BYTES 0
#endif

// The second cursor visits the same task order as the loader. It only yields
// operand ranges with no producer in the current step: packed weights and
// historical KV rows. Activation and newly appended KV never enter this list.
struct PageStream {
  Params const* params;
  unsigned step=0,steps=1;
  bool l2;
  unsigned slot=0,stage=0;
  int task=0,part=0,offset=0;
  __device__ PageStream(Params const* first,unsigned count,bool placed)
      :params(first),steps(count),l2(placed) {
    if(l2)slot=params[0].schedule_offsets[blockIdx.x];
    else task=blockIdx.x;
  }
  __device__ bool Next(executor::PrefetchRange* out) {
    for(;;) {
      if(step>=steps)return false;
      Params const& p=params[step];
      if(l2) {
        if(slot>=p.schedule_offsets[blockIdx.x+1]) {
          ++step;
          if(step<steps)slot=params[step].schedule_offsets[blockIdx.x];
          part=0;offset=0;continue;
        }
        stage=p.schedule[slot].stage;task=p.schedule[slot].logical_task;
      }else {
        while(stage<p.stage_count &&
              (p.stages[stage].handoff_elided ||
               task>=ActiveBlocks(p,p.stages[stage]))) {
          ++stage;task=blockIdx.x;part=0;offset=0;
        }
        if(stage>=p.stage_count) {
          ++step;stage=0;task=blockIdx.x;part=0;offset=0;continue;
        }
      }
      auto const& s=p.stages[stage];
      char const* source=nullptr;int total=0;
      if(s.kind==TaskKind::kGemm) {
#if TILEMEGA_WEIGHT_LAYOUT_TILED
        auto const* table=static_cast<GemmInvocation const*>(p.gemms);
        auto const& first=table[s.gemm];
        auto point=DecodeSplitTask(task,first.tiles_m*first.tiles_n,first.chunks);
        auto const& inv=table[s.gemm+point.chunk];
        if(inv.serving_weight_base && inv.serving_k_total_full>0) {
          int tn=point.tile%inv.tiles_n;
          int stage_bytes=inv.tile_n*inv.serving_tile_k*2;
          int kt=CeilDiv(inv.serving_k_total_full,inv.serving_tile_k);
          source=reinterpret_cast<char const*>(inv.serving_weight_base)+
              (std::size_t(tn)*kt+inv.serving_k_begin/inv.serving_tile_k)*stage_bytes;
          total=CeilDiv(cute::get<2>(inv.problem),inv.serving_tile_k)*stage_bytes;
        }
#endif
      }else if(s.kind==TaskKind::kFusedAttention) {
        auto point=DecodeServingAttentionTaskGMajor(task,p.dims.batch,
            CeilDiv(p.dims.capacity,s.attention_kv_block));
        int begin=point.cache_block*s.attention_kv_block;
        int end=min(begin+s.attention_kv_block,p.dims.past);
        if(end>begin) {
          std::size_t element=((std::size_t(point.batch)*s.extent+point.group)*
              p.dims.capacity+begin)*s.width;
          source=reinterpret_cast<char const*>(p.buffers[s.operand[part?2:1]])+
              element*2;
          total=(end-begin)*int(s.width)*2;
        }
      }
      if(source && offset<total) {
        int bytes=min(TILEMEGA_PAGE_BYTES,total-offset);
        *out={source+offset,unsigned(bytes)};
        offset+=bytes;return true;
      }
      offset=0;
      if(s.kind==TaskKind::kFusedAttention && part==0) {part=1;continue;}
      part=0;
      if(l2)++slot;else task+=gridDim.x;
    }
  }
};

#ifndef TILEMEGA_EVICT_LAST
#define TILEMEGA_EVICT_LAST 1
#endif
__device__ inline void PrefetchL2(executor::PrefetchRange range) {
  using Copy=typename Ring::Copy;
  if constexpr(Copy::Caps::kBulkPrefetch) {
    if(executor::LoaderLane()==0)
      if constexpr(TILEMEGA_EVICT_LAST)
        Copy::PrefetchEvictLast(range.address,range.bytes&~15u);
      else Copy::Prefetch(range.address,range.bytes&~15u);
  }else {
    for(unsigned off=executor::LoaderLane()*128;off<range.bytes;off+=32*128)
      if constexpr(TILEMEGA_EVICT_LAST)
        Copy::PrefetchEvictLast(static_cast<char const*>(range.address)+off,128);
      else Copy::Prefetch(static_cast<char const*>(range.address)+off,128);
  }
}

struct Lookahead {
  PageStream* ahead;unsigned long long* prefetched;unsigned long long* loaded;
  __device__ void operator()(unsigned stream_bytes) const {
#if TILEMEGA_LOOKAHEAD_BYTES > 0
    executor::PrefetchRange range{};
    while(*prefetched<*loaded+TILEMEGA_LOOKAHEAD_BYTES && ahead->Next(&range)) {
      PrefetchL2(range);*prefetched+=range.bytes;
    }
    *loaded+=stream_bytes;
#endif
  }
};

__device__ inline ServingGemmOperands Operands(GemmInvocation const& inv) {
  auto [m,n,k,batch]=inv.problem;(void)batch;
  ServingGemmOperands p;
  p.a=inv.mainloop.ptr_A-inv.serving_k_begin;
  p.b=inv.mainloop.ptr_B-inv.serving_k_begin;
  p.residual=inv.epilogue.ptr_C;p.output=inv.epilogue.ptr_D;p.partial=inv.serving_partial;
  p.argmax_value=reinterpret_cast<float*>(inv.epilogue.ptr_D);p.argmax_index=inv.serving_argmax_index;
  p.m=m;p.n=n;p.k_total=inv.k_total;p.k_begin=inv.serving_k_begin;
  p.k_count=k;p.output_stride=inv.serving_output_stride;
  p.partial_stride=inv.serving_partial_stride;
  p.a_row_stride=p.b_row_stride=inv.k_total;
  p.weight_base=inv.serving_weight_base;
  p.k_total_full=inv.serving_k_total_full;
  p.norm_ss=inv.serving_norm_ss;p.ss_out=inv.serving_ss_out;
  p.norm_k=inv.k_total;p.norm_eps=TILEMEGA_NORM_EPSILON;
  p.epilogue=inv.chunks>1?backend::ServingEpilogueOp::kPartial:inv.serving_op;
  return p;
}
struct PhaseGate {
  Params const* params=nullptr;
  PhaseGateDesc const* desc=nullptr;
  EventCounter* events=nullptr;
  unsigned long long iteration=0;
  int tile=0;
  unsigned* shared=nullptr;
  bool enabled=false;
  Watch* watch=nullptr;
  __device__ bool Ready() const {
    if(!enabled)return true;
    if(ComputeThread()==0) {
      auto source=desc->producer;
      auto need=static_cast<unsigned long long>(
          ActiveBlocks(*params,params->stages[source]))*(iteration+1);
      *shared=LoadAcquire(&events[EventIndex(*params,source,
          kWholeStageEventGroup)].arrivals)>=need;
    }
    ComputeSync();return *shared!=0;
  }
  __device__ void Wait(int global_k) const {
    if(!enabled)return;
    if(ComputeThread()==0) {
      // The phase ISL proof projects away the output N tile after checking
      // that every N tile has exactly the same producer relation.
      int q=global_k;
      int begin=(q/int(desc->div))*desc->scale+desc->offset;
      int end=begin+int(desc->count);
      int live=ActiveBlocks(*params,params->stages[desc->producer]);
      for(int producer=max(0,begin);producer<min(live,end);++producer)
        {Watch here=watch?*watch:Watch{};here.site=2;
         here.producer_stage=desc->producer;here.group=producer;
         here.row=EventIndex(*params,desc->producer,producer);
         WaitAtLeast(&events[here.row].arrivals,iteration+1,here);}
    }
    ComputeSync();
  }
};
#ifndef TILEMEGA_KPHASE
#define TILEMEGA_KPHASE 1
#endif
#ifndef TILEMEGA_KPHASE_CLASS_MASK
#define TILEMEGA_KPHASE_CLASS_MASK 31
#endif
template<bool Loader,bool L2,int Variant=0>
__device__ void Gemm(Params const& params,GemmInvocation const& inv,int local,Ring const& ring,
                     std::uint64_t& sequence,char* work,EventCounter* events,
                     unsigned long long iteration,Lookahead const* lookahead=nullptr) {
  if(inv.variant==Variant) {
    using V=GemmVariant<Variant>;
    using Body=PagedGemmTaskBody<PageArch,V::kTileM,V::kTileN,V::kTileK,
        TILEMEGA_PAGE_BYTES,TILEMEGA_PAGE_COUNT,TILEMEGA_ARCH_PATH_SM80!=0>;
    static_assert(TILEMEGA_PAGE_POOL_OFFSET-TILEMEGA_PAGE_WORKSPACE_OFFSET>=Body::kActivationBytes);
    static_assert(TILEMEGA_PAGE_POOL_OFFSET-TILEMEGA_PAGE_WORKSPACE_OFFSET>=Body::kScratchBytes);
    auto operands=Operands(inv);
#if TILEMEGA_TRACE_TASK
    operands.profile=ring.profile;
    if(operands.profile)operands.profile->bytes=2ull*min(V::kTileN,operands.n-(local%inv.tiles_n)*V::kTileN)*operands.k_count;
#endif
    if(params.serving_tensor_maps) {
      operands.tensor_map=static_cast<executor::TensorMap const*>(params.serving_tensor_maps)+inv.serving_weight_buffer;
      operands.tensor_k_begin=inv.serving_k_begin;
    }
    if constexpr(Loader) {
#if TILEMEGA_LOOKAHEAD_BYTES > 0
      Body::Load(operands,local%inv.tiles_n,ring,sequence,*lookahead);
#else
      Body::Load(operands,local%inv.tiles_n,ring,sequence);
#endif
    }
    else {
      PhaseGate gate{&params,&inv.serving_phase_gate,events,iteration,
          local,ring.SharedLastFlag(),L2 && TILEMEGA_KPHASE &&
          (TILEMEGA_KPHASE_CLASS_MASK & (1u<<inv.serving_phase_class)) &&
          inv.serving_phase_gate.enabled,ring.watch};
      Body::Run(operands,local/inv.tiles_n,local%inv.tiles_n,ring,
          sequence,work,gate);
    }
  }else if constexpr(Variant + 1 < TILEMEGA_GEMM_VARIANT_COUNT)
    Gemm<Loader,L2,Variant+1>(params,inv,local,ring,sequence,work,events,iteration,lookahead);
  else asm volatile("trap;");
}
template<bool Last=false,int Variant=0>
__device__ bool Combine(Params const& p,StageDesc const& stage,int task,char* work,
                        unsigned* ticket=nullptr,unsigned* shared_last=nullptr) {
  auto const& inv=static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
  if(inv.variant==Variant) {
    using V=GemmVariant<Variant>;
    bool last=false;
    auto run=[&](auto op){
      auto reduction=[&](auto body){
        body(
          reinterpret_cast<float const*>(p.buffers[stage.operand[0]]),inv.chunks,
          task/inv.tiles_n,task%inv.tiles_n,stage.batch_rows?p.dims.batch:p.dims.tokens(),
          stage.width,stage.width,inv.serving_output_stride,
          reinterpret_cast<cutlass::bfloat16_t*>(p.buffers[stage.operand[1]]),
          reinterpret_cast<cutlass::bfloat16_t const*>(inv.residual),
          reinterpret_cast<float*>(p.buffers[stage.operand[1]]),inv.serving_argmax_index,
          reinterpret_cast<float*>(work),inv.serving_norm_ss,inv.serving_ss_out,
          inv.k_total,TILEMEGA_NORM_EPSILON);
      };
      if constexpr(Last)reduction([&](auto... args){
        last=LastArriverGemmTaskBody<V::kTileM,V::kTileN,decltype(op)::value>::Run(
            ticket,inv.chunks,shared_last,args...);
      });
      else reduction([&](auto... args){
        ServingGemmCombineTaskBody<V::kTileM,V::kTileN,decltype(op)::value>::Run(args...);
      });
    };
    switch(inv.serving_op) {
      case backend::ServingEpilogueOp::kStore:run(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kStore>{});break;
      case backend::ServingEpilogueOp::kResidual:run(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kResidual>{});break;
      case backend::ServingEpilogueOp::kSwiGLU:run(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kSwiGLU>{});break;
      case backend::ServingEpilogueOp::kArgmaxPartial:run(std::integral_constant<backend::ServingEpilogueOp,backend::ServingEpilogueOp::kArgmaxPartial>{});break;
      default:asm volatile("trap;");
    }
    return last;
  }else if constexpr(Variant + 1 < TILEMEGA_GEMM_VARIANT_COUNT)
    return Combine<Last,Variant+1>(p,stage,task,work,ticket,shared_last);
  else asm volatile("trap;");
  return false;
}
__device__ inline ServingAttentionOperands AttentionOperands(Params const& p,StageDesc const& s) {
  using E=cutlass::bfloat16_t;
  auto ptr=[&](int i){return s.operand[i]==kNoOperand?nullptr:p.buffers[s.operand[i]];};
  ServingAttentionOperands operands{reinterpret_cast<E const*>(ptr(0)),reinterpret_cast<E*>(ptr(1)),reinterpret_cast<E*>(ptr(2)),
      reinterpret_cast<E const*>(ptr(3)),reinterpret_cast<E const*>(ptr(4)),reinterpret_cast<E const*>(ptr(5)),
      reinterpret_cast<E const*>(ptr(6)),reinterpret_cast<E*>(ptr(7)),reinterpret_cast<float*>(ptr(8)),
      reinterpret_cast<float*>(ptr(9)),p.dims.batch,int(s.extent),p.dims.capacity,p.dims.past,
      s.attention_kv_block,TILEMEGA_NORM_EPSILON};
  if(p.serving_tensor_maps) {
    auto* maps=static_cast<executor::TensorMap const*>(p.serving_tensor_maps);
    operands.key_tensor_map=maps+s.operand[1];operands.value_tensor_map=maps+s.operand[2];
  }
  return operands;
}
template<int Variant=0>
__device__ bool ArgmaxLast(Params const& p,GemmInvocation const& inv,
                          StageDesc const& reducer,int tile_m,char* work,
                          unsigned* ticket,unsigned* shared_last) {
  if(inv.variant==Variant) {
    using V=GemmVariant<Variant>;
    return LastArriverArgmaxTaskBody<V::kTileM>::Run(ticket,shared_last,
        inv.tiles_n,reinterpret_cast<float const*>(p.buffers[reducer.operand[0]]),
        reinterpret_cast<int const*>(p.buffers[reducer.operand[1]]),
        reinterpret_cast<int32_t*>(p.buffers[reducer.operand[2]]),
        tile_m,p.dims.batch,reducer.width,p.dims.capacity,
        p.dims.past+p.dims.seq,work);
  }
  if constexpr(Variant + 1 < TILEMEGA_GEMM_VARIANT_COUNT)
    return ArgmaxLast<Variant+1>(p,inv,reducer,tile_m,work,ticket,shared_last);
  asm volatile("trap;");return false;
}
__device__ inline void Publish(Params const&,EventCounter*,unsigned,unsigned,
                               unsigned long long);
__device__ inline void TraceReducer(Params const& p,unsigned reducer,unsigned reducer_task,
                                    unsigned producer,unsigned producer_task) {
#if TILEMEGA_TRACE_V2
  if(ComputeThread()==0 && p.reducer_trace) {
    auto& r=p.reducer_trace[reducer*p.reducer_trace_stride+reducer_task];
    r={TraceNow(),reducer,reducer_task,producer,producer_task,unsigned(blockIdx.x)};
  }
#endif
}
template<bool Loader,bool L2>
__device__ void Task(Params const& p,unsigned stage_index,int task,Ring const& input_ring,
                     std::uint64_t& sequence,char* work,EventCounter* events,
                     unsigned long long iteration,
                     unsigned long long* step_ns=nullptr,unsigned step=0,
                     Lookahead const* lookahead=nullptr) {
  auto const& s=p.stages[stage_index];using E=cutlass::bfloat16_t;
  if(s.handoff_elided)return;
#if TILEMEGA_TRACE_TASK
  executor::ProfileScope profile{!Loader && (s.kind==TaskKind::kGemm || s.kind==TaskKind::kFusedAttention)
      ?executor::BeginProfile(p,stage_index,task,iteration):nullptr};
  Ring ring=input_ring;ring.profile=profile.row;
#else
  auto const& ring=input_ring;
#endif
  if constexpr(Loader)if(lookahead)(*lookahead)(0);
  if constexpr(!Loader && L2)if(s.kind==TaskKind::kEmbedding && iteration) {
#if TILEMEGA_TRACE_STEP
    auto lag_begin=executor::ServingTraceNow();
#endif
    if(ComputeThread()==0)for(unsigned edge=0;edge<p.lag_dependency_count;++edge) {
      auto const& lag=p.lag_dependencies[edge];
      if(lag.kind==LagDependency::Kind::kToken && lag.consumer==stage_index)
        {Watch here=ring.watch?*ring.watch:Watch{};here.site=3;
         here.producer_stage=lag.producer;
         here.row=EventIndex(p,lag.producer,kWholeStageEventGroup);
         WaitAtLeast(&events[here.row].arrivals,
                    static_cast<unsigned long long>(p.dims.batch)*iteration,here);}
    }
    ComputeSync();
#if TILEMEGA_TRACE_STEP
    if(ComputeThread()==0)executor::StepDelay(p,iteration,0,lag_begin);
#endif
  }
  if(s.kind==TaskKind::kGemm) {
    auto const* table=static_cast<GemmInvocation const*>(p.gemms);
    auto point=DecodeSplitTask(task,table[s.gemm].tiles_m*table[s.gemm].tiles_n,table[s.gemm].chunks);
    Gemm<Loader,L2>(p,table[s.gemm+point.chunk],point.tile,ring,sequence,work,
        events,iteration,lookahead);
    if constexpr(!Loader)if(s.handoff_reduce_stage!=kNoOperand) {
      auto const& reducer=p.stages[s.handoff_reduce_stage];
      auto* ticket=p.serving_handoff_tickets+
          stage_index*p.serving_handoff_ticket_stride+
          (reducer.kind==TaskKind::kArgmaxReduce?
              point.tile/table[s.gemm].tiles_n:point.tile);
      if(reducer.kind==TaskKind::kArgmaxReduce) {
        int tile_m=point.tile/table[s.gemm].tiles_n;
        bool last=ArgmaxLast(p,table[s.gemm],reducer,tile_m,work,ticket,
                            ring.SharedLastFlag());
        if constexpr(L2)if(last) {
          // A reducer event belongs to each request row, not the M tile.
          for(int row=tile_m*int(table[s.gemm].tile_m);
              row<(tile_m+1)*int(table[s.gemm].tile_m) && row<p.dims.batch;++row)
            {Publish(p,events,s.handoff_reduce_stage,row,iteration);
             TraceReducer(p,s.handoff_reduce_stage,row,stage_index,task);}
          if(step_ns && tile_m==0 && ComputeThread()==0) {
            {Watch here=ring.watch?*ring.watch:Watch{};here.site=5;
             here.producer_stage=s.handoff_reduce_stage;
             here.row=EventIndex(p,s.handoff_reduce_stage,kWholeStageEventGroup);
             WaitAtLeast(&events[here.row].arrivals,
                static_cast<unsigned long long>(p.dims.batch)*(iteration+1),here);}
            step_ns[step+1]=executor::PageTraceNow();
          }
        }
      }else {
        bool last=Combine<true>(p,reducer,point.tile,work,ticket,
            ring.SharedLastFlag());
        if constexpr(L2)if(last) {
          Publish(p,events,s.handoff_reduce_stage,point.tile,iteration);
          TraceReducer(p,s.handoff_reduce_stage,point.tile,stage_index,task);
        }
      }
    }
    return;
  }
  if(s.kind==TaskKind::kFusedAttention) {
    auto point=DecodeServingAttentionTaskGMajor(task,p.dims.batch,
        CeilDiv(p.dims.capacity,s.attention_kv_block));
    if constexpr(Loader) {
      if(iteration) {
#if TILEMEGA_TRACE_STEP
        auto lag_begin=executor::ServingTraceNow();
#endif
        if(executor::LoaderLane()==0)for(unsigned edge=0;
            edge<p.lag_dependency_count;++edge) {
          auto const& lag=p.lag_dependencies[edge];
          if(lag.kind!=LagDependency::Kind::kHistoricalKv ||
             lag.consumer!=stage_index)continue;
          if constexpr(L2) {
            unsigned source=lag.producer;
            {Watch here=ring.watch?*ring.watch:Watch{};here.site=4;
             here.producer_stage=source;
             here.row=EventIndex(p,source,kWholeStageEventGroup);
             WaitAtLeast(&events[here.row].arrivals,
                static_cast<unsigned long long>(ActiveBlocks(p,p.stages[source]))*iteration,here);}
          }
        }
        __syncwarp();
#if TILEMEGA_TRACE_STEP
        if(executor::LoaderLane()==0)executor::StepDelay(p,iteration,1,lag_begin);
#endif
        Ring::Copy::ProxyAsyncGlobalFence();
      }
#if TILEMEGA_LOOKAHEAD_BYTES > 0
      Attention::Load(AttentionOperands(p,s),point.batch,point.group,point.cache_block,ring,sequence,*lookahead);
#else
      Attention::Load(AttentionOperands(p,s),point.batch,point.group,point.cache_block,ring,sequence);
#endif
    }
    else {
      auto operands=AttentionOperands(p,s);
#if TILEMEGA_TRACE_TASK
      operands.profile=profile.row;
#endif
      Attention::Run(operands,point.batch,point.group,point.cache_block,ring,sequence,
                     *reinterpret_cast<Attention::SharedStorage*>(work));
#if TILEMEGA_TRACE_TASK
      auto la_begin=TaskProfileNow(profile.row);
#endif
      if(s.handoff_reduce_stage!=kNoOperand) {
        auto* ticket=p.serving_handoff_tickets+
            stage_index*p.serving_handoff_ticket_stride+
            point.group*p.dims.batch+point.batch;
        bool last=LastArriverAttentionTaskBody<TILEMEGA_SERVING_HEAD_DIM,
            TILEMEGA_SERVING_QPERKV,TILEMEGA_SERVING_SEQ>::Run(
                ticket,ring.SharedLastFlag(),point.cache_block,
                operands.partial,operands.lse,operands.context,
                point.batch,point.group,int(s.extent),p.dims.capacity,
                s.attention_kv_block,p.dims.past);
        if constexpr(L2)if(last) {
          unsigned rt=point.group*p.dims.batch+point.batch;
          Publish(p,events,s.handoff_reduce_stage,rt,iteration);
          TraceReducer(p,s.handoff_reduce_stage,rt,stage_index,task);
        }
      }
#if TILEMEGA_TRACE_TASK
      if(profile.row)profile.row->la_ns=TaskProfileNow(profile.row)-la_begin;
#endif
    }
    return;
  }
  if constexpr(!Loader) {
    auto ptr=[&](int i){return p.buffers[s.operand[i]];};
    switch(s.kind) {
      case TaskKind::kGemmCombine:Combine(p,s,task,work);break;
      case TaskKind::kEmbedding:ServingEmbeddingTaskBody::RunRow(reinterpret_cast<int const*>(ptr(0)),
          reinterpret_cast<E const*>(ptr(1)),reinterpret_cast<E*>(ptr(2)),task,p.dims.seq,p.dims.past,
          p.dims.capacity,s.width,s.extent,
          s.operand[3]==kNoOperand?nullptr:reinterpret_cast<float*>(ptr(3)));break;
      case TaskKind::kRMSNorm:ServingRMSNormTaskBody::RunRow(reinterpret_cast<E const*>(ptr(0)),
          reinterpret_cast<E const*>(ptr(1)),reinterpret_cast<E*>(ptr(2)),task,s.row_stride,s.row_offset,
          s.width,TILEMEGA_NORM_EPSILON,reinterpret_cast<float*>(work));break;
      case TaskKind::kArgmaxReduce:ServingArgmaxReduceTaskBody::RunRow(reinterpret_cast<float const*>(ptr(0)),
          reinterpret_cast<int const*>(ptr(1)),reinterpret_cast<int*>(ptr(2)),task,s.width,p.dims.capacity,
          p.dims.past+p.dims.seq,reinterpret_cast<ServingArgmaxReduceTaskBody::SharedStorage*>(work));break;
      case TaskKind::kAttentionMerge:RunServingMergeTask(p,s,task);break;
      default:asm volatile("trap;");
    }
  }
}
__device__ inline void WaitDependencies(Params const& p,EventCounter* events,TaskRef const& task,
                                       unsigned long long iteration) {
  if(task.wait_count && ComputeThread()==0)executor::PageTraceTransition(p.serving_page_trace ? p.serving_page_trace+blockIdx.x : nullptr,1u,true);
  for(unsigned i=ComputeThread();i<task.wait_count;i+=kComputeThreads) {
    auto const& w=p.task_waits[task.wait_begin+i];
#if TILEMEGA_SYNC_V3
    WaitAtLeast(&events[EventIndex(p,w.producer,w.group)].arrivals,
                EventTriggers(p,w.producer,w.group)*(iteration+1),
                Watch{p.serving_watchdog,p.serving_watchdog_ns,1,task.stage,
                    task.logical_task,w.producer,w.group,
                    EventIndex(p,w.producer,w.group),iteration});
#elif TILEMEGA_EVENT_RED_PUBLISH
    GradedWait(&events[EventIndex(p,w.producer,w.group)].arrivals,
               EventTriggers(p,w.producer,w.group)*(iteration+1));
#else
    GradedWait(&events[EventIndex(p,w.producer,w.group)].epoch,iteration+1);
#endif
  }
  ComputeSync();
#if !TILEMEGA_SYNC_V3
  if(task.wait_count)__threadfence();
#endif
  if(task.wait_count && ComputeThread()==0)executor::PageTraceTransition(p.serving_page_trace ? p.serving_page_trace+blockIdx.x : nullptr,1u,false);
}
__device__ inline void Publish(Params const& p,EventCounter* events,unsigned stage,unsigned task,
                               unsigned long long iteration) {
  auto flags=p.event_flags[stage];if(!flags)return;
#if !TILEMEGA_SYNC_V3 && !TILEMEGA_RELEASE_AFTER_BARRIER
  __threadfence();
#endif
  ComputeSync();
  if(ComputeThread()==0) {
#if !TILEMEGA_SYNC_V3 && TILEMEGA_RELEASE_AFTER_BARRIER
    __threadfence();
#endif
    int count=ActiveBlocks(p,p.stages[stage]);
    if(flags&kNeedsFineEvents) {
      int k=StageKappa(p,stage),group=task/k;
      ArriveEvent(p,events,EventIndex(p,stage,group),min(k,count-group*k),iteration);
    }
    if(flags&kNeedsAggregateEvent)ArriveEvent(p,events,EventIndex(p,stage,kWholeStageEventGroup),count,iteration);
  }
#if !TILEMEGA_SYNC_V3 && !TILEMEGA_BARRIER_V2
  ComputeSync();
#endif
}
__device__ inline void StageBarrier(EventCounter* events,unsigned stage,unsigned long long iteration,
                                    Watch const* watch=nullptr,Params const* params=nullptr) {
#if TILEMEGA_SYNC_V3
  ComputeSync();
  if(ComputeThread()==0) {
    RedRelease(&events[stage].arrivals,1ull);
    executor::StageTasksEnd(params,stage,iteration);
    Watch here=watch?*watch:Watch{};here.site=9;here.row=stage;
    WaitAtLeast(&events[stage].arrivals,
                static_cast<unsigned long long>(gridDim.x)*(iteration+1),here);
  }
  ComputeSync();
#else
  __threadfence();ComputeSync();
  if(ComputeThread()==0) {

    auto ticket=atomicAdd(&events[stage].arrivals,1ull);
    if(ticket+1==static_cast<unsigned long long>(gridDim.x)*(iteration+1)) {
      __threadfence();TILEMEGA_GENERATED_NOTIFY_global(&events[stage].epoch,iteration+1);
    }else GradedWait(&events[stage].epoch,iteration+1);
  }
  ComputeSync();__threadfence();
#endif
}
template<bool Loader,bool L2>
__device__ void Execute(Params const& p,EventCounter* events,unsigned long long iteration,
                        Ring const& source_ring,char* work,
                        std::uint64_t* persistent_sequence=nullptr,
                        unsigned long long* step_ns=nullptr,unsigned step=0,
                        bool first_step=true,bool final_step=true,
                        PageStream* persistent_ahead=nullptr,
                        unsigned long long* persistent_prefetched=nullptr,
                        unsigned long long* persistent_loaded=nullptr) {
  if constexpr(!Loader)executor::StepBegin(p,iteration);
  Watch watch{p.serving_watchdog,p.serving_watchdog_ns};
  watch.iteration=iteration;
  Ring ring=source_ring;ring.watch=&watch;
  std::uint64_t local_sequence=0;
  std::uint64_t& sequence=persistent_sequence?*persistent_sequence:local_sequence;
  unsigned long long local_prefetched=0,local_loaded=0;
  unsigned long long& prefetched=persistent_prefetched?*persistent_prefetched:local_prefetched;
  unsigned long long& loaded=persistent_loaded?*persistent_loaded:local_loaded;
  PageStream local_ahead(&p,1,L2);
  PageStream& ahead=persistent_ahead?*persistent_ahead:local_ahead;
  Lookahead lookahead{&ahead,&prefetched,&loaded};
  bool previous_grid_ready=false;
  auto wait_previous=[&](unsigned stage) {
    // Loader reads weights before the wait, but never a historical KV page.
    // Every compute lane waits before any activation or token access.
    if(first_step && !previous_grid_ready &&
       (!Loader || p.stages[stage].kind==TaskKind::kFusedAttention)) {
      executor::GridDependency<PageArch>::Wait();previous_grid_ready=true;
    }
  };
  if constexpr(L2) {
    for(unsigned slot=p.schedule_offsets[blockIdx.x];slot<p.schedule_offsets[blockIdx.x+1];++slot) {
      auto const& task=p.schedule[slot];
      watch.waiter_stage=task.stage;watch.waiter_task=task.logical_task;
      wait_previous(task.stage);
#if TILEMEGA_TRACE_V2
      if constexpr(!Loader)if(ComputeThread()==0 && p.task_trace_v2)
        p.task_trace_v2[slot].wait_begin=TraceNow();
#endif
      if constexpr(!Loader)WaitDependencies(p,events,task,iteration);
#if TILEMEGA_TRACE_V2
      if constexpr(!Loader)if(ComputeThread()==0 && p.task_trace_v2) {
        auto& row=p.task_trace_v2[slot];row.ready=TraceNow();row.run_begin=TraceNow();
        row.run_begin_clk=clock64();row.smid=TraceSmid();row.worker=blockIdx.x;
        row.stage=task.stage;row.logical_task=task.logical_task;
      }
#endif
      if constexpr(!Loader)executor::TaskBegin(p,iteration);
      Task<Loader,L2>(p,task.stage,task.logical_task,ring,sequence,work,
          events,iteration,step_ns,step,Loader?&lookahead:nullptr);
#if TILEMEGA_TRACE_V2
      if constexpr(!Loader)if(ComputeThread()==0 && p.task_trace_v2) {
        p.task_trace_v2[slot].run_end=TraceNow();p.task_trace_v2[slot].run_end_clk=clock64();
      }
#endif
      if constexpr(!Loader)executor::TaskEnd(p,iteration);
      if constexpr(!Loader)Publish(p,events,task.stage,task.logical_task,iteration);
#if TILEMEGA_TRACE_V2
      if constexpr(!Loader)if(ComputeThread()==0 && p.task_trace_v2)
        p.task_trace_v2[slot].publish_end=TraceNow();
#endif
    }
  }else {
    for(unsigned stage=0;stage<p.stage_count;++stage) {
      if(p.stages[stage].handoff_elided)continue;
      watch.waiter_stage=stage;
      wait_previous(stage);
      int count=ActiveBlocks(p,p.stages[stage]);
      if constexpr(!Loader)executor::StageBegin(p,stage,iteration,
          count>int(blockIdx.x)?(count-blockIdx.x+gridDim.x-1)/gridDim.x:0);
      for(int task=blockIdx.x;task<count;task+=gridDim.x)
        {
          watch.waiter_task=task;
                  Task<Loader,L2>(p,stage,task,ring,sequence,work,events,iteration,
              step_ns,step,Loader?&lookahead:nullptr);
              if constexpr(!Loader)ComputeSync();
        }
      if constexpr(!Loader) {
        StageBarrier(events,stage,iteration,&watch,&p);
        executor::StageRelease(p,stage,iteration);
      }
    }
  }
  if constexpr(!Loader)executor::StepEnd(p,iteration);
  if constexpr(!Loader)if(final_step)executor::GridDependency<PageArch>::Release();
}
} // namespace paged
__global__ __launch_bounds__(160,1)
void tilemega_l1_kernel(Params const* p,EventCounter* events,unsigned long long iteration) {
#if TILEMEGA_PAGE_TRACE
  if(threadIdx.x==0 && p->serving_page_trace)
    p->serving_page_trace[blockIdx.x].kernel_begin_ns=executor::PageTraceNow();
#endif
  executor::StepBegin(*p,iteration);
  extern __shared__ __align__(1024) char page_storage[];
  paged::Ring ring{reinterpret_cast<paged::Ring::Slot*>(page_storage),page_storage+TILEMEGA_PAGE_POOL_OFFSET,
      p->serving_page_trace ? p->serving_page_trace+blockIdx.x : nullptr};ring.Initialize();
  if(executor::IsCompute())paged::Execute<false,false>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
  else paged::Execute<true,false>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
#if TILEMEGA_TRACE_STEP
  if(executor::LoaderLane()==0)if(auto* row=executor::StepRow(*p,iteration))
    atomicMax(&row->kernel_end,executor::ServingTraceNow());
#endif
#if TILEMEGA_PAGE_TRACE
  // The compute group can finish before the loader warp, or vice versa.
  // Record the later completion so adjacent-launch gaps use the full CTA span.
  if(executor::IsCompute())executor::ComputeSync();
  else __syncwarp();
  if((threadIdx.x==0 || executor::LoaderLane()==0) && p->serving_page_trace)
    atomicMax(&p->serving_page_trace[blockIdx.x].kernel_end_ns,executor::PageTraceNow());
#endif
}
__global__ __launch_bounds__(160,1)
void tilemega_l2_kernel(Params const* p,EventCounter* events,unsigned long long iteration) {
#if TILEMEGA_PAGE_TRACE
  if(threadIdx.x==0 && p->serving_page_trace)
    p->serving_page_trace[blockIdx.x].kernel_begin_ns=executor::PageTraceNow();
#endif
  executor::StepBegin(*p,iteration);
  extern __shared__ __align__(1024) char page_storage[];
  paged::Ring ring{reinterpret_cast<paged::Ring::Slot*>(page_storage),page_storage+TILEMEGA_PAGE_POOL_OFFSET,
      p->serving_page_trace ? p->serving_page_trace+blockIdx.x : nullptr};ring.Initialize();
  if(executor::IsCompute())paged::Execute<false,true>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
  else paged::Execute<true,true>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
#if TILEMEGA_TRACE_STEP
  if(executor::LoaderLane()==0)if(auto* row=executor::StepRow(*p,iteration))
    atomicMax(&row->kernel_end,executor::ServingTraceNow());
#endif
#if TILEMEGA_PAGE_TRACE
  if(executor::IsCompute())executor::ComputeSync();
  else __syncwarp();
  if((threadIdx.x==0 || executor::LoaderLane()==0) && p->serving_page_trace)
    atomicMax(&p->serving_page_trace[blockIdx.x].kernel_end_ns,executor::PageTraceNow());
#endif
}
#ifndef TILEMEGA_PAGE_LOOP_SPLIT
#define TILEMEGA_PAGE_LOOP_SPLIT 0
#endif
__global__ __launch_bounds__(160,1)
void tilemega_loop_kernel(Params const* params,unsigned steps,EventCounter* events,
                          unsigned long long base_iteration,
                          unsigned long long* step_ns) {
  extern __shared__ __align__(1024) char page_storage[];
  paged::Ring ring{reinterpret_cast<paged::Ring::Slot*>(page_storage),
      page_storage+TILEMEGA_PAGE_POOL_OFFSET,nullptr};
  ring.Initialize();
  std::uint64_t sequence=0;
  bool const compute=executor::IsCompute();
  paged::PageStream ahead(params,steps,true);
  unsigned long long prefetched=0,loaded=0;
#if TILEMEGA_PAGE_LOOP_SPLIT
  // The role is uniform for the lifetime of each warp. Keeping the compute
  // loop outside the loader loop removes loader cursor liveness from the
  // compute path. The page sequence and all lag/acquire waits are unchanged.
  if(compute) {
    for(unsigned step=0;step<steps;++step) {
      Params const& p=params[step];
      if(step==0 && blockIdx.x==0 && paged::ComputeThread()==0 && step_ns)
        step_ns[0]=executor::PageTraceNow();
      paged::Execute<false,true>(p,events,base_iteration+step,ring,
          page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET,&sequence,step_ns,step,
          step==0,step+1==steps);
    }
  }else {
    for(unsigned step=0;step<steps;++step) {
      Params const& p=params[step];
      paged::Execute<true,true>(p,events,base_iteration+step,ring,
          page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET,&sequence,step_ns,step,
          step==0,step+1==steps,&ahead,&prefetched,&loaded);
#if TILEMEGA_TRACE_STEP
      if(executor::LoaderLane()==0)
        if(auto* row=executor::StepRow(p,base_iteration+step))
          atomicMax(&row->kernel_end,executor::ServingTraceNow());
#endif
    }
  }
#else
  for(unsigned step=0;step<steps;++step) {
    Params const& p=params[step];
    if(step==0 && blockIdx.x==0 && compute && paged::ComputeThread()==0 && step_ns)
      step_ns[0]=executor::PageTraceNow();
    if(compute)paged::Execute<false,true>(p,events,base_iteration+step,ring,
        page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET,&sequence,step_ns,step,
        step==0,step+1==steps);
    else paged::Execute<true,true>(p,events,base_iteration+step,ring,
        page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET,&sequence,step_ns,step,
        step==0,step+1==steps,&ahead,&prefetched,&loaded);
#if TILEMEGA_TRACE_STEP
    if(!compute && executor::LoaderLane()==0)
      if(auto* row=executor::StepRow(p,base_iteration+step))
        atomicMax(&row->kernel_end,executor::ServingTraceNow());
#endif
  }
#endif
}
