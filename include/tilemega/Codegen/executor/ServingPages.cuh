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

__device__ inline ServingGemmOperands Operands(GemmInvocation const& inv) {
  auto [m,n,k,batch]=inv.problem;(void)batch;
  ServingGemmOperands p;p.a=inv.mainloop.ptr_A;p.b=inv.mainloop.ptr_B;
  p.residual=inv.epilogue.ptr_C;p.output=inv.epilogue.ptr_D;p.partial=inv.serving_partial;
  p.argmax_value=reinterpret_cast<float*>(inv.epilogue.ptr_D);p.argmax_index=inv.serving_argmax_index;
  p.m=m;p.n=n;p.k_total=k;p.k_count=k;p.output_stride=inv.serving_output_stride;
  p.partial_stride=inv.serving_partial_stride;
  p.a_row_stride=p.b_row_stride=inv.k_total;
  p.epilogue=inv.chunks>1?backend::ServingEpilogueOp::kPartial:inv.serving_op;
  return p;
}
template<bool Loader,int Variant=0>
__device__ void Gemm(Params const& params,GemmInvocation const& inv,int local,Ring const& ring,
                     std::uint64_t& sequence,char* work) {
  if(inv.variant==Variant) {
    using V=GemmVariant<Variant>;
    using Body=PagedGemmTaskBody<PageArch,V::kTileM,V::kTileN,V::kTileK,
        TILEMEGA_PAGE_BYTES,TILEMEGA_PAGE_COUNT,TILEMEGA_ARCH_PATH_SM80!=0>;
    static_assert(TILEMEGA_PAGE_POOL_OFFSET-TILEMEGA_PAGE_WORKSPACE_OFFSET>=Body::kActivationBytes);
    static_assert(TILEMEGA_PAGE_POOL_OFFSET-TILEMEGA_PAGE_WORKSPACE_OFFSET>=Body::kScratchBytes);
    auto operands=Operands(inv);
    if(params.serving_tensor_maps) {
      operands.tensor_map=static_cast<executor::TensorMap const*>(params.serving_tensor_maps)+inv.serving_weight_buffer;
      operands.tensor_k_begin=inv.serving_k_begin;
    }
    if constexpr(Loader)Body::Load(operands,local%inv.tiles_n,ring,sequence);
    else Body::Run(operands,local/inv.tiles_n,local%inv.tiles_n,ring,sequence,work,
                   inv.serving_norm_input,inv.serving_norm_weight,
                   TILEMEGA_NORM_EPSILON);
  }else if constexpr(Variant+1<TILEMEGA_GEMM_VARIANT_COUNT)Gemm<Loader,Variant+1>(params,inv,local,ring,sequence,work);
  else asm volatile("trap;");
}
template<bool Last=false,int Variant=0>
__device__ void Combine(Params const& p,StageDesc const& stage,int task,char* work,
                        unsigned* ticket=nullptr,unsigned* shared_last=nullptr) {
  auto const& inv=static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
  if(inv.variant==Variant) {
    using V=GemmVariant<Variant>;
    auto run=[&](auto op){
      auto reduction=[&](auto body){
        body(
          reinterpret_cast<float const*>(p.buffers[stage.operand[0]]),inv.chunks,
          task/inv.tiles_n,task%inv.tiles_n,stage.batch_rows?p.dims.batch:p.dims.tokens(),
          stage.width,stage.width,inv.serving_output_stride,
          reinterpret_cast<cutlass::bfloat16_t*>(p.buffers[stage.operand[1]]),
          reinterpret_cast<cutlass::bfloat16_t const*>(inv.residual),
          reinterpret_cast<float*>(p.buffers[stage.operand[1]]),inv.serving_argmax_index,reinterpret_cast<float*>(work));
      };
      if constexpr(Last)reduction([&](auto... args){
        LastArriverGemmTaskBody<V::kTileM,V::kTileN,decltype(op)::value>::Run(
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
  }else if constexpr(Variant+1<TILEMEGA_GEMM_VARIANT_COUNT)
    Combine<Last,Variant+1>(p,stage,task,work,ticket,shared_last);
  else asm volatile("trap;");
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
template<bool Loader>
__device__ void Task(Params const& p,unsigned stage_index,int task,Ring const& ring,
                     std::uint64_t& sequence,char* work) {
  auto const& s=p.stages[stage_index];using E=cutlass::bfloat16_t;
  if(s.handoff_elided)return;
  if(s.kind==TaskKind::kGemm) {
    auto const* table=static_cast<GemmInvocation const*>(p.gemms);
    auto point=DecodeSplitTask(task,table[s.gemm].tiles_m*table[s.gemm].tiles_n,table[s.gemm].chunks);
    Gemm<Loader>(p,table[s.gemm+point.chunk],point.tile,ring,sequence,work);
    if constexpr(!Loader)if(s.handoff_reduce_stage!=kNoOperand) {
      auto* ticket=p.serving_handoff_tickets+
          stage_index*p.serving_handoff_ticket_stride+point.tile;
      auto const& reducer=p.stages[s.handoff_reduce_stage];
      Combine<true>(p,reducer,point.tile,work,ticket,
          reinterpret_cast<unsigned*>(work));
    }
    return;
  }
  if(s.kind==TaskKind::kFusedAttention) {
    auto point=DecodeServingAttentionTask(task,1,int(s.extent),CeilDiv(p.dims.capacity,s.attention_kv_block));
    if constexpr(Loader)Attention::Load(AttentionOperands(p,s),point.batch,point.group,point.cache_block,ring,sequence);
    else {
      auto operands=AttentionOperands(p,s);
      Attention::Run(operands,point.batch,point.group,point.cache_block,ring,sequence,
                     *reinterpret_cast<Attention::SharedStorage*>(work));
      if(s.handoff_reduce_stage!=kNoOperand) {
        auto* ticket=p.serving_handoff_tickets+
            stage_index*p.serving_handoff_ticket_stride+
            point.batch*int(s.extent)+point.group;
        LastArriverAttentionTaskBody<TILEMEGA_SERVING_HEAD_DIM,
            TILEMEGA_SERVING_QPERKV,TILEMEGA_SERVING_SEQ>::Run(
                ticket,reinterpret_cast<unsigned*>(work),point.cache_block,
                operands.partial,operands.lse,operands.context,
                point.batch,point.group,int(s.extent),p.dims.capacity,
                s.attention_kv_block,p.dims.past);
      }
    }
    return;
  }
  if constexpr(!Loader) {
    auto ptr=[&](int i){return p.buffers[s.operand[i]];};
    switch(s.kind) {
      case TaskKind::kGemmCombine:Combine(p,s,task,work);break;
      case TaskKind::kEmbedding:ServingEmbeddingTaskBody::RunRow(reinterpret_cast<int const*>(ptr(0)),
          reinterpret_cast<E const*>(ptr(1)),reinterpret_cast<E*>(ptr(2)),task,p.dims.seq,p.dims.past,
          p.dims.capacity,s.width,s.extent);break;
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
#if TILEMEGA_EVENT_RED_PUBLISH
    GradedWait(&events[EventIndex(p,w.producer,w.group)].arrivals,
               EventTriggers(p,w.producer,w.group)*(iteration+1));
#else
    GradedWait(&events[EventIndex(p,w.producer,w.group)].epoch,iteration+1);
#endif
  }
  ComputeSync();if(task.wait_count)__threadfence();
  if(task.wait_count && ComputeThread()==0)executor::PageTraceTransition(p.serving_page_trace ? p.serving_page_trace+blockIdx.x : nullptr,1u,false);
}
__device__ inline void Publish(Params const& p,EventCounter* events,unsigned stage,unsigned task,
                               unsigned long long iteration) {
  auto flags=p.event_flags[stage];if(!flags)return;
#if !TILEMEGA_RELEASE_AFTER_BARRIER
  __threadfence();
#endif
  ComputeSync();
  if(ComputeThread()==0) {
#if TILEMEGA_RELEASE_AFTER_BARRIER
    __threadfence();
#endif
    int count=ActiveBlocks(p,p.stages[stage]);
    if(flags&kNeedsFineEvents) {
      int k=StageKappa(p,stage),group=task/k;
      ArriveEvent(p,events,EventIndex(p,stage,group),min(k,count-group*k),iteration);
    }
    if(flags&kNeedsAggregateEvent)ArriveEvent(p,events,EventIndex(p,stage,kWholeStageEventGroup),count,iteration);
  }
#if !TILEMEGA_BARRIER_V2
  ComputeSync();
#endif
}
__device__ inline void StageBarrier(EventCounter* events,unsigned stage,unsigned long long iteration) {
  __threadfence();ComputeSync();
  if(ComputeThread()==0) {

    auto ticket=atomicAdd(&events[stage].arrivals,1ull);
    if(ticket+1==static_cast<unsigned long long>(gridDim.x)*(iteration+1)) {
      __threadfence();TILEMEGA_GENERATED_NOTIFY_global(&events[stage].epoch,iteration+1);
    }else GradedWait(&events[stage].epoch,iteration+1);
  }
  ComputeSync();__threadfence();
}
template<bool Loader,bool L2>
__device__ void Execute(Params const& p,EventCounter* events,unsigned long long iteration,
                        Ring const& ring,char* work) {
  std::uint64_t sequence=0;
  bool previous_grid_ready=false;
  auto wait_previous=[&](unsigned stage) {
    // Loader reads weights before the wait, but never a historical KV page.
    // Every compute lane waits before any activation or token access.
    if(!previous_grid_ready && (!Loader || p.stages[stage].kind==TaskKind::kFusedAttention)) {
      executor::GridDependency<PageArch>::Wait();previous_grid_ready=true;
    }
  };
  if constexpr(L2) {
    for(unsigned slot=p.schedule_offsets[blockIdx.x];slot<p.schedule_offsets[blockIdx.x+1];++slot) {
      auto const& task=p.schedule[slot];
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
      Task<Loader>(p,task.stage,task.logical_task,ring,sequence,work);
#if TILEMEGA_TRACE_V2
      if constexpr(!Loader)if(ComputeThread()==0 && p.task_trace_v2) {
        p.task_trace_v2[slot].run_end=TraceNow();p.task_trace_v2[slot].run_end_clk=clock64();
      }
#endif
      if constexpr(!Loader)Publish(p,events,task.stage,task.logical_task,iteration);
#if TILEMEGA_TRACE_V2
      if constexpr(!Loader)if(ComputeThread()==0 && p.task_trace_v2)
        p.task_trace_v2[slot].publish_end=TraceNow();
#endif
    }
  }else {
    for(unsigned stage=0;stage<p.stage_count;++stage) {
      wait_previous(stage);
      int count=ActiveBlocks(p,p.stages[stage]);
      for(int task=blockIdx.x;task<count;task+=gridDim.x)
        {
          Task<Loader>(p,stage,task,ring,sequence,work);
          if constexpr(!Loader)ComputeSync();
        }
      if constexpr(!Loader)StageBarrier(events,stage,iteration);
    }
  }
  if constexpr(!Loader)executor::GridDependency<PageArch>::Release();
}
} // namespace paged
__global__ __launch_bounds__(160,1)
void tilemega_l1_kernel(Params const* p,EventCounter* events,unsigned long long iteration) {
#if TILEMEGA_PAGE_TRACE
  if(threadIdx.x==0 && p->serving_page_trace)
    p->serving_page_trace[blockIdx.x].kernel_begin_ns=executor::PageTraceNow();
#endif
  extern __shared__ __align__(1024) char page_storage[];
  paged::Ring ring{reinterpret_cast<paged::Ring::Slot*>(page_storage),page_storage+TILEMEGA_PAGE_POOL_OFFSET,
      p->serving_page_trace ? p->serving_page_trace+blockIdx.x : nullptr};ring.Initialize();
  if(executor::IsCompute())paged::Execute<false,false>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
  else paged::Execute<true,false>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
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
  extern __shared__ __align__(1024) char page_storage[];
  paged::Ring ring{reinterpret_cast<paged::Ring::Slot*>(page_storage),page_storage+TILEMEGA_PAGE_POOL_OFFSET,
      p->serving_page_trace ? p->serving_page_trace+blockIdx.x : nullptr};ring.Initialize();
  if(executor::IsCompute())paged::Execute<false,true>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
  else paged::Execute<true,true>(*p,events,iteration,ring,page_storage+TILEMEGA_PAGE_WORKSPACE_OFFSET);
#if TILEMEGA_PAGE_TRACE
  if(executor::IsCompute())executor::ComputeSync();
  else __syncwarp();
  if((threadIdx.x==0 || executor::LoaderLane()==0) && p->serving_page_trace)
    atomicMax(&p->serving_page_trace[blockIdx.x].kernel_end_ns,executor::PageTraceNow());
#endif
}
