// SPDX-License-Identifier: BSD-3-Clause
// Included inside tilemega::codegen after task dispatch and event publication.
namespace nonpaged {
__device__ inline unsigned long long* Ticket(Params const& p,unsigned producer,
                                             unsigned task) {
  if(!p.serving_epoch_handoff_tickets || producer>=p.stage_count ||
     task>=p.serving_epoch_handoff_stride) {
    asm volatile("trap;");return nullptr;
  }
  return p.serving_epoch_handoff_tickets+
      std::size_t(producer)*p.serving_epoch_handoff_stride+task;
}
__device__ inline void TraceReducer(Params const& p,unsigned reducer,unsigned task,
                                    unsigned producer,unsigned producer_task) {
#if TILEMEGA_TRACE_V2
  if(executor::ComputeThread()==0 && p.reducer_trace)
    p.reducer_trace[reducer*p.reducer_trace_stride+task]=
        {TraceNow(),reducer,task,producer,producer_task,unsigned(blockIdx.x)};
#endif
}
template<bool L2>
__device__ inline void RunTask(Params const& p,unsigned stage_index,unsigned task,
    TaskSmem& smem,EventCounter* events,unsigned long long iteration
    TILEMEGA_PHASE_ARG TILEMEGA_PREFETCH_ARG) {
  auto const& stage=p.stages[stage_index];
  tilemega::codegen::RunTask(p,stage_index,task,smem
      TILEMEGA_PHASE_PASS TILEMEGA_PREFETCH_PASS);
#if TILEMEGA_DM_REDUCTIONS
  CompleteDmTask<L2>(p,stage_index,task,smem,events,iteration);
#endif
  if(stage.handoff_reduce_stage==kNoOperand)return;
  auto reducer_index=stage.handoff_reduce_stage;
  if(reducer_index>=p.stage_count){asm volatile("trap;");return;}
  auto const& reducer=p.stages[reducer_index];
  unsigned reducer_task=0,producers=0;
  auto reduce=[&] {
    if(stage.kind==TaskKind::kGemm && reducer.kind==TaskKind::kGemmCombine)
      T_GemmCombine::DispatchServing(p,reducer,smem,reducer_task);
#if TILEMEGA_SERVING_DECODER_ATTENTION && TILEMEGA_SERVING_SEQ==1
    else if(stage.kind==TaskKind::kFusedAttention && reducer.kind==TaskKind::kAttentionMerge)
      RunServingMergeTask(p,reducer,reducer_task);
#endif
    else asm volatile("trap;");
  };
  if(stage.kind==TaskKind::kGemm && reducer.kind==TaskKind::kGemmCombine) {
    auto const& inv=static_cast<GemmInvocation const*>(p.gemms)[stage.gemm];
    reducer_task=DecodeSplitTask(task,inv.tiles_m*inv.tiles_n,inv.chunks).tile;
    producers=inv.chunks;
  }
#if TILEMEGA_SERVING_DECODER_ATTENTION && TILEMEGA_SERVING_SEQ==1
  else if(stage.kind==TaskKind::kFusedAttention && reducer.kind==TaskKind::kAttentionMerge) {
    producers=CeilDiv(p.dims.capacity,stage.attention_kv_block);
    auto point=DecodeServingAttentionTaskGMajor(task,p.dims.batch,producers);
    reducer_task=point.group*p.dims.batch+point.batch;
  }
#endif
  else {asm volatile("trap;");return;}
  __shared__ unsigned last;
  // The acq_rel ticket follows every writer's fence and convergence. Only
  // the final arrival reuses the task workspace and reads all partials.
  bool completed=executor::EpochLastArriver::Run(Ticket(p,stage_index,reducer_task),
      producers,iteration,&last,reduce);
  if(completed) {
#if TILEMEGA_DM_REDUCTIONS
    CompleteDmTask<L2>(p,reducer_index,reducer_task,smem,events,iteration);
#endif
    if constexpr(L2)NotifyTask(p,events,reducer_index,reducer_task,iteration);
    TraceReducer(p,reducer_index,reducer_task,stage_index,task);
  }
}
__device__ inline void RunStage(Params const& p,unsigned stage,TaskSmem& smem,
                                EventCounter* events,unsigned long long iteration) {
  for(int task=blockIdx.x;task<ActiveBlocks(p,p.stages[stage]);task+=gridDim.x)
    RunTask<false>(p,stage,task,smem,events,iteration);
}
} // namespace nonpaged
