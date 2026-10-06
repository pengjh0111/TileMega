// SPDX-License-Identifier: BSD-3-Clause
// Included in tilemega::codegen after task descriptors and event primitives.
namespace prefetch {
using NativeArch=std::conditional_t<std::is_void_v<arch::CurrentArch>,GemmVariantArch,arch::CurrentArch>;
using Issuer=executor::L2Prefetch<NativeArch,TILEMEGA_ARCH_PATH_SM80!=0>;
template<int Variant=0>
__device__ void Gemm(GemmInvocation const& inv,int task,unsigned& budget) {
  if(inv.variant==Variant) {
    using V=GemmVariant<Variant>;
    using Body=ServingGemmTaskBody<NativeArch,V::kTileM,V::kTileN,V::kTileK,V::kStages>;
    auto [m,n,k,batch]=inv.problem;(void)batch;
    ServingGemmOperands p;p.b=inv.mainloop.ptr_B;p.m=m;p.n=n;p.k_total=k;p.k_count=k;
    p.b_row_stride=inv.k_total;
#if TILEMEGA_NONPAGED_TILED
    p.weight_base=inv.serving_weight_base;p.k_total_full=inv.serving_k_total_full;
    p.k_begin=inv.serving_k_begin;
#endif
    Body::PrefetchRanges(p,task%inv.tiles_n,[&](executor::PrefetchRange range){
      Issuer::Issue(range,budget,TILEMEGA_L2_PREFETCH_STRIDE);
    });
  }else if constexpr(Variant+1<TILEMEGA_GEMM_VARIANT_COUNT)Gemm<Variant+1>(inv,task,budget);
}
__device__ inline void Task(Params const& p,unsigned stage_index,int task,unsigned& budget) {
  auto const& s=p.stages[stage_index];
  if(s.kind==TaskKind::kGemm) {
    auto* table=static_cast<GemmInvocation const*>(p.gemms);
    auto point=DecodeSplitTask(task,table[s.gemm].tiles_m*table[s.gemm].tiles_n,table[s.gemm].chunks);
    auto const& inv=table[s.gemm+point.chunk];
    // This byte is emitted from lifting's written frontier, not a buffer-name list.
    if(p.serving_no_producer[inv.serving_weight_buffer])Gemm(inv,point.tile,budget);
  }else if(s.kind==TaskKind::kFusedAttention && s.prefetch_history_mask) {
    using E=cutlass::bfloat16_t;
    int blocks=CeilDiv(p.dims.capacity,s.attention_kv_block);
    int qb=CeilDiv(int(s.group)*p.dims.seq,s.attention_query_rows);
    auto point=p.dims.seq==1
        ? DecodeServingAttentionTaskGMajor(task,p.dims.batch,blocks)
        : DecodeServingAttentionTask(task,qb,s.extent,blocks);
    ServingAttentionOperands op{};op.key_cache=reinterpret_cast<E*>(p.buffers[s.operand[1]]);
    op.value_cache=reinterpret_cast<E*>(p.buffers[s.operand[2]]);op.past=p.dims.past;
    op.heads_kv=s.extent;op.capacity=p.dims.capacity;op.block_extent=s.attention_kv_block;
    using Body=FusedAttentionTaskBody<NativeArch,TILEMEGA_SERVING_HEAD_DIM,
        TILEMEGA_SERVING_QPERKV,TILEMEGA_SERVING_SEQ,TILEMEGA_SERVING_QROWS,
        TILEMEGA_SERVING_KV_TILE,TILEMEGA_SERVING_QK_NORM!=0>;
    Body::PrefetchRanges(op,point.batch,point.group,point.cache_block,s.prefetch_history_mask,
        [&](executor::PrefetchRange range){Issuer::Issue(range,budget,TILEMEGA_L2_PREFETCH_STRIDE);});
  }
}
__device__ inline void Upcoming(Params const& p,unsigned slot,unsigned end) {
  unsigned budget=TILEMEGA_L2_PREFETCH_BYTES;
  for(unsigned next=slot+1;next<end && next<=slot+TILEMEGA_L2_PREFETCH_DEPTH && budget;++next)
    Task(p,p.schedule[next].stage,p.schedule[next].logical_task,budget);
}
__device__ inline void NextStage(Params const& p,unsigned stage) {
#if TILEMEGA_NONPAGED_LA
  while(stage<p.stage_count && p.stages[stage].handoff_elided)++stage;
#endif
  if(stage>=p.stage_count)return;
  unsigned budget=TILEMEGA_L2_PREFETCH_BYTES;
  int count=ActiveBlocks(p,p.stages[stage]);
  for(int ordinal=PlacedBlock();ordinal<ServingL1TaskLimit(p,p.stages[stage],count) && budget;ordinal+=gridDim.x) {
    int task=ServingL1Task(p,p.stages[stage],ordinal);
    if(task<count)Task(p,stage,task,budget);
  }
}
// Same monotonically counted grid event; only its wait is delayed.
__device__ inline void Arrive(EventCounter* events,unsigned stage,unsigned long long iteration,
                               Params const* params=nullptr) {
#if TILEMEGA_SYNC_V3
  (void)iteration;
  executor::ComputeSync();
  if(executor::ComputeThread()==0) {
    RedRelease(&events[stage].arrivals,1ull);
    executor::StageTasksEnd(params,stage,iteration);
  }
#else
  __threadfence();executor::ComputeSync();
  if(executor::ComputeThread()==0) {
    auto ticket=atomicAdd(&events[stage].arrivals,1ull);
    if(ticket+1==static_cast<unsigned long long>(gridDim.x)*(iteration+1)) {
      __threadfence();TILEMEGA_GENERATED_NOTIFY_global(&events[stage].epoch,iteration+1);
    }
  }
#endif
}
__device__ inline void Wait(EventCounter* events,unsigned stage,unsigned long long iteration,
                            Params const* params=nullptr) {
#if TILEMEGA_SYNC_V3
  if(executor::ComputeThread()==0) {
    Watch watch{params?params->serving_watchdog:nullptr,
        params?params->serving_watchdog_ns:0,12,stage,0,stage,0,stage,iteration};
    WaitAtLeast(&events[stage].arrivals,
      static_cast<unsigned long long>(gridDim.x)*(iteration+1),watch);
  }
  executor::ComputeSync();
#else
  if(executor::ComputeThread()==0)GradedWait(&events[stage].epoch,iteration+1);
  executor::ComputeSync();__threadfence();
#endif
}
} // namespace prefetch
