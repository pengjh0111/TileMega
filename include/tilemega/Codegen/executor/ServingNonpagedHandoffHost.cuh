// SPDX-License-Identifier: BSD-3-Clause
// Included inside tilemega::codegen::harness after DeviceModel.
inline Params EpochL2Params(Params const& input) {
  Params result=input;
  if(result.serving_epoch_handoff_tickets)
    result.serving_epoch_handoff_tickets+=
        std::size_t(result.stage_count)*result.serving_epoch_handoff_stride;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  if(result.dm_reductions.tickets)result.dm_reductions.tickets+=result.dm_reductions.ticket_count;
  if(result.counted_dependencies)
    result.counted_dependencies+=result.counted_dependency_count;
#endif
  return result;
}
inline void UploadEpochL2Params(DeviceModel& model) {
  if(!model.params.serving_epoch_handoff_tickets)return;
  if(!model.device_epoch_l2_params)
    TILEMEGA_CUDA_CHECK(cudaMalloc(&model.device_epoch_l2_params,sizeof(Params)));
  Params parameters=EpochL2Params(model.params);
  TILEMEGA_CUDA_CHECK(cudaMemcpy(model.device_epoch_l2_params,&parameters,
                               sizeof(Params),cudaMemcpyHostToDevice));
}
inline void PrepareEpochHandoffs(DeviceModel& model,
                                 std::vector<GemmInvocation> const& gemms) {
  std::vector<unsigned> owners(model.stages.size(),0);
  unsigned stride=0;
#if TILEMEGA_DM_REDUCTIONS
  for(unsigned source=0;source<model.stages.size();++source) {
    auto target=model.stages[source].dm_reduce_stage;
    if(target!=kNoOperand) {
      if(target<=source || target>=model.stages.size() || ++owners[target]>1 ||
          !model.stages[target].handoff_elided)
        throw std::invalid_argument("DM last-arriver owner is invalid");
    }
  }
#endif
  for(unsigned source=0;source<model.stages.size();++source) {
    auto const& producer=model.stages[source];
    auto target=producer.handoff_reduce_stage;
    if(target==kNoOperand)continue;
    if(target<=source || target>=model.stages.size() || ++owners[target]>1)
      throw std::invalid_argument("nonpaged reducer requires one earlier producer owner");
    auto const& reducer=model.stages[target];
    if(!reducer.handoff_elided)
      throw std::invalid_argument("nonpaged reducer must be elided");
    unsigned tasks=0;
    if(producer.kind==TaskKind::kGemm && reducer.kind==TaskKind::kGemmCombine) {
      if(producer.gemm>=gemms.size() || reducer.gemm!=producer.gemm ||
         !(model.params.ownership_flags&kCombinerTileOwnership))
        throw std::invalid_argument("nonpaged split-K requires matching tile ownership");
      auto const& inv=gemms[producer.gemm];
      if(inv.chunks<=1 || inv.tiles_m<=0 || inv.tiles_n<=0 ||
         std::uint64_t(inv.tiles_m)*inv.tiles_n>std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("nonpaged split-K tile count is invalid");
      tasks=inv.tiles_m*inv.tiles_n;
    }else if(producer.kind==TaskKind::kFusedAttention &&
             reducer.kind==TaskKind::kAttentionMerge) {
      if(model.params.dims.seq!=1 || producer.attention_kv_block<=0 ||
         producer.extent!=reducer.extent || producer.width!=reducer.width ||
         producer.attention_kv_block!=reducer.attention_kv_block ||
         producer.operand[8]!=reducer.operand[0] ||
         producer.operand[9]!=reducer.operand[1] ||
         producer.operand[7]!=reducer.operand[2] ||
         std::uint64_t(model.params.dims.batch)*producer.extent>
             std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("nonpaged attention reduction has incompatible partials");
      tasks=model.params.dims.batch*producer.extent;
    }else throw std::invalid_argument("unsupported nonpaged last-arriver reduction");
    stride=std::max(stride,tasks);
  }
  for(unsigned stage=0;stage<model.stages.size();++stage)
    if(model.stages[stage].handoff_elided && owners[stage]!=1)
      throw std::invalid_argument("nonpaged elided stage has no reduction owner");
  if(!stride)return;
  auto rows=std::uint64_t(model.stages.size())*stride;
  if(rows>std::numeric_limits<std::size_t>::max()/(2*sizeof(unsigned long long)))
    throw std::overflow_error("nonpaged ticket bank allocation overflow");
  // Iteration numbers advance independently in the two execution modes.
  // Their tickets and partially completed arrival epochs cannot share a bank.
  auto bytes=std::size_t(rows)*2*sizeof(unsigned long long);
  TILEMEGA_CUDA_CHECK(cudaMalloc(&model.params.serving_epoch_handoff_tickets,bytes));
  TILEMEGA_CUDA_CHECK(cudaMemset(model.params.serving_epoch_handoff_tickets,0,bytes));
  model.params.serving_epoch_handoff_stride=stride;
}
