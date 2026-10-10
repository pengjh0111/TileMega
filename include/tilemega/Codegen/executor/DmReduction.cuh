// SPDX-License-Identifier: BSD-3-Clause
// Included inside tilemega::codegen after task dispatch and publication.
template<class Arch,bool L2,unsigned Depth=0,class Publish>
__device__ void RunDmReductions(Params const& p,unsigned producer,unsigned task,
    char* scratch,unsigned long long iteration,Publish const& publish) {
  auto const& view=p.dm_reductions;
  if(!view.stages)return;
  auto const& program=view.stages[producer];
  if(program.target==kNoOperand)return;
  if constexpr(Depth>=6) {asm volatile("trap;");return;}
  else {
    auto const& reducer=p.stages[program.target];
    __shared__ unsigned last;
    auto arrive=[&](unsigned target,unsigned expected,unsigned contribution) {
      if(std::uint64_t(program.ticket_offset)+target>=view.ticket_count || !view.tickets)
        asm volatile("trap;");
      bool completed=executor::EpochLastArriver::RunWeighted(
          view.tickets+program.ticket_offset+target,expected,iteration,contribution,&last,[&] {
            DispatchDmStage(unsigned(reducer.kind),reducer.width,reducer.group,
                DmStageRunner<Arch>{p,reducer,target,scratch});
          });
      if(completed) {
        RunDmReductions<Arch,L2,Depth+1>(p,program.target,target,scratch,iteration,publish);
        if constexpr(L2)publish(program.target,target);
      }
    };
    if(program.counted_edge==kNoOperand) {
      if(!view.offsets)asm volatile("trap;");
      auto row=program.task_offset+task;
      for(unsigned i=view.offsets[row];i<view.offsets[row+1];++i) {
        auto a=view.arrivals[i];arrive(a.task,a.expected,1);
      }
      return;
    }
    if(program.counted_edge>=p.dependency_count)asm volatile("trap;");
    auto const& dep=p.dependencies[program.counted_edge];
    auto const& source=p.stages[producer];
    auto const& inv=static_cast<GemmInvocation const*>(p.gemms)[source.gemm];
    auto const& access=inv.access;
    if(access.b!=DmBAccess::kExpertIndirect || access.write.kind!=DmWriteKind::kRowScatter ||
        !access.block_rows || !inv.tile_m || !inv.tiles_n || !reducer.group ||
        reducer.width!=unsigned(inv.tile_n))asm volatile("trap;");
    unsigned tile=IsGemmStage(source.kind)?
        DecodeSplitTask(task,inv.tiles_m*inv.tiles_n,inv.chunks).tile:task;
    unsigned subtiles=(access.block_rows+inv.tile_m-1)/inv.tile_m;
    unsigned m=tile/inv.tiles_n,n=tile%inv.tiles_n;
    MoeBindingView bindings{static_cast<MoeBindingRecord const*>(inv.binding),
        static_cast<MoeBindingRow const*>(inv.rows),access.binding_blocks,
        access.binding_rows,access.experts,access.block_rows};
    MoeBindingRecord block;auto status=bindings.Lookup(m/subtiles,&block);
    if(status==MoeBindingStatus::kInvalid)asm volatile("trap;");
    unsigned begin=(m%subtiles)*inv.tile_m;
    if(status==MoeBindingStatus::kEmpty || begin>=block.row_count)return;
    unsigned count=min(unsigned(inv.tile_m),block.row_count-begin);
    unsigned columns=(reducer.extent+reducer.width-1)/reducer.width;
    if(n>=columns)asm volatile("trap;");
    auto target=[&](unsigned row) {
      auto const& entry=bindings.rows[block.row_begin+begin+row];
      if(entry.token>=unsigned(p.dims.tokens()) || entry.rank>=access.routing_topk)
        asm volatile("trap;");
      return (entry.token/reducer.group)*columns+n;
    };
    // Every compute lane traverses the same distinct targets. The last
    // arrival must run a CTA-cooperative reducer, not a divergent warp body.
    for(unsigned row=0;row<count;++row) {
      unsigned t=target(row),contribution=0;bool first=true;
      for(unsigned r=0;r<count;++r)if(target(r)==t) {
        ++contribution;if(r<row)first=false;
      }
      if(!first)continue;
      unsigned expected=0;
      if(t>=dep.table_rows || !ReadCountedThreshold(p.counted_thresholds,
          dep.counted_threshold_offset,dep.table_rows,t,dep.count,&expected))asm volatile("trap;");
      arrive(t,expected,contribution);
    }
  }
}

template<bool L2>
__device__ inline void CompleteDmTask(Params const& p,unsigned producer,unsigned task,
    TaskSmem& smem,EventCounter* events,unsigned long long iteration) {
  RunDmReductions<HarnessArch,L2>(p,producer,task,reinterpret_cast<char*>(&smem),iteration,
      [&](unsigned reducer,unsigned target){NotifyTask(p,events,reducer,target,iteration);});
}
__device__ inline void RunDmStage(Params const& p,unsigned stage,TaskSmem& smem,
    EventCounter* events,unsigned long long iteration) {
  for(int task=blockIdx.x;task<ActiveBlocks(p,p.stages[stage]);task+=gridDim.x) {
    RunTask(p,stage,task,smem);
    CompleteDmTask<false>(p,stage,task,smem,events,iteration);
  }
}
