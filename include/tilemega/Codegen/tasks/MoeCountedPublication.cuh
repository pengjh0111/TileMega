// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/MoeBinding.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cuda/atomic>

namespace tilemega::codegen {
// Invocations describe physical TM padding, while the binding describes the
// live BM rows. Empty records and empty row subtiles contribute no arrivals.
template<class Invocation>
__device__ void PublishMoeCountedRows(Params const& p,unsigned producer,
    unsigned tile,Invocation const& inv) {
  auto const& access=inv.access;
  if(access.b!=DmBAccess::kExpertIndirect || access.write.kind!=DmWriteKind::kRowScatter)return;
  if(!inv.tile_m || !inv.tile_n || !inv.tiles_n || !access.block_rows)
    asm volatile("trap;");
  unsigned subtiles=(access.block_rows+inv.tile_m-1)/inv.tile_m;
  unsigned tile_m=tile/inv.tiles_n,tile_n=tile%inv.tiles_n;
  MoeBindingView bindings{static_cast<MoeBindingRecord const*>(inv.binding),
      static_cast<MoeBindingRow const*>(inv.rows),access.binding_blocks,
      access.binding_rows,access.experts,access.block_rows};
  for(unsigned edge=0;edge<p.dependency_count;++edge) {
    auto const& dep=p.dependencies[edge];
    if(dep.producer!=producer || dep.map!=StageDependency::Map::kCounted)continue;
    if(dep.consumer>=p.stage_count)asm volatile("trap;");
    auto const& combine=p.stages[dep.consumer];
    if(combine.kind!=TaskKind::kMoECombine || !combine.group || !combine.width ||
       combine.width!=unsigned(inv.tile_n) || combine.moe.top_k!=access.routing_topk)
      asm volatile("trap;");
    MoeBindingRecord block;
    auto status=bindings.Lookup(tile_m/subtiles,&block);
    if(status==MoeBindingStatus::kInvalid)asm volatile("trap;");
    if(status==MoeBindingStatus::kEmpty || (tile_m%subtiles)*inv.tile_m>=block.row_count)continue;
    unsigned begin=(tile_m%subtiles)*inv.tile_m;
    unsigned count=min(unsigned(inv.tile_m),block.row_count-begin);
    unsigned columns=(combine.extent+combine.width-1)/combine.width;
    if(tile_n>=columns)asm volatile("trap;");
    // All writer threads release their stores before any warp publishes the
    // weighted arrival; the consumer acquires the complete release sequence.
    __threadfence();executor::ComputeSync();
    for(unsigned wave=0;wave<unsigned(inv.tile_m);wave+=executor::kComputeThreads) {
      unsigned row=wave+executor::ComputeThread();
      bool active=row<count;unsigned target=0;
      if(active) {
        auto const& entry=bindings.rows[block.row_begin+begin+row];
        if(entry.token>=unsigned(p.dims.tokens()) || entry.rank>=access.routing_topk)
          asm volatile("trap;");
        target=(entry.token/combine.group)*columns+tile_n;
        if(target>=dep.table_rows || std::uint64_t(dep.counted_offset)+target>=p.counted_dependency_count ||
            !p.counted_dependencies)asm volatile("trap;");
      }
      unsigned lanes=__ballot_sync(0xffffffffu,active);
      if(active) {
        unsigned peers=__match_any_sync(lanes,target);
        if(unsigned(executor::ComputeThread()%32)==unsigned(__ffs(peers)-1)) {
          cuda::atomic_ref<unsigned long long,cuda::thread_scope_device> counter(
              p.counted_dependencies[dep.counted_offset+target]);
          counter.fetch_add(unsigned(__popc(peers)),cuda::memory_order_acq_rel);
        }
      }
    }
    executor::ComputeSync();
  }
}
} // namespace tilemega::codegen
