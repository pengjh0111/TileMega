// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnStorage.h>
#include <limits>
#include <map>
#include <stdexcept>

namespace tilemega::frontend {
void MaterializeDnnStorage(ModelPlan& plan,std::vector<GemmGranularity> const& gemms,int batch) {
  if(!plan.dm || !plan.forward || plan.forward_token_axis)return;
  if(gemms.size()!=plan.gemms.size())
    throw std::invalid_argument("DNN side storage requires every GEMM geometry");
  std::map<unsigned,unsigned> pool_tiles;
  for(unsigned index=0;index<plan.gemms.size();++index) {
    auto const& g=plan.gemms[index];auto const& impl=gemms[index];
    for(unsigned i=0;i<g.chain.side_count;++i) {
      auto const& side=g.chain.side[i];
      if(side.kind!=codegen::DmSideOutputKind::kRowStats &&
         side.kind!=codegen::DmSideOutputKind::kChannelPartialSums)
        throw std::invalid_argument("DNN side storage has no materialization recipe");
      auto& buffer=plan.buffers.at(side.buffer);
      if(buffer.dtype!="f32" || impl.tile_m<=0 || impl.tile_n<=0 || batch<=0)
        throw std::invalid_argument("DNN side storage requires FP32 and bound positive geometry");
      unsigned rows=g.access.rows_per_batch?g.access.rows_per_batch:plan.serving_seq;
      bool channels=side.kind==codegen::DmSideOutputKind::kChannelPartialSums;
      auto parts=channels?(std::uint64_t(batch)*rows+impl.tile_m-1)/impl.tile_m:
          (std::uint64_t(g.n)+impl.tile_n-1)/impl.tile_n;
      std::uint64_t outer=channels?1:rows,inner=channels?g.n:2;
      auto elements=outer*parts*inner;
      if(!parts || elements>std::numeric_limits<std::uint32_t>::max() ||
         outer*batch>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("DNN side storage exceeds descriptor capacity");
      codegen::DmBufferLayout layout;layout.rank=3;
      layout.logical[0]=layout.physical[0]=outer*batch;
      layout.logical[1]=layout.physical[1]=parts;
      layout.logical[2]=layout.physical[2]=inner;
      layout.strides[2]=1;layout.strides[1]=inner;layout.strides[0]=parts*inner;
      buffer.layout=layout;buffer.constant=buffer.per_seq=buffer.per_past=buffer.per_total=0;
      buffer.per_batch=elements;
      if(channels && !pool_tiles.emplace(side.buffer,impl.tile_m).second)
        throw std::invalid_argument("DNN channel partial storage has multiple producers");
    }
  }
  for(auto& stage:plan.stages)if(stage.kind==PlanTaskKind::kGlobalPoolReduce) {
    auto found=pool_tiles.find(stage.operands[0]);
    if(found!=pool_tiles.end()) {
      stage.group=found->second;stage.partial_rows_per_image=0;
    }
  }
}
} // namespace tilemega::frontend
