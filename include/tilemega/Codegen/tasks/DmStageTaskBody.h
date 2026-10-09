// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/LayerNormTaskBody.h>
#include <tilemega/Codegen/tasks/EmbeddingSumTaskBody.h>
#include <tilemega/Codegen/tasks/LayoutConvertTaskBody.h>

namespace tilemega::codegen {
#ifndef TILEMEGA_DM_STAGE_DISPATCH
template<class Runner>
__device__ inline void DispatchDmStage(std::uint32_t,std::uint32_t,std::uint32_t,Runner const&) {
  asm volatile("trap;");
}
#endif

__host__ __device__ inline int DmStageRows(StageDesc const& stage,ModelDims const& dims) {
  return stage.rows_per_batch ? int(stage.rows_per_batch)*dims.batch : dims.tokens();
}
__host__ __device__ inline int DmStageTaskCount(StageDesc const& stage,ModelDims const& dims) {
  int rows=DmStageRows(stage,dims);
  return stage.group ? (rows+int(stage.group)-1)/int(stage.group) : 0;
}

// LN: x, gamma, beta, y, optional statistics. Embedding sum: ids, type ids,
// word/type/position tables, y, optional statistics. Table shapes give the vocabulary/type/position counts.
// Layout conversion: external NCHW x, internal NHWC y.
template<class Arch>
struct DmStageRunner {
  Params const& params;
  StageDesc const& stage;
  unsigned task;
  char* scratch;
  template<TaskKind Kind,int Width,int RowsPerTask>
  __device__ void Run() const {
    using E=cutlass::bfloat16_t;
    auto pointer=[&](int i)->void* {
      auto id=stage.operand[i];
      return id==kNoOperand?nullptr:params.dm_buffers.data[id];
    };
    if constexpr(Kind==TaskKind::kLayerNorm) {
      LayerNormOperands inputs{static_cast<E const*>(pointer(0)),
          static_cast<E const*>(pointer(1)),static_cast<E const*>(pointer(2)),
          static_cast<E*>(pointer(3)),
          params.dm_buffers.layouts[stage.operand[0]],
          params.dm_buffers.layouts[stage.operand[3]],
          std::uint64_t(DmStageRows(stage,params.dims)),stage.norm_epsilon,
          static_cast<float*>(pointer(4))};
      LayerNormTaskBody<Arch,Width,RowsPerTask>::Run(inputs,task);
    }else if constexpr(Kind==TaskKind::kEmbeddingSum) {
      EmbeddingSumOperands inputs{static_cast<std::int64_t const*>(pointer(0)),
          static_cast<std::int64_t const*>(pointer(1)),
          static_cast<E const*>(pointer(2)),static_cast<E const*>(pointer(3)),
          static_cast<E const*>(pointer(4)),static_cast<E*>(pointer(5)),
          static_cast<float*>(pointer(6)),
          std::uint64_t(DmStageRows(stage,params.dims)),unsigned(params.dims.seq),
          stage.extent,params.dm_buffers.layouts[stage.operand[3]].logical[0],
          params.dm_buffers.layouts[stage.operand[4]].logical[0]};
      for(unsigned i=0;i<RowsPerTask && task*RowsPerTask+i<inputs.rows;++i)
        EmbeddingSumTaskBody<Arch,Width>::Run(inputs,task*RowsPerTask+i,
                                            reinterpret_cast<float*>(scratch));
    }else if constexpr(Kind==TaskKind::kLayoutConvert) {
      LayoutConvertTaskBody<Arch,RowsPerTask>::Run({static_cast<E const*>(pointer(0)),
          static_cast<E*>(pointer(1)),params.dm_buffers.layouts[stage.operand[1]]},task);
    }else {
      static_assert(Kind==TaskKind::kLayerNorm || Kind==TaskKind::kEmbeddingSum ||
                    Kind==TaskKind::kLayoutConvert,"unsupported DM scalar stage");
    }
  }
};
} // namespace tilemega::codegen
