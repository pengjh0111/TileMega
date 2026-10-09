// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <tilemega/Codegen/tasks/LayerNormTaskBody.h>
#include <tilemega/Codegen/tasks/EmbeddingSumTaskBody.h>
#include <tilemega/Codegen/tasks/LayoutConvertTaskBody.h>
#include <tilemega/Codegen/tasks/PoolTaskBody.h>
#include <tilemega/Codegen/tasks/GlobalPoolReduceTaskBody.h>
#include <tilemega/Codegen/tasks/EncoderAttentionTaskBody.h>
#include <tilemega/Codegen/tasks/DepthwiseConvTaskBody.h>

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
  if(stage.kind==TaskKind::kGlobalPoolReduce)
    return stage.width?dims.batch*((stage.extent+stage.width-1)/stage.width):0;
  if(stage.kind==TaskKind::kEncoderAttention)
    return stage.group?dims.batch*stage.extent*((stage.width+stage.group-1)/stage.group):0;
  if(stage.kind==TaskKind::kDepthwiseConv)
    return stage.group && stage.width && stage.spatial_width?
        dims.batch*((stage.rows_per_batch/stage.spatial_width+stage.group-1)/stage.group)*
            ((stage.extent+stage.width-1)/stage.width):0;
  int rows=DmStageRows(stage,dims);
  int count=stage.group ? (rows+int(stage.group)-1)/int(stage.group) : 0;
  if(stage.kind==TaskKind::kPool)
    count*=stage.width?(stage.extent+stage.width-1)/stage.width:0;
  return count;
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
  template<int RowBand,int ChannelTile,class Program>
  __device__ void RunDepthwise() const {
    using E=cutlass::bfloat16_t;
    auto const& buffers=params.dm_buffers;
    auto conv=params.dm_convolutions[stage.conv];conv.n=params.dims.batch;
    DepthwiseConvOperands inputs{static_cast<E const*>(buffers.data[stage.operand[0]]),
        static_cast<E const*>(buffers.data[stage.operand[1]]),
        static_cast<E*>(buffers.data[stage.operand[2]]),conv,
        buffers.layouts[stage.operand[0]],buffers.layouts[stage.operand[2]],8,buffers,stage.chain};
    if(stage.operand[3]!=kNoOperand) {
      inputs.channel_partials=static_cast<float*>(buffers.data[stage.operand[3]]);
      inputs.partial_layout=buffers.layouts[stage.operand[3]];
    }
    DepthwiseConvTaskBody<Arch,RowBand,ChannelTile,Program>::Run(inputs,task,scratch);
  }
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
    }else if constexpr(Kind==TaskKind::kPool) {
      auto window=params.dm_convolutions[stage.conv];window.n=params.dims.batch;
      PoolTaskBody<Arch,RowsPerTask,Width>::Run({static_cast<E const*>(pointer(0)),
          static_cast<E*>(pointer(1)),window,params.dm_buffers.layouts[stage.operand[0]],
          params.dm_buffers.layouts[stage.operand[1]]},task);
    }else if constexpr(Kind==TaskKind::kEncoderAttention) {
      EncoderAttentionOperands inputs{static_cast<E const*>(pointer(0)),
          static_cast<E*>(pointer(1)),static_cast<std::int64_t const*>(pointer(2)),
          unsigned(params.dims.batch),stage.extent};
      using Masked=EncoderAttentionTaskBody<Arch,Width,RowsPerTask,true>;
      using Unmasked=EncoderAttentionTaskBody<Arch,Width,RowsPerTask,false>;
      auto& shared=*reinterpret_cast<typename Masked::SharedStorage*>(scratch);
      if(inputs.key_padding)Masked::Run(inputs,task,shared);
      else Unmasked::Run(inputs,task,*reinterpret_cast<typename Unmasked::SharedStorage*>(scratch));
    }else if constexpr(Kind==TaskKind::kGlobalPoolReduce) {
      auto const& output=params.dm_buffers.layouts[stage.operand[1]];
      GlobalPoolReduceTaskBody<Arch,Width>::Run({static_cast<float const*>(pointer(0)),
          static_cast<float*>(pointer(1)),params.dm_buffers.layouts[stage.operand[0]],
          unsigned(params.dims.batch),stage.extent,stage.rows_per_batch,RowsPerTask,
          unsigned(output.rank?output.strides[0]:stage.extent),stage.partial_rows_per_image},task);
    }else {
      static_assert(Kind==TaskKind::kLayerNorm || Kind==TaskKind::kEmbeddingSum ||
                    Kind==TaskKind::kLayoutConvert || Kind==TaskKind::kPool ||
                    Kind==TaskKind::kGlobalPoolReduce,"unsupported DM scalar stage");
    }
  }
};
} // namespace tilemega::codegen
