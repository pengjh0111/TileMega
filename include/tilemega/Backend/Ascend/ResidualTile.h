// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/BatchedResidualContract.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
struct ResidualTileBuffers {
  AscendC::LocalTensor<bfloat16_t> left,right,output;
  AscendC::LocalTensor<float> x,y,sum;
};

__aicore__ inline bool RunResidualTile(ResidualTileBuffers const& b,GM_ADDR left,
    GM_ADDR right,GM_ADDR output,unsigned rows,unsigned columns,unsigned left_stride,
    unsigned right_stride,unsigned output_stride) {
  if(!batched_residual::Tile(rows,columns,left_stride,right_stride,output_stride))return false;
  const unsigned padded=(columns+15)/16*16,count=rows*padded;
  AscendC::GlobalTensor<bfloat16_t> lhs,rhs,dst;
  lhs.SetGlobalBuffer((__gm__ bfloat16_t*)left);rhs.SetGlobalBuffer((__gm__ bfloat16_t*)right);
  dst.SetGlobalBuffer((__gm__ bfloat16_t*)output);
  AscendC::DataCopyPadExtParams<bfloat16_t> pad(true,0,padded-columns,bfloat16_t(0));
  AscendC::DataCopyExtParams load_left{static_cast<uint16_t>(rows),columns*2,(left_stride-columns)*2,0,0};
  AscendC::DataCopyExtParams load_right{static_cast<uint16_t>(rows),columns*2,(right_stride-columns)*2,0,0};
  AscendC::DataCopyPad(b.left,lhs,load_left,pad);AscendC::DataCopyPad(b.right,rhs,load_right,pad);
  CompleteDma<AscendC::HardEvent::MTE2_V>();
  // Both operands already have the graph's BF16 storage rounding. The only
  // new cast is the same FP32-add -> BF16 result as the row-wise control.
  AscendC::Cast(b.x,b.left,AscendC::RoundMode::CAST_NONE,count);
  AscendC::Cast(b.y,b.right,AscendC::RoundMode::CAST_NONE,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Add(b.sum,b.x,b.y,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(b.output,b.sum,AscendC::RoundMode::CAST_RINT,count);
  CompleteDma<AscendC::HardEvent::V_MTE3>();
  // Each padded UB row has a 32B stride; only the valid GM columns are stored.
  AscendC::DataCopyExtParams store{static_cast<uint16_t>(rows),columns*2,0,(output_stride-columns)*2,0};
  AscendC::DataCopyPad(dst,b.output,store);CompleteDma<AscendC::HardEvent::MTE3_S>();
  return true;
}
} // namespace tilemega::ascend
