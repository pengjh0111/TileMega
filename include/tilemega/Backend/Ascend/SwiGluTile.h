// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
struct SwiGluTileBuffers {
  AscendC::LocalTensor<bfloat16_t> raw,gate,output;
  AscendC::LocalTensor<float> x,y,work;
};

// Borrow disjoint UB arrays (2048 elements each). The input packs alternating
// gate16/up16 groups. Preserve the BF16 SiLU rounding before multiplication.
__aicore__ inline bool RunSwiGluTile(SwiGluTileBuffers const& b,GM_ADDR input,
    GM_ADDR output,unsigned rows,unsigned columns,unsigned input_stride,unsigned output_stride) {
  if(!rows || rows>16 || !columns || columns>128 || columns%32 ||
      input_stride<columns || output_stride<columns/2 ||
      input_stride>32768 || output_stride>16384 || input_stride%16 || output_stride%16)return false;
  AscendC::GlobalTensor<bfloat16_t> src,dst;
  src.SetGlobalBuffer((__gm__ bfloat16_t*)input);
  dst.SetGlobalBuffer((__gm__ bfloat16_t*)output);
  AscendC::DataCopyExtParams load{static_cast<uint16_t>(rows),columns*2,(input_stride-columns)*2,0,0};
  AscendC::DataCopyPadExtParams<bfloat16_t> pad(false,0,0,0);
  AscendC::DataCopyPad(b.raw,src,load,pad);
  CompleteDma<AscendC::HardEvent::MTE2_V>();
  AscendC::DataCopyParams select{static_cast<uint16_t>(rows*columns/32),1,1,0};
  AscendC::DataCopy(b.gate,b.raw,select);
  AscendC::DataCopy(b.output,b.raw[16],select);
  AscendC::PipeBarrier<PIPE_V>();
  const unsigned count=rows*columns/2;
  AscendC::Cast(b.x,b.gate,AscendC::RoundMode::CAST_NONE,count);
  AscendC::Cast(b.y,b.output,AscendC::RoundMode::CAST_NONE,count);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Muls(b.work,b.x,-1.0f,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Exp(b.work,b.work,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Adds(b.work,b.work,1.0f,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Div(b.work,b.x,b.work,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(b.output,b.work,AscendC::RoundMode::CAST_RINT,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(b.work,b.output,AscendC::RoundMode::CAST_NONE,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Mul(b.work,b.work,b.y,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(b.output,b.work,AscendC::RoundMode::CAST_RINT,count);
  CompleteDma<AscendC::HardEvent::V_MTE3>();
  AscendC::DataCopyExtParams store{static_cast<uint16_t>(rows),columns,0,(output_stride-columns/2)*2,0};
  AscendC::DataCopyPad(dst,b.output,store);
  CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
}
} // namespace tilemega::ascend
