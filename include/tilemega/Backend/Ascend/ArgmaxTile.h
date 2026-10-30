// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/ArgmaxContract.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
struct ArgmaxTileBuffers {
  AscendC::LocalTensor<bfloat16_t> raw;
  AscendC::LocalTensor<float> values,indices,result,work,scratch;
  AscendC::LocalTensor<uint8_t> mask;
};

// All supported token IDs are exactly representable in FP32. The sentinel is
// strictly greater than every real ID; masked padding never enters a reduction.
__aicore__ inline bool ArgmaxPair(ArgmaxTileBuffers const& b,unsigned n,
                                float& maximum,uint32_t& chosen) {
  AscendC::Abs(b.work,b.values,n);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Compares(b.mask,b.work,3.4028234663852886e38f,AscendC::CMPMODE::LE,n);
  AscendC::Duplicate(b.result,1.0f,n);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Select(b.work,b.mask,b.result,0.0f,AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
  AscendC::PipeBarrier<PIPE_V>();AscendC::ReduceMin(b.result,b.work,b.scratch,n);
  CompleteDma<AscendC::HardEvent::V_S>();
  if(b.result.GetValue(0)!=1.0f)return false;
  AscendC::ReduceMax(b.result,b.values,b.scratch,n);CompleteDma<AscendC::HardEvent::V_S>();
  maximum=b.result.GetValue(0);
  AscendC::Compares(b.mask,b.values,maximum,AscendC::CMPMODE::EQ,n);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Select(b.work,b.mask,b.indices,float(argmax::kFloatIndexSentinel),
      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
  AscendC::PipeBarrier<PIPE_V>();AscendC::ReduceMin(b.result,b.work,b.scratch,n);
  CompleteDma<AscendC::HardEvent::V_S>();
  const float id=b.result.GetValue(0);
  if(!(id>=0 && id<float(argmax::kFloatIndexSentinel)))return false;
  chosen=static_cast<uint32_t>(id);return true;
}

__aicore__ inline bool RunArgmaxRows(ArgmaxTileBuffers const& b,GM_ADDR input,
    GM_ADDR values,GM_ADDR indices,unsigned rows,unsigned columns,unsigned input_stride,
    unsigned output_stride,unsigned first,unsigned vocabulary) {
  if(!argmax::Tile(rows,columns,input_stride,output_stride,first,vocabulary))return false;
  const unsigned padded=(columns+15)/16*16;
  AscendC::GlobalTensor<bfloat16_t> src;src.SetGlobalBuffer((__gm__ bfloat16_t*)input);
  AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows),columns*2,(input_stride-columns)*2,0,0};
  AscendC::DataCopyPadExtParams<bfloat16_t> pad(true,0,padded-columns,bfloat16_t(0));
  AscendC::DataCopyPad(b.raw,src,copy,pad);CompleteDma<AscendC::HardEvent::MTE2_V>();
  AscendC::Cast(b.values,b.raw,AscendC::RoundMode::CAST_NONE,rows*padded);AscendC::PipeBarrier<PIPE_V>();
  AscendC::CreateVecIndex(b.indices,float(first),columns);AscendC::PipeBarrier<PIPE_V>();
  for(unsigned row=0;row<rows;++row) {
    auto local=b;local.values=b.values[row*padded];float maximum=0;uint32_t chosen=0;
    if(!ArgmaxPair(local,columns,maximum,chosen))return false;
    union {float f;uint32_t u;} bits;bits.f=maximum;
    WriteControl((__gm__ uint32_t*)values+uint64_t(row)*output_stride,bits.u);
    WriteControl((__gm__ uint32_t*)indices+uint64_t(row)*output_stride,chosen);
  }
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();return true;
}

__aicore__ inline bool RunArgmaxMerge(ArgmaxTileBuffers const& b,GM_ADDR values,
    GM_ADDR indices,unsigned n,unsigned vocabulary,uint32_t& chosen) {
  if(!argmax::Merge(n,vocabulary))return false;
  AscendC::GlobalTensor<float> v;v.SetGlobalBuffer((__gm__ float*)values);
  AscendC::GlobalTensor<uint32_t> ids;ids.SetGlobalBuffer((__gm__ uint32_t*)indices);
  AscendC::DataCopyExtParams copy{1,n*4,0,0,0};
  AscendC::DataCopyPadExtParams<float> pad(false,0,0,0);
  AscendC::DataCopyPad(b.values,v,copy,pad);
  auto raw=b.raw.ReinterpretCast<uint32_t>();
  AscendC::DataCopyPadExtParams<uint32_t> id_pad(false,0,0,0);
  AscendC::DataCopyPad(raw,ids,copy,id_pad);CompleteDma<AscendC::HardEvent::MTE2_V>();
  AscendC::Cast(b.indices,raw,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Compares(b.mask,b.indices,float(vocabulary),AscendC::CMPMODE::LT,n);
  AscendC::Duplicate(b.result,1.0f,n);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Select(b.work,b.mask,b.result,0.0f,AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
  AscendC::PipeBarrier<PIPE_V>();AscendC::ReduceMin(b.result,b.work,b.scratch,n);
  CompleteDma<AscendC::HardEvent::V_S>();
  if(b.result.GetValue(0)!=1.0f)return false;
  float maximum=0;return ArgmaxPair(b,n,maximum,chosen);
}
} // namespace tilemega::ascend
