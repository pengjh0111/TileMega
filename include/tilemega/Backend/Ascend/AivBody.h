// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>
namespace tilemega::ascend {
class AivBody {
  static constexpr unsigned kElements=256;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
  AscendC::LocalTensor<bfloat16_t> bx_,by_,bo_;
  AscendC::LocalTensor<float> x_,y_,work_,reduce_,reduce_work_;
  __aicore__ inline void Load(AscendC::LocalTensor<bfloat16_t> local,GM_ADDR pointer,unsigned n) {
    AscendC::GlobalTensor<bfloat16_t> global;global.SetGlobalBuffer((__gm__ bfloat16_t*)pointer);
    AscendC::DataCopyExtParams copy{1,uint32_t(n*2),0,0,0};
    AscendC::DataCopyPadExtParams<bfloat16_t> pad(false,0,0,0);
    AscendC::DataCopyPad(local,global,copy,pad);
  }
 public:
  static constexpr unsigned kLocalBytes=3*kElements*2+5*kElements*4;
  __aicore__ inline AivBody() {
    pipe_.InitBuffer(storage_,kLocalBytes);auto raw=storage_.Get<uint8_t>();
    bx_=raw[0].ReinterpretCast<bfloat16_t>();by_=raw[512].ReinterpretCast<bfloat16_t>();
    bo_=raw[1024].ReinterpretCast<bfloat16_t>();
    x_=raw[1536].ReinterpretCast<float>();y_=raw[2560].ReinterpretCast<float>();
    work_=raw[3584].ReinterpretCast<float>();reduce_=raw[4608].ReinterpretCast<float>();
    reduce_work_=raw[5632].ReinterpretCast<float>();
  }
  __aicore__ inline bool Run(unsigned kind,GM_ADDR input,GM_ADDR rhs,GM_ADDR output,
                             unsigned n,float epsilon=0.0f) {
    if(n<1 || n>kElements || (kind!=1 && kind!=4 && kind!=10) || (kind==1 && epsilon<=0))return false;
    Load(bx_,input,n);if(kind!=4)Load(by_,rhs,n);
    CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,n);
    if(kind!=4)AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,n);
    AscendC::PipeBarrier<PIPE_V>();
    if(kind==10) {
      AscendC::Add(work_,x_,y_,n);
    } else if(kind==4) {
      AscendC::Muls(work_,x_,-1.0f,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(work_,work_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(work_,work_,1.0f,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(work_,x_,work_,n);
    } else {
      AscendC::Mul(work_,x_,x_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::ReduceSum(reduce_,work_,reduce_work_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(reduce_,reduce_,1.0f/float(n),1);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(reduce_,reduce_,epsilon,1);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Rsqrt(reduce_,reduce_,1);
      CompleteDma<AscendC::HardEvent::V_S>();
      float scale=reduce_.GetValue(0);
      AscendC::Muls(work_,x_,scale,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(work_,work_,y_,n);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(bo_,work_,AscendC::RoundMode::CAST_RINT,n);
    CompleteDma<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<bfloat16_t> global;global.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    AscendC::DataCopyExtParams copy{1,uint32_t(n*2),0,0,0};
    AscendC::DataCopyPad(global,bo_,copy);
    CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
  }
};
} // namespace tilemega::ascend
