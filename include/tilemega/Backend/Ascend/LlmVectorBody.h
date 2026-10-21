// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <adv_api/pad/broadcast.h>
#include <adv_api/reduce/reduce.h>

namespace tilemega::ascend {
class LlmVectorBody {
  static constexpr unsigned kElements=4096;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
  AscendC::LocalTensor<bfloat16_t> bx_,by_,bo_;
  AscendC::LocalTensor<float> x_,y_,z_,work_,scratch_;
  __aicore__ inline void Load(AscendC::LocalTensor<bfloat16_t> dst,GM_ADDR pointer,unsigned n) {
    AscendC::GlobalTensor<bfloat16_t> src;src.SetGlobalBuffer((__gm__ bfloat16_t*)pointer);
    AscendC::DataCopyExtParams copy{1,n*2,0,0,0};
    AscendC::DataCopyPadExtParams<bfloat16_t> pad(false,0,0,0);
    AscendC::DataCopyPad(dst,src,copy,pad);
  }
  __aicore__ inline void Store(GM_ADDR pointer,unsigned n) {
    CompleteDma<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<bfloat16_t> dst;dst.SetGlobalBuffer((__gm__ bfloat16_t*)pointer);
    AscendC::DataCopyExtParams copy{1,n*2,0,0,0};AscendC::DataCopyPad(dst,bo_,copy);
    CompleteDma<AscendC::HardEvent::MTE3_S>();
  }
 public:
  static constexpr unsigned kLocalBytes=kElements*(3*2+5*4);
  __aicore__ inline LlmVectorBody() {
    pipe_.InitBuffer(storage_,kLocalBytes);auto raw=storage_.Get<uint8_t>();
    bx_=raw[0].ReinterpretCast<bfloat16_t>();by_=raw[kElements*2].ReinterpretCast<bfloat16_t>();
    bo_=raw[kElements*4].ReinterpretCast<bfloat16_t>();
    x_=raw[kElements*6].ReinterpretCast<float>();y_=raw[kElements*10].ReinterpretCast<float>();
    z_=raw[kElements*14].ReinterpretCast<float>();work_=raw[kElements*18].ReinterpretCast<float>();
    scratch_=raw[kElements*22].ReinterpretCast<float>();
  }
  // n is the entire normalization domain; splitting an H3072 row is illegal.
  __aicore__ inline bool RmsNorm(GM_ADDR input,GM_ADDR gamma,GM_ADDR output,unsigned n,float epsilon) {
    if(!n || n>kElements || epsilon<=0)return false;
    Load(bx_,input,n);Load(by_,gamma,n);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,n);
    AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(work_,x_,x_,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(z_,work_,scratch_,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(z_,z_,1.0f/n,1);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(z_,z_,epsilon,1);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Rsqrt(z_,z_,1);CompleteDma<AscendC::HardEvent::V_S>();
    AscendC::Muls(work_,x_,z_.GetValue(0),n);AscendC::PipeBarrier<PIPE_V>();
    // The pinned HF graph casts normalized values before the BF16 scale multiply.
    AscendC::Cast(bo_,work_,AscendC::RoundMode::CAST_RINT,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(work_,bo_,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(work_,work_,y_,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(bo_,work_,AscendC::RoundMode::CAST_RINT,n);Store(output,n);return true;
  }
  __aicore__ inline bool Pointwise(GM_ADDR input,GM_ADDR rhs,GM_ADDR output,unsigned n,bool swiglu) {
    if(!n || n>kElements)return false;
    Load(bx_,input,n);Load(by_,rhs,n);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,n);
    AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
    if(swiglu) {
      AscendC::Muls(work_,x_,-1.0f,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(work_,work_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(work_,work_,1.0f,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(work_,x_,work_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(bo_,work_,AscendC::RoundMode::CAST_RINT,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(work_,bo_,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(work_,work_,y_,n);
    }else AscendC::Add(work_,x_,y_,n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(bo_,work_,AscendC::RoundMode::CAST_RINT,n);Store(output,n);return true;
  }
  __aicore__ inline bool Rope(GM_ADDR input,GM_ADDR cos,GM_ADDR sin,GM_ADDR output,unsigned d) {
    if(d!=64 && d!=128)return false;
    Load(bx_,input,d);Load(by_,cos,d);Load(bo_,sin,d);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,d);
    AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,d);
    AscendC::Cast(z_,bo_,AscendC::RoundMode::CAST_NONE,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(work_,x_,y_,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(bx_,work_,AscendC::RoundMode::CAST_RINT,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(work_,bx_,AscendC::RoundMode::CAST_NONE,d);
    AscendC::DataCopy(y_,x_[d/2],d/2);AscendC::DataCopy(y_[d/2],x_,d/2);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(y_,y_,-1.0f,d/2);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(y_,y_,z_,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(by_,y_,AscendC::RoundMode::CAST_RINT,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(work_,work_,y_,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(bo_,work_,AscendC::RoundMode::CAST_RINT,d);Store(output,d);return true;
  }
  // A task owns one query head and one contiguous K/V chunk, at most 32 keys.
  // The output is unnormalized (m,l,o), suitable for a separate CTG merge task.
  __aicore__ inline bool AttentionChunk(GM_ADDR query,GM_ADDR key,GM_ADDR value,
      GM_ADDR partial,GM_ADDR statistics,unsigned keys,unsigned d,float scale) {
    if(!keys || keys>32 || (d!=64 && d!=128))return false;
    Load(bx_,key,keys*d);Load(by_,value,keys*d);Load(bo_,query,d);
    CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,keys*d);
    AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,keys*d);
    AscendC::Cast(z_,bo_,AscendC::RoundMode::CAST_NONE,d);AscendC::PipeBarrier<PIPE_V>();
    uint32_t shape[2]={keys,d},query_shape[2]={1,d};
    auto tmp=scratch_.ReinterpretCast<uint8_t>();
    AscendC::Broadcast<float,2,0>(work_,z_,shape,query_shape,tmp);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(work_,work_,x_,keys*d);AscendC::PipeBarrier<PIPE_V>();
    auto scores=z_[256],stat=z_[512];
    AscendC::ReduceSum<float,AscendC::Pattern::Reduce::AR>(scores,work_,tmp,shape,true);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(scores,scores,scale,keys);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(stat,scores,scratch_,keys);CompleteDma<AscendC::HardEvent::V_S>();
    float maximum=stat.GetValue(0);
    AscendC::Adds(scores,scores,-maximum,keys);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(scores,scores,keys);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(stat,scores,scratch_,keys);CompleteDma<AscendC::HardEvent::V_S>();
    float sum=stat.GetValue(0);if(!(sum>0))return false;
    uint32_t probability_shape[2]={keys,1};
    AscendC::Broadcast<float,2,1>(work_,scores,shape,probability_shape,tmp);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(work_,work_,y_,keys*d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float,AscendC::Pattern::Reduce::RA>(x_,work_,tmp,shape,true);
    CompleteDma<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<float> out;out.SetGlobalBuffer((__gm__ float*)partial);
    AscendC::DataCopyExtParams copy{1,d*4,0,0,0};AscendC::DataCopyPad(out,x_,copy);
    CompleteDma<AscendC::HardEvent::MTE3_S>();
    z_.SetValue(0,maximum);z_.SetValue(1,sum);CompleteDma<AscendC::HardEvent::S_MTE3>();
    out.SetGlobalBuffer((__gm__ float*)statistics);
    copy.blockLen=8;AscendC::DataCopyPad(out,z_,copy);CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
  }
  __aicore__ inline bool AttentionMerge(GM_ADDR partial,GM_ADDR statistics,GM_ADDR output,unsigned chunks,unsigned d) {
    if(!chunks || chunks>34 || (d!=64 && d!=128))return false;
    AscendC::GlobalTensor<float> stats,parts;
    stats.SetGlobalBuffer((__gm__ float*)statistics);parts.SetGlobalBuffer((__gm__ float*)partial);
    float maximum=-3.4e38f;
    for(unsigned c=0;c<chunks;++c)if(stats.GetValue(2*c)>maximum)maximum=stats.GetValue(2*c);
    AscendC::Duplicate(x_,0.0f,d);AscendC::PipeBarrier<PIPE_V>();float denominator=0;
    for(unsigned c=0;c<chunks;++c) {
      z_.SetValue(0,stats.GetValue(2*c)-maximum);CompleteDma<AscendC::HardEvent::S_V>();
      AscendC::Exp(z_,z_,1);CompleteDma<AscendC::HardEvent::V_S>();
      float factor=z_.GetValue(0);denominator+=factor*stats.GetValue(2*c+1);
      AscendC::DataCopy(y_,parts[c*d],d);CompleteDma<AscendC::HardEvent::MTE2_V>();
      AscendC::Muls(work_,y_,factor,d);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(x_,x_,work_,d);AscendC::PipeBarrier<PIPE_V>();
    }
    if(!(denominator>0))return false;
    AscendC::Muls(x_,x_,1.0f/denominator,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(bo_,x_,AscendC::RoundMode::CAST_RINT,d);Store(output,d);return true;
  }
};
} // namespace tilemega::ascend
