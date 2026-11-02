// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
struct StreamingAttentionBuffers {
  AscendC::LocalTensor<bfloat16_t> raw_k,raw_v,raw_q;
  AscendC::LocalTensor<float> k,v,state,work,scratch;
};

// One independently normalized query. Q/K/V and output are BF16, statistics
// and o are FP32; P is rounded to BF16 before PV. This is a distinct precision
// contract from FP32 replay. No score/P/KV expansion or partial o goes to GM.
__aicore__ inline bool RunStreamingAttention(StreamingAttentionBuffers const& b,
    GM_ADDR query,GM_ADDR key,GM_ADDR value,GM_ADDR output,unsigned keys,unsigned d) {
  if(!keys || keys>1088 || (d!=64 && d!=128))return false;
  AscendC::GlobalTensor<bfloat16_t> q,k,v,out;
  q.SetGlobalBuffer((__gm__ bfloat16_t*)query);k.SetGlobalBuffer((__gm__ bfloat16_t*)key);
  v.SetGlobalBuffer((__gm__ bfloat16_t*)value);out.SetGlobalBuffer((__gm__ bfloat16_t*)output);
  AscendC::DataCopy(b.raw_q,q,d);CompleteDma<AscendC::HardEvent::MTE2_V>();
  AscendC::Cast(b.state,b.raw_q,AscendC::RoundMode::CAST_NONE,d);
  auto scores=b.state[256],stat=b.state[512],alpha=b.state[768],acc=b.state[1024];
  AscendC::Duplicate(acc,0.0f,d);AscendC::PipeBarrier<PIPE_V>();
  float maximum=0.0f,denominator=0.0f;
  for(unsigned begin=0;begin<keys;begin+=32) {
    const unsigned count=keys-begin<32?keys-begin:32;
    CompleteDma<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(b.raw_k,k[uint64_t(begin)*d],count*d);
    AscendC::DataCopy(b.raw_v,v[uint64_t(begin)*d],count*d);
    CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(b.k,b.raw_k,AscendC::RoundMode::CAST_NONE,count*d);
    AscendC::Cast(b.v,b.raw_v,AscendC::RoundMode::CAST_NONE,count*d);AscendC::PipeBarrier<PIPE_V>();
    uint32_t shape[2]={count,d},qshape[2]={1,d},pshape[2]={count,1};
    auto tmp=b.scratch.ReinterpretCast<uint8_t>();
    AscendC::Broadcast<float,2,0>(b.work,b.state,shape,qshape,tmp);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(b.work,b.work,b.k,count*d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float,AscendC::Pattern::Reduce::AR>(scores,b.work,tmp,shape,true);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(scores,scores,d==128?0.08838834764831845f:0.125f,count);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(stat,scores,b.scratch,count);CompleteDma<AscendC::HardEvent::V_S>();
    const float current=stat.GetValue(0);
    if(!(current<=3.4028234663852886e38f && current>=-3.4028234663852886e38f))return false;
    const float next=begin && maximum>current?maximum:current;
    alpha.SetValue(0,begin?maximum-next:0.0f);CompleteDma<AscendC::HardEvent::S_V>();
    AscendC::Exp(alpha,alpha,1);CompleteDma<AscendC::HardEvent::V_S>();
    const float factor=alpha.GetValue(0);
    AscendC::Adds(scores,scores,-next,count);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(scores,scores,count);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(stat,scores,b.scratch,count);CompleteDma<AscendC::HardEvent::V_S>();
    denominator=factor*denominator+stat.GetValue(0);maximum=next;
    AscendC::Cast(b.raw_q,scores,AscendC::RoundMode::CAST_RINT,count);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(scores,b.raw_q,AscendC::RoundMode::CAST_NONE,count);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Broadcast<float,2,1>(b.work,scores,shape,pshape,tmp);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(b.work,b.work,b.v,count*d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float,AscendC::Pattern::Reduce::RA>(b.k,b.work,tmp,shape,true);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(acc,acc,factor,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(acc,acc,b.k,d);AscendC::PipeBarrier<PIPE_V>();
  }
  if(!(denominator>0.0f && denominator<=1088.0f))return false;
  auto mask=b.raw_k.ReinterpretCast<uint8_t>();
  AscendC::Abs(b.work,acc,d);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Compares(mask,b.work,3.4028234663852886e38f,AscendC::CMPMODE::LE,d);
  AscendC::Duplicate(b.k,1.0f,d);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Select(b.v,mask,b.k,0.0f,AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,d);
  AscendC::PipeBarrier<PIPE_V>();AscendC::ReduceMin(stat,b.v,b.scratch,d);
  CompleteDma<AscendC::HardEvent::V_S>();if(stat.GetValue(0)!=1.0f)return false;
  AscendC::Muls(acc,acc,1.0f/denominator,d);AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(b.raw_q,acc,AscendC::RoundMode::CAST_RINT,d);CompleteDma<AscendC::HardEvent::V_MTE3>();
  AscendC::DataCopy(out,b.raw_q,d);CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
}
} // namespace tilemega::ascend
