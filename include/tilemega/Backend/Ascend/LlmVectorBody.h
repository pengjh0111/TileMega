// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <adv_api/pad/broadcast.h>
#include <adv_api/reduce/reduce.h>
#if defined(TILEMEGA_AS2_CANN_NORM_PRIMITIVES) && __CCE_AICORE__ == 310
#include <reduce_common_regbase.h>
#endif

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
  __aicore__ inline bool Copy(GM_ADDR input,GM_ADDR output,unsigned n) {
    if(!n || n>kElements)return false;
    Load(bo_,input,n);CompleteDma<AscendC::HardEvent::MTE2_MTE3>();Store(output,n);return true;
  }
  __aicore__ inline bool Bf16ToFloat(GM_ADDR input,GM_ADDR output,unsigned n,float scale=1.0f) {
    if(!n || n>kElements)return false;
    Load(bx_,input,n);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
    if(scale!=1.0f){AscendC::Muls(x_,x_,scale,n);AscendC::PipeBarrier<PIPE_V>();}
    CompleteDma<AscendC::HardEvent::V_MTE3>();AscendC::GlobalTensor<float> dst;
    dst.SetGlobalBuffer((__gm__ float*)output);AscendC::DataCopyExtParams copy{1,n*4,0,0,0};
    AscendC::DataCopyPad(dst,x_,copy);CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
  }
  __aicore__ inline bool FloatToBf16(GM_ADDR input,GM_ADDR output,unsigned n) {
    if(!n || n>kElements)return false;
    AscendC::GlobalTensor<float> src;src.SetGlobalBuffer((__gm__ float*)input);
    AscendC::DataCopyExtParams copy{1,n*4,0,0,0};AscendC::DataCopyPadExtParams<float> pad(false,0,0,0);
    AscendC::DataCopyPad(x_,src,copy,pad);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(bo_,x_,AscendC::RoundMode::CAST_RINT,n);Store(output,n);return true;
  }
  __aicore__ inline bool ZeroFloat(GM_ADDR output,unsigned n) {
    if(!n || n>kElements)return false;
    AscendC::Duplicate(x_,0.0f,n);CompleteDma<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<float> dst;dst.SetGlobalBuffer((__gm__ float*)output);
    AscendC::DataCopyExtParams copy{1,n*4,0,0,0};AscendC::DataCopyPad(dst,x_,copy);
    CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
  }
  __aicore__ inline bool ServingScores(GM_ADDR query,GM_ADDR key,GM_ADDR scores,
      unsigned keys,unsigned d,float scale,bool bf16_scores=true,bool split_scale=false) {
    if(!keys || keys>32 || (d!=64 && d!=128))return false;
    Load(bx_,key,keys*d);Load(bo_,query,d);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,keys*d);
    AscendC::Cast(z_,bo_,AscendC::RoundMode::CAST_NONE,d);AscendC::PipeBarrier<PIPE_V>();
    if(split_scale) {
      const float root=d==128?0.29730177875068026f:0.3535533905932738f;
      AscendC::Muls(x_,x_,root,keys*d);AscendC::Muls(z_,z_,root,d);AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t shape[2]={keys,d},qshape[2]={1,d};auto tmp=scratch_.ReinterpretCast<uint8_t>();
    AscendC::Broadcast<float,2,0>(work_,z_,shape,qshape,tmp);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(work_,work_,x_,keys*d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float,AscendC::Pattern::Reduce::AR>(y_,work_,tmp,shape,true);AscendC::PipeBarrier<PIPE_V>();
    // The BF16-score implementation preserves both graph storage roundings.
    // The separately declared FP32-score candidate omits those two casts.
    if(bf16_scores) {
      AscendC::Cast(bx_,y_,AscendC::RoundMode::CAST_RINT,keys);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(y_,bx_,AscendC::RoundMode::CAST_NONE,keys);AscendC::PipeBarrier<PIPE_V>();
    }
    if(!split_scale){AscendC::Muls(y_,y_,scale,keys);AscendC::PipeBarrier<PIPE_V>();}
    if(bf16_scores) {
      AscendC::Cast(bx_,y_,AscendC::RoundMode::CAST_RINT,keys);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(y_,bx_,AscendC::RoundMode::CAST_NONE,keys);AscendC::PipeBarrier<PIPE_V>();
    }
    CompleteDma<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<float> dst;dst.SetGlobalBuffer((__gm__ float*)scores);
    AscendC::DataCopyExtParams copy{1,keys*4,0,0,0};AscendC::DataCopyPad(dst,y_,copy);
    CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
  }
  __aicore__ inline bool ServingSoftmax(GM_ADDR scores,GM_ADDR probabilities,unsigned keys,
      bool fp32=false,unsigned width=0,unsigned output_width=0,bool vector_divide=false) {
    if(!width)width=keys;
    if(!output_width)output_width=keys;
    if(!keys || keys>output_width || output_width>width || width>1088)return false;
    AscendC::GlobalTensor<float> src;src.SetGlobalBuffer((__gm__ float*)scores);
    AscendC::DataCopyExtParams copy{1,keys*4,0,0,0};AscendC::DataCopyPadExtParams<float> pad(false,0,0,0);
    AscendC::DataCopyPad(x_,src,copy,pad);CompleteDma<AscendC::HardEvent::MTE2_V>();
    // The static-cache math baseline reduces over capacity. Materialize its
    // masked score tail locally; no invalid/padding K or V is loaded.
    AscendC::Duplicate(work_,-3.3895313892515355e38f,width);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(work_,x_,0.0f,keys);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(z_,work_,scratch_,width);CompleteDma<AscendC::HardEvent::V_S>();float maximum=z_.GetValue(0);
    AscendC::Adds(work_,work_,-maximum,width);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(work_,work_,width);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(z_,work_,scratch_,width);CompleteDma<AscendC::HardEvent::V_S>();float sum=z_.GetValue(0);
    if(!(sum>0))return false;
    if(vector_divide) {
      AscendC::Duplicate(z_,sum,output_width);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(work_,work_,z_,output_width);
    }else AscendC::Muls(work_,work_,1.0f/sum,output_width);
    AscendC::PipeBarrier<PIPE_V>();
    if(fp32) {
      CompleteDma<AscendC::HardEvent::V_MTE3>();AscendC::GlobalTensor<float> dst;
      dst.SetGlobalBuffer((__gm__ float*)probabilities);
      AscendC::DataCopyExtParams copy{1,output_width*4,0,0,0};AscendC::DataCopyPad(dst,work_,copy);
      CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
    }
    AscendC::Cast(bo_,work_,AscendC::RoundMode::CAST_RINT,output_width);Store(probabilities,output_width);return true;
  }
  __aicore__ inline bool ServingPV(GM_ADDR probabilities,GM_ADDR values,GM_ADDR output,unsigned keys,unsigned d,bool fp32=false) {
    if(!keys || keys>32 || (d!=64 && d!=128))return false;
    if(fp32) {
      AscendC::GlobalTensor<float> src;src.SetGlobalBuffer((__gm__ float*)probabilities);
      AscendC::DataCopyExtParams copy{1,keys*4,0,0,0};AscendC::DataCopyPadExtParams<float> pad(false,0,0,0);
      AscendC::DataCopyPad(z_,src,copy,pad);
    }else Load(bx_,probabilities,keys);
    Load(by_,values,keys*d);CompleteDma<AscendC::HardEvent::MTE2_V>();
    if(!fp32)AscendC::Cast(z_,bx_,AscendC::RoundMode::CAST_NONE,keys);
    AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,keys*d);AscendC::PipeBarrier<PIPE_V>();
    uint32_t shape[2]={keys,d},pshape[2]={keys,1};auto tmp=scratch_.ReinterpretCast<uint8_t>();
    AscendC::Broadcast<float,2,1>(work_,z_,shape,pshape,tmp);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(work_,work_,y_,keys*d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float,AscendC::Pattern::Reduce::RA>(x_,work_,tmp,shape,true);
    CompleteDma<AscendC::HardEvent::V_MTE3>();AscendC::GlobalTensor<float> dst;
    dst.SetGlobalBuffer((__gm__ float*)output);AscendC::DataCopy(dst,x_,d);CompleteDma<AscendC::HardEvent::MTE3_S>();return true;
  }
  __aicore__ inline bool SumServingPV(GM_ADDR partial,GM_ADDR output,unsigned chunks,unsigned d) {
    if(!chunks || chunks>34 || (d!=64 && d!=128))return false;
    AscendC::GlobalTensor<float> src;src.SetGlobalBuffer((__gm__ float*)partial);
    AscendC::Duplicate(x_,0.0f,d);AscendC::PipeBarrier<PIPE_V>();
    for(unsigned c=0;c<chunks;++c) {
      // y_ is reused by MTE2. A PIPE_V barrier alone does not stop the next
      // DMA overwriting it while the preceding vector addition is in flight.
      CompleteDma<AscendC::HardEvent::V_MTE2>();
      AscendC::DataCopy(y_,src[c*d],d);CompleteDma<AscendC::HardEvent::MTE2_V>();
      AscendC::Add(x_,x_,y_,d);AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Cast(bo_,x_,AscendC::RoundMode::CAST_RINT,d);Store(output,d);return true;
  }
  __aicore__ inline bool ArgmaxTile(GM_ADDR input,GM_ADDR value,GM_ADDR index,unsigned n,unsigned first) {
    if(!n || n>kElements)return false;
    Load(bx_,input,n);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(z_,x_,scratch_,n);CompleteDma<AscendC::HardEvent::V_S>();
    float maximum=z_.GetValue(0);unsigned chosen=n;
    for(unsigned i=0;i<n;++i)if(x_.GetValue(i)==maximum){chosen=i;break;}
    if(chosen==n)return false;
    union {float f;uint32_t u;} bits;bits.f=maximum;
    WriteControl((__gm__ uint32_t*)value,bits.u);WriteControl((__gm__ uint32_t*)index,first+chosen);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();return true;
  }
  // A serving CTG partial is normalized FP32 output plus log2(LSE), unlike
  // the diagnostic chunk ABI (unnormalized o, maximum, denominator).
  __aicore__ inline bool ServingPartial(GM_ADDR partial,GM_ADDR statistics,GM_ADDR output,
      GM_ADDR lse,unsigned chunks,unsigned d) {
    if(chunks>4 || (d!=64 && d!=128))return false;
    if(!chunks) {
      AscendC::Duplicate(x_,0.0f,d);CompleteDma<AscendC::HardEvent::V_MTE3>();
      AscendC::GlobalTensor<float> dst;dst.SetGlobalBuffer((__gm__ float*)output);
      AscendC::DataCopy(dst,x_,d);CompleteDma<AscendC::HardEvent::MTE3_S>();
      WriteControl((__gm__ uint32_t*)lse,0xff800000u);return true;
    }
    AscendC::GlobalTensor<float> stats,parts;
    stats.SetGlobalBuffer((__gm__ float*)statistics);parts.SetGlobalBuffer((__gm__ float*)partial);
    float maximum=-3.4e38f;
    for(unsigned c=0;c<chunks;++c)if(stats.GetValue(2*c)>maximum)maximum=stats.GetValue(2*c);
    AscendC::Duplicate(x_,0.0f,d);AscendC::PipeBarrier<PIPE_V>();float denominator=0;
    for(unsigned c=0;c<chunks;++c) {
      z_.SetValue(0,stats.GetValue(2*c)-maximum);CompleteDma<AscendC::HardEvent::S_V>();
      AscendC::Exp(z_,z_,1);CompleteDma<AscendC::HardEvent::V_S>();
      float factor=z_.GetValue(0);denominator+=factor*stats.GetValue(2*c+1);
      CompleteDma<AscendC::HardEvent::V_MTE2>();
      AscendC::DataCopy(y_,parts[c*d],d);CompleteDma<AscendC::HardEvent::MTE2_V>();
      AscendC::Muls(work_,y_,factor,d);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(x_,x_,work_,d);AscendC::PipeBarrier<PIPE_V>();
    }
    if(!(denominator>0))return false;
    AscendC::Muls(x_,x_,1.0f/denominator,d);CompleteDma<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<float> dst;dst.SetGlobalBuffer((__gm__ float*)output);
    AscendC::DataCopy(dst,x_,d);CompleteDma<AscendC::HardEvent::MTE3_S>();
    z_.SetValue(0,denominator);CompleteDma<AscendC::HardEvent::S_V>();
    AscendC::Ln(z_,z_,1);CompleteDma<AscendC::HardEvent::V_S>();
    union {float f;uint32_t u;} bits;bits.f=(maximum+z_.GetValue(0))*1.4426950408889634f;
    WriteControl((__gm__ uint32_t*)lse,bits.u);return true;
  }
  __aicore__ inline bool ServingMerge(GM_ADDR partial,GM_ADDR lse,GM_ADDR output,
      unsigned blocks,unsigned head,unsigned heads,unsigned d) {
    if(!blocks || blocks>9 || head>=heads || (d!=64 && d!=128))return false;
    AscendC::GlobalTensor<float> stats,parts;
    stats.SetGlobalBuffer((__gm__ float*)lse);parts.SetGlobalBuffer((__gm__ float*)partial);
    float maximum=-3.4e38f;
    for(unsigned c=0;c<blocks;++c)if(stats.GetValue(c*heads+head)>maximum)maximum=stats.GetValue(c*heads+head);
    if(maximum==-3.4e38f)return false;
    AscendC::Duplicate(x_,0.0f,d);AscendC::PipeBarrier<PIPE_V>();float denominator=0;
    for(unsigned c=0;c<blocks;++c) {
      float l=stats.GetValue(c*heads+head);if(l< -3.4e38f)continue;
      z_.SetValue(0,(l-maximum)*0.6931471805599453f);CompleteDma<AscendC::HardEvent::S_V>();
      AscendC::Exp(z_,z_,1);CompleteDma<AscendC::HardEvent::V_S>();float weight=z_.GetValue(0);
      denominator+=weight;CompleteDma<AscendC::HardEvent::V_MTE2>();
      AscendC::DataCopy(y_,parts[(c*heads+head)*d],d);
      CompleteDma<AscendC::HardEvent::MTE2_V>();AscendC::Muls(work_,y_,weight,d);
      AscendC::PipeBarrier<PIPE_V>();AscendC::Add(x_,x_,work_,d);AscendC::PipeBarrier<PIPE_V>();
    }
    if(!(denominator>0))return false;
    AscendC::Muls(x_,x_,1.0f/denominator,d);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(bo_,x_,AscendC::RoundMode::CAST_RINT,d);Store(output,d);return true;
  }
  // n is the entire normalization domain; splitting an H3072 row is illegal.
  __aicore__ inline bool RmsNorm(GM_ADDR input,GM_ADDR gamma,GM_ADDR output,unsigned n,float epsilon,unsigned variant=0) {
    if(!n || n>kElements || epsilon<=0 || variant>2)return false;
    Load(bx_,input,n);Load(by_,gamma,n);CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,n);
    AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
    if(variant==2) {
#if defined(TILEMEGA_AS2_CANN_NORM_PRIMITIVES) && __CCE_AICORE__ == 310
      // Version-pinned device primitives only, no whole-op scheduler/launch.
      // Larger reduction families require their own fold/resource validation.
      if(n!=128 && n!=1024)return false;
      NormCommon::NormCommonRegbase::CalculateSquareReduceSum<bfloat16_t>(bx_,z_,scratch_,1,n,n,n,8,n);
      AscendC::PipeBarrier<PIPE_V>();
      NormCommon::ComputeRstdNewtonRaphson<false,true>(z_,z_,1,epsilon,1.0f/n);
#else
      return false;
#endif
    }else {
    AscendC::Mul(work_,x_,x_,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(z_,work_,scratch_,n);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(z_,z_,1.0f/n,1);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(z_,z_,epsilon,1);AscendC::PipeBarrier<PIPE_V>();
    if(variant==1) {
      // Installed 3510 RmsNorm uses ln -> -0.5 -> exp. Keep this measured
      // alternative explicit: its FP32 rounding differs from Rsqrt.
      AscendC::Ln(z_,z_,1);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(z_,z_,-0.5f,1);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(z_,z_,1);
    }else AscendC::Rsqrt(z_,z_,1);
    }
    CompleteDma<AscendC::HardEvent::V_S>();
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
      CompleteDma<AscendC::HardEvent::V_MTE2>();
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
