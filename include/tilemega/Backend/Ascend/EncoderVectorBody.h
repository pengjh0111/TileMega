// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <lib/math/erf.h>
namespace tilemega::ascend {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
__simd_vf__ inline void EncoderPvStep(__ubuf__ float* accumulator,__ubuf__ bfloat16_t* values,__ubuf__ float* probability) {
  using namespace AscendC::MicroAPI;
  static constexpr CastTrait convertZero={RegLayout::ZERO,SatMode::UNKNOWN,MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
  static constexpr CastTrait convertOne={RegLayout::ONE,SatMode::UNKNOWN,MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
  auto mask=CreateMask<float,MaskPattern::ALL>();auto halfMask=CreateMask<bfloat16_t,MaskPattern::ALL>();
  RegTensor<bfloat16_t> packed;
  RegTensor<float> low,high,value,scale,accumulated,product;
  LoadAlign(packed,values);
  Cast<float,bfloat16_t,convertZero>(low,packed,mask);
  Cast<float,bfloat16_t,convertOne>(high,packed,halfMask);
  Interleave(value,high,low,high);
  LoadAlign<float,LoadDist::DIST_BRC_B32>(scale,probability);
  LoadAlign(accumulated,accumulator);
  Mul(product,value,scale,mask);Add(accumulated,accumulated,product,mask);
  StoreAlign<float,StoreDist::DIST_NORM_B32>(accumulator,accumulated,mask);
}
#endif
class EncoderVectorBody {
  static constexpr unsigned kElements=512;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
  AscendC::LocalTensor<bfloat16_t> bx_,by_,bz_,bo_;
  AscendC::LocalTensor<float> x_,y_,z_,work_,reduce_,reduce_work_;
  __aicore__ inline void Load(AscendC::LocalTensor<bfloat16_t> local,GM_ADDR pointer,unsigned n) {
    AscendC::GlobalTensor<bfloat16_t> global;global.SetGlobalBuffer((__gm__ bfloat16_t*)pointer);
    AscendC::DataCopyExtParams copy{1,uint32_t(n*2),0,0,0};
    AscendC::DataCopyPadExtParams<bfloat16_t> pad(false,0,0,0);
    AscendC::DataCopyPad(local,global,copy,pad);
  }
  __aicore__ inline void Store(AscendC::LocalTensor<float> value,GM_ADDR output,unsigned n) {
    AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(bo_,value,AscendC::RoundMode::CAST_RINT,n);
    CompleteDma<AscendC::HardEvent::V_MTE3>();
    AscendC::GlobalTensor<bfloat16_t> global;global.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    AscendC::DataCopyExtParams copy{1,uint32_t(n*2),0,0,0};
    AscendC::DataCopyPad(global,bo_,copy);CompleteDma<AscendC::HardEvent::MTE3_S>();
  }
 public:
  static constexpr unsigned kLocalBytes=16384;
  __aicore__ inline EncoderVectorBody() {
    pipe_.InitBuffer(storage_,kLocalBytes);auto raw=storage_.Get<uint8_t>();
    bx_=raw[0].ReinterpretCast<bfloat16_t>();by_=raw[1024].ReinterpretCast<bfloat16_t>();
    bz_=raw[2048].ReinterpretCast<bfloat16_t>();bo_=raw[3072].ReinterpretCast<bfloat16_t>();
    x_=raw[4096].ReinterpretCast<float>();y_=raw[6144].ReinterpretCast<float>();
    z_=raw[8192].ReinterpretCast<float>();work_=raw[10240].ReinterpretCast<float>();
    reduce_=raw[12288].ReinterpretCast<float>();reduce_work_=raw[14336].ReinterpretCast<float>();
  }
  __aicore__ inline bool Run(unsigned kind,GM_ADDR input,GM_ADDR gamma,GM_ADDR beta,
                             GM_ADDR output,unsigned n,float epsilon=0) {
    if(!n || n>kElements || (kind!=4 && kind!=10 && kind!=19) || (kind==19 && epsilon<=0))return false;
    Load(bx_,input,n);if(kind!=4)Load(by_,gamma,n);if(kind==19)Load(bz_,beta,n);
    CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,n);
    if(kind!=4)AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,n);
    if(kind==19)AscendC::Cast(z_,bz_,AscendC::RoundMode::CAST_NONE,n);
    AscendC::PipeBarrier<PIPE_V>();
    if(kind==4) {
      AscendC::Muls(work_,x_,0.7071067811865475f,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Erf<float,false>(y_,work_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(y_,y_,1.0f,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(work_,x_,y_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(work_,work_,0.5f,n);
    }else if(kind==10) {
      AscendC::Add(work_,x_,y_,n);
    }else {
      AscendC::ReduceSum(reduce_,x_,reduce_work_,n);CompleteDma<AscendC::HardEvent::V_S>();
      float mean=reduce_.GetValue(0)/float(n);
      CompleteDma<AscendC::HardEvent::S_V>();
      AscendC::Adds(work_,x_,-mean,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(work_,work_,work_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::ReduceSum(reduce_,work_,reduce_work_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(reduce_,reduce_,1.0f/float(n),1);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(reduce_,reduce_,epsilon,1);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Rsqrt(reduce_,reduce_,1);CompleteDma<AscendC::HardEvent::V_S>();
      float scale=reduce_.GetValue(0);
      CompleteDma<AscendC::HardEvent::S_V>();
      AscendC::Adds(work_,x_,-mean,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(work_,work_,scale,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(work_,work_,y_,n);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(work_,work_,z_,n);
    }
    Store(work_,output,n);return true;
  }
  __aicore__ inline bool Attention(GM_ADDR packed,GM_ADDR mask,GM_ADDR output,
                                   unsigned sequence,unsigned heads,unsigned query,GM_ADDR debug=nullptr) {
    if(!sequence || sequence>16 || !heads || heads>4 || query>=sequence*heads)return false;
    unsigned head=query/sequence,row=query%sequence,hidden=heads*64;
    AscendC::GlobalTensor<int64_t> valid;valid.SetGlobalBuffer((__gm__ int64_t*)mask);
    Load(bx_,packed+(row*hidden*3+head*192)*2,64);
    CompleteDma<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(x_,bx_,AscendC::RoundMode::CAST_NONE,64);AscendC::PipeBarrier<PIPE_V>();
    float maximum=-3.4e38f;unsigned keys=0;
    for(unsigned key=0;key<sequence;++key) {
      float score=-3.4e38f;
      if(valid.GetValue(key)) {
        ++keys;Load(by_,packed+(key*hidden*3+head*192+64)*2,64);
        CompleteDma<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(y_,by_,AscendC::RoundMode::CAST_NONE,64);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(work_,x_,y_,64);AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(reduce_,work_,reduce_work_,64);CompleteDma<AscendC::HardEvent::V_S>();
        score=reduce_.GetValue(0)*0.125f;if(score>maximum)maximum=score;
      }
      z_.SetValue(key,score);
      if(debug) {union {float f;uint32_t u;} v;v.f=score;WriteControl((__gm__ uint32_t*)debug+key,v.u);}
      CompleteDma<AscendC::HardEvent::S_V>();
    }
    if(!keys) {
      AscendC::Duplicate(work_,0.0f,64);Store(work_,output+(row*hidden+head*64)*2,64);return true;
    }
    CompleteDma<AscendC::HardEvent::S_V>();
    AscendC::Adds(z_,z_,-maximum,sequence);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(z_,z_,sequence);AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(reduce_,z_,reduce_work_,sequence);CompleteDma<AscendC::HardEvent::V_S>();
    float total=reduce_.GetValue(0);if(!(total>0))return false;
    CompleteDma<AscendC::HardEvent::S_V>();
    AscendC::Muls(z_,z_,1.0f/total,sequence);CompleteDma<AscendC::HardEvent::V_S>();
    AscendC::Duplicate(x_,0.0f,64);AscendC::PipeBarrier<PIPE_V>();
    for(unsigned key=0;key<sequence;++key)if(valid.GetValue(key)) {
      Load(by_,packed+(key*hidden*3+head*192+128)*2,64);CompleteDma<AscendC::HardEvent::MTE2_V>();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
      EncoderPvStep((__ubuf__ float*)x_.GetPhyAddr(),(__ubuf__ bfloat16_t*)by_.GetPhyAddr(),
          (__ubuf__ float*)z_.GetPhyAddr()+key);
#endif
      CompleteDma<AscendC::HardEvent::V_MTE2>();
    }
    Store(x_,output+(row*hidden+head*64)*2,64);return true;
  }
};
} // namespace tilemega::ascend
