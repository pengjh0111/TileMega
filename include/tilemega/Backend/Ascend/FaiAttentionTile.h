/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE.CATLASS in this directory for the full text of the License.
 */
// Adapted from CATLASS49 at 2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52:
// caller-owned tile coordinates, private resources and single-owner completion.
#pragma once
#include <tilemega/Target/Ascend/FaiTileContract.h>
#include <tilemega/Target/Ascend/ProgressPrimitives.h>
#include <catlass/arch/resource.hpp>
#include <catlass/gemm/block/block_mmad.hpp>
#include <catlass/gemm/dispatch_policy.hpp>
#include <catlass/gemm/gemm_type.hpp>
#include <catlass/epilogue/block/block_epilogue.hpp>
#include <catlass/epilogue/dispatch_policy.hpp>

namespace tilemega::ascend {
struct FaiTileTask {
  GM_ADDR query;
  GM_ADDR key;
  GM_ADDR value;
  GM_ADDR output;
  GM_ADDR control;
  unsigned rows,width,keys,capacity,epoch;
  uint64_t wait_ticks;
};

// Assembly of the pinned CATLASS49 Block/epilogue components, not its Kernel
// or block scheduler. The caller supplies one independent query/KV domain and
// role. BF16 P is a distinct precision recipe, pending frozen model gates.
template<unsigned Width> class FaiAttentionTile {
  using Arch=Catlass::Arch::Ascend950;
  using Row=Catlass::layout::RowMajor;
  using Col=Catlass::layout::ColumnMajor;
  using Packed=Catlass::layout::zN;
  using Shape=tla::Shape<tla::Int<32>,tla::Int<128>,tla::Int<Width>>;
  using MmadShape=tla::Shape<tla::Int<32>,tla::Int<128>,tla::Int<fai_tile::kMmadK>>;
  using QCopy=Catlass::Gemm::Tile::PackedTileCopyTlaToUB<Arch,bfloat16_t,Row,bfloat16_t,Col,float,Row,
      void,Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
  using PCopy=Catlass::Gemm::Tile::PackedTileCopyTlaToUB<Arch,bfloat16_t,Packed,bfloat16_t,Row,float,Row,
      void,Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
  using QMicro=Catlass::Gemm::Tile::TileMmadTla<Arch,bfloat16_t,typename QCopy::LayoutTagL1A>;
  using PMicro=Catlass::Gemm::Tile::TileMmadTla<Arch,bfloat16_t,typename PCopy::LayoutTagL1A>;
  using QK=Catlass::Gemm::Block::BlockMmadTla<Catlass::Gemm::MmadFAIQK<Arch,false>,
      MmadShape,MmadShape,bfloat16_t,bfloat16_t,float,void,QCopy,QMicro>;
  using PV=Catlass::Gemm::Block::BlockMmadTla<Catlass::Gemm::MmadFAIPV<Arch,false>,
      MmadShape,MmadShape,bfloat16_t,bfloat16_t,float,void,PCopy,PMicro>;
  static_assert(QK::L0A_TILE_SIZE==PV::L0A_TILE_SIZE && QK::L0B_TILE_SIZE==PV::L0B_TILE_SIZE);
  static_assert(QK::BLOCK_L1_SIZE+PV::BLOCK_L1_SIZE+32*128*2==fai_tile::Budget(Width).l1);
  static_assert(QK::BLOCK_L0C_SIZE+PV::BLOCK_L0C_SIZE==fai_tile::Budget(Width).l0c);
  using PType=Catlass::Gemm::GemmType<bfloat16_t,Packed>;
  using SType=Catlass::Gemm::GemmType<float,Row>;
  using MaskType=Catlass::Gemm::GemmType<uint8_t,Row>;
  using OType=Catlass::Gemm::GemmType<bfloat16_t,Row>;
  using Softmax=Catlass::Epilogue::Block::BlockEpilogue<
      Catlass::Epilogue::EpilogueAscend950FASoftmax<false>,Shape,PType,SType,MaskType>;
  using Rescale=Catlass::Epilogue::Block::BlockEpilogue<
      Catlass::Epilogue::EpilogueAscend950FARescaleO,Shape,OType,SType>;
 public:
  __aicore__ inline bool RunTask(FaiTileTask const& t,unsigned lane) {
    using namespace fai_tile;
    if(!ShapeValid(t) || lane>2)return false; // lane2 is the AIC owner.
    if ASCEND_IS_AIC {if(lane!=2)return false;}
    else {if(lane>1)return false;}
    Catlass::Arch::Resource<Arch> resource;
    unsigned ub=kUbBase,l1=kRows*kKeys*2,l0c=0;
    AscendC::LocalTensor<float> score[2],partial[2];
    for(unsigned i=0;i<2;++i) {
      score[i]=resource.ubBuf.template GetBufferByByte<float>(ub);ub+=kRows/2*kKeys*4;
      partial[i]=resource.ubBuf.template GetBufferByByte<float>(ub);ub+=kRows/2*Width*4;
    }
    auto sum=resource.ubBuf.template GetBufferByByte<float>(ub);ub+=512;
    auto maximum=resource.ubBuf.template GetBufferByByte<float>(ub);ub+=512;
    auto rescale=resource.ubBuf.template GetBufferByByte<float>(ub);ub+=512;
    auto probability=resource.l1Buf.template GetBufferByByte<bfloat16_t>(0);
    QK qk(resource,l1,l0c);PV pv(resource,l1,l0c);
    Softmax softmax(resource,Width==128?0.08838834764831845f:0.125f,ub);
    Rescale update(resource,ub);
    if ASCEND_IS_AIV {
      for(unsigned i=0;i<2;++i) {
        AscendC::CrossCoreSetFlag<4,PIPE_V>(kFreeScore[i]);
        AscendC::CrossCoreSetFlag<4,PIPE_V>(kFreeOutput[i]);
      }
    }
    AscendC::GlobalTensor<bfloat16_t> q,k,v,o;
    q.SetGlobalBuffer((__gm__ bfloat16_t*)t.query);k.SetGlobalBuffer((__gm__ bfloat16_t*)t.key);
    v.SetGlobalBuffer((__gm__ bfloat16_t*)t.value);o.SetGlobalBuffer((__gm__ bfloat16_t*)t.output);
    auto tq=tla::MakeTensor(q,tla::MakeLayout<bfloat16_t,Row>(kRows,Width),Catlass::Arch::PositionGM{});
    auto tk=tla::MakeTensor(k,tla::MakeLayout<bfloat16_t,Col>(Width,t.capacity),Catlass::Arch::PositionGM{});
    auto tv=tla::MakeTensor(v,tla::MakeLayout<bfloat16_t,Row>(t.capacity,Width),Catlass::Arch::PositionGM{});
    auto to=tla::MakeTensor(o,tla::MakeLayout<bfloat16_t,Row>(kRows,Width),Catlass::Arch::PositionGM{});
    auto tp=tla::MakeTensor(probability,tla::MakeLayout<bfloat16_t,Packed>(kRows,kKeys),Catlass::Arch::PositionL1{});
    AscendC::GlobalTensor<int32_t> table;table.SetGlobalBuffer((__gm__ int32_t*)t.query);
    AscendC::GlobalTensor<uint8_t> mask;mask.SetGlobalBuffer((__gm__ uint8_t*)t.query);
    auto unused_mask=tla::MakeTensor(mask,tla::MakeLayout<uint8_t,Row>(kRows,kKeys),Catlass::Arch::PositionGM{});
    for(unsigned begin=0;begin<t.keys;begin+=kKeys) {
      const unsigned count=t.keys-begin<kKeys?t.keys-begin:kKeys,slot=begin/kKeys%2;
      const bool first=begin==0,last=begin+count==t.keys;
      if ASCEND_IS_AIC {
        auto s=tla::MakeTensor(score[slot],tla::MakeLayout<float,Row>(kRows,kKeys),Catlass::Arch::PositionUB{});
        auto key=tla::GetTile(tk,tla::MakeCoord(0,begin),tla::MakeShape(Width,count));
        auto qshape=tla::MakeShape(kRows,count,Width);
        qk(tq,key,s,table,qshape,slot,kKeys,first,last);
        Signal<PIPE_FIX>(kScoreReady);
        Wait<PIPE_MTE1>(kProbabilityReady);
        auto out=tla::MakeTensor(partial[slot],tla::MakeLayout<float,Row>(kRows,Width),Catlass::Arch::PositionUB{});
        auto value=tla::GetTile(tv,tla::MakeCoord(begin,0),tla::MakeShape(count,Width));
        auto pshape=tla::MakeShape(kRows,Width,count);
        pv(tp,value,out,table,pshape,slot,kKeys);
        Signal<PIPE_FIX>(kOutputReady);
      }else {
        AscendC::CrossCoreWaitFlag<4,PIPE_V>(kScoreReady);
        auto layout=tla::MakeLayout(tla::MakeShape(kRows/2,count),tla::MakeStride(kKeys,tla::Int<1>{}));
        auto s=tla::MakeTensor(score[slot],layout,Catlass::Arch::PositionUB{});
        auto pdst=tla::GetTile(tp,tla::MakeCoord(lane*16,0),tla::MakeShape(16,kKeys));
        softmax(pdst,sum,maximum,rescale,s,unused_mask,!first,slot,0,kFreeScore[slot],kProbabilityReady);
        AscendC::CrossCoreWaitFlag<4,PIPE_V>(kOutputReady);
        auto src=tla::MakeTensor(partial[slot],tla::MakeLayout<float,Row>(16,Width),Catlass::Arch::PositionUB{});
        auto dst=tla::GetTile(to,tla::MakeCoord(lane*16,0),tla::MakeShape(16,Width));
        update(dst,rescale,sum,src,first,last,kFreeOutput[slot]);
      }
    }
    auto* cells=(__gm__ uint32_t*)t.control;
    if ASCEND_IS_AIC {
      for(unsigned i=0;i<2;++i) {Wait<PIPE_FIX>(kFreeScore[i]);Wait<PIPE_FIX>(kFreeOutput[i]);}
      if(!WaitProgress(cells+128,t.epoch,cells+384,t.wait_ticks) ||
         !WaitProgress(cells+256,t.epoch,cells+384,t.wait_ticks))return false;
      PublishSingleOwnerEvent(cells,t.epoch);
    }else {
      CompleteDma<AscendC::HardEvent::MTE3_S>();
      PublishSingleOwnerEvent(cells+(lane+1)*128,t.epoch);
      // Do not seed the next Task's credits while the owner is draining this
      // generation. An acknowledgement alone does not release the channel.
      if(!WaitProgress(cells,t.epoch,cells+384,t.wait_ticks))return false;
    }
    return true;
  }
 private:
  __aicore__ static inline bool ShapeValid(FaiTileTask const& t) {
    return fai_tile::Shape(t.rows,t.width,t.keys,t.capacity) && t.width==Width &&
      t.query && t.key && t.value && t.output && t.control && t.epoch && t.wait_ticks;
  }
  template<pipe_t Pipe> __aicore__ static inline void Signal(unsigned id) {
    AscendC::CrossCoreSetFlag<4,Pipe>(id);AscendC::CrossCoreSetFlag<4,Pipe>(16+id);
  }
  template<pipe_t Pipe> __aicore__ static inline void Wait(unsigned id) {
    AscendC::CrossCoreWaitFlag<4,Pipe>(id);AscendC::CrossCoreWaitFlag<4,Pipe>(16+id);
  }
};
} // namespace tilemega::ascend
