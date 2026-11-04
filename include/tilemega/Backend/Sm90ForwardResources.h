// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseLayout.h>

namespace tilemega::backend {
constexpr int Sm90ForwardAlign(int bytes) {return (bytes+1023)/1024*1024;}
constexpr int Sm90ForwardChainScratchBytes(int rows,bool needs_stats) {
  return needs_stats?8*rows:16;
}
constexpr int Sm90ForwardGemmSlotBytes(int tm,int tn,int tk) {return 2*tk*(tm+tn);}
constexpr Sm90DenseLayout Sm90ForwardGemmLayout(int tm,int tn,int tk,int slots,int chain_scratch) {
  int scratch=1024+slots*Sm90ForwardGemmSlotBytes(tm,tn,tk);
  return {1024,1024+2*slots*tk*tm,scratch,
      Sm90ForwardAlign(scratch+4*tm*tn+chain_scratch+16)};
}
constexpr int Sm90ForwardDepthwiseSlotBytes(int tw,int th,int channels,int stride,int pack) {
  return 2*(((tw-1)*stride+3)*((th-1)*stride+3)*channels+channels*9*pack);
}
constexpr Sm90DenseLayout Sm90ForwardDepthwiseLayout(int tw,int th,int channels,int stride,
    int slots,int pack,int chain_scratch) {
  int scratch=1024+slots*Sm90ForwardDepthwiseSlotBytes(tw,th,channels,stride,pack);
  return {1024,1024,scratch,Sm90ForwardAlign(scratch+4*tw*th*channels+chain_scratch)};
}
struct Sm90ForwardFusedResources {
  int pages=0,filters=0,panel=0,slot_bytes=0;
  Sm90DenseLayout layout;
};
constexpr Sm90ForwardFusedResources Sm90ForwardFusedLayout(int tw,int th,int cb,int tn,int stride,
    int slots,int pack,int chain_scratch) {
  int halo=((tw-1)*stride+3)*((th-1)*stride+3)*cb;
  int page=tn*cb,filter=cb*9*pack;
  int pages=Sm90ForwardAlign(1024+slots*2*halo);
  int filters=Sm90ForwardAlign(pages+slots*2*page);
  int panel=Sm90ForwardAlign(filters+slots*2*filter);
  int scratch=panel+2*64*cb;
  return {pages,filters,panel,2*(halo+page+filter),
      {1024,pages,scratch,Sm90ForwardAlign(scratch+4*64*tn+chain_scratch)}};
}
constexpr int Sm90ForwardAttentionSlotBytes(int seq,int rows) {return 4*seq*64+2*rows*64;}
constexpr Sm90DenseLayout Sm90ForwardAttentionLayout(int seq,int rows,int slots,
    bool scalar_reference=false) {
  int scratch=1024+slots*Sm90ForwardAttentionSlotBytes(seq,rows);
  int probability=Sm90ForwardAlign(scratch+4*rows*seq);
  return {1024,1024,scratch,
      Sm90ForwardAlign(probability+(scalar_reference?0:2*rows*seq))};
}
constexpr Sm90DenseLayout Sm90ForwardGemvLayout(int k,int block) {
  int page=Sm90ForwardAlign(1024+2*k);
  int end=Sm90ForwardAlign(page+2*block*64);
  return {1024,page,end,end};
}
// The cooperative GAP reducer owns 32 partial rows by 128 channel lanes.
constexpr Sm90DenseLayout Sm90ForwardGlobalPoolLayout() {
  return {1024,1024,1024,1024+32*128*4};
}
}  // namespace tilemega::backend
