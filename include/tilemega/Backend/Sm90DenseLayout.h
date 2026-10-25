// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>

namespace tilemega::backend {
struct Sm90DenseInterval {
  int begin=0,end=0;
  constexpr bool Overlaps(Sm90DenseInterval other) const {
    return begin<other.end && other.begin<end && begin<end && other.begin<other.end;
  }
};
struct Sm90DensePayload {
  std::array<Sm90DenseInterval,16> ranges{};
  int count=0;
  constexpr void Add(int begin,int bytes) {
    if(bytes)ranges[count++]={begin,begin+bytes};
  }
  constexpr bool Overlaps(Sm90DenseInterval other) const {
    for(int i=0;i<count;++i)if(ranges[i].Overlaps(other))return true;
    return false;
  }
  constexpr Sm90DensePayload Shift(int offset) const {
    auto result=*this;
    for(int i=0;i<count;++i) {
      result.ranges[i].begin+=offset;result.ranges[i].end+=offset;
    }
    return result;
  }
};
struct Sm90DenseLayout {
  int key=0,value=0,scratch=1024,end=1024;
  constexpr Sm90DenseInterval Live(int offset) const {return {offset+1024,offset+end};}
  constexpr Sm90DenseInterval Scratch(int offset) const {return {offset+scratch,offset+end};}
  static constexpr Sm90DenseLayout Argmax() {
    return {0,0,1024,1024+int(4*(sizeof(float)+sizeof(int)))};
  }
  // TMA activation rows per slot: eight rows alias the upper MMA rows for B<=8.
  static constexpr int ActivationRows(int batch) {return batch<=8?8:16;}
  // Scratch: WK fp32 partial tiles, 512 B of per-row squares, then (residual
  // stages) two prefetched BF16 residual tiles of Batch x TN.
  static constexpr Sm90DenseLayout Gemm(int tn,int tk,int slots,int wk,int batch,bool residual=false) {
    int weight=1024+2*ActivationRows(batch)*slots*tk,partial=weight+2*slots*tn*tk;
    return {weight,0,partial,partial+64*wk*tn+512+(residual?4*batch*tn:0)};
  }
  static constexpr Sm90DensePayload GemmPayload(int tn,int tk,int slots,int k,int batch,int bytes) {
    Sm90DensePayload result;auto layout=Gemm(tn,tk,slots,1,batch);
    for(int i=0;i<slots && i<k/tk && bytes>0;++i) {
      int part=bytes<2*tn*tk?bytes:2*tn*tk;
      result.Add(layout.key+i*2*tn*tk,part);bytes-=part;
    }
    return result;
  }
  // Sm90DenseAttentionBody::Storage: K/V ring, Q and two new-row panels,
  // then the four-warp merge area; the struct is 1024-byte aligned.
  static constexpr Sm90DenseLayout Attention(int d,int q,int chunks,int keys,int slots) {
    int key=1024,value=key+2*keys*d,scratch=key+4*slots*keys*d+3*32*d;
    int merge=16*q*d+32*q+4*(chunks>4?chunks:4)*q+4;
    return {key,value,scratch,key+(4*slots*keys*d+3*32*d+merge+1023)/1024*1024};
  }
};
}  // namespace tilemega::backend
