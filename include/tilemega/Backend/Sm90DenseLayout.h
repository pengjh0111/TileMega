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
struct Sm90DenseActivationStage {
  int tile_k=0,slots=0,offset=0;
  Sm90DenseInterval written;
  bool argmax=false;
};
struct Sm90DenseActivationReuse {
  bool valid=false;
  int first=0,tile_k=0,slots=0,offset=0;
  constexpr int End() const {return offset+1024+32*slots*tile_k;}
  constexpr int ArgmaxShift() const {return End()-1024;}
  template<std::size_t N>
  static constexpr Sm90DenseActivationReuse Prove(
      std::array<Sm90DenseActivationStage,N> const& stages,int batch) {
    Sm90DenseActivationReuse result;result.valid=batch<16;
    int maximum=1024;
    for(unsigned i=0;i<N;++i) {
      auto const& s=stages[i];if(s.written.end>maximum)maximum=s.written.end;
      if(!s.tile_k)continue;
      if(!result.tile_k) {
        result.first=i;result.tile_k=s.tile_k;result.slots=s.slots;result.offset=s.offset;
      }else if(s.tile_k!=result.tile_k || s.slots!=result.slots || s.offset!=result.offset)
        result.valid=false;
    }
    if(!result.tile_k)result.valid=false;
    Sm90DenseInterval activation{result.offset+1024,result.End()};
    for(auto const& s:stages)if(!s.tile_k) {
      if(s.argmax) {
        if(s.written.end+result.ArgmaxShift()>maximum)result.valid=false;
      }else if(s.written.Overlaps(activation))result.valid=false;
    }
    return result;
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
  static constexpr Sm90DenseLayout Gemm(int tn,int tk,int slots,int wk) {
    int weight=1024+32*slots*tk,partial=weight+2*slots*tn*tk;
    return {weight,0,partial,partial+64*wk*tn+512};
  }
  static constexpr Sm90DensePayload GemmPayload(int tn,int tk,int slots,int k,int bytes) {
    Sm90DensePayload result;auto layout=Gemm(tn,tk,slots,1);
    for(int i=0;i<slots && i<k/tk && bytes>0;++i) {
      int part=bytes<2*tn*tk?bytes:2*tn*tk;
      result.Add(layout.key+i*2*tn*tk,part);bytes-=part;
    }
    return result;
  }
  static constexpr int AttentionKeys(int chunks,int row_work,int capacity) {
    return (capacity+row_work+chunks-1)/chunks;
  }
  static constexpr Sm90DenseLayout Attention(int batch,int d,int q,int chunks,
      int keys,int slots,int row_work,int capacity) {
    int key=1024+(batch<=4?2*q*d:36*d);
    int payload=2*d*(batch<=4?AttentionKeys(chunks,row_work,capacity):slots*keys);
    int scratch=key+2*payload;
    int end=scratch+16*q*d+32*q+4*(chunks>4?chunks:4)*q+4*q+4;
    return {key,key+payload,scratch,(end+15)/16*16};
  }
  static constexpr Sm90DensePayload AttentionPayload(int batch,int d,int q,int chunks,
      int keys,int slots,int row_work,int capacity,int bytes) {
    Sm90DensePayload result;
    auto layout=Attention(batch,d,q,chunks,keys,slots,row_work,capacity);
    int count=batch<=4?1:slots,rows=batch<=4?AttentionKeys(chunks,row_work,capacity):keys;
    for(int i=0;i<count && bytes>0;++i) {
      int part=bytes<4*d*rows?bytes:4*d*rows;
      result.Add(layout.key+i*2*d*rows,part/2);
      result.Add(layout.value+i*2*d*rows,part/2);bytes-=part;
    }
    return result;
  }
};
}  // namespace tilemega::backend
