// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseLayout.h>

namespace tilemega::backend {

struct Sm80AttentionScratch {
  static constexpr int NewRowsBytes(int d) {return 64*d;}
  static constexpr int OutputBytes(int d,int q) {return 16*q*d;}
  static constexpr int SharedBytes(int d,int q,bool compact) {
    int rows=NewRowsBytes(d),output=OutputBytes(d,q);
    return compact?(rows>output?rows:output):rows+output;
  }
  static constexpr Sm90DenseLayout Layout(int d,int q,int chunks,int keys,int slots,
      bool dsmem,bool compact) {
    auto layout=Sm90DenseLayout::Attention(d,q,chunks,keys,slots,dsmem);
    if(compact) {
      layout.scratch-=NewRowsBytes(d);
      layout.end-=SharedBytes(d,q,false)-SharedBytes(d,q,true);
    }
    return layout;
  }
};

}  // namespace tilemega::backend
