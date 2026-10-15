// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90Gmma.h>
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>
#include <cstdio>
#include <stdexcept>
#include <set>

template<int D,int Rows>
void Check() {
  using namespace cute;
  using Atom=decltype(composition(Swizzle<3,3,3>{},Layout<Shape<_8,_64>,Stride<_64,_1>>{}));
  auto chunk=tile_to_shape(Atom{},Shape<_16,Int<D>>{});
  auto uniform=as_position_independent_swizzle_layout(tilemega::backend::Sm90KLayout<Rows,D>{});
  std::set<int> offsets;
  for(int row=0;row<Rows;++row)for(int column=0;column<D;++column) {
    int old=(row/16)*16*D+chunk(row%16,column);
    int actual=tilemega::codegen::AttentionTileLayout<D,Rows,false>::Offset(row,column);
    int hopper=tilemega::codegen::AttentionTileLayout<D,Rows,true>::Offset(row,column);
    if(actual!=old || hopper!=uniform(row,column) || hopper<0 || hopper>=Rows*D || !offsets.insert(hopper).second)
      throw std::runtime_error("chunk16/uniform storage layout differs from its CuTe description");
  }
}
int main() {
  Check<64,16>();Check<64,32>();Check<64,64>();Check<64,128>();
  Check<128,16>();Check<128,32>();Check<128,64>();Check<128,128>();
  using namespace tilemega::codegen;
  for(int count:{1,15,16,63,64,65,128,129,1024}) {
    AttentionPageLayout layout(64,count,AttentionPagePolicy::WarpGroup,32);
    std::set<int> positions;
    for(int wave=0;wave<layout.waves;++wave) {
      for(int owner=0;owner<4;++owner)
        if(layout.PageOf(owner,wave)!=wave || layout.RowOffset(owner)!=0 || layout.ReleaseArrivals(wave)!=1)
          throw std::runtime_error("warpgroup page ownership/release differs");
      for(int row=0;row<64;++row)if(layout.Valid(0,wave,row,32+count,32+count))
        positions.insert(layout.Position(0,wave,row));
      if(layout.Full(wave,32+count,32+count)!=(wave*64+64<=count))
        throw std::runtime_error("warpgroup full-page predicate differs");
    }
    if(int(positions.size())!=count || layout.pages_in_flight!=1)
      throw std::runtime_error("warpgroup logical coverage differs");
  }
  std::puts("attention storage: 8 chunk16/uniform CuTe matrices and 9 warpgroup ranges PASS");
}
