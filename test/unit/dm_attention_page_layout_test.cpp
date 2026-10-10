#define TILEMEGA_DM_SUPPORT 1
// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>
#include <cassert>
#include <iostream>
#include <vector>

namespace tilemega::tests::dm_attention_page_layout_test {
template<int D,int Bytes,codegen::AttentionPagePolicy Policy>
void Check(unsigned& cases) {
  using Layout=codegen::AttentionPageLayout<D,Bytes,Policy>;
  for(int count:{0,1,15,16,17,32,33,48,63,64,65,128,129,256,1087,4096}) {
    Layout layout{count};std::vector<unsigned> seen(count);
    for(int wave=0;wave<layout.waves();++wave)for(int owner=0;owner<4;++owner) {
      auto page=layout.PageOf(owner,wave);
      assert(page>=0 && page<layout.pages_per_task());
      assert(layout.ReleaseArrivals(page)*layout.ReleaseIterations(page)==128);
      for(int row=0;row<layout.RowsPerOwner();++row) {
        int physical=layout.RowOffset(owner)+row;
        assert(physical<Layout::page_rows && layout.OwnerOf(page,physical)==owner);
        assert(layout.LocalRow(physical)==row);
        int position=layout.Position(owner,wave,row);
        if(layout.Valid(owner,wave,row)) {assert(position>=0 && position<count);++seen[position];}
      }
    }
    for(auto visits:seen)assert(visits==1);
    if(Policy==codegen::AttentionPagePolicy::kWarpPrivate)assert(!layout.needs_cta_sync());
    assert(layout.pages_in_flight()==2*layout.pages_per_wave());++cases;
  }
}
int TestDmAttentionPageLayout(int,char**) {
  unsigned cases=0;
  Check<64,8192,codegen::AttentionPagePolicy::kWarpPrivate>(cases);
  Check<64,8192,codegen::AttentionPagePolicy::kPacked>(cases);
  Check<64,16384,codegen::AttentionPagePolicy::kWarpPrivate>(cases);
  Check<64,16384,codegen::AttentionPagePolicy::kPacked>(cases);
  Check<128,8192,codegen::AttentionPagePolicy::kWarpPrivate>(cases);
  Check<128,8192,codegen::AttentionPagePolicy::kPacked>(cases);
  Check<128,16384,codegen::AttentionPagePolicy::kWarpPrivate>(cases);
  Check<128,16384,codegen::AttentionPagePolicy::kPacked>(cases);
  std::cout<<"Attention page layout: "<<cases<<" exact position coverage and release counts PASS\n";
  return 0;
}
}
