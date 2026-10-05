// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>
#include <cassert>
#include <set>
#include <stdexcept>
#include <iostream>
namespace tilemega::tests::attention_page_layout_test {
using namespace tilemega::codegen;
int TestAttentionPageLayout(int,char**) {
  int cases=0;
  for(int rows:{16,32,64})for(int count:{0,1,15,16,17,32,33,48,63,64,65,128,129,256})
    for(auto policy:{AttentionPagePolicy::WarpPrivate,AttentionPagePolicy::Packed}) {
      AttentionPageLayout l(rows,count,policy,64);
      std::set<int> positions;std::set<std::pair<int,int>> addresses;
      for(int w=0;w<4;++w)for(int wave=0;wave<l.waves;++wave)
        for(int r=0;r<l.RowsPerOwner();++r) {
          auto page=l.PageOf(w,wave);auto pr=l.RowOffset(w)+r;
          if(pr>=rows)throw std::runtime_error("page row overflow");
          if(!addresses.emplace(page,pr).second)throw std::runtime_error("overlapping owner regions");
          if(l.OwnerOf(page,pr)!=w || l.LocalRow(pr)!=r)throw std::runtime_error("inverse owner map");
          if(l.Valid(w,wave,r,64+count,64+count))positions.insert(l.Position(w,wave,r));
        }
      if(int(positions.size())!=count)throw std::runtime_error("logical row coverage");
      for(int page=0;page<l.pages_per_task;++page) {
        int arrivals=0;for(int w=0;w<4;++w)
          if(l.PageOf(w,page/l.pages_per_wave)==page)arrivals+=32*l.ReleaseArrivals(page);
        if(arrivals!=128)throw std::runtime_error("release quorum");
        bool full=true;
        for(int r=0;r<rows;++r)full=full && l.Valid(l.OwnerOf(page,r),page/l.pages_per_wave,l.LocalRow(r),64+count,64+count);
        if(l.Full(page,64+count,64+count)!=full)throw std::runtime_error("full page predicate");
      }
      if(policy==AttentionPagePolicy::WarpPrivate && l.needs_cta_sync)
        throw std::runtime_error("private pages require no inter-warp synchronization");
      ++cases;
    }
  std::cout<<"attention_page_layout PASS cases="<<cases<<'\n';return 0;
}
}
#ifdef TILEMEGA_LAYOUT_STANDALONE
int main(){return tilemega::tests::attention_page_layout_test::TestAttentionPageLayout(0,nullptr);}
#endif
