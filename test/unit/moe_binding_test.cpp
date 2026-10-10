// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MoeBinding.h>
#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

namespace tilemega::tests::moe_binding_test {
int TestMoeBinding(int,char**) {
  using namespace tilemega::codegen;
  for(unsigned tokens:{1,2,4,8,16,32,64,128,256,512,1024,2048,4096})
    for(unsigned block_rows:{16,32,64,128}) {
      unsigned capacity=0,slots=0;
      assert(MoeVirtualCapacity(tokens,8,128,block_rows,true,&capacity));
      assert(capacity==(tokens*8+block_rows-1)/block_rows+std::min(128u,tokens*8));
      assert(MoeVirtualCapacity(tokens,8,128,1,false,&slots) && slots==tokens*8);
      for(unsigned seed:{0,1,7,31}) {
        std::vector<unsigned> counts(128);
        for(unsigned row=0;row<tokens*8;++row)++counts[(row*seed+row/8)%128];
        unsigned used=0;
        for(auto count:counts)used+=(count+block_rows-1)/block_rows;
        assert(used<=capacity);
      }
    }
  unsigned capacity=0;
  assert(!MoeVirtualCapacity(0,8,128,16,true,&capacity));
  assert(!MoeVirtualCapacity(1,129,128,16,true,&capacity));
  assert(!MoeVirtualCapacity(1,8,128,0,true,&capacity));
  assert(!MoeVirtualCapacity(std::numeric_limits<unsigned>::max(),8,128,1,true,&capacity));
  MoeBindingRecord records[]={{3,2,7,1},
    {0xffffffffu,0xffffffffu,0xffffffffu,0}};
  MoeBindingRow rows[9]{};
  MoeBindingView view{records,rows,2,9,128,16};
  MoeBindingRecord active,empty;
  assert(view.Lookup(0,&active)==MoeBindingStatus::kActive);
  assert(view.Lookup(1,&empty)==MoeBindingStatus::kEmpty);
  assert(view.Lookup(2,&empty)==MoeBindingStatus::kInvalid);
  assert(view.Lookup(0,nullptr)==MoeBindingStatus::kInvalid);
  std::uint64_t offset=0;
  assert(MoeExpertOffset(active,9'437'184/2,&offset) && offset==3ull*9'437'184/2);
  assert(!MoeExpertOffset(empty,9'437'184/2,&offset));
  assert(!MoeExpertOffset(active,std::numeric_limits<std::uint64_t>::max(),&offset));
  for(auto wrong:std::vector<MoeBindingRecord>{{128,2,7,1},{3,2,8,1},
      {3,0,17,1},{3,0,0,1},{3,0,1,2}}) {
    records[0]=wrong;
    assert(view.Lookup(0,&active)==MoeBindingStatus::kInvalid);
  }
  return 0;
}
}
