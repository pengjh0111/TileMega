// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeOwnership.h>
#include <cassert>
#include <set>
int main() {
  using namespace tilemega::codegen;
  for(int m:{1,2,3,8,16})for(int n:{1,3,64})for(int chunks:{1,2,4})
    for(bool cg:{false,true}) {
      std::set<int> seen;
      for(int ordinal=0;ordinal<m*n*chunks;++ordinal) {
        int task=NMajorGemmTask(ordinal,m,n,chunks,cg);
        auto before=DecodeSplitTask(ordinal,m*n,chunks,cg);
        auto after=DecodeSplitTask(task,m*n,chunks,cg);
        assert(after.tile/n==before.tile%m && after.tile%n==before.tile/m);
        assert(after.chunk==before.chunk);
        assert(task>=0 && task<m*n*chunks && seen.insert(task).second);
      }
      assert(seen.size()==unsigned(m*n*chunks));
      assert(NMajorGemmTask(m*n*chunks,m,n,chunks,cg)==m*n*chunks);
    }
  assert(NMajorGemmTask(0,0,1,1)==-1);
  assert(NMajorGemmTask(0,INT32_MAX,2,1)==-1);
}
