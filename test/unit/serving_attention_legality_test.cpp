// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ServingAttentionLegality.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace tilemega::tests::serving_attention_legality_test {
int TestServingAttentionLegality(int,char**) {
  using namespace tilemega::solver;
  int checks=0;
  for(int page:{8192,16384})for(int d:{64,128})for(int ec:{32,64,128,256,512,1088}) {
    tilemega::codegen::AttentionPageLayout max_layout(page/(4*d),ec,
        tilemega::codegen::AttentionPagePolicy::Packed);
    for(int count=1;count<=ec;++count) {
      tilemega::codegen::AttentionPageLayout live(page/(4*d),count,
          tilemega::codegen::AttentionPagePolicy::Packed);
      if(live.pages_in_flight>max_layout.pages_in_flight)
        throw std::runtime_error("maximum chunk did not bound all partial chunks");
    }
    for(int slots=1;slots<=6;++slots) {
      if(ServingAttentionFrontierFits(page,slots,d,ec)!=(max_layout.pages_in_flight<=slots))
        throw std::runtime_error("attention frontier legality mismatch");
      ++checks;
    }
  }
  if(ServingAttentionFrontierFits(4096,4,128,32) || ServingAttentionFrontierFits(8192,4,0,32) ||
     ServingAttentionFrontierFits(8192,4,64,0) || ServingAttentionFrontierFits(8192,0,64,32))
    throw std::runtime_error("invalid attention shape accepted");
  // Audit existing g-major grid-stride coverage without silently changing it.
  for(int grid:{128,170})for(int batch:{1,16})for(int ec:{32,64})for(int past:{64,575,1000}) {
    int chunks=(1088+ec-1)/ec,active=0;std::vector<int> assigned(grid);
    for(int t=0;t<8*batch*chunks;++t)if((t%chunks)*ec<=past){++assigned[t%grid];++active;}
    int max=*std::max_element(assigned.begin(),assigned.end());
    std::printf("ATTENTION_DISTRIBUTION grid=%d batch=%d Ec=%d past=%d active=%d max_per_worker=%d ideal=%d\n",
        grid,batch,ec,past,active,max,(active+grid-1)/grid);
  }
  std::printf("attention frontier: %d legality checks PASS\n",checks);return 0;
}
}
#ifdef TILEMEGA_ATTENTION_LEGALITY_STANDALONE
int main(){return tilemega::tests::serving_attention_legality_test::TestServingAttentionLegality(0,nullptr);}
#endif
