// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DmGemmTraits.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>

namespace tilemega::tests::dm_gemm_traits_test {
int TestDmGemmTraits(int,char**) {
  using namespace tilemega::solver;
  unsigned all=0,added=0;
  for(int m:{16,32,64,128})for(int n:{16,32,64,128,256})
    for(int k:{16,32,64,128})for(int stages:{2,3,8}) {
      bool legal=m*n<=16384;
      assert(DmServingBF16ShapeLegal(m,n,k,stages)==legal);
      auto traits=DmServingBF16Traits(m,n,k,stages);
      assert(traits.shape_legal==legal && traits.threads==128);
      int staged_bytes=2*(m+n)*k*stages;
      int partial_bytes=4*m*n*(m==16 && n==16?2:1);
      int statistics_bytes=4*(m*n+2*m);
      assert(traits.smem_bytes==std::max({staged_bytes,partial_bytes,statistics_bytes}));
      assert(traits.alignment.a==8 && traits.alignment.b==8);
      if(legal && stages==2) {++all;added+=!ServingBF16ShapeLegal(m,n,k,stages);}
    }
  assert(all==76 && added==46);
  assert(!ServingBF16ShapeLegal(16,16,64,2));
  assert(!ServingBF16ShapeLegal(16,128,32,2));
  assert(DmServingBF16SmemBytes(16,16,16,2)==2048);
  assert(DmServingBF16SmemBytes(128,128,16,2)==66560);
  for(auto shape:{std::array<int,4>{8,16,16,2},{16,8,16,2},{16,16,8,2},
                  {16,16,16,1},{128,256,16,2}})
    assert(!DmServingBF16ShapeLegal(shape[0],shape[1],shape[2],shape[3]));
  std::cout<<"DM GEMM traits: 76 legal geometries, 46 additions, K partial/statistics storage and legacy domain PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_gemm_traits_test
