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
      // DM legality was tightened on this branch (4d134101f) to the shapes
      // ServingDmGemm actually implements: m in {16,32,64}, n in {32,64,128},
      // k in {32,64,128}, m*n <= 8192. The expectation here still described
      // the pre-port rule (any m/n with m*n <= 16384, k down to 16), so this
      // test had been failing since that commit.
      bool legal=(m==16||m==32||m==64) && (n==32||n==64||n==128) &&
          (k==32||k==64||k==128) && m*n<=8192;
      assert(DmServingBF16ShapeLegal(m,n,k,stages)==legal);
      auto traits=DmServingBF16Traits(m,n,k,stages);
      assert(traits.shape_legal==legal && traits.threads==128);
      // DmServingBF16SmemBytes was reduced to the FP32 accumulator tile plus
      // per-row statistics (4*m*n + 8*m) when the DM mainloop moved to the
      // source-policy MMA loop; it no longer stages A/B through shared memory,
      // so the old three-way maximum over staged/partial/statistics bytes is
      // not what it returns.
      assert(traits.smem_bytes==4*m*n+8*m);
      assert(traits.alignment.a==8 && traits.alignment.b==8);
      if(legal && stages==2) {++all;added+=!ServingBF16ShapeLegal(m,n,k,stages);}
    }
  // `added` counts DM-legal depth-2 shapes the dense rule rejects. Since
  // MX-3 the dense rule also bounds pipeline depth by registers
  // (ServingGemmDepthFits), so wider DM tiles at depth 2 join it: 9 -> 19.
  assert(all==27 && added==19);
  assert(!ServingBF16ShapeLegal(16,16,64,2));
  assert(!ServingBF16ShapeLegal(16,128,32,2));
  // A dense tile must satisfy the rule of the body that will run it. Under
  // TILEMEGA_DM_SUPPORT the serving GEMM body executes through ServingDmGemm,
  // so tile_k=32 is legal; the codegen variant used to assert the non-DM rule
  // instead and rejected shapes the MoE solver had legitimately selected
  // (ServingGemmConfig<Xcore1000,16,64,32,2>). Keep the two predicates
  // disagreeing on exactly that shape, so a regression is visible here.
  assert(DmServingBF16ShapeLegal(16,64,32,2));
  assert(!ServingBF16ShapeLegal(16,64,32,2));
  // k=16 is outside both rules and must stay rejected either way.
  assert(!DmServingBF16ShapeLegal(16,64,16,2));
  assert(!ServingBF16ShapeLegal(16,64,16,2));
  assert(DmServingBF16SmemBytes(16,16,16,2)==4*16*16+8*16);
  assert(DmServingBF16SmemBytes(128,128,16,2)==4*128*128+8*128);
  for(auto shape:{std::array<int,4>{8,16,16,2},{16,8,16,2},{16,16,8,2},
                  {16,16,16,1},{128,256,16,2}})
    assert(!DmServingBF16ShapeLegal(shape[0],shape[1],shape[2],shape[3]));
  std::cout<<"DM GEMM traits: 27 legal geometries, 19 beyond the non-DM rule, "
             "accumulator-only shared storage and legacy domain PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_gemm_traits_test
