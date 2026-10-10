// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DmGemmCandidates.h>
#include <cassert>
#include <iostream>
#include <set>
#include <tuple>

namespace tilemega::tests::dm_gemm_candidates_test {
int TestDmGemmCandidates(int,char**) {
  using namespace tilemega;
  using namespace codegen;
  using namespace solver;
  frontend::ModelPlan plan;plan.dm=true;
  frontend::PlanGemm gemm;gemm.n=128;gemm.k=192;
  TargetSpec target;target.caps.cp_async=true;target.res.max_dynamic_smem_per_cta=98304;
  auto legal=[&](GemmConfig g){return DmGemmCandidateRejection(gemm,plan,g,target).empty();};
  assert(legal({16,16,16,2,1}) && legal({128,128,16,2,1}));
  assert(!legal({128,256,16,2,1}) && !legal({16,16,16,1,1}));
  assert(!legal({16,16,16,2,32}) && legal({16,16,128,2,1}));
  // Two full iterations are needed only when split=2. The shortest K tail
  // is allowed for split=1, preserving RGB/small-channel geometries.
  gemm.k=27;assert(legal({16,16,16,2,1}) && legal({16,16,16,2,2}));
  assert(!legal({16,16,16,2,4}));
  gemm.k=192;
  auto domain=DmGemmCandidates(gemm,plan,target);
  std::set<std::tuple<int,int,int,int,int>> seen;
  for(auto const& g:domain) {
    assert(legal(g) && g.stages<=8);
    assert(seen.emplace(g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k).second);
  }
  auto shared=DmGemmCandidates(gemm,plan,target,{64,32});assert(!shared.empty());
  for(auto const& g:shared)assert(g.tile_n==64 && g.tile_k==32);
  auto large=domain.size();target.res.max_dynamic_smem_per_cta=16384;
  auto small=DmGemmCandidates(gemm,plan,target);assert(!small.empty() && small.size()<large);
  for(auto const& g:small)assert(DmServingBF16SmemBytes(g.tile_m,g.tile_n,g.tile_k,g.stages)<=16384);
  target.res.max_dynamic_smem_per_cta=98304;
  plan.buffers.resize(1);auto& layout=plan.buffers[0].layout;
  layout.kind=DmLayout::kNHWC;layout.rank=4;
  plan.convolutions.resize(1);auto& conv=plan.convolutions[0];
  conv.input_layout=0;conv.r=conv.s=3;
  gemm.access.a=DmAAccess::kIm2Col;gemm.access.conv=0;
  for(unsigned channels:{4,8,16,24,32,64,96})for(int tile:{16,32,64,128}) {
    layout.physical[3]=channels;gemm.k=channels*9;
    unsigned iterations=0;
    if(channels>=unsigned(tile)) {
      for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s)
        for(unsigned c=0;c<channels;c+=tile)++iterations;
    }else {
      for(unsigned k=0;k<channels*9;k+=tile)++iterations;
    }
    assert(DmGemmIterations(gemm,plan,tile)==iterations);
    assert(legal({16,16,tile,2,1})==(channels>=unsigned(tile) || unsigned(tile)%channels==0));
  }
  layout.physical[3]=24;gemm.k=216;
  assert(DmGemmIterations(gemm,plan,16)==18);
  assert(legal({16,16,16,2,2}) && !legal({16,16,16,2,4}));
  gemm.access={};gemm.access.b=DmBAccess::kExpertIndirect;
  gemm.access.a=DmAAccess::kRowGather;gemm.access.experts=128;
  gemm.access.expert_stride=1536ull*2048;gemm.access.binding=0;
  gemm.n=1536;gemm.k=2048;gemm.chain.count=1;
  gemm.chain.operations[0].kind=DmEpilogueKind::kGatePair;
  for(int n:{16,32,64,128,256})for(int k:{16,32,64,128}) {
    bool expected=n>=32 && std::max(4*k*(16+n),4*16*n+8*16)<=
        target.res.max_dynamic_smem_per_cta;
    assert(legal({16,n,k,2,1})==expected);
  }
  gemm.n=1504;assert(!legal({16,64,32,2,1}));
  assert(!legal({16,32,64,2,1}));
  gemm.n=2048;gemm.k=768;gemm.chain.count=0;
  assert(legal({16,16,128,2,1}));
  gemm.access={};gemm.n=128;gemm.k=192;
  gemm.chain.count=1;auto& gate=gemm.chain.operations[0];
  gate.kind=DmEpilogueKind::kGatePair;gate.gate=DmGatePair::kSimpleGate;gate.unit=8;
  assert(legal({16,16,16,2,1}));gate.unit=16;assert(!legal({16,16,16,2,1}));
  gemm.chain.count=0;gemm.epilogue=frontend::PlanGemm::Epilogue::kArgmaxPartial;
  assert(legal({16,16,16,2,1}) && !legal({16,16,16,2,2}));
  target.caps.cp_async=false;assert(DmGemmCandidates(gemm,plan,target).empty());
  bool rejected=false;
  try{DmGemmCandidates(gemm,plan,target,{64,0});}catch(std::invalid_argument const&){rejected=true;}
  assert(rejected);
  std::cout<<"DM GEMM candidates: target budgets, halo K packing, expert/gate legality and shared weights PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_gemm_candidates_test
