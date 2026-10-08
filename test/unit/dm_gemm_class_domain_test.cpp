// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DmGemmClassDomain.h>
#include <cassert>
#include <iostream>
#include <set>
#include <tuple>

namespace tilemega::tests::dm_gemm_class_domain_test {
int TestDmGemmClassDomain(int,char**) {
  using namespace tilemega;
  using namespace solver;
  frontend::ModelPlan plan;plan.dm=true;plan.serving=true;plan.dtype="bf16";
  plan.gemms.resize(2);plan.stages.resize(2);
  for(unsigned i=0;i<2;++i) {
    plan.stages[i].gemm=i;plan.gemms[i].n=96;plan.gemms[i].k=16;
  }
  auto& gemm=plan.gemms[0];gemm.access.rows_per_batch=12544;
  TargetSpec target;target.caps.cp_async=true;target.res.max_dynamic_smem_per_cta=98304;
  auto contains=[](auto const& domain,int m,int n,int k,int stages=2) {
    return std::any_of(domain.candidates.begin(),domain.candidates.end(),[&](auto const& g) {
      return g.tile_m==m && g.tile_n==n && g.tile_k==k && g.stages==stages && g.split_k==1;
    });
  };
  assert(DmGemmActiveRows(gemm,plan,0,1,1)==12544);
  auto spatial=DmClassCandidates({0},plan,target,1,1);
  plan.serving=false;plan.forward=true;
  auto forward=DmClassCandidates({0},plan,target,1,1);
  assert(forward.candidates.size()==spatial.candidates.size());
  plan.serving=true;plan.forward=false;
  assert(contains(spatial,128,128,16));
  // RGB/output tails have one K iteration. They require the minimum
  // two-stage body, while narrower N=16 avoids padded output columns.
  gemm.n=3;gemm.k=27;gemm.access.rows_per_batch=1;
  auto head=DmClassCandidates({0},plan,target,1,1);
  assert(!head.candidates.empty() && contains(head,16,16,32));
  assert(contains(head,32,16,32) && contains(head,16,32,32));
  auto batched=DmClassCandidates({0},plan,target,32,1);
  assert(contains(batched,32,16,32) && contains(batched,64,16,32));
  gemm.access.rows_per_batch=0;plan.stages[0].batch_rows=true;
  assert(DmGemmActiveRows(gemm,plan,0,32,128)==32);
  plan.stages[0].batch_rows=false;
  assert(DmGemmActiveRows(gemm,plan,0,8,128)==1024);
  gemm.n=2048;gemm.k=768;gemm.access.b=codegen::DmBAccess::kExpertIndirect;
  gemm.access.experts=128;gemm.access.expert_stride=2048*768;
  gemm.access.binding=0;gemm.access.block_rows=1;gemm.access.binding_blocks=32768;
  auto slot=DmClassCandidates({0},plan,target,1,4096);
  assert(contains(slot,16,16,16));
  assert(contains(slot,32,16,16));
  assert(DmGemmActiveRows(gemm,plan,0,1,4096)==1);
  gemm.access.block_rows=64;
  auto group=DmClassCandidates({0},plan,target,1,4096);
  assert(contains(group,64,16,16) && contains(group,128,16,16));
  assert(DmGemmActiveRows(gemm,plan,0,1,4096)==64);
  gemm.access={};gemm.n=96;gemm.k=192;gemm.access.rows_per_batch=1;
  plan.gemms[1].access.rows_per_batch=12544;
  auto members=DmClassCandidates({0,1},plan,target,1,1);
  assert(contains(members,128,128,16));
  auto unpruned=members;
  using Key=std::tuple<int,int,int,int,int>;
  auto key=[](GemmConfig const& g){return Key{g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k};};
  std::set<Key> expected,actual;
  for(auto const& g:DmGemmCandidates(gemm,plan,target))
    if(DmGemmCandidateRejection(plan.gemms[1],plan,g,target).empty())expected.insert(key(g));
  for(auto const& g:unpruned.candidates)assert(actual.insert(key(g)).second);
  assert(expected==actual && unpruned.removed_r2==0);
  assert(unpruned.raw==unpruned.removed_r1+unpruned.candidates.size());
  assert(members.raw==members.removed_r1+members.removed_r2+members.candidates.size());
  auto packed=DmClassCandidates({0,1},plan,target,1,1,{{0,{64,32}},{1,{64,32}}});
  assert(!packed.candidates.empty());
  for(auto const& g:packed.candidates)assert(g.tile_n==64 && g.tile_k==32);
  assert(DmClassCandidates({0,1},plan,target,1,1,{{0,{64,32}},{1,{32,32}}}).candidates.empty());
  for(auto const& ids:std::vector<std::vector<std::size_t>>{{},{0,0}}) {
    bool rejected=false;
    try{DmClassCandidates(ids,plan,target,1,1);}catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);
  }
  gemm.access.rows_per_batch=0xffffffffu;
  bool overflow=false;
  try{DmClassCandidates({0},plan,target,64,1);}catch(std::overflow_error const&){overflow=true;}
  assert(overflow);
  std::cout<<"DM class domain: spatial rows, one-iteration tails, slot/group M, class intersection and shared layout PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_gemm_class_domain_test
