// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ServingSeedGeometry.h>
#include <cassert>
#include <cstdio>
int main(){using namespace tilemega;using Op=frontend::PlanGemm::Epilogue;
 // Llama-3.1-8B decode shapes: qkv, o, gate_up, down, lm_head.
 frontend::ModelPlan plan;plan.gemms.resize(5);
 auto set=[&](int i,Op e,unsigned n,unsigned k){plan.gemms[i].epilogue=e;plan.gemms[i].n=n;plan.gemms[i].k=k;};
 set(0,Op::kStore,6144,4096);set(1,Op::kResidual,4096,4096);set(2,Op::kSwiGLU,28672,4096);
 set(3,Op::kResidual,4096,14336);set(4,Op::kArgmaxPartial,128256,4096);
 assert(solver::SeedGeometryFamily(plan,1)=="o" && solver::SeedGeometryFamily(plan,3)=="down");
 auto policy=solver::ParseSeedGeometryPolicy(json::Parse(
   R"({"base":{"tile_n":128,"tile_k":64,"stages":4,"impl":"mma16","min_tasks":118,"max_split":16},
       "families":{"lm_head":{"split":1},"gate_up":{"tile_n":64}}})"));
 std::vector<frontend::GemmGranularity> g(5,{16,128,128,2,1,0});
 solver::ApplySeedGeometryPolicy(plan,g,policy,132,true);
 // 48/32/448/32/1002 N tiles: narrow GEMMs split until they cover 118 tasks.
 assert(g[0].split_k==4 && g[1].split_k==4 && g[2].split_k==1 && g[3].split_k==4 && g[4].split_k==1);
 assert(g[2].tile_n==64 && g[0].stages==4 && g[0].tile_k==64);
 assert(solver::SeedSplitK(32,14336,64,118,2)==2 && solver::SeedSplitK(2,192,64,132,16)==1);
 // Qwen3-0.6B gate_up (N=6144): shrink TN instead of splitting the SwiGLU GEMM.
 frontend::ModelPlan qwen;qwen.gemms.resize(1);qwen.gemms[0].epilogue=Op::kSwiGLU;qwen.gemms[0].n=6144;qwen.gemms[0].k=1024;
 std::vector<frontend::GemmGranularity> q(1,{16,128,64,2,1,0});
 solver::ApplySeedGeometryPolicy(qwen,q,solver::ParseSeedGeometryPolicy(json::Parse(
   R"({"base":{"tile_n":128,"min_tasks":118},"families":{"gate_up":{"max_split":1,"min_tile_n":32}}})")),132,true);
 assert(q[0].tile_n==32 && q[0].split_k==1);
 int rejected=0;
 for(auto text:{R"({"base":{"stages":1}})",R"({"base":{"impl":"gemv","tile_n":128}})",
                R"({"families":{"lm_head":{"split":2}}})",R"({"base":{"impl":"wgmma","stages":4}})",
                R"({"base":{"tile_k":96}})",R"({"base":{"bogus":1}})"}) {
  auto copy=std::vector<frontend::GemmGranularity>(5,{16,128,64,2,1,0});
  try{solver::ApplySeedGeometryPolicy(plan,copy,solver::ParseSeedGeometryPolicy(json::Parse(text)),132,true);}
  catch(std::invalid_argument const&){++rejected;}
 }
 assert(rejected==6);
 std::puts("serving seed geometry: SM-filling split, TN shrink, family overrides, six rejection contracts PASS");
}
