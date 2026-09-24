// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/VariantResourceCache.h>
#include <iostream>
using namespace tilemega;
int main(){
 solver::SolverTiming timing;int calls=0;
 solver::VariantResourceCache cache([&](std::string const&,solver::GemmConfig const* g,solver::ScalarType){
   ++calls;return g ? solver::VariantResources{38,3072,128,true}:solver::VariantResources{40,16384,128,true};
 },&timing);
 TargetSpec target;target.res.regs_per_sm=65536;target.res.max_smem_per_sm=102400;
 target.res.max_dynamic_smem_per_cta=101376;target.res.max_threads_per_sm=1536;
 std::vector<solver::OperatorClass> classes={{"a",{0},{"a"}},{"b",{1},{"b"}}};
 std::vector<solver::GemmConfig> g(2,{32,16,16,2,1});
 auto a=cache.Estimate(classes,g,target,solver::ScalarType::kBF16);
 auto b=cache.Estimate(classes,g,target,solver::ScalarType::kBF16);
 if(a.resident_limit!=6 || b.resident_limit!=6 || calls!=3)return 1;
 if(solver::VariantResourceCache::ResidentLimit({256,16384,128,0},target)!=2)return 2;
 if(solver::VariantResourceCache::ResidentLimit({40,102400,128,0},target)!=0)return 3;
 std::cout<<"RESOURCE_CACHE calls="<<calls<<" repeat=0 resident=6 register_limited=2 oversized=0 PASS\n";
}
