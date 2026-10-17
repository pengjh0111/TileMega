// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ServingImplementation.h>
#include <cassert>
#include <cstdio>
int main(){using namespace tilemega;
 frontend::ModelPlan plan;plan.gemms.resize(5);using Op=frontend::PlanGemm::Epilogue;
 plan.gemms[0].epilogue=Op::kStore;plan.gemms[1].epilogue=Op::kResidual;plan.gemms[1].k=64;plan.gemms[1].n=128;
 plan.gemms[2].epilogue=Op::kSwiGLU;plan.gemms[3].epilogue=Op::kResidual;plan.gemms[3].k=128;plan.gemms[3].n=64;
 plan.gemms[4].epilogue=Op::kArgmaxPartial;
 for(int i=0;i<5;++i)assert(solver::ServingGemmFamily(plan.gemms[i])==i);
 auto geometry=std::vector<frontend::GemmGranularity>(5,{16,128,64,2,1,0});
 solver::ApplyServingSeedImplementations(plan,geometry,{2,2,2,2,2},true,1,true,16384);
 for(auto g:geometry)assert(g.impl==2);
 solver::ApplyServingSeedImplementations(plan,geometry,{-1,0,-1,0,-1},true,1,true,16384);
 assert(geometry[0].impl==2 && geometry[1].impl==0 && geometry[2].impl==2 && geometry[3].impl==0);
 int rejected=0;
 for(int arm=0;arm<5;++arm){auto g=geometry;
  if(arm==1)g[0].tile_k=128;if(arm==2)g[0].tile_m=64;if(arm==3)g[0].stages=9;
  try{solver::ApplyServingSeedImplementations(plan,g,{2,2,2,2,2},arm!=0,arm==4?17:1,true,16384);}
  catch(std::invalid_argument const&){++rejected;}
 }
 assert(rejected==5);
 auto m8=std::vector<frontend::GemmGranularity>(5,{8,128,64,2,1,0});
 solver::ApplyServingSeedImplementations(plan,m8,{2,2,2,2,2},true,8,true,16384);
 for(auto g:m8)assert(g.impl==2 && g.tile_m==8);
 bool m8_batch_rejected=false;
 try{solver::ApplyServingSeedImplementations(plan,m8,{2,2,2,2,2},true,9,true,16384);}
 catch(std::invalid_argument const&){m8_batch_rejected=true;}
 assert(m8_batch_rejected);
 std::puts("serving seed: five families, explicit propagation, five rejection contracts PASS");
}
