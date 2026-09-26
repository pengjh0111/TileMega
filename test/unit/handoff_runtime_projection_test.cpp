// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/HandoffRuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
namespace tilemega::tests::handoff_runtime_projection_test {
int TestHandoffRuntimeProjection(int,char**) {
  analysis::IslContext context;
  using R=analysis::CouplingRelation;using solver::RuntimeHandoff;
  auto read=[](char const* s){return R::FromIslText(s);};
  auto same=[](R const& a,R const& b){assert(a.IsSubset(b)&&b.IsSubset(a));};
  auto tasks=read("{ [] -> [s,t] : 0<=s<4 and 0<=t<4 }");
  auto deps=read("{ [1,t] -> [0,t] : 0<=t<4; [2,t] -> [1,u] : 0<=t<4 and 2*floor(t/2)<=u<2*floor(t/2)+2; [3,t] -> [2,t] : 0<=t<4 }");
  auto map=read("{ [t] -> [u] : 0<=t<4 and 2*floor(t/2)<=u<2*floor(t/2)+2 }");
  auto result=solver::ProjectHandoffRuntime(tasks,deps,4,{{1,2,"recompute",map}});
  same(result.dependencies,read("{ [1,t] -> [0,u] : 0<=t<4 and 2*floor(t/2)<=u<2*floor(t/2)+2; [2,t] -> [1,t] : 0<=t<4 }"));
  assert((result.surviving_stages==std::vector<int>{0,2,3}));
  tasks=read("{ [] -> [s,t] : (s=0 or s=1) and 0<=t<6; [] -> [s,t] : (s=2 or s=3) and 0<=t<2 }");
  deps=read("{ [1,t] -> [0,t] : 0<=t<6; [2,t] -> [1,u] : 0<=t<2 and 3*t<=u<3*t+3; [3,t] -> [2,t] : 0<=t<2 }");
  map=read("{ [t] -> [u] : 0<=t<2 and 3*t<=u<3*t+3 }");
  result=solver::ProjectHandoffRuntime(tasks,deps,4,{{1,2,"last_arriver",map}});
  same(result.dependencies,read("{ [1,t] -> [0,t] : 0<=t<6; [2,t] -> [1,u] : 0<=t<2 and 3*t<=u<3*t+3 }"));
  assert((result.surviving_stages==std::vector<int>{0,1,3}));
  auto reject=[&](auto action){bool bad=false;try{action();}catch(std::invalid_argument const&){bad=true;}assert(bad);};
  reject([&]{solver::ProjectHandoffRuntime(tasks,deps,4,{{1,2,"last_arriver",read("{ [t] -> [u] : 0<=t<2 and 0<=u<6 }")}});});
  reject([&]{solver::ProjectHandoffRuntime(tasks,deps,4,{{1,2,"smem_direct",map}});});
  reject([&]{solver::ProjectHandoffRuntime(tasks,deps,4,{{1,2,"unknown",map}});});
  assert(context.ReferenceCount()==0);
  std::cout<<"HANDOFF_RUNTIME multirow_recompute=PASS complete_fanin=PASS negative=3 remaining=0\n";
  return 0;
}
}
