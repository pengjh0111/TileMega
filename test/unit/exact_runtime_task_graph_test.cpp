// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Analysis/ISLContext.h>
#include <iostream>
#include <stdexcept>

int main() try {
  using namespace tilemega;
  analysis::IslContext context;
  codegen::RuntimeExactDependencyDesc descriptor{
      "[S,P] -> { [] -> [s=0,t] : 0<=t<S and 0<=P<=3; [] -> [s=1,t] : 0<=t<2*S and 0<=P<=3 }",
      "[S,P] -> { [s=1,t] -> [p=0,q] : 0<=t<2*S and q=floord(t,2) and 0<=P<=3 }","S","P"};
  int cases=0,errors=0;
  for (int seq:{1,4,128}) for (int workers:{1,4,256}) {
    auto graph=codegen::MaterializeExactRuntimeTaskGraph({seq,2*seq},descriptor,seq,3,workers);
    if (graph.successors.size()!=std::size_t(3*seq)) throw std::runtime_error("incorrect exact task count");
    for (int p=0;p<seq;++p)
      if (graph.successors[p]!=std::vector<int>{seq+2*p,seq+2*p+1})
        throw std::runtime_error("exact dependency does not match independently enumerated tasks");
    ++cases;
  }
  auto reject=[&](auto desc,std::vector<int> counts,int seq,int past) {
    int refs=context.ReferenceCount(); bool failed=false;
    try { (void)codegen::MaterializeExactRuntimeTaskGraph(counts,desc,seq,past,4); }
    catch (std::exception const&) { failed=true; }
    if (!failed || refs!=context.ReferenceCount()) throw std::runtime_error("exact runtime rejection failed or leaked");
    ++errors;
  };
  reject(descriptor,{4,7},4,3);
  reject(descriptor,{4,8},4,4);
  auto missing=descriptor; missing.dependencies=nullptr; reject(missing,{4,8},4,3);
  auto backwards=descriptor;
  backwards.dependencies="{ [s=0,t=0] -> [p=1,q=0] }";
  reject(backwards,{4,8},4,3);
  auto outside=descriptor; outside.dependencies="{ [s=1,t=0] -> [p=0,q=4] }";
  reject(outside,{4,8},4,3);
  auto unbound=descriptor; unbound.dependencies="[X] -> { [s=1,t] -> [p=0,q=0] : 0<=t<X }";
  reject(unbound,{4,8},4,3);
  if (context.ReferenceCount()) throw std::runtime_error("exact runtime adapter retained references");
  std::cout << "EXACT_RUNTIME_GRAPH cases=" << cases << " errors=" << errors << " remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
