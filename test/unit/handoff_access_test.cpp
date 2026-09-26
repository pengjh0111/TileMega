// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/HandoffAccess.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
namespace tilemega::tests::handoff_access_test {
using namespace analysis;
int TestHandoffAccess(int,char**) {
  IslContext context;
  auto rel=[](char const* s){return CouplingRelation::FromIslText(s);};
  auto reject=[](auto&& call){bool rejected=false;try{call();}catch(std::invalid_argument const&){rejected=true;}assert(rejected);};
  TaskAccesses p,c,available;
  p.writes["mid"]=rel("[B] -> { [r] -> [r,h] : 0<=r<B and 0<=h<128 }");
  p.reads["x"]=p.writes["mid"];
  p.reads["w"]=rel("[B] -> { [r] -> [h] : 0<=r<B and 0<=h<128 }");
  c.reads["mid"]=rel("[B] -> { [m,n] -> [r,h] : 0<=16m<B and 0<=n<8 and 16m<=r<16m+16 and r<B and 0<=h<128 }");
  c.writes["out"]=rel("[B] -> { [m,n] -> [r,j] : 0<=16m<B and 0<=n<8 and 16m<=r<16m+16 and r<B and 32n<=j<32n+32 }");
  available.reads["x"]=c.reads["mid"];
  available.reads["w"]=rel("[B] -> { [m,n] -> [h] : 0<=16m<B and 0<=n<8 and 0<=h<128 }");
  auto good=ProveRecompute(p,c,{"mid"},available);
  assert(!good.consumer_to_producer.IsSingleValued()); // B16 spans 16 row phases.
  assert(good.composed.reads.at("x").IsSubset(c.reads.at("mid")) &&
         c.reads.at("mid").IsSubset(good.composed.reads.at("x")));
  auto missing=available;missing.reads.erase("x");
  reject([&]{ProveRecompute(p,c,{"mid"},missing);});
  auto bad=p;bad.reads["x"]=rel("[B] -> { [r] -> [i,h] : 0<=r<B and 0<=i<B and 0<=h<128 }");
  reject([&]{ProveRecompute(bad,c,{"mid"},available);});
  TaskAccesses partial,reduce;
  partial.writes["partial"]=rel("{ [m,s] -> [s,m,j] : 0<=m<4 and 0<=s<4 and 0<=j<32 }");
  reduce.reads["partial"]=rel("{ [m] -> [s,m,j] : 0<=m<4 and 0<=s<4 and 0<=j<32 }");
  reduce.writes["out"]=rel("{ [m] -> [m,j] : 0<=m<4 and 0<=j<32 }");
  auto collapse=rel("{ [s,m,j] -> [m,j] }");
  auto fanin=ProveLastArriver(partial,reduce,{{"partial",collapse}},"out");
  assert(fanin.consumer_to_producer.Reverse().IsSingleValued());
  auto incomplete=reduce;
  incomplete.reads["partial"]=rel("{ [m] -> [s,m,j] : 0<=m<4 and 0<=s<3 and 0<=j<32 }");
  reject([&]{ProveLastArriver(partial,incomplete,{{"partial",collapse}},"out");});
  auto strided=reduce;
  strided.reads["partial"]=rel("{ [m] -> [s,m,j] : 0<=m<4 and 0<=s<4 and 0<=j<32 and j%2=0 }");
  reject([&]{ProveLastArriver(partial,strided,{{"partial",collapse}},"out");});
  TaskAccesses direct;
  direct.reads["mid"]=p.writes["mid"];direct.writes["y"]=p.writes["mid"];
  auto pp=rel("[B] -> { [r] -> [w,s] : 0<=r<B and w=r%4 and s=2*floor(r/4) }");
  auto cp=rel("[B] -> { [r] -> [w,s] : 0<=r<B and w=r%4 and s=2*floor(r/4)+1 }");
  ProveDirectHandoff(p,direct,{"mid"},pp,cp);
  reject([&]{ProveDirectHandoff(p,direct,{"mid"},pp,pp);});
  auto wrong=rel("[B] -> { [r] -> [w,s] : 0<=r<B and w=(r+1)%4 and s=2*floor(r/4)+1 }");
  reject([&]{ProveDirectHandoff(p,direct,{"mid"},pp,wrong);});
  std::cout<<"PASS handoff access proofs: multirow recompute, complete reduction fibre, same-worker adjacency; six negative controls\n";
  return 0;
}
}
