// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/SymbolicOracle.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
using namespace tilemega::analysis;
int main() {
 try {
  IslContext context;
  struct Case {char const* text;OracleKind kind;};
  std::vector<Case> cases={
    {"[N] -> { [p] -> [q] : 0<=p<N and 2*p<=q<=2*p+1 }",OracleKind::Rectangular},
    {"[N] -> { [q] -> [p] : 0<=p<N and 2*p<=q<=2*p+1 }",OracleKind::Unique},
    {"[N] -> { [q] -> [p] : 0<=q<N and p=q+1 }",OracleKind::Unique},
    {"[N] -> { [q] -> [i,j] : 0<=q<N and q<=i<=q+1 and 0<=j<=2 }",OracleKind::Rectangular},
    {"[N] -> { [q] -> [i,j] : 0<=q<N and ((i=0 and j=5) or (i=1 and j=0)) }",OracleKind::General},
    {"[N] -> { [q] -> [i] : 0<=q<N and 0<=i<=8 and i%2=0 }",OracleKind::General},
    {"[N] -> { [q] -> [i] : 0<=q<N and 0<=i<8 and (q=0 or i%2=0) }",OracleKind::General},
    {"[N] -> { [i,j] -> [p,q] : 0<=i<N and 0<=j<N and p=i and q=j }",OracleKind::Unique}};
  int comparisons=0;
  for(auto const& c:cases) {
    OraclePair pair(c.text);
    if(pair.forward.kind()!=c.kind)throw std::runtime_error(std::string("wrong oracle kind: ")+c.text);
    for(int n:{1,3,7}) {
      ParamBinding theta;theta.Bind("N",n);
      auto relation=CouplingRelation::FromIslText(c.text).BindParams(theta);
      std::map<std::vector<long>,std::set<std::vector<long>>> forward,reverse;
      for(auto const& [a,b]:relation.Points()){forward[a].insert(b);reverse[b].insert(a);}
      auto check=[&](SymbolicOracle const& oracle,auto const& expected) {
        for(auto const& [source,points]:expected) {
          auto image=oracle.Query(source,theta);std::set<std::vector<long>> actual;
          image.ForEach([&](auto const& p){actual.insert(p);});
          if(actual!=points || image.Count()!=long(points.size()))throw std::runtime_error("oracle differs from exact expanded set");
          ++comparisons;
        }
      };check(pair.forward,forward);check(pair.reverse,reverse);
    }
    std::cout<<"ORACLE "<<ToString(pair.forward.kind())<<" reverse="<<ToString(pair.reverse.kind())<<" structure="<<ToString(pair.structure)<<" PASS\n";
  }
  SymbolicOracle all("{ [q] -> [p] : 0<=q<4 and 0<=p<8 }");
  if(!all.IsAllBox({{0,7}}) || all.IsAllBox({{0,8}}))throw std::runtime_error("kAll proof must be equality");
  if(all.Query({4}).Count()!=0)throw std::runtime_error("out-of-domain query nonempty");
  SymbolicOracle mixed("[N] -> { [q] -> [i] : 0<=q<N and 0<=i<8 and (q=0 or i%2=0) }");
  ParamBinding theta;theta.Bind("N",3);
  if(!mixed.Query({0},theta).rectangular || mixed.Query({1},theta).rectangular)
    throw std::runtime_error("local box proof must preserve strided General fibers");
  std::cout<<"ORACLE_SET_EQUAL comparisons="<<comparisons<<" PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
