// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/SymbolicOracle.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include "../../lib/Analysis/OracleExpression.h"
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
using namespace tilemega::analysis;
int main() {
  IslContext context;
  {
    SymbolicOracle release("{ [q] -> [p] : 0<=q<4 and 0<=p<=2*q and p%2=0 }");
    for(long q=0;q<4;++q)if(release.MaximumLinear({q})!=2*q)
      throw std::runtime_error("General scalar release maximum changed a strided fiber");
    if(release.MaximumLinear({5})!=-1)throw std::runtime_error("empty release fiber acquired a predecessor");
  }
 try {
  struct Case {char const* text;OracleKind kind;};
  std::vector<Case> cases={
    {"[N] -> { [p] -> [q] : 0<=p<N and 2*p<=q<=2*p+1 }",OracleKind::Rectangular},
    {"[N] -> { [q] -> [p] : 0<=p<N and 2*p<=q<=2*p+1 }",OracleKind::Unique},
    {"[N] -> { [q] -> [p] : 0<=q<N and p=q+1 }",OracleKind::Unique},
    {"[N] -> { [q] -> [] : 0<=q<N }",OracleKind::Unique},
    {"[N] -> { [q] -> [i,j] : 0<=q<N and q<=i<=q+1 and 0<=j<=2 }",OracleKind::Rectangular},
    {"[N] -> { [q] -> [i,j] : 0<=q<N and ((i=0 and j=5) or (i=1 and j=0)) }",OracleKind::General},
    {"[N] -> { [q] -> [i] : 0<=q<N and 0<=i<=8 and i%2=0 }",OracleKind::General},
    {"[N] -> { [q] -> [i] : 0<=q<N and 0<=i<8 and (q=0 or i%2=0) }",OracleKind::General},
    {"[N] -> { [q] -> [i] : 0<=q<N and (i=0 or i=1000000) }",OracleKind::General},
    {"[N] -> { [i,j] -> [p,q] : 0<=i<N and 0<=j<N and p=i and q=j }",OracleKind::Unique}};
  int comparisons=0,releases=0;
  {
    auto* f=isl_pw_aff_read_from_str(context.raw(),"[x] -> { [(floor(x/3))] : -20<=x<=20 and x%2=0 }");
    auto program=OracleProgram::Build(isl_pw_aff_domain(isl_pw_aff_copy(f)),{f});
    for(long x=-22;x<=22;++x) {
      bool nonempty=false;std::vector<long> values;
      if(!program.Eval({x},nonempty,values))throw std::runtime_error("integer expression was not compiled");
      bool expected=x>=-20 && x<=20 && x%2==0;
      long quotient=x/3-(x<0 && x%3!=0);
      if(nonempty!=expected || (expected && values!=std::vector<long>{quotient}))
        throw std::runtime_error("compiled expression lost floor division or domain holes");
    }
    isl_pw_aff_free(f);
    std::cout<<"ORACLE_EXPRESSION signed_floor_domain_holes=45 PASS\n";
  }
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
          if(!points.empty() && points.begin()->size()==1) {
            auto release=oracle.LinearRelease(source,theta);
            long maximum=points.rbegin()->front();
            bool prefix=points.begin()->front()==0 && long(points.size())==maximum+1;
            if(release.maximum!=maximum || release.prefix!=prefix)throw std::runtime_error("scalar release differs from exact expanded fiber");
            ++releases;
          }
          ++comparisons;
        }
      };check(pair.forward,forward);check(pair.reverse,reverse);
    }
    std::cout<<"ORACLE "<<ToString(pair.forward.kind())<<" reverse="<<ToString(pair.reverse.kind())<<" structure="<<ToString(pair.structure)<<" PASS\n";
  }
  SymbolicOracle all("{ [q] -> [p] : 0<=q<4 and 0<=p<8 }");
  if(!all.IsAllBox({{0,7}}) || all.IsAllBox({{0,8}}))throw std::runtime_error("kAll proof must be equality");
  if(all.Query({4}).Count()!=0)throw std::runtime_error("out-of-domain query nonempty");
  SymbolicOracle unique("[N] -> { [p] -> [q] : 0<=p<N and q=p+1 }");
  ParamBinding three;three.Bind("N",3);
  if(unique.Query({-1},three).Count()!=0 || unique.Query({3},three).Count()!=0)
    throw std::runtime_error("out-of-domain affine evaluation nonempty");
  SymbolicOracle folded("[N] -> { [q] -> [p] : 0<=p<N and 2*p<=q<=2*p+1 }");
  for(long q=-1;q<=6;++q) {
    auto image=folded.Query({q},three);
    bool valid=q>=0 && q<6;
    if(image.Count()!=int(valid) ||
       (valid && image.points.front()!=std::vector<long>{q/2}))
      throw std::runtime_error("bound many-to-one fiber lost its exact domain");
  }
  SymbolicOracle mixed("[N] -> { [q] -> [i] : 0<=q<N and 0<=i<8 and (q=0 or i%2=0) }");
  ParamBinding theta;theta.Bind("N",3);
  if(mixed.Query({0},theta).Count()!=8 || mixed.Query({1},theta).Count()!=4)
    throw std::runtime_error("local enumeration must preserve strided General fibers");
  SymbolicOracle theta_box("[N] -> { [q] -> [i] : 0<=q<3 and 0<=i<8 and (N=1 or i%2=0) }");
  if(theta_box.kind()!=OracleKind::General)throw std::runtime_error("theta-dependent relation must remain General");
  for(int n:{1,3,1,3}) {
    ParamBinding bound;bound.Bind("N",n);
    for(int q:{0,1,2,3}) {
      auto image=theta_box.Query({q},bound);std::set<std::vector<long>> actual,expected;
      image.ForEach([&](auto const& p){actual.insert(p);});
      if(q<3)for(int i=0;i<8;++i)if(n==1 || i%2==0)expected.insert({i});
      if(actual!=expected)throw std::runtime_error("theta box cache reused stale bounds");
      auto release=theta_box.LinearRelease({q},bound);
      long maximum=expected.empty()?-1:expected.rbegin()->front();
      bool prefix=expected.empty() || (expected.begin()->front()==0 && long(expected.size())==maximum+1);
      if(release.maximum!=maximum || release.prefix!=prefix)throw std::runtime_error("theta-bound scalar release changed");
      ++releases;++comparisons;
    }
  }
  std::cout<<"ORACLE_RELEASE_EQUAL comparisons="<<releases<<" PASS\n";
  std::cout<<"ORACLE_SET_EQUAL comparisons="<<comparisons<<" PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
