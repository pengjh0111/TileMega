// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/PageLayout.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::dm_page_resources_test {
int TestDmPageResources(int,char**) {
  using namespace tilemega::solver;
  unsigned shapes=0;
  for(int m:{16,32,64,128})for(int n:{16,32,64,128,256})for(int k:{16,32,64,128}) {
    if(m*n>16384)continue;
    bool added=n==16 || k<64;
    int activation=2*m*k*(added?2:4);
    int partials=4*m*n*(m==16 && n==16?2:1);
    int statistics=4*m*n+8*m;
    assert(DmServingPageActivationBytes(m,n,k)==activation);
    assert(DmServingPageScratchBytes(m,n)==std::max(partials,statistics));
    assert(DmServingPageWorkspaceBytes(m,n,k)==std::max({activation,partials,statistics}));
    auto dm=PageLayout::ServingWorkspace({{m,n,k}},{},false,true);
    auto legacy=PageLayout::ServingWorkspace({{m,n,k}},{});
    assert(dm.first==activation && dm.second==std::max(partials,statistics));
    assert(legacy.first==8*m*k && legacy.second==4*m*n+4*m);
    ++shapes;
  }
  assert(shapes==76);
  assert(DmServingPageWorkspaceBytes(16,16,16)==2048);
  auto reduced=PageLayout::ServingWorkspace({{128,16,128}},{},false,true);
  assert(reduced.first==65536 && reduced.second==9216);
  for(int stages:{2,3,4,8})
    assert(DmServingPageActivationBytes(16,16,32,stages)==2*16*32*stages);
  for(int page:{8192,16384}) {
    tilemega::TargetSpec target;
    int workspace=std::max(reduced.first,reduced.second);
    int minimum=PageLayout::Align(128+workspace,1024)+2*page;
    target.res.max_dynamic_smem_per_cta=minimum;
    auto fit=PageLayout::Build(target,page,reduced.first,reduced.second,0,2);
    assert(fit.pages==2 && fit.shared_bytes==minimum);
    assert(fit.activation_offset==fit.scratch_offset && fit.pages_offset%1024==0);
    target.res.max_dynamic_smem_per_cta=minimum-1;
    bool rejected=false;
    try{PageLayout::Build(target,page,reduced.first,reduced.second,0,2);}
    catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);
  }
  auto mixed=PageLayout::ServingWorkspace({{16,16,16},{64,128,64}},{},false,true,40000);
  assert(mixed.first==32768 && mixed.second==40000);
  std::cout<<"DM page resources: 76 geometries, K partials, activation stages, target budgets and legacy defaults PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_page_resources_test
