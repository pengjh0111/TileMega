// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/PageLayout.h>
#include <cassert>

namespace tilemega::tests::dm_page_layout_test {
int TestDmPageLayout(int,char**) {
  using tilemega::solver::PageLayout;
  auto legacy=PageLayout::ServingWorkspace({{16,128,64}},{{64,4}});
  assert(legacy.first==8192 && legacy.second==8256);
  auto dm=PageLayout::ServingWorkspace({{16,128,64}},{{64,4}},false,true);
  assert(dm.first==8192 && dm.second==8320);
  for(int m:{16,32,64,128})for(int d:{64,128}) {
    auto prefill=PageLayout::ServingWorkspace({{m,64,64}},{{d,4}},true,true,64);
    assert(prefill.first==8*m*64);
    assert(prefill.second==std::max(4*m*66,tilemega::codegen::ServingAttentionSharedBytes(d)));
    tilemega::TargetSpec target;target.res.max_dynamic_smem_per_cta=131072;
    for(int page:{8192,16384}) {
      auto full=PageLayout::Build(target,page,prefill.first,prefill.second);
      assert(full.shared_bytes<=target.res.max_dynamic_smem_per_cta);
      assert(full.pages_offset>=full.activation_offset+std::max(prefill.first,prefill.second));
      assert(full.pages_offset%1024==0);
      auto one=PageLayout::Build(target,page,prefill.first,prefill.second,0,1);
      assert(one.pages==1 && one.shared_bytes<=full.shared_bytes);
      bool refused=false;
      try{(void)PageLayout::Build(target,page,prefill.first,prefill.second,0,full.pages+1);}
      catch(std::invalid_argument const&){refused=true;}
      assert(refused);
    }
  }
  auto extra=PageLayout::ServingWorkspace({}, {},false,true,32768);
  assert(extra.first==0 && extra.second==32768);
  tilemega::TargetSpec target;target.res.max_dynamic_smem_per_cta=4096;
  bool refused=false;
  try{(void)PageLayout::Build(target,8192,0,0,0,1);}
  catch(std::invalid_argument const&){refused=true;}
  assert(refused);
  return 0;
}
}
