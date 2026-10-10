// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MoeOpaqueControl.h>
#include <cassert>

namespace tilemega::tests::moe_opaque_control_test {
int TestMoeOpaqueControl(int,char**) {
  using namespace codegen;
  OpaqueMoeStage attention{},norm{false,true},router{true,false,false,false,7};
  OpaqueMoeStage select{false,false,true,false,0,7},expert{true,false,false,false,8};
  OpaqueMoeStage combine{false,false,false,true,0,7};
  auto regions=FindOpaqueMoeRegions({attention,norm,router,select,select,expert,combine,attention});
  assert(regions.size()==1 && regions[0].first==1 && regions[0].last==6);
  // Deferred normalization starts at the router, and adjacent blocks retain
  // their own dispatch/combine boundaries.
  auto two=FindOpaqueMoeRegions({router,select,expert,combine,router,select,expert,combine});
  assert(two.size()==2 && two[0].first==0 && two[0].last==3 &&
         two[1].first==4 && two[1].last==7);
  auto reject=[](std::vector<OpaqueMoeStage> const& stages) {
    bool rejected=false;
    try{(void)FindOpaqueMoeRegions(stages);}catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);
  };
  reject({attention});reject({select,combine});reject({router,select});reject({combine});
  auto other=select;other.router_gemm=9;reject({router,select,other,combine});
  return 0;
}
} // namespace tilemega::tests::moe_opaque_control_test
