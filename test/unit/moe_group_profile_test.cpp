// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MoeRoutingProfile.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::moe_group_profile_test {
int TestMoeGroupProfile(int,char**) {
  using namespace tilemega;
  // Two hand-enumerated windows have counts [4,2,1,1] and [2,2,2,2].
  // Their BM=2 virtual prefixes have five and four active blocks.
  auto point=json::Parse(R"({"tokens":4,"windows":2,"assignments_per_window":8,
    "distinct_experts_histogram":{"4":2},"expected_distinct_experts":4,
    "tokens_per_expert_histograms":[{"2":1,"4":1},{"2":2},
                                   {"1":1,"2":1},{"1":1,"2":1}],
    "group_blocks_histograms":{"1":{"8":2},"2":{"4":1,"5":1},"16":{"4":2}}})");
  auto parse=[&](json::Value const& coordinate) {
    json::Value profile(json::Object{{"schema","tilemega.dm1.routing.profile.v1"},
      {"evidence","verified"},{"profile_id",std::string(64,'a')},
      {"sampling",json::Object{{"layers",1},{"experts",4},{"top_k",2}}},
      {"layers",json::Array{json::Object{{"layer",0},
          {"coordinates",json::Object{{"4",coordinate}}}}}}});
    return solver::MoeRoutingProfile::Read(profile,1,4,2);
  };
  auto profile=parse(point);auto const& p=profile.At(0,4);
  for(unsigned block:{1,2,16}) {
    double total=0;
    for(std::uint64_t v=0;v<p.GroupCapacity(block);++v) {
      double expected=block==1?(v<8?1:0):block==16?(v<4?1:0):
          (v<4?1:v==4?.5:0);
      assert(p.ActiveVirtualProbability(block,v)==expected);
      total+=p.ActiveVirtualProbability(block,v);
    }
    assert(total==p.ExpectedGroupBlocks(block));
  }
  auto rejects=[](auto action) {
    bool rejected=false;try{action();}catch(std::exception const&){rejected=true;}assert(rejected);
  };
  rejects([&]{p.ActiveVirtualProbability(4,0);});
  rejects([&]{p.ActiveVirtualProbability(2,p.GroupCapacity(2));});
  for(auto bins:std::vector<json::Value>{
      json::Object{{"2",json::Object{{"4",2}}}},
      json::Object{{"2",json::Object{{"4",1}}}},
      json::Object{{"02",json::Object{{"4",1},{"5",1}}}},
      json::Object{{"0",json::Object{{"4",1},{"5",1}}}},
      json::Object{{"2",json::Object{{"9",2}}}},
      json::Object{{"2",json::Object{{"4",1},{"5",1}}},
                   {"2",json::Object{{"4",1},{"5",1}}}}}) {
    auto corrupted=point;corrupted.Set("group_blocks_histograms",bins);
    rejects([&]{parse(corrupted);});
  }
  std::cout<<"MoE joint groups: static prefix probabilities, exact marginal means and corruption rejection PASS\n";
  return 0;
}
} // namespace tilemega::tests::moe_group_profile_test
