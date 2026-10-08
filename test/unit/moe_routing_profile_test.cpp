// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MoeRoutingProfile.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::moe_routing_profile_test {
int TestMoeRoutingProfile(int, char**) {
  using namespace tilemega;
  auto point=json::Parse(R"({"tokens":2,"windows":2,"assignments_per_window":4,
    "distinct_experts_histogram":{"3":2},"expected_distinct_experts":3,
    "tokens_per_expert_histograms":[{"1":1,"2":1},{"0":1,"1":1},
                                   {"1":1,"2":1},{"0":1,"1":1}]})");
  json::Value layer(json::Object{{"layer",0},{"coordinates",json::Object{{"2",point}}}});
  json::Value fixture(json::Object{{"schema","tilemega.dm1.routing.profile.v1"},
    {"evidence","verified"},{"profile_id",std::string(64,'a')},
    {"sampling",json::Object{{"layers",1},{"experts",4},{"top_k",2}}},
    {"layers",json::Array{layer}}});
  auto parse=[&](json::Value const& v){return solver::MoeRoutingProfile::Read(v,1,4,2);};
  auto profile=parse(fixture);auto const& p=profile.At(0,2);
  assert(p.expected_distinct_experts==3 && p.SlotCapacity()==4);
  assert(p.GroupCapacity(2)==6 && p.ExpectedGroupBlocks(1)==4 && p.ExpectedGroupBlocks(2)==3);
  assert(p.UniqueExpertWeightBytes(12288)==36864 && p.SlotWeightReadBytes(12288)==49152);
  assert(p.GatheredAReadBytes(64,2)==512);
  auto rejects=[](auto action) {
    bool failed=false;try {action();}catch(std::exception const&){failed=true;}assert(failed);
  };
  rejects([&]{profile.At(1,2);});rejects([&]{profile.At(0,1);});
  rejects([&]{p.GroupCapacity(0);});rejects([&]{p.ExpectedGroupBlocks(0);});
  rejects([&]{p.UniqueExpertWeightBytes(0);});rejects([&]{p.GatheredAReadBytes(0,2);});
  rejects([&]{solver::MoeRoutingProfile::Read(fixture,1,8,2);});
  unsigned corruptions=0;
  auto corrupt=[&](auto change) {auto v=fixture;change(v);rejects([&]{parse(v);});++corruptions;};
  corrupt([](auto& v){v.Set("evidence","inferred");});
  corrupt([](auto& v){v.Set("profile_id","bad");});
  corrupt([](auto& v){v.Set("layers",json::Array{});});
  auto corrupt_point=[&](auto change) {
    auto q=point;change(q);auto v=fixture;
    v.Set("layers",json::Array{json::Object{{"layer",0},{"coordinates",json::Object{{"2",q}}}}});
    rejects([&]{parse(v);});++corruptions;
  };
  corrupt_point([](auto& v){v.Set("expected_distinct_experts",2.5);});
  corrupt_point([](auto& v){v.Set("windows",1.5);});
  corrupt_point([](auto& v){v.Set("assignments_per_window",3);});
  corrupt_point([](auto& v){v.Set("distinct_experts_histogram",json::Object{{"03",2}});});
  corrupt_point([](auto& v){v.Set("distinct_experts_histogram",json::Object{{"3",1}});});
  corrupt_point([](auto& v){v.Set("distinct_experts_histogram",json::Object{{"3",1},{"3",1}});});
  corrupt_point([](auto& v){v.Set("tokens_per_expert_histograms",json::Array{});});
  corrupt_point([](auto& v){v.Set("tokens_per_expert_histograms",json::Array{
    json::Object{{"1",2}},json::Object{{"0",1},{"1",1}},
    json::Object{{"1",1},{"2",1}},json::Object{{"0",1},{"1",1}}});});
  std::cout<<"MoE routing costs: exact coordinates, capacities, traffic and "
           <<corruptions<<" corruption checks PASS\n";
  return 0;
}
} // namespace tilemega::tests::moe_routing_profile_test
