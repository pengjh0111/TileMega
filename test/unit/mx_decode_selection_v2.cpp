// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxDecodeSelectionV2.h>
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
using namespace tilemega;
int main() {
  // An integral-valued float in real attention evidence must remain a float.
  assert(json::Parse("23460.0").Dump(0) == "23460.0\n");
  assert(json::Parse("104").Dump(0) == "104\n");
  assert(json::Parse("1.25e+2").Dump(0) == "1.25e+2\n");
  assert(json::Parse("[23460.0, 104, 1.25e+2]").Dump(0) == "[23460.0, 104, 1.25e+2]\n");
  for (double number : {0.12345678901234566, std::nextafter(1.0, 2.0),
                        std::numeric_limits<double>::min(), std::numeric_limits<double>::max()})
    assert(json::Parse(json::Value(number).Dump(0)).AsNumber("roundtrip") == number);

  solver::MxDecodeSelectionV2 selection;
  char const* names[]={"qkv","o","gate_up","down","head","attention"};
  for (char const* name:names) {
    solver::MxDecodeCandidateV2 c;c.evidence={json::Object{{"synthetic_fixture",true}}};
    selection.candidates[name].push_back(c);
  }
  for (int i=0;i<4;++i) {
    solver::MxDecodePredictionV2 p;p.tpot_ns=100+i;
    for (char const* name:names) p.stage_ns[name]=10;
    selection.top.push_back(p);
  }
  auto preview=solver::MxDecodeCandidatesJsonV2(selection);
  assert(preview.At("diagnostic").AsBool("diagnostic") && !preview.At("final").AsBool("final"));
  assert(!preview.At("solver_selected").AsBool("solver_selected"));
  for (auto const& c:preview.At("candidates").AsArray("candidates")) {
    auto const& pin=c.At("pin").At("mx_decode_v2");
    assert(pin.At("purpose").AsString("purpose")=="manual");
    assert(pin.At("diagnostic").AsBool("diagnostic") && !pin.At("final").AsBool("final"));
    assert(c.At("evidence").At("qkv").AsArray("evidence")[0].At("synthetic_fixture").AsBool("fixture"));
  }
  std::vector<std::array<double,2>> pairs{{{100,100}},{{99.1,99.1}},{{110,110}},{{120,120}}};
  auto result=solver::MxDecodeMeasuredJsonV2(selection,pairs);
  assert(result.At("selected_index").AsNumber("selected_index")==0);
  pairs[1]={99,99};result=solver::MxDecodeMeasuredJsonV2(selection,pairs);
  assert(result.At("selected_index").AsNumber("selected_index")==1);
  assert(!result.At("selection_applied").AsBool("selection_applied") && !result.At("final").AsBool("final"));
  assert(result.At("candidates").AsArray("candidates")[1].At("pred_over_meas").AsNumber("ratio")==101.0/99);
  // Synthetic domain reports exercise serialization without claiming measured eligibility.
  selection.domain_report=json::Object{{"resource_rejections",json::Array{json::Object{{"synthetic_fixture",true}}}},
      {"ranking_claim","exact_top4_of_certified_executable_stage_domain"}};
  selection.status="predicted_top4_resource_review_required";
  auto diagnostic=solver::MxDecodeQualifiedDomainJsonV3(selection);
  assert(diagnostic.At("domain_report").Dump()==selection.domain_report.Dump());
  assert(!diagnostic.At("execution_authorized").AsBool("execution_authorized"));
  assert(!diagnostic.At("results_consumption_allowed").AsBool("results_consumption_allowed"));
  auto const& diagnostic_plans=diagnostic.At("candidates").AsArray("candidates");
  auto const& original_plans=preview.At("candidates").AsArray("candidates");
  for(std::size_t i=0;i<4;++i) {
    for(char const* field:{"pin","predicted_tpot_ns","predicted_stage_ns","ranks","evidence"})
      assert(diagnostic_plans[i].At(field).Dump()==original_plans[i].At(field).Dump());
    assert(diagnostic_plans[i].At("diagnostic").AsBool("diagnostic"));
    assert(!diagnostic_plans[i].At("final").AsBool("final"));
    assert(!diagnostic_plans[i].At("execution_authorized").AsBool("execution_authorized"));
  }
  for(std::size_t count=0;count<4;++count) {
    auto partial=selection;partial.top.resize(count);
    partial.status=count?"insufficient_top4":"no_executable_plan";
    partial.domain_report.Set("ranking_claim","no_top4_claim");
    auto doc=solver::MxDecodeQualifiedDomainJsonV3(partial);
    assert(doc.At("domain_report").Dump()==partial.domain_report.Dump());
    assert(doc.At("available_candidate_count").AsNumber("count")==double(count));
    assert(doc.At("candidates").AsArray("candidates").empty());
    assert(doc.Dump().find("\"pin\"")==std::string::npos);
    partial.status="selected_top4";bool rejected=false;
    try{solver::MxDecodeQualifiedDomainJsonV3(partial);}catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);
  }
  std::cout<<"mx_decode_selection_v2: synthetic CPU serialization and exact 1% consumer passed\n";
}
