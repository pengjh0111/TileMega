// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxDecodeSelectionV2.h>
#include <cassert>
#include <iostream>
using namespace tilemega;
int main() {
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
  std::cout<<"mx_decode_selection_v2: synthetic CPU serialization and exact 1% consumer passed\n";
}
