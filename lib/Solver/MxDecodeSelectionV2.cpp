// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxDecodeSelectionV2.h>
#include <stdexcept>
namespace tilemega::solver {
namespace {
using json::Value;
using json::Object;
using json::Array;
Value Class(codegen::MxDecodeClassV2 const& c) {
  return Object{{"tn",c.tn},{"wn",c.wn},{"d",c.d},{"part",c.stream_k ? "kStreamK" : "kTileRange"},
                {"prefetch",c.prefetch},{"r_next",c.r_next}};
}
}
Value MxDecodeCandidatesJsonV2(MxDecodeSelectionV2 const& selection) {
  if (selection.top.size()!=4) throw std::invalid_argument("MX-5 preview requires exactly four solver candidates");
  Array plans;
  char const* names[]={"qkv","o","gate_up","down","head","attention"};
  for (std::size_t index=0;index<4;++index) {
    auto const& p=selection.top[index];auto const& c=p.config;
    Value pin=Object{{"qkv",Class(c.qkv)},{"o",Class(c.o)},{"gate_up",Class(c.gate_up)},
      {"down",Class(c.down)},{"head",Class(c.head)},{"threads",c.threads},{"b_min",c.b_min},
      {"attn_prefetch",c.attn_prefetch},{"purpose","manual"},{"diagnostic",true},{"final",false},
      {"solver_selected",false},{"default_enabled",false}};
    Value evidence=Object{}, stages=Object{};
    Array ranks;
    for (int i=0;i<6;++i) {
      auto const& candidate=selection.candidates.at(names[i]).at(p.ranks[i]);
      evidence.Set(names[i],Array(candidate.evidence));stages.Set(names[i],p.stage_ns.at(names[i]));
      ranks.emplace_back(double(p.ranks[i]));
    }
    plans.emplace_back(Object{{"index",int(index)},{"pin",Object{{"mx_decode_v2",pin}}},
      {"predicted_tpot_ns",p.tpot_ns},{"predicted_stage_ns",stages},{"ranks",ranks},{"evidence",evidence}});
  }
  return Object{{"schema_version",1},{"kind","mx5_decode_candidate_preview"},{"diagnostic",true},{"final",false},
    {"solver_selected",false},{"default_enabled",false},{"status","pending_per_plan_resources_and_top4_measurement"},
    {"resource_policy","stop_on_unresolved_per_plan_reduction_or_D7_fallback"},
    {"fit",Object{{"r2",selection.fit_r2},{"rows",selection.fit_rows}}},{"candidates",plans}};
}
Value MxDecodeMeasuredJsonV2(MxDecodeSelectionV2 const& selection,std::vector<std::array<double,2>> const& samples) {
  auto selected=SelectMxDecodeMeasuredV2(selection,samples);
  Array plans;
  for (std::size_t i=0;i<4;++i) plans.emplace_back(Object{{"index",int(i)},
    {"predicted_tpot_ns",selection.top[i].tpot_ns},{"process_tpot_ns",Array{samples[i][0],samples[i][1]}},
    {"measured_tpot_ns",selected.measured_ns[i]},{"pred_over_meas",selected.pred_over_meas[i]}});
  return Object{{"schema_version",1},{"kind","mx5_decode_top4_selection"},{"status","measured_candidate_selection"},
    {"selected_index",int(selected.selected)},{"diagnostic",true},{"final",false},{"selection_applied",false},
    {"default_enabled",false},{"policy","retain predicted winner unless fastest improves at least one percent"},{"candidates",plans}};
}
}  // namespace tilemega::solver
