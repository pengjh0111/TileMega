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
Value MxDecodeQualifiedDomainJsonV3(MxDecodeSelectionV2 const& selection) {
  bool const complete=selection.top.size()==4;
  if ((complete && selection.status!="predicted_top4_resource_review_required") ||
      (!complete && !((selection.top.empty() && selection.status=="no_executable_plan") ||
        (!selection.top.empty() && selection.top.size()<4 && selection.status=="insufficient_top4"))))
    throw std::invalid_argument("MX-5 schema3 diagnostic status/top4 mismatch");
  selection.domain_report.AsObject("schema3 domain_report");
  Value doc=complete ? MxDecodeCandidatesJsonV2(selection) : Value(Object{
    {"schema_version",1},{"kind","mx5_decode_candidate_preview"},{"diagnostic",true},{"final",false},
    {"solver_selected",false},{"default_enabled",false},{"candidates",Array{}},
    {"fit",Object{{"r2",selection.fit_r2},{"rows",selection.fit_rows}}}});
  doc.Set("preview_mode","schema3_qualified_domain_diagnostic");
  doc.Set("source_schema_version",3);doc.Set("schema3_diagnostic_only",true);
  doc.Set("execution_authorized",false);doc.Set("results_consumption_allowed",false);
  doc.Set("status",selection.status);doc.Set("available_candidate_count",int(selection.top.size()));
  doc.Set("domain_report",selection.domain_report);
  doc.Set("resource_policy","diagnostic_only; no execution, results, default or 512 approval");
  auto plans=doc.At("candidates").AsArray("candidates");
  for(auto& candidate:plans) {
    candidate.Set("diagnostic",true);candidate.Set("final",false);candidate.Set("solver_selected",false);
    candidate.Set("default_enabled",false);candidate.Set("execution_authorized",false);
  }
  doc.Set("candidates",plans);return doc;
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
