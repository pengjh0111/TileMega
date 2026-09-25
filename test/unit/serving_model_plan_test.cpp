// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/DramFloor.h>

#include <cassert>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  tilemega::analysis::IslContext isl;
  if (argc != 3) {
    std::cerr << "usage: serving_model_plan_test BRIDGE.json prefill|decode\n";
    return 2;
  }
  auto bridge = tilemega::frontend::ReadExportBridge(argv[1]);
  tilemega::frontend::ServingOptions options;
  options.phase = std::string(argv[2]) == "prefill"
      ? tilemega::frontend::ServingOptions::Phase::kPrefill
      : tilemega::frontend::ServingOptions::Phase::kDecode;
  options.seq = options.phase == tilemega::frontend::ServingOptions::Phase::kPrefill
      ? 64 : 1;
  options.kv_block = 64;
  options.query_rows = 16;
  options.argmax_tile_n = 64;
  auto plan = tilemega::frontend::BuildModelPlan(
      bridge.nodes, bridge.inputs, bridge.outputs, options);
  assert(plan.serving);
  assert(plan.stages.front().kind == tilemega::frontend::PlanTaskKind::kEmbedding);
  assert(plan.stages.back().kind == tilemega::frontend::PlanTaskKind::kArgmaxReduce);
  int qkv = 0, attention = 0;
  bool packed_qkv = false, packed_gate_up = false;
  bool selected_final_row = false;
  bool tied_vocabulary = false;
  for (auto const& buffer : plan.buffers) {
    packed_qkv |= buffer.pack_json.find("qkv_group_interleave") != std::string::npos;
    packed_gate_up |= buffer.pack_json.find("gate_up_interleave") != std::string::npos;
    tied_vocabulary |= buffer.name == "model.embed_tokens.weight";
    if (buffer.name == "serving.tokens")
      assert(buffer.per_batch == 1088 && buffer.dtype == "i32" &&
             buffer.role == "external");
  }
  for (auto const& stage : plan.stages) {
    if (stage.kind == tilemega::frontend::PlanTaskKind::kFusedAttention)
      { ++attention;
        assert(stage.attention_kv_block ==
            (options.seq == 64 ? options.capacity : 64)); }
    if (stage.kind == tilemega::frontend::PlanTaskKind::kRMSNorm &&
        stage.batch_rows && stage.row_stride == options.seq &&
        stage.row_offset == options.seq - 1)
      selected_final_row = true;
    if (stage.kind == tilemega::frontend::PlanTaskKind::kGemm &&
        plan.gemms[stage.gemm].n > 0) ++qkv;
  }
  bool matched_argmax_tile = false;
  for (auto const& gemm : plan.gemms)
    if (gemm.epilogue == tilemega::frontend::PlanGemm::Epilogue::kArgmaxPartial)
      matched_argmax_tile = gemm.partial_tile_n == 64;
  assert(matched_argmax_tile);
  assert(attention > 0 && qkv == 4 * attention + 1);
  assert(packed_qkv && packed_gate_up && selected_final_row && tied_vocabulary);
  tilemega::frontend::LiftOptions dims;
  dims.serving = true;
  dims.batch_symbol = "B";
  dims.past_symbol = options.seq == 1 ? "past" : "";
  dims.static_seq = options.seq;
  auto lifted = tilemega::frontend::LiftSemantics(plan, dims);
  assert(lifted.sem.ops.size() == plan.stages.size());
  auto geometry = tilemega::frontend::LaunchGranularity(lifted, plan, {});
  auto instances = tilemega::analysis::Instantiate(lifted.sem, geometry);
  assert(instances.nodes.size() == plan.stages.size());
  for (auto const& node : instances.nodes)
    for (auto const& operand : node.operands)
      if (!operand.producer.empty()) {
        auto const* source = instances.Find(operand.producer);
        if (source && source->output.axes.size() != operand.tensor.axes.size())
          std::cerr << "rank mismatch " << source->name << " -> " << node.name
                    << " " << source->output.axes.size() << " vs "
                    << operand.tensor.axes.size() << '\n';
      }
  auto edges = tilemega::analysis::CouplingDerivation{}.Derive(instances, {});
  assert(!edges.empty());
  tilemega::analysis::DramFloorOptions floor_options;
  floor_options.dram_gbps = 981.582923;
  floor_options.tc_gflops = 179997;
  for (auto const& buffer : plan.buffers)
    if (buffer.dtype == "i32") floor_options.element_bytes[buffer.name] = 4;
  floor_options.indirect_read_images["model.embed_tokens.weight"] =
      tilemega::analysis::CouplingRelation::FromIslText(
          "[B] -> { [] -> [v,h] : 0 <= v < B and 0 <= h < 2048 }");
  auto floor = tilemega::analysis::DeriveDramFloor(
      lifted.sem, floor_options, {});
  tilemega::analysis::ParamBinding theta;
  theta.Bind("B", 2).Bind("past", 64);
  auto floor_at_point = floor.Evaluate(theta);
  assert(floor_at_point.floor_ns > 0);
  tilemega::analysis::ParamBinding endpoint;
  endpoint.Bind("B", 16).Bind("past", 1086);
  auto floor_at_endpoint = floor.Evaluate(endpoint);
  std::cout << "serving model plan: stages=" << plan.stages.size()
            << " gemms=" << plan.gemms.size()
            << " attention=" << attention << '\n';
  std::cout << "serving floor B=2 past=64 dram_bytes="
            << floor_at_point.read_bytes + floor_at_point.write_bytes
            << " floor_ns=" << floor_at_point.floor_ns << '\n';
  std::cout << "serving floor B=16 past=1086 dram_bytes="
            << floor_at_endpoint.read_bytes + floor_at_endpoint.write_bytes
            << " floor_ns=" << floor_at_endpoint.floor_ns << '\n';
}
