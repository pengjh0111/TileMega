// SPDX-License-Identifier: BSD-3-Clause
// The vocabulary gather in front of the first layer (R7 A4). The exported
// reference graph starts at hidden states; this test rewrites its entry into
// the form a real checkpoint exports -- token identifiers through an
// `aten.embedding.default` -- and checks that the importer recognizes it
// structurally, without any name convention.
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>

using namespace tilemega;

namespace {

/// Put an embedding between the model input and the rest of the graph: the
/// user input becomes int64 identifiers, and a new gather produces the tensor
/// the first layer used to read directly.
frontend::ExportBridge WithEmbedding(frontend::ExportBridge bridge,
                                     std::string const& hidden,
                                     std::string const& dtype) {
  frontend::FxNodeRecord ids;
  frontend::FxNodeRecord table;
  frontend::FxNodeRecord gather;
  for (auto& node : bridge.nodes) {
    if (node.name != hidden) continue;
    ids = node;
    ids.name = "input_ids";
    ids.target = "input_ids";
    ids.shape = {"1", node.shape.at(1)};
    ids.dtype = dtype;
    table = node;
    table.name = "p_embed_weight";
    table.target = "p_embed_weight";
    table.shape = {"32", node.shape.back()};
    gather = node;
    gather.op = "call_function";
    gather.target = "aten.embedding.default";
    gather.inputs = {table.name, ids.name};
    break;
  }
  assert(!gather.name.empty());

  std::vector<frontend::FxNodeRecord> nodes;
  for (auto const& node : bridge.nodes) {
    if (node.name == hidden) {
      nodes.push_back(ids);
      nodes.push_back(table);
      nodes.push_back(gather);
      continue;
    }
    nodes.push_back(node);
  }
  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) nodes[i].index = i;
  bridge.nodes = std::move(nodes);

  for (auto& input : bridge.inputs) {
    if (input.name != hidden) continue;
    input.name = ids.name;
    input.target = ids.target;
  }
  bridge.inputs.push_back({table.name, "PARAMETER", "embed.weight", false});
  return bridge;
}

}  // namespace

int main() {
  std::string const path = std::string(TILEMEGA_SOURCE_DIR) +
                           "/docs/experiments/E2E_GEN/raw/export_bridge.json";
  auto reference = frontend::ReadExportBridge(path);
  auto base = frontend::BuildModelPlan(reference.nodes, reference.inputs,
                                       reference.outputs);
  assert(base.token_id_bits == 0);
  assert(base.stages.front().kind != frontend::PlanTaskKind::kEmbedding);

  auto bridge = WithEmbedding(frontend::ReadExportBridge(path), "hidden",
                              "torch.int64");
  auto plan = frontend::BuildModelPlan(bridge.nodes, bridge.inputs,
                                       bridge.outputs);
  // One stage more than the hidden-state entry, and it is the gather.
  assert(plan.stages.size() == base.stages.size() + 1);
  frontend::PlanStage const& stage = plan.stages.front();
  assert(stage.kind == frontend::PlanTaskKind::kEmbedding);
  assert(stage.width == 512 && stage.extent == 32);
  assert(plan.token_id_bits == 64);
  // The identifiers keep their own width in the model-element buffer table,
  // and the gather's result is what the first normalization reads.
  assert(plan.buffers.at(stage.operands[0]).per_seq == 4);
  assert(plan.buffers.at(stage.operands[0]).source ==
         frontend::PlanBuffer::Source::kFixture);
  assert(plan.buffers.at(stage.operands[1]).source ==
         frontend::PlanBuffer::Source::kWeight);
  assert(plan.stages.at(1).operands[0] == stage.operands[2]);
  assert(plan.dtype == base.dtype);

  auto narrow_bridge = WithEmbedding(frontend::ReadExportBridge(path),
                                     "hidden", "torch.int32");
  auto narrow = frontend::BuildModelPlan(narrow_bridge.nodes,
                                         narrow_bridge.inputs,
                                         narrow_bridge.outputs);
  assert(narrow.token_id_bits == 32);
  assert(narrow.buffers.at(narrow.stages.front().operands[0]).per_seq == 2);

  std::cout << "embedding_plan_test PASS\n";
  return 0;
}
