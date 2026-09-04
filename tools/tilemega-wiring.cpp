// SPDX-License-Identifier: BSD-3-Clause
// Prints the coupling table the *frontend* derives for an exported model:
// FX -> ModelPlan -> L-sem -> Instantiate(g) -> CouplingDerivation. Nothing
// here is adjusted to match §2.7; the comparison against it lives in
// docs/experiments/WIRING/result.md.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>

#include <iostream>
#include <string>
#include <unordered_map>

using namespace tilemega;

namespace {

/// isl needs a literal floor/ceildiv divisor, so every tile symbol must be
/// bound before the derivation builds a map. §2.7's instantiation, and only
/// it: the workload symbols stay free (invariant I1).
analysis::ParamBinding TileBinding() {
  analysis::ParamBinding known;
  known.Bind("Tm", 128);
  known.Bind("Tn", 128);
  known.Bind("Tkv", 128);
  return known;
}

}  // namespace

int main(int argc, char** argv) {
  std::string path = argc > 1 ? argv[1]
                              : std::string(TILEMEGA_SOURCE_DIR) +
                                    "/docs/experiments/E2E_GEN/raw/export_bridge.json";
  std::string which = argc > 2 ? argv[2] : "reference";

  frontend::ExportBridge bridge = frontend::ReadExportBridge(path);
  frontend::ModelPlan plan =
      frontend::BuildModelPlan(bridge.nodes, bridge.inputs, bridge.outputs);
  frontend::LiftOptions options;
  options.seq_symbol = "S";
  options.past_symbol = "past";
  frontend::LiftedModel lifted =
      plan.stages.empty()
          ? frontend::LiftGenericSemantics(
                bridge.tasks, frontend::FormSemanticStages(bridge.tasks, plan),
                options)
          : frontend::LiftSemantics(plan, options);
  if (which != "reference" && which != "launch") {
    std::cerr << "unknown granularity: " << which << "\n";
    return 2;
  }
  analysis::Granularity g = which == "reference"
                                ? frontend::ReferenceGranularity(lifted)
                                : frontend::LaunchGranularity(lifted);
  analysis::OperatorGraph graph = analysis::Instantiate(lifted.sem, g);

  std::unordered_map<std::string, frontend::LiftedOp const*> origin;
  for (auto const& op : lifted.ops) origin.emplace(op.name, &op);
  auto role = [&](std::string const& node) {
    auto found = origin.find(node);
    if (found != origin.end()) return frontend::ToString(found->second->role);
    std::size_t dot = node.find_last_of('.');
    found = dot == std::string::npos ? origin.end()
                                     : origin.find(node.substr(0, dot));
    return found == origin.end() ? std::string("?")
                                 : frontend::ToString(found->second->role) +
                                       node.substr(dot);
  };

  std::cout << "## " << path << " (" << which << " granularity, Tm=Tn=Tkv=128)\n\n"
            << "| # | edge | roles | C | event shape | wait | fanout | volume "
               "| count | tier | attributes | guard | relaxation |\n"
            << "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
  int index = 0;
  for (auto const& edge :
       analysis::CouplingDerivation{}.Derive(graph, TileBinding())) {
    std::cout << "| " << ++index << " | " << edge.src.name << " -> "
              << edge.dst.name << " | " << role(edge.src.name) << " -> "
              << role(edge.dst.name) << " | `" << edge.C.ToString() << "` | `"
              << edge.EventShapeString() << "` | `"
              << edge.metrics.wait.ToString() << "` | `"
              << edge.metrics.fanout.ToString() << "` | `"
              << edge.metrics.volume.ToString() << "` | `"
              << edge.metrics.count.ToString() << "` | "
              << analysis::ToString(edge.tier) << " | "
              << edge.attributes.ToString() << " | "
              << (edge.guard.empty() ? "-" : "`" + edge.guard + "`") << " | "
              << (edge.relaxation.empty() ? "-" : edge.relaxation) << " |\n";
  }
  return 0;
}
