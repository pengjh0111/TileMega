// SPDX-License-Identifier: BSD-3-Clause
//
// Part 3.3: what `kIdentity` is actually worth, asked of the production path.
//
// `kIdentity` needs two things at once: a coupling relation C contained in the
// identity, so consumer task `b` depends only on producer task `b`; and two
// stages whose TaskBodies map `blockIdx.x` onto their task spaces the same
// way, so that `b` names the same thing on both sides.
//
// Both now come from the pipeline that generates the kernel. C is the isl
// relation `CouplingDerivation` derives from the lifted model, and the
// ownership is `LiftedOp::ownership` -- assigned per ModelPlan stage kind by
// `LiftSemantics` and cross-checked here against `OwnershipOf`, the table the
// TaskBodies themselves return from `Ownership` (TaskBase.h). The earlier
// version of this probe answered the second question from a hand-written
// table keyed on operator name prefixes; a name is not a declaration, and
// nothing kept it in step with the TaskBodies.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>

#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tilemega;
using namespace tilemega::analysis;

namespace {

codegen::TaskKind TaskKindOf(frontend::PlanTaskKind kind) {
  switch (kind) {
    case frontend::PlanTaskKind::kGemm: return codegen::TaskKind::kGemm;
    case frontend::PlanTaskKind::kRMSNorm: return codegen::TaskKind::kRMSNorm;
    case frontend::PlanTaskKind::kRoPE: return codegen::TaskKind::kRoPE;
    case frontend::PlanTaskKind::kKVAppend: return codegen::TaskKind::kKVAppend;
    case frontend::PlanTaskKind::kElementwise:
      return codegen::TaskKind::kElementwise;
    case frontend::PlanTaskKind::kAttention:
      return codegen::TaskKind::kAttention;
  }
  return codegen::TaskKind::kElementwise;
}

char const* Name(codegen::TaskOwnershipKind kind) {
  return kind == codegen::TaskOwnershipKind::kTilePerBlock ? "kTilePerBlock"
                                                           : "kElementChunk";
}

/// `{ [a,b,...] -> [a,b,...] }` over the same arity as `C`'s two tuples.
CouplingRelation IdentityLike(CouplingRelation const& C) {
  std::string tuple;
  for (std::size_t i = 0; i < C.DomainDimNames().size(); ++i) {
    if (i) tuple += ",";
    tuple += "x" + std::to_string(i);
  }
  return CouplingRelation::FromIslText("{ [" + tuple + "] -> [" + tuple + "] }");
}

ParamBinding TileBinding() {
  ParamBinding known;
  known.Bind("Tm", 128);
  known.Bind("Tn", 128);
  known.Bind("Tkv", 128);
  return known;
}

int Report(char const* label, std::string const& bridge_path) {
  frontend::ExportBridge bridge = frontend::ReadExportBridge(bridge_path);
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

  // The lifted ownership must be the TaskBody's own declaration, not a second
  // opinion about it. A disagreement is a hard failure: it would mean the
  // generator and the analysis disagree about what `blockIdx.x` owns.
  std::unordered_map<std::string, codegen::TaskOwnershipKind> owns;
  int drift = 0;
  for (auto const& op : lifted.ops) {
    auto declared = static_cast<codegen::TaskOwnershipKind>(op.ownership);
    if (lifted.has_plan && op.stage < static_cast<int>(plan.stages.size())) {
      auto body = codegen::OwnershipOf(TaskKindOf(plan.stages[op.stage].kind));
      if (body != declared) {
        ++drift;
        std::printf("DRIFT model=%s op=%s lifted=%s taskbody=%s\n", label,
                    op.name.c_str(), Name(declared), Name(body));
      }
      declared = body;
    }
    owns.emplace(op.name, declared);
  }

  auto ownership = [&](std::string const& node) {
    auto found = owns.find(node);
    if (found != owns.end()) return found->second;
    // Instantiation splits one lifted operator into `name.suffix` task
    // spaces; the ownership is the operator's.
    std::size_t dot = node.find_last_of('.');
    if (dot != std::string::npos) {
      found = owns.find(node.substr(0, dot));
      if (found != owns.end()) return found->second;
    }
    return codegen::TaskOwnershipKind::kElementChunk;
  };

  OperatorGraph graph =
      Instantiate(lifted.sem, frontend::LaunchGranularity(lifted));
  std::vector<CouplingEdge> edges =
      CouplingDerivation().Derive(graph, TileBinding());
  int identity = 0, arity_mismatch = 0, admissible = 0, chunk_pairs = 0;
  for (auto const& edge : edges) {
    if (edge.C.empty()) continue;
    if (edge.C.DomainDimNames().size() != edge.C.RangeDimNames().size()) {
      // A consumer task space of a different rank than its producer's cannot
      // have CTA `b` depend on producer CTA `b` at all -- there is no
      // identity to be contained in. Reported so the edge list stays whole.
      ++arity_mismatch;
      std::printf("EDGE model=%s src=%s dst=%s identity=no-arity C=%s\n", label,
                  edge.src.name.c_str(), edge.dst.name.c_str(),
                  edge.C.ToString().c_str());
      continue;
    }
    bool is_identity = edge.C.IsSubset(IdentityLike(edge.C));
    auto producer = ownership(edge.src.name), consumer = ownership(edge.dst.name);
    bool same = producer == consumer &&
                producer == codegen::TaskOwnershipKind::kTilePerBlock;
    if (is_identity) ++identity;
    if (is_identity && same) ++admissible;
    // Two kElementChunk stages would also share a CTA->task map, but only if
    // they linearize the same number of elements over the same grid, which
    // the CG task space does not record. Counted separately rather than
    // claimed: admitting them needs an element count in the ABI.
    if (is_identity && producer == consumer && !same) ++chunk_pairs;
    std::printf("EDGE model=%s src=%s dst=%s identity=%s owns=%s->%s "
                "kidentity_admissible=%s C=%s\n", label,
                edge.src.name.c_str(), edge.dst.name.c_str(),
                is_identity ? "yes" : "no", Name(producer), Name(consumer),
                (is_identity && same) ? "yes" : "no", edge.C.ToString().c_str());
  }
  std::printf("SUMMARY model=%s edges=%zu identity_candidates=%d "
              "arity_mismatch=%d kidentity_admissible=%d chunk_pairs=%d "
              "ownership_drift=%d\n", label, edges.size(), identity,
              arity_mismatch, admissible, chunk_pairs, drift);
  return drift;
}

}  // namespace

int main(int argc, char** argv) {
  std::string root = argc > 1 ? argv[1] : std::string(TILEMEGA_SOURCE_DIR);
  int drift = Report("gqa2", root + "/docs/experiments/E2E_GEN/raw/export_bridge.json");
  drift += Report("mha4",
                  root + "/docs/experiments/P3_GENERALIZATION/raw/export_bridge.json");
  return drift == 0 ? 0 : 1;
}
