// SPDX-License-Identifier: BSD-3-Clause
// Part 2's precondition: MPK-style `EventDesc{num_triggers, first, last}` is
// only admissible if a consumer's wait set is a *contiguous* run of producer
// task ids under the order Place launches them in. isl has no operator for
// that question, so this enumerates C's integer points at concrete dimensions
// and measures it, per edge.
//
// It also answers the two questions the E2E_L2 identity probe answered from a
// hand-written name -> ownership table: whether C is contained in the identity,
// and whether both endpoints declare the same TaskOwnership. Ownership now
// comes from the lifted plan (LiftedOp::ownership), not from an operator name.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/DependencyForm.h>
#include <tilemega/Analysis/ReferenceModels.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace tilemega;
using analysis::ClosedForm;
using analysis::ParamBinding;

namespace {

struct Dims {
  long S = 16, past = 8, Tm = 4, Tn = 4, Tkv = 4;
};

ParamBinding Binding(Dims const& d) {
  ParamBinding known;
  known.Bind("S", d.S);
  known.Bind("past", d.past);
  known.Bind("Tm", d.Tm);
  known.Bind("Tn", d.Tn);
  known.Bind("Tkv", d.Tkv);
  return known;
}

/// Row-major linear id of a task coordinate, over the node's own coordinate
/// extents. This is the order the harness walks a task space in: the last
/// coordinate is the fastest-varying one, which is what `blockIdx.x` indexes
/// for a `kTilePerBlock` TaskBody.
struct Linearizer {
  std::vector<long> extent;
  long Of(std::vector<long> const& point) const {
    long id = 0;
    for (std::size_t i = 0; i < point.size(); ++i)
      id = id * (i < extent.size() ? extent[i] : 1) + point[i];
    return id;
  }
};

Linearizer LinearizerOf(analysis::OperatorNode const& node,
                        ParamBinding const& known) {
  Linearizer lin;
  for (std::size_t axis = 0; axis < node.tile.size(); ++axis)
    if (node.IsTiled(axis))
      lin.extent.push_back(node.CoordinateExtent(axis).Eval(known, {}));
  return lin;
}

analysis::CouplingRelation IdentityLike(analysis::CouplingRelation const& C) {
  std::string tuple;
  for (std::size_t i = 0; i < C.DomainDimNames().size(); ++i)
    tuple += (i ? ",x" : "x") + std::to_string(i);
  return analysis::CouplingRelation::FromIslText("{ [" + tuple + "] -> [" +
                                                 tuple + "] }");
}

void Report(char const* label, analysis::OperatorGraph const& graph,
            ParamBinding const& known,
            std::map<std::string, std::string> const& ownership,
            std::map<std::string, bool> const& runtime_space) {
  std::vector<analysis::CouplingEdge> edges =
      analysis::CouplingDerivation().Derive(graph, known);
  int contiguous_edges = 0, identity_edges = 0, kidentity_ok = 0;
  std::map<std::string, int> form_count;
  long total_exact = 0, total_hull = 0;
  int index = 0;
  for (auto const& edge : edges) {
    ++index;
    if (edge.C.empty()) continue;
    analysis::OperatorNode const* src = graph.Find(edge.src.name);
    analysis::OperatorNode const* dst = graph.Find(edge.dst.name);
    Linearizer plin = LinearizerOf(*src, known);
    Linearizer clin = LinearizerOf(*dst, known);

    // C's range is not clamped to the producer's task space (see
    // CouplingDerivation.cpp): unclamped it offers whole-tile producer
    // coordinates that no launch creates, so the wait columns below would
    // count tasks that do not exist.
    analysis::CouplingRelation C = edge.C;
    try {
      C = edge.C.IntersectRange(
          analysis::ProducerTaskSpaceText(edge.C, *src, known));
    } catch (std::exception const&) {
    }
    if (C.empty()) continue;

    long const points = C.Card().Eval(known);
    if (points > (1L << 22)) {
      std::printf("EDGE model=%s n=%d src=%s dst=%s form=all points=%ld "
                  "skipped=too_large\n", label, index, edge.src.name.c_str(),
                  edge.dst.name.c_str(), points);
      ++form_count["all"];
      continue;
    }
    std::map<long, std::vector<long>> wait;  // consumer id -> producer ids
    for (auto const& [consumer, producer] : C.Points())
      wait[clin.Of(consumer)].push_back(plin.Of(producer));

    bool contiguous = true;
    long wait_min = -1, wait_max = 0, hull_max = 0, worst_overwait = 0;
    long stride = 0;
    bool uniform_stride = true;
    for (auto& [consumer, producers] : wait) {
      std::sort(producers.begin(), producers.end());
      producers.erase(std::unique(producers.begin(), producers.end()),
                      producers.end());
      long const exact = static_cast<long>(producers.size());
      long const hull = producers.back() - producers.front() + 1;
      if (hull != exact) contiguous = false;
      for (std::size_t k = 1; k < producers.size(); ++k) {
        long const step = producers[k] - producers[k - 1];
        if (stride == 0) stride = step;
        else if (step != stride) uniform_stride = false;
      }
      if (wait_min < 0 || exact < wait_min) wait_min = exact;
      wait_max = std::max(wait_max, exact);
      hull_max = std::max(hull_max, hull);
      worst_overwait = std::max(worst_overwait, hull - exact);
      total_exact += exact;
      total_hull += hull;
    }

    long maxp = -1, maxc = -1;
    for (auto const& [consumer, producers] : wait) {
      maxc = std::max(maxc, consumer);
      for (long p : producers) maxp = std::max(maxp, p);
    }
    long const producer_tasks = src->Count().Eval(known, {});
    long const consumer_tasks = dst->Count().Eval(known, {});
    analysis::WaitWindow const window =
        analysis::FitWaitWindow(edge, *src, *dst, known);
    std::string const form = window.ToString();

    bool const identity =
        C.DomainDimNames().size() == C.RangeDimNames().size() &&
        C.IsSubset(IdentityLike(C));
    // Same rule Frontend.cpp uses to attribute an Instantiate-added combiner
    // to the operator it combines, so the probe reads the ownership the CG
    // actually carries rather than a second guess at it.
    auto own = [&](std::string const& name) {
      auto it = ownership.find(name);
      if (it != ownership.end()) return it->second;
      std::size_t dot = name.rfind('.');
      if (dot != std::string::npos) {
        it = ownership.find(name.substr(0, dot));
        if (it != ownership.end()) return it->second;
      }
      return std::string("unknown");
    };
    std::string const src_own = own(edge.src.name), dst_own = own(edge.dst.name);
    // `kIdentity` needs more than a semantic identity C: both TaskBodies must
    // index their task space the same way, which only `kTilePerBlock` on both
    // ends guarantees, and both spaces must have the same size.
    bool const admissible = window.IsIdentity() && src_own == dst_own &&
                            src_own == "tile_per_block";
    bool const ragged = runtime_space.count(edge.src.name)
                            ? runtime_space.at(edge.src.name)
                            : false;

    if (contiguous) ++contiguous_edges;
    if (identity) ++identity_edges;
    if (admissible) ++kidentity_ok;
    std::printf(
        "EDGE model=%s n=%d src=%s dst=%s form=%s P=%ld C=%ld contiguous=%s "
        "stride=%ld wait=%ld..%ld overwait_max=%ld dims=%zu/%zu max=%ld/%ld identity=%s owns=%s->%s "
        "kidentity=%s ragged=%s tier=%s\n",
        label, index, edge.src.name.c_str(), edge.dst.name.c_str(),
        form.c_str(), producer_tasks, consumer_tasks,
        contiguous ? "yes" : "no", uniform_stride ? stride : -1, wait_min,
        wait_max, worst_overwait, C.DomainDimNames().size(),
        C.RangeDimNames().size(), maxc, maxp,
        identity ? "yes" : "no", src_own.c_str(),
        dst_own.c_str(), admissible ? "yes" : "no", ragged ? "yes" : "no",
        analysis::ToString(edge.tier).c_str());
    ++form_count[form];
  }
  std::printf(
      "SUMMARY model=%s edges=%zu contiguous=%d identity=%d kidentity=%d "
      "wait_points_exact=%ld wait_points_hull=%ld overwait_ratio=%.4f\n",
      label, edges.size(), contiguous_edges, identity_edges, kidentity_ok,
      total_exact, total_hull,
      total_exact ? static_cast<double>(total_hull) / total_exact : 0.0);
  for (auto const& [form, count] : form_count)
    std::printf("FORMS model=%s %s=%d\n", label, form.c_str(), count);
}

}  // namespace

int main(int argc, char** argv) {
  std::string which = argc > 1 ? argv[1] : "fixture";
  Dims d;
  if (argc > 2) d.S = std::atol(argv[2]);
  if (argc > 3) d.past = std::atol(argv[3]);
  if (argc > 4) d.Tm = d.Tn = d.Tkv = std::atol(argv[4]);
  ParamBinding known = Binding(d);

  std::map<std::string, std::string> ownership;
  std::map<std::string, bool> runtime_space;
  analysis::OperatorGraph graph;

  if (which == "fixture" || which == "launch") {
    frontend::ExportBridge bridge = frontend::ReadExportBridge(
        std::string(TILEMEGA_SOURCE_DIR) +
        "/docs/experiments/E2E_GEN/raw/export_bridge.json");
    frontend::ModelPlan plan =
        frontend::BuildModelPlan(bridge.nodes, bridge.inputs, bridge.outputs);
    frontend::LiftedModel model =
        frontend::LiftSemantics(plan, frontend::LiftOptions{});
    for (std::size_t i = 0; i < model.ops.size(); ++i)
      ownership[model.sem.ops[i].name] = ToString(model.ops[i].ownership);
    graph = analysis::Instantiate(
        model.sem, which == "launch" ? frontend::LaunchGranularity(model)
                                     : frontend::ReferenceGranularity(model));
  } else {
    analysis::DecoderShape shape;
    shape.S = ClosedForm::Symbol("S");
    shape.past = ClosedForm::Symbol("past");
    shape.L_s = shape.S + shape.past;
    shape.H = ClosedForm::Constant(16);
    shape.n_h = ClosedForm::Constant(4);
    shape.n_kv = ClosedForm::Constant(2);
    shape.group = ClosedForm::Constant(2);
    shape.d = ClosedForm::Constant(4);
    shape.I = ClosedForm::Constant(32);
    shape.Tm = ClosedForm::Symbol("Tm");
    shape.Tn = ClosedForm::Symbol("Tn");
    shape.Tkv = ClosedForm::Symbol("Tkv");
    graph = analysis::LlamaDecoderLayerSem(shape).Task();
  }
  // A node whose own task space has a run-time extent is what makes an edge
  // ragged; recorded per producer so the shape column can say so.
  for (auto const& node : graph.nodes)
    runtime_space[node.name] = node.HasRuntimeTaskSpace();
  Report(which.c_str(), graph, known, ownership, runtime_space);
  return 0;
}
