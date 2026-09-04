// SPDX-License-Identifier: BSD-3-Clause
// Part 1 acceptance: the FX stage list lifts to the same L-sem the reference
// model states by hand, item by item -- operator count, iteration domain,
// indexing maps, memory effects, reduction semantics and the granularity.
// Differences that do exist are pinned in `kExpectedDifferences` rather than
// absorbed into the comparison.
#include <tilemega/Analysis/ReferenceModels.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/SemanticLifting.h>

#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace tilemega;
using analysis::ClosedForm;

namespace {

std::vector<std::string> differences;

void Note(std::string const& text) { differences.push_back(text); }

/// Two bindings, so an accidental numeric coincidence at one point does not
/// pass for equality.
std::vector<analysis::ParamBinding> const& Points() {
  static std::vector<analysis::ParamBinding> points = [] {
    analysis::ParamBinding a, b;
    a.Bind("s11", 7).Bind("s14", 11).Bind("Tm", 128).Bind("Tn", 64).Bind(
        "Tkv", 32);
    b.Bind("s11", 13).Bind("s14", 5).Bind("Tm", 8).Bind("Tn", 16).Bind("Tkv",
                                                                       4);
    return std::vector<analysis::ParamBinding>{a, b};
  }();
  return points;
}

bool SameValue(ClosedForm const& lhs, ClosedForm const& rhs) {
  for (auto const& point : Points())
    if (lhs.Eval(point, point) != rhs.Eval(point, point)) return false;
  return true;
}

/// OperandAxisMap has no textual form of its own; the comparison needs one
/// that shows every field it distinguishes.
std::string AxisText(analysis::OperandAxisMap const& axis) {
  char const* kind = "?";
  switch (axis.kind) {
    case analysis::OperandAxisMap::Kind::kIndexed: kind = "indexed"; break;
    case analysis::OperandAxisMap::Kind::kFullRange: kind = "full"; break;
    case analysis::OperandAxisMap::Kind::kBroadcast: kind = "broadcast"; break;
    case analysis::OperandAxisMap::Kind::kDataDependent: kind = "gather"; break;
  }
  std::string text = std::string(kind) + "(";
  for (auto const& term : axis.terms)
    text += std::to_string(term.output_axis) + ":" + term.scale.ToString() +
            "/" + term.group.ToString() + " ";
  return text + "off=" + axis.offset.ToString() + " span=" +
         axis.span.ToString() + ")";
}

/// Producers are compared as graph positions, not as strings: the two graphs
/// name their operators differently, and a consumer may legally name either a
/// splittable operator or its combiner (Instantiate resolves both).
struct Positions {
  std::map<std::string, int> of;
  explicit Positions(analysis::SemanticGraph const& graph) {
    for (std::size_t i = 0; i < graph.ops.size(); ++i) {
      of[graph.ops[i].name] = static_cast<int>(i);
      if (graph.ops[i].reduction.splittable)
        of[graph.ops[i].reduction.combiner] = static_cast<int>(i);
    }
  }
  int At(std::string const& name) const {
    if (name.empty()) return -1;
    auto found = of.find(name);
    return found == of.end() ? -2 : found->second;
  }
};

std::string Where(int op, char const* what) {
  return "op " + std::to_string(op) + ": " + what;
}

void CompareMap(int index, char const* what, analysis::IndexingMap const& lhs,
                analysis::IndexingMap const& rhs) {
  if (lhs.Serialize() != rhs.Serialize())
    Note(Where(index, what) + " " + lhs.Serialize() + " != " + rhs.Serialize());
}

void CompareSpace(int index, char const* what, analysis::TensorSpace const& lhs,
                  analysis::TensorSpace const& rhs) {
  if (lhs.axes.size() != rhs.axes.size()) {
    Note(Where(index, what) + " rank differs");
    return;
  }
  for (std::size_t a = 0; a < lhs.axes.size(); ++a) {
    if (!SameValue(lhs.axes[a].extent, rhs.axes[a].extent))
      Note(Where(index, what) + " axis " + std::to_string(a) + " extent " +
           lhs.axes[a].extent.ToString() + " != " +
           rhs.axes[a].extent.ToString());
    if (!SameValue(lhs.axes[a].origin, rhs.axes[a].origin))
      Note(Where(index, what) + " axis " + std::to_string(a) + " origin " +
           lhs.axes[a].origin.ToString() + " != " +
           rhs.axes[a].origin.ToString());
    if (lhs.axes[a].runtime != rhs.axes[a].runtime)
      Note(Where(index, what) + " axis " + std::to_string(a) + " runtime flag");
  }
  if (lhs.layout_id != rhs.layout_id)
    Note(Where(index, what) + " layout " + lhs.layout_id + " != " +
         rhs.layout_id);
}

void CompareEffect(int index, char const* what,
                   analysis::MemoryEffect const& lhs,
                   analysis::MemoryEffect const& rhs) {
  if (lhs.Serialize() != rhs.Serialize())
    Note(Where(index, what) + " effect " + lhs.Serialize() + " != " +
         rhs.Serialize());
}

void CompareOps(analysis::SemanticGraph const& lifted,
                analysis::SemanticGraph const& reference) {
  Positions lifted_at(lifted), reference_at(reference);
  for (std::size_t i = 0; i < lifted.ops.size(); ++i) {
    int index = static_cast<int>(i);
    auto const& a = lifted.ops[i];
    auto const& b = reference.ops[i];
    if (a.kind != b.kind)
      Note(Where(index, "kind") + " " + analysis::ToString(a.kind) + " != " +
           analysis::ToString(b.kind));
    if (a.generic != b.generic) Note(Where(index, "generic flag"));
    if (a.domain.size() != b.domain.size()) {
      Note(Where(index, "domain rank differs"));
      continue;
    }
    for (std::size_t d = 0; d < a.domain.size(); ++d) {
      auto const& x = a.domain[d];
      auto const& y = b.domain[d];
      if (x.name != y.name)
        Note(Where(index, "domain dim") + " " + std::to_string(d) + " name " +
             x.name + " != " + y.name);
      if (!SameValue(x.extent, y.extent))
        Note(Where(index, "domain dim") + " " + std::to_string(d) +
             " extent " + x.extent.ToString() + " != " + y.extent.ToString());
      if (!SameValue(x.origin, y.origin))
        Note(Where(index, "domain dim") + " " + std::to_string(d) + " origin");
      if (x.type != y.type)
        Note(Where(index, "domain dim") + " " + std::to_string(d) + " type");
      if (x.runtime != y.runtime)
        Note(Where(index, "domain dim") + " " + std::to_string(d) + " runtime");
    }
    CompareSpace(index, "result", a.result, b.result);
    CompareMap(index, "result map", a.result_map, b.result_map);
    CompareEffect(index, "result", a.result_effect, b.result_effect);
    if (a.operands.size() != b.operands.size()) {
      Note(Where(index, "operand count") + " " +
           std::to_string(a.operands.size()) + " != " +
           std::to_string(b.operands.size()));
      continue;
    }
    for (std::size_t o = 0; o < a.operands.size(); ++o) {
      std::string what = "operand " + std::to_string(o);
      if (lifted_at.At(a.operands[o].producer) !=
          reference_at.At(b.operands[o].producer))
        Note(Where(index, what.c_str()) + " producer " +
             a.operands[o].producer + " != " + b.operands[o].producer);
      CompareSpace(index, (what + " tensor").c_str(), a.operands[o].tensor,
                   b.operands[o].tensor);
      CompareMap(index, (what + " map").c_str(), a.operands[o].map,
                 b.operands[o].map);
      CompareEffect(index, (what).c_str(), a.operands[o].effect,
                    b.operands[o].effect);
    }
    auto const& ra = a.reduction;
    auto const& rb = b.reduction;
    if (ra.splittable != rb.splittable || ra.dim != rb.dim ||
        ra.reduction_operator != rb.reduction_operator ||
        ra.ownership != rb.ownership)
      Note(Where(index, "reduction") + " " + ra.Serialize() + " != " +
           rb.Serialize());
  }
}

void CompareGranularity(analysis::SemanticGraph const& lifted,
                        analysis::Granularity const& lifted_g,
                        analysis::SemanticGraph const& reference,
                        analysis::Granularity const& reference_g) {
  for (std::size_t i = 0; i < lifted.ops.size(); ++i) {
    int index = static_cast<int>(i);
    std::string const& a = lifted.ops[i].name;
    std::string const& b = reference.ops[i].name;
    std::set<std::string> dims;
    auto ta = lifted_g.tiles.find(a);
    auto tb = reference_g.tiles.find(b);
    if (ta != lifted_g.tiles.end())
      for (auto const& [dim, _] : ta->second) dims.insert(dim);
    if (tb != reference_g.tiles.end())
      for (auto const& [dim, _] : tb->second) dims.insert(dim);
    for (auto const& dim : dims) {
      ClosedForm x, y;
      bool has_x = lifted_g.TileOf(a, dim, &x);
      bool has_y = reference_g.TileOf(b, dim, &y);
      if (has_x != has_y) {
        Note(Where(index, "tile") + " " + dim + (has_x ? " only lifted"
                                                       : " only reference"));
        continue;
      }
      if (x.ToString() != y.ToString())
        Note(Where(index, "tile") + " " + dim + " " + x.ToString() + " != " +
             y.ToString());
    }
    ClosedForm ca, cb;
    bool has_a = lifted_g.ChunkOf(a, &ca), has_b = reference_g.ChunkOf(b, &cb);
    if (has_a != has_b)
      Note(Where(index, "reduction chunk presence"));
    else if (has_a && ca.ToString() != cb.ToString())
      Note(Where(index, "reduction chunk") + " " + ca.ToString() + " != " +
           cb.ToString());
  }
}

void CompareTasks(analysis::OperatorGraph const& lifted,
                  analysis::OperatorGraph const& reference) {
  if (lifted.nodes.size() != reference.nodes.size()) {
    Note("task node count " + std::to_string(lifted.nodes.size()) + " != " +
         std::to_string(reference.nodes.size()));
    return;
  }
  std::map<std::string, int> lifted_at, reference_at;
  for (std::size_t i = 0; i < lifted.nodes.size(); ++i) {
    lifted_at[lifted.nodes[i].name] = static_cast<int>(i);
    reference_at[reference.nodes[i].name] = static_cast<int>(i);
  }
  for (std::size_t i = 0; i < lifted.nodes.size(); ++i) {
    int index = static_cast<int>(i);
    auto const& a = lifted.nodes[i];
    auto const& b = reference.nodes[i];
    if (a.kind != b.kind) Note(Where(index, "task kind"));
    CompareSpace(index, "task output", a.output, b.output);
    if (a.tile.size() != b.tile.size()) {
      Note(Where(index, "task tile rank"));
      continue;
    }
    for (std::size_t t = 0; t < a.tile.size(); ++t)
      if (!SameValue(a.tile[t], b.tile[t]))
        Note(Where(index, "task tile") + " " + std::to_string(t) + " " +
             a.tile[t].ToString() + " != " + b.tile[t].ToString());
    if (a.operands.size() != b.operands.size()) {
      Note(Where(index, "task operand count"));
      continue;
    }
    for (std::size_t o = 0; o < a.operands.size(); ++o) {
      auto pa = lifted_at.find(a.operands[o].producer);
      auto pb = reference_at.find(b.operands[o].producer);
      int ia = pa == lifted_at.end() ? -1 : pa->second;
      int ib = pb == reference_at.end() ? -1 : pb->second;
      if (ia != ib)
        Note(Where(index, "task operand") + " " + std::to_string(o) +
             " producer " + a.operands[o].producer + " != " +
             b.operands[o].producer);
      if (a.operands[o].axes.size() != b.operands[o].axes.size()) {
        Note(Where(index, "task operand") + " " + std::to_string(o) + " rank");
        continue;
      }
      for (std::size_t x = 0; x < a.operands[o].axes.size(); ++x)
        if (AxisText(a.operands[o].axes[x]) !=
            AxisText(b.operands[o].axes[x]))
          Note(Where(index, "task operand") + " " + std::to_string(o) +
               " axis " + std::to_string(x) + " " +
               AxisText(a.operands[o].axes[x]) + " != " +
               AxisText(b.operands[o].axes[x]));
    }
  }
}

/// Every difference the lifting genuinely has against the hand-written
/// reference. They are recorded, not removed: docs/experiments/WIRING states
/// what each one is and why the reference, not the lifting, is the odd side.
/// All ten are one difference: the reference names the second RMSNorm's
/// parallel dim `i` while every other operator on both sides names it `m`.
char const* const kExpectedDifferences[] = {
    "op 11: domain dim 0 name m != i",
    "op 11: result map (1*m, 1*h) != (1*i, 1*h)",
    "op 11: operand 0 map (1*m, 1*r) != (1*i, 1*r)",
    "op 28: result map (1*m, 1*h) != (1*i, 1*h)",
    "op 28: operand 0 map (1*m, 1*r) != (1*i, 1*r)",
    "op 11: tile i only reference",
    "op 11: tile m only lifted",
    "op 28: domain dim 0 name m != i",
    "op 28: tile i only reference",
    "op 28: tile m only lifted",
};

}  // namespace

int main() {
  auto bridge = frontend::ReadExportBridge(
      std::string(TILEMEGA_SOURCE_DIR) +
      "/docs/experiments/E2E_GEN/raw/export_bridge.json");
  auto plan = frontend::BuildModelPlan(bridge.nodes, bridge.inputs,
                                       bridge.outputs);
  assert(plan.stages.size() == 30);

  frontend::LiftOptions options;
  options.seq_symbol = "s11";
  options.past_symbol = "s14";
  frontend::LiftedModel lifted = frontend::LiftSemantics(plan, options);
  assert(lifted.has_plan && lifted.degraded.empty());
  // 15 plan stages per layer, two of which carry a fused residual epilogue.
  assert(lifted.sem.ops.size() == 34 && lifted.ops.size() == 34);

  analysis::DecoderShape shape;
  shape.S = ClosedForm::Symbol("s11");
  shape.past = ClosedForm::Symbol("s14");
  shape.L_s = shape.S + shape.past;
  shape.H = ClosedForm::Constant(512);
  shape.n_h = ClosedForm::Constant(4);
  shape.n_kv = ClosedForm::Constant(2);
  shape.group = ClosedForm::Constant(2);
  shape.d = ClosedForm::Constant(128);
  shape.I = ClosedForm::Constant(1024);
  analysis::ReferenceModel reference = analysis::LlamaStackSem(shape, 2);
  assert(reference.sem.ops.size() == 34);

  analysis::Granularity lifted_g = frontend::ReferenceGranularity(lifted);
  CompareOps(lifted.sem, reference.sem);
  CompareGranularity(lifted.sem, lifted_g, reference.sem, reference.g);
  CompareTasks(analysis::Instantiate(lifted.sem, lifted_g), reference.Task());

  std::set<std::string> expected(std::begin(kExpectedDifferences),
                                 std::end(kExpectedDifferences));
  std::set<std::string> found(differences.begin(), differences.end());
  for (auto const& item : found)
    if (!expected.count(item)) std::cerr << "UNEXPECTED " << item << "\n";
  for (auto const& item : expected)
    if (!found.count(item)) std::cerr << "MISSING " << item << "\n";
  assert(found == expected);

  // The degraded path: one conservative task space per call_function, and no
  // silent placeholder -- every operator it produced is reported.
  auto stages = frontend::FormSemanticStages(bridge.tasks, plan);
  frontend::LiftedModel generic =
      frontend::LiftGenericSemantics(bridge.tasks, stages, options);
  assert(!generic.has_plan);
  assert(generic.sem.ops.size() == bridge.tasks.size());
  assert(generic.degraded.size() == bridge.tasks.size());
  for (auto const& op : generic.sem.ops) assert(op.generic);
  return 0;
}
