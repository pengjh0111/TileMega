// SPDX-License-Identifier: BSD-3-Clause
//
// Skeleton §1.5.1: the derived C must reach the production path. This checks
// the claim that makes §2.7's table apply to the *frontend*: put the lifted FX
// model through the same granularity the reference model uses, and the
// coupling derivation must return the same edges -- same C, same wait, fanout,
// volume, count, tier, attributes and guard.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/SemanticLifting.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

using namespace tilemega;

namespace {

/// The exported fixture's own dimensions, so the two sides describe the same
/// model. Only the tiles are bound: S and past stay isl parameters (I1).
analysis::DecoderShape FixtureShape() {
  analysis::DecoderShape shape;
  shape.S = analysis::ClosedForm::Symbol("S");
  shape.H = analysis::ClosedForm::Constant(512);
  shape.n_h = analysis::ClosedForm::Constant(4);
  shape.n_kv = analysis::ClosedForm::Constant(2);
  shape.group = analysis::ClosedForm::Constant(2);
  shape.d = analysis::ClosedForm::Constant(128);
  shape.I = analysis::ClosedForm::Constant(1024);
  shape.past = analysis::ClosedForm::Symbol("past");
  shape.L_s = shape.S + shape.past;
  shape.Tm = analysis::ClosedForm::Symbol("Tm");
  shape.Tn = analysis::ClosedForm::Symbol("Tn");
  shape.Tkv = analysis::ClosedForm::Symbol("Tkv");
  return shape;
}

analysis::ParamBinding TileBinding() {
  analysis::ParamBinding known;
  known.Bind("Tm", 128);
  known.Bind("Tn", 128);
  known.Bind("Tkv", 128);
  return known;
}

std::vector<analysis::CouplingEdge> Lifted() {
  frontend::ExportBridge bridge = frontend::ReadExportBridge(
      std::string(TILEMEGA_SOURCE_DIR) +
      "/docs/experiments/E2E_GEN/raw/export_bridge.json");
  frontend::ModelPlan plan =
      frontend::BuildModelPlan(bridge.nodes, bridge.inputs, bridge.outputs);
  frontend::LiftOptions options;  // the reference model's own symbol names
  frontend::LiftedModel model = frontend::LiftSemantics(plan, options);
  return analysis::CouplingDerivation{}.Derive(
      analysis::Instantiate(model.sem, frontend::ReferenceGranularity(model)),
      TileBinding());
}

std::vector<analysis::CouplingEdge> Reference() {
  analysis::ReferenceModel model =
      analysis::LlamaStackSem(FixtureShape(), 2);
  return analysis::CouplingDerivation{}.Derive(model.Task(), TileBinding());
}

/// The reference names the second RMSNorm's parallel dim `i` where the lifting
/// names it `m`, so an edge *into* an rmsnorm2 carries that name in its C and
/// its wait. Ten records of the same fact are pinned in semantic_lifting_test;
/// these four are the same fact reaching the derived relation.
char const* const kExpectedDifferences[] = {
    "edge 14 C: [m] != [i]",
    "edge 14 wait: [m] != [i]",
    "edge 37 C: [m] != [i]",
    "edge 37 wait: [m] != [i]",
};

/// `text` with every standalone occurrence of the reference's `i` replaced by
/// the lifting's `m`, used only to decide whether a difference is *that*
/// difference and nothing more.
std::string RenameI(std::string text) {
  for (std::size_t at = 0; at < text.size(); ++at) {
    if (text[at] != 'i') continue;
    // A digit to the left is an isl coefficient (`128i`), not an identifier.
    bool left = at == 0 || !(std::isalpha(static_cast<unsigned char>(text[at - 1])) ||
                             text[at - 1] == '_');
    bool right = at + 1 == text.size() ||
                 !(std::isalnum(static_cast<unsigned char>(text[at + 1])) ||
                   text[at + 1] == '_');
    if (left && right) text[at] = 'm';
  }
  return text;
}

}  // namespace

int main() {
  std::vector<analysis::CouplingEdge> lifted = Lifted();
  std::vector<analysis::CouplingEdge> reference = Reference();
  assert(lifted.size() == reference.size());

  std::vector<std::string> differences;
  auto compare = [&](std::size_t edge, char const* field,
                     std::string const& a, std::string const& b) {
    if (a == b) return;
    // Only the naming difference is tolerated, and only by being recorded.
    std::string note = "edge " + std::to_string(edge + 1) + " " + field + ": " +
                       (RenameI(b) == a ? "[m] != [i]" : a + " != " + b);
    differences.push_back(note);
  };
  for (std::size_t i = 0; i < lifted.size(); ++i) {
    analysis::CouplingEdge const& a = lifted[i];
    analysis::CouplingEdge const& b = reference[i];
    compare(i, "C", a.C.ToString(), b.C.ToString());
    compare(i, "event shape", a.EventShapeString(), b.EventShapeString());
    compare(i, "wait", a.metrics.wait.ToString(), b.metrics.wait.ToString());
    compare(i, "fanout", a.metrics.fanout.ToString(), b.metrics.fanout.ToString());
    compare(i, "volume", a.metrics.volume.ToString(), b.metrics.volume.ToString());
    compare(i, "count", a.metrics.count.ToString(), b.metrics.count.ToString());
    compare(i, "tier", analysis::ToString(a.tier), analysis::ToString(b.tier));
    compare(i, "attributes", a.attributes.ToString(), b.attributes.ToString());
    compare(i, "guard", a.guard, b.guard);
    compare(i, "relaxation", a.relaxation, b.relaxation);
  }

  std::vector<std::string> expected(std::begin(kExpectedDifferences),
                                    std::end(kExpectedDifferences));
  bool ok = differences == expected;
  for (auto const& note : differences)
    if (std::find(expected.begin(), expected.end(), note) == expected.end())
      std::fprintf(stderr, "UNEXPECTED %s\n", note.c_str());
  for (auto const& note : expected)
    if (std::find(differences.begin(), differences.end(), note) ==
        differences.end())
      std::fprintf(stderr, "MISSING %s\n", note.c_str());
  assert(ok);

  std::printf("WIRING edges=%zu differences=%zu (all naming)\n", lifted.size(),
              differences.size());
  return 0;
}
