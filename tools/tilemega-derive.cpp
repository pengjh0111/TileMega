// SPDX-License-Identifier: BSD-3-Clause
// Prints the coupling table the analysis layer derives for a reference model.
// The output is the raw derivation: nothing here is adjusted to match §2.7.
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <iostream>
#include <string>

using namespace tilemega::analysis;

namespace {

/// isl can only take a literal floor/ceildiv divisor (docs/experiments/
/// P3_ISL/result.md), so every symbol a derivation might use as a tile size
/// or a GQA group divisor -- everything Table27Theta/Table27G fix -- must be
/// bound before DeriveCoupling builds an isl_map. Only S/L_s/past stay free,
/// exactly as §2.7 presents them.
ParamBinding KnownBinding() {
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& [name, value] : DecoderShape::Table27G().values)
    known.Bind(name, value);
  return known;
}

void Dump(std::string const& title, OperatorGraph const& graph) {
  std::cout << "## " << title << "\n\n";
  std::cout << "| # | edge | C | event shape | wait | fanout | volume | count "
               "| tier | attributes | guard | relaxation |\n";
  std::cout << "|---|---|---|---|---|---|---|---|---|---|---|---|\n";
  CouplingDerivation derivation;
  std::vector<CouplingEdge> edges = derivation.Derive(graph, KnownBinding());
  int index = 0;
  for (auto const& edge : edges) {
    std::cout << "| " << ++index << " | " << edge.src.name << " -> "
              << edge.dst.name << " | `" << edge.C.ToString() << "` | `"
              << edge.EventShapeString() << "` | `"
              << edge.metrics.wait.ToString() << "` | `"
              << edge.metrics.fanout.ToString() << "` | `"
              << edge.metrics.volume.ToString() << "` | `"
              << edge.metrics.count.ToString() << "` | " << ToString(edge.tier)
              << " | " << edge.attributes.ToString()
              << " | " << (edge.guard.empty() ? "-" : "`" + edge.guard + "`")
              << " | " << (edge.relaxation.empty() ? "-" : edge.relaxation)
              << " |\n";
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string which = argc > 1 ? argv[1] : "llama";
  DecoderShape shape;
  if (which == "llama") {
    Dump("Llama decoder layer (one layer, symbolic S)",
         LlamaDecoderLayer(shape));
  } else if (which == "decode") {
    shape.S = ClosedForm::Constant(1);
    Dump("Llama decoder layer (decode instantiation, S = 1)",
         LlamaDecoderLayer(shape));
  } else if (which == "llama4") {
    Dump("Llama stack, four layers", LlamaStack(shape, 4));
  } else if (which == "mlp") {
    Dump("MLP stack, three blocks", MlpStack(shape, 3));
  } else if (which == "mha") {
    Dump("MHA model (no GQA), two layers", MhaModel(shape, 2));
  } else if (which == "gather") {
    Dump("Data-dependent gather", GatherModel(shape));
  } else {
    std::cerr << "unknown model: " << which << "\n";
    return 2;
  }
  return 0;
}
