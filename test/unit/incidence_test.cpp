// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ReferenceModels.h>

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace tilemega::analysis;

int main() try {
  IslContext context;
  DecoderShape shape;
  ParamBinding known = DecoderShape::Table27Theta();
  for (auto const& item : DecoderShape::Table27G().values)
    known.Bind(item.first, item.second);
  std::vector<std::pair<std::string, OperatorGraph>> graphs{
      {"llama", LlamaDecoderLayer(shape)}, {"llama4", LlamaStack(shape, 4)},
      {"mlp", MlpStack(shape, 3)}, {"mha", MhaModel(shape, 2)},
      {"gather", GatherModel(shape)}, {"affine_gather", GatherModel(shape, false)},
      {"misaligned", MisalignedTileModel(shape, 16, 24)},
      {"unknown", UnknownOperatorModel(shape)}};
  std::cout << "model\tsrc\tdst\tseq\tpast\told_wait_sum\twait_sum\tfanout_sum\n";
  std::size_t checks = 0;
  for (auto const& [name, graph] : graphs) {
    auto edges = CouplingDerivation{}.Derive(graph, known);
    for (auto const& edge : edges) {
      auto const& producer = *graph.Find(edge.src.name);
      auto const& consumer = *graph.Find(edge.dst.name);
      std::size_t operand = 0;
      while (operand < consumer.operands.size() &&
             consumer.operands[operand].producer != producer.name) ++operand;
      if (operand == consumer.operands.size()) throw std::runtime_error("missing operand");
      // Recreate the historical nominal relation independently from accesses,
      // not by relaxing or altering the corrected metric under test.
      auto nominal = DeriveCoupling(BuildWriteMap(producer),
          BuildReadMap(consumer, operand), producer, consumer, known);
      for (long seq : {1L,4L,128L,512L,2048L}) for (long past : {0L,3L,512L}) {
        auto point = known;
        point.Bind("S", seq).Bind("past", past).Bind("L_s", seq+past);
        auto wait = edge.metrics.wait.SubstituteParams(point).SumDomain().Eval(point);
        auto fanout = edge.metrics.fanout.SubstituteParams(point).SumDomain().Eval(point);
        auto old = nominal.BindParams(point).Card().SumDomain().Eval(point);
        std::cout << name << '\t' << producer.name << '\t' << consumer.name << '\t'
                  << seq << '\t' << past << '\t' << old << '\t' << wait << '\t'
                  << fanout << '\n';
        if (wait != fanout) throw std::runtime_error("physical incidence identity failed");
        ++checks;
      }
    }
  }
  std::cerr << "INCIDENCE_GATE checks=" << checks << " failures=0\n";
  auto const& graph = graphs.front().second;
  auto const& producer = *graph.Find("rmsnorm1");
  auto const& consumer = *graph.Find("wq");
  auto relation = DeriveCoupling(BuildWriteMap(producer), BuildReadMap(consumer, 0),
                                 producer, consumer, known);
  int const before = context.ReferenceCount();
  bool rejected = false;
  try {
    // isl cannot divide by an unbound tile parameter; exercise the actual
    // ComputeMetrics early exit, not just a normal successful teardown.
    (void)ComputeMetrics(relation, BuildWriteMap(producer), BuildReadMap(consumer, 0),
                         producer, consumer, {});
  } catch (std::exception const&) { rejected = true; }
  if (!rejected || context.ReferenceCount() != before)
    throw std::runtime_error("ComputeMetrics error path did not reject cleanly");
  std::cerr << "ISL_ERROR_BRANCH function=ComputeMetrics rejected=1 before=" << before
            << " after=" << context.ReferenceCount() << '\n';
  return 0;
} catch (std::exception const& error) {
  std::cerr << "incidence: " << error.what() << '\n';
  return 1;
}
