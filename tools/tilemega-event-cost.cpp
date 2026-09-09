// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ModelDescription.h>
#include <mlir/IR/MLIRContext.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#ifndef TILEMEGA_EVENT_COST_DIAGNOSTICS
#define TILEMEGA_EVENT_COST_DIAGNOSTICS 1
#endif

int main(int argc, char** argv) try {
  tilemega::analysis::IslContext isl_context;
#if !TILEMEGA_EVENT_COST_DIAGNOSTICS
  throw std::runtime_error("event metric audit disabled at compile time");
#endif
  if (argc != 2) throw std::invalid_argument("usage: tilemega-event-cost REPO");
  std::string root = argv[1];
  mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  std::cout << "model\tseq\tproducer\tconsumer\twait_sum\tfanout_sum\tcount\tvolume\n"
            << std::setprecision(17);
  for (std::string name : {"gqa2", "mha4"}) {
    tilemega::frontend::ImportOptions import;
    import.rope_tile_per_block = import.kv_tile_per_block = true;
    import.activation_tile_per_block = import.combiner_tile_per_block = true;
    auto cg = tilemega::frontend::TorchExportImporter{}.Import(
        root+"/docs/experiments/SEQSCAN/raw/export/"+name+".json", context, nullptr, import);
    auto symbolic = tilemega::solver::ModelDescription::FromCouplingGraph(
        *cg, tilemega::solver::ModelDims::Symbolic("S",3), name);
    if (symbolic.dtype != tilemega::solver::ScalarType::kBF16)
      throw std::runtime_error("event gate requires BF16 input");
    for (int seq : {4,128}) {
      tilemega::analysis::ParamBinding theta; theta.Bind("S",seq);
      auto model = symbolic.SubstituteParams(theta);
      for (auto const& edge : model.coupling_metrics) {
        auto eval = [&](tilemega::analysis::QuasiPolynomial const& q) {
          return q.SubstituteParams(model.metric_bindings).Eval(model.metric_bindings);
        };
        auto wait = eval(edge.wait.SumDomain()), fanout = eval(edge.fanout.SumDomain());
        auto count = eval(edge.count), volume = eval(edge.volume);
        std::cout << name << '\t' << seq << '\t' << edge.producer << '\t' << edge.consumer
                  << '\t' << wait << '\t' << fanout << '\t' << count << '\t' << volume << '\n';
        // Incidence equality is required before these quantities can price
        // one physical graph. Never silently replace one metric by another.
        if (wait != fanout) {
          std::cerr << "METRIC_DOMAIN_MISMATCH relation=" << edge.relation.ToString() << '\n';
          throw std::runtime_error("wait/fanout sum different task domains; event pricing refused");
        }
      }
    }
  }
  std::cerr << "METRIC_AUDIT_ONLY: no event price or symbolic DP acceptance claimed\n";
} catch (std::exception const& e) {
  std::cerr << "event-cost: " << e.what() << '\n'; return 2;
}
