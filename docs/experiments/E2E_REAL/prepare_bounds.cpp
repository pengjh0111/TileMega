// SPDX-License-Identifier: BSD-3-Clause
// C1-b before/after: the outer bound pass with the dense edge set materialized
// against the same pass reading the relation's intervals.  Both arms answer the
// same two numbers, so the run also re-checks equality on real relations rather
// than only on the unit test's synthetic ones.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Solver/ExecutionSimulator.h>
#include <algorithm>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace tilemega;
using Clock=std::chrono::steady_clock;
static double micros(Clock::time_point t) {
  return std::chrono::duration<double,std::micro>(Clock::now()-t).count();
}
int main(int argc,char** argv) try {
  if (argc<4) throw std::runtime_error("usage: prepare_bounds TARGET OUT CG:SEQ...");
  analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  auto target=TargetSpec::FromJson(argv[1]);
  std::ofstream out(std::string(argv[2])+"/prepare_bounds.tsv");
  out<<std::setprecision(12)<<"cell\tseq\tnodes\tedges\tdense_us\tinterval_us\tspeedup\t"
       "dense_total_us\tinterval_total_us\tdense_work_ns\tinterval_work_ns\t"
       "dense_cp_ns\tinterval_cp_ns\tidentical\n";
  std::cout<<"cell\tseq\tnodes\tedges\tdense_us\tinterval_us\tspeedup\tidentical\n";
  for (int arg=3;arg<argc;++arg) {
    std::string spec=argv[arg];auto colon=spec.rfind(':');
    std::string path=spec.substr(0,colon);int seq=std::stoi(spec.substr(colon+1));
    auto module=mlir::parseSourceFile<mlir::ModuleOp>(path,&context);
    if (!module) throw std::runtime_error("invalid CG: "+path);
    auto attr=[&](char const* key){
      return int(module->getOperation()->getAttrOfType<mlir::IntegerAttr>(key).getInt());};
    dialect::PlacementSolveOptions options;options.target=target;
    int past=attr("tilemega.solved_past");
    options.dims={seq,past,seq+past};
    options.residency=attr("tilemega.solved_residency");
    options.verified_resident_limit=options.residency;
    options.kappa=attr("tilemega.solved_kappa");
    options.requested_grid=attr("tilemega.solved_grid");

    // The whole preparation, both ways, and inside it the one stage C1-b
    // changes: building the bound from the projected dependency relation.
    auto start=Clock::now();
    auto dense=dialect::PreparePlacementProblem(*module,options);
    double const dense_total_us=micros(start);
    solver::RelationBounds interval_bounds;
    start=Clock::now();
    dialect::PreparePlacementProblem(*module,options,&interval_bounds);
    double const interval_total_us=micros(start);

    std::string const relation=dense.projection.dependencies.ToString();
    start=Clock::now();
    auto graph=codegen::MaterializeRuntimeTaskGraph(dense.counts,{},dense.grid);
    analysis::VisitFiniteRelation(analysis::SharedIslContext(),relation,4,[&](long const* e) {
      graph.successors[graph.stage_offsets[e[2]]+int(e[3])]
          .push_back(graph.stage_offsets[e[0]]+int(e[1]));
    });
    for (auto& row:graph.successors) {
      std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
    }
    solver::SimulatorInput input;input.graph=&graph;input.task_ns=dense.task_ns;
    solver::PreparedPlanBounds dense_bounds;std::string error;
    if (!solver::PreparePlanBounds(input,&dense_bounds,&error)) throw std::runtime_error(error);
    double const dense_us=micros(start);

    solver::RelationBounds checked;
    start=Clock::now();
    if (!solver::PrepareRelationBounds(graph.stage_offsets,dense.counts,dense.task_ns,
            relation,&checked,&error)) throw std::runtime_error(error);
    double const interval_us=micros(start);
    if (checked.work_ns!=interval_bounds.work_ns ||
        checked.critical_path_ns!=interval_bounds.critical_path_ns)
      throw std::runtime_error("bounds differ between the pass and the isolated call");

    long edges=0;for (auto const& row:graph.successors) edges+=long(row.size());
    bool const identical=
        dense_bounds.work_ns==interval_bounds.work_ns &&
        dense_bounds.critical_path_ns==interval_bounds.critical_path_ns;
    auto name=path.substr(path.rfind("experiments/")+12);
    out<<name<<'\t'<<seq<<'\t'<<graph.stage_offsets.back()<<'\t'<<edges<<'\t'
       <<dense_us<<'\t'<<interval_us<<'\t'<<dense_us/interval_us<<'\t'
       <<dense_total_us<<'\t'<<interval_total_us<<'\t'
       <<dense_bounds.work_ns<<'\t'<<interval_bounds.work_ns<<'\t'
       <<dense_bounds.critical_path_ns<<'\t'<<interval_bounds.critical_path_ns<<'\t'
       <<int(identical)<<'\n'<<std::flush;
    std::cout<<name<<'\t'<<seq<<'\t'<<graph.stage_offsets.back()<<'\t'<<edges<<'\t'
       <<dense_us<<'\t'<<interval_us<<'\t'<<dense_us/interval_us<<'\t'<<int(identical)
       <<'\n'<<std::flush;
    if (!identical) throw std::runtime_error("interval bounds disagree with the dense edge set");
  }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
