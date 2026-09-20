// SPDX-License-Identifier: BSD-3-Clause
// §5.4 B3: the S5 ISL certificate applied to every point of every segment of a
// segmented theta interval. The placements are EFT tables, which are outside
// the four template families, so the certificate is built from the table
// itself: pi/sigma are the materialized worker and slot, and the rank is a
// topological order of dependency and FIFO edges that ISL then verifies is a
// strict lexicographic order on those edges.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Solver/ParametricPlacement.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <mlir/Parser/Parser.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <queue>
#include <set>
#include <sstream>
using namespace tilemega;
using analysis::CouplingRelation;

namespace {
std::string Conjuncts(std::vector<std::string> const& pieces,char const* empty) {
  if (pieces.empty()) return empty;
  std::string text="{ ";
  for (std::size_t i=0;i<pieces.size();++i) text+=(i ? "; " : "")+pieces[i];
  return text+" }";
}
}  // namespace

int main(int argc,char** argv) try {
  if (argc<3) throw std::invalid_argument(
      "segment_proof OUT_DIR [--only SEQ,SEQ,...] SEGMENT_CG [SEGMENT_CG ...]");
  analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  std::filesystem::path out=argv[1];std::filesystem::create_directories(out);
  // One certificate costs minutes at a few thousand nodes and the points are
  // independent, so a campaign runs one process per point; the partition check
  // still reads every segment's whole table.
  std::set<int> only;int first=2;
  if (argc>3 && std::string(argv[2])=="--only") {
    std::stringstream list(argv[3]);std::string item;
    while (std::getline(list,item,',')) only.insert(std::stoi(item));
    first=4;
  }
  std::ofstream proofs(out/"proofs.tsv"),coverage(out/"coverage.tsv");
  proofs<<"segment\tgeometry\tseq\tnodes\tgrid\ttotal\tbijective\tdense\tacyclic\tresident\tstatus\n";
  coverage<<"segment\tbegin\tend\tmacro_begin\tmacro_end\tpoints\n";
  std::vector<std::pair<int,int>> covered;
  int macro_begin=0,macro_end=0,failures=0,proved=0;
  for (int index=first;index<argc;++index) {
    auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[index],&context);
    if (!module) throw std::invalid_argument(std::string("invalid segment CG: ")+argv[index]);
    std::vector<dialect::PlacementTable> tables;std::string error;
    if (!dialect::ReadPlacementIntervalTables(*module,&tables,&error) || tables.empty())
      throw std::runtime_error("segment carries no interval table: "+error);
    auto integer=[&](char const* key,int fallback) {
      auto attr=(*module)->getAttrOfType<mlir::IntegerAttr>(key);
      return attr ? int(attr.getInt()) : fallback;
    };
    int const segment=index-first,kappa=integer("tilemega.solved_kappa",1);
    int const begin=int(tables.front().seq),end=int(tables.back().seq);
    if (macro_begin==0) {macro_begin=integer("tilemega.solved_seq_begin",0);
                         macro_end=integer("tilemega.solved_seq_end",0);}
    else if (macro_begin!=integer("tilemega.solved_seq_begin",0) ||
             macro_end!=integer("tilemega.solved_seq_end",0))
      throw std::runtime_error("segments disagree on the compiled interval macros");
    std::vector<std::int64_t> stage_kappa;
    if (auto attr=(*module)->getAttrOfType<mlir::DenseI64ArrayAttr>("tilemega.solved_stage_kappa"))
      stage_kappa.assign(attr.asArrayRef().begin(),attr.asArrayRef().end());
    auto runtime=codegen::ReadRuntimePlan(*module);
    auto model=solver::ModelDescription::FromCouplingGraph(*module,
        solver::ModelDims::Symbolic("S",int(tables.front().past)),"segment-proof");
    std::string geometry=std::to_string(runtime.gemms.front().tile_m)+"x"+
        std::to_string(runtime.gemms.front().tile_n)+"x"+
        std::to_string(runtime.gemms.front().tile_k)+"s"+
        std::to_string(runtime.gemms.front().stages)+"split"+
        std::to_string(runtime.gemms.front().split_k);
    int const threads=model.dtype==solver::ScalarType::kBF16 ? 128 : 256;
    solver::RuntimeProjectionOptions po{int(tables.front().grid),threads,kappa};
    po.count_wait_entries=false;
    po.stage_kappa.assign(stage_kappa.begin(),stage_kappa.end());
    auto projection=solver::ProjectRuntimeQueues(model,runtime,po);
    // The wait a consumer performs is on the event group, so the dependency it
    // must respect is on every member of that group, not only its own producer.
    std::vector<std::string> members;
    for (std::size_t s=0;s<projection.stages.size();++s) {
      int const k=stage_kappa.empty() ? kappa : int(stage_kappa[s]);
      members.push_back("[w,"+std::to_string(s)+",kind,g] -> ["+std::to_string(s)+
          ",t] : kind=0 or (kind=1 and g*"+std::to_string(k)+"<=t<(g+1)*"+
          std::to_string(k)+") or (kind=2 and t=g)");
    }
    auto grouped=projection.dependencies.Union(projection.requested_events
        .ApplyRange(CouplingRelation::FromIslText(Conjuncts(members,"{ [w,s,kind,g] -> [s,t] : false }")))
        .ApplyRange(projection.tasks.ImageIdentity()));
    covered.push_back({begin,end});
    coverage<<segment<<'\t'<<begin<<'\t'<<end<<'\t'<<macro_begin<<'\t'<<macro_end
        <<'\t'<<tables.size()<<'\n';
    for (auto const& table:tables) {
      if (!only.empty() && !only.count(int(table.seq))) continue;
      analysis::ParamBinding theta;theta.Bind("S",int(table.seq));
      std::vector<int> counts,offsets{0};
      for (auto const& stage:projection.stages) {
        counts.push_back(stage.task_count.Eval(theta));
        offsets.push_back(offsets.back()+counts.back());
      }
      if (offsets.back()!=int(table.worker.size()))
        throw std::runtime_error("projected task count disagrees with the materialized table");
      int const grid=int(table.grid),nodes=offsets.back();
      std::vector<std::string> task_pieces,place_pieces;
      for (std::size_t s=0;s<counts.size();++s) {
        if (!counts[s]) continue;
        task_pieces.push_back("[] -> ["+std::to_string(s)+",t] : 0<=t<"+std::to_string(counts[s]));
        for (int t=0;t<counts[s];++t) {
          int const node=offsets[s]+t;
          place_pieces.push_back("["+std::to_string(s)+","+std::to_string(t)+"] -> ["+
              std::to_string(table.worker[node])+","+std::to_string(table.slot[node])+"]");
        }
      }
      // Kahn over dependency and FIFO edges. The order is the certificate; the
      // ISL check below is what proves it orders every edge strictly.
      std::vector<std::vector<int>> successors(nodes);
      std::vector<int> indegree(nodes,0);
      auto edge=[&](int producer,int consumer) {
        successors[producer].push_back(consumer);++indegree[consumer];
      };
      analysis::VisitFiniteRelation(isl,grouped.BindParams(theta).ToString(),4,[&](long const* e) {
        int const consumer=offsets[e[0]]+int(e[1]),producer=offsets[e[2]]+int(e[3]);
        if (consumer!=producer) edge(producer,consumer);
      });
      std::vector<std::vector<int>> queue(grid);
      for (int node=0;node<nodes;++node) queue[table.worker[node]].push_back(node);
      for (auto& row:queue) {
        std::sort(row.begin(),row.end(),[&](int a,int b) {
          return table.slot[a]<table.slot[b];
        });
        for (std::size_t i=1;i<row.size();++i) edge(row[i-1],row[i]);
      }
      std::vector<int> order;std::queue<int> ready;
      for (int node=0;node<nodes;++node) if (!indegree[node]) ready.push(node);
      while (!ready.empty()) {
        int const node=ready.front();ready.pop();order.push_back(node);
        for (int next:successors[node]) if (!--indegree[next]) ready.push(next);
      }
      std::vector<std::string> rank_pieces;
      bool const ordered=order.size()==std::size_t(nodes);
      for (std::size_t i=0;i<order.size();++i) {
        int const node=order[i];
        auto stage=std::upper_bound(offsets.begin(),offsets.end(),node)-offsets.begin()-1;
        rank_pieces.push_back("["+std::to_string(stage)+","+std::to_string(node-offsets[stage])+
            "] -> [0,"+std::to_string(i)+"]");
      }
      solver::ParametricPlacement finite;
      finite.family="eft";
      finite.tasks=CouplingRelation::FromIslText(Conjuncts(task_pieces,"{ [] -> [s,t] : false }"));
      finite.dependencies=grouped.BindParams(theta);
      finite.pi_sigma=CouplingRelation::FromIslText(Conjuncts(place_pieces,"{ [s,t] -> [w,q] : false }"));
      finite.rank=CouplingRelation::FromIslText(Conjuncts(rank_pieces,"{ [s,t] -> [a,b] : false }"));
      finite.grid_limit=CouplingRelation::FromIslText("{ [] -> ["+std::to_string(grid)+","+
          std::to_string(std::max(grid,integer("tilemega.solved_residency",1)*grid))+"] }");
      auto proof=ordered ? solver::ProveParametricPlacement(finite,nullptr,grid)
                         : solver::ParametricProof{};
      bool const pass=ordered && proof.passed();
      proofs<<segment<<'\t'<<geometry<<'\t'<<table.seq<<'\t'<<nodes<<'\t'<<grid<<'\t'
          <<proof.total<<'\t'<<proof.bijective<<'\t'<<proof.dense<<'\t'<<proof.acyclic
          <<'\t'<<proof.resident<<'\t'<<(pass ? "PASS" : "FAIL")<<'\n';
      proofs.flush();
      if (pass) ++proved; else ++failures;
      std::cout<<"SEGMENT_POINT segment="<<segment<<" seq="<<table.seq<<" nodes="<<nodes
          <<" status="<<(pass ? "PASS" : "FAIL")<<std::endl;
    }
  }
  std::sort(covered.begin(),covered.end());
  int cursor=macro_begin;
  for (auto const& [begin,end]:covered) {
    if (begin!=cursor) throw std::runtime_error("segments do not partition the compiled interval");
    cursor=end+1;
  }
  if (cursor!=macro_end+1) throw std::runtime_error("segments do not reach the interval end");
  std::cout<<"SEGMENT_PROOF segments="<<covered.size()<<" points="<<proved+failures
      <<" proved="<<proved<<" failed="<<failures<<" interval="<<macro_begin<<".."<<macro_end<<"\n";
  return failures ? 1 : 0;
} catch (std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
