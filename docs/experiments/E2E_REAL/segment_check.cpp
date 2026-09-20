// SPDX-License-Identifier: BSD-3-Clause
//
// §5.4 B3: the materialization half.  A segmented interval carries one plan
// table per integer seq, split across two variants that were solved from two
// different geometries.  This re-solves every point on its own segment's
// coupling graph, under the launch the binary actually pins, and requires the
// table to come back unchanged -- at the endpoints of each segment and at every
// interior point alike, which is what makes the cut a partition of the interval
// rather than a relabelling of one geometry's schedule.
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <mlir/Parser/Parser.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace tilemega;

namespace {
struct Segment {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<dialect::PlacementTable> tables;
  std::string path;
  long long begin=0,end=0;
};
long long Integer(mlir::ModuleOp module,char const* key) {
  auto attr=module->getAttrOfType<mlir::IntegerAttr>(key);
  if (!attr) throw std::runtime_error(std::string("segment is missing ")+key);
  return attr.getInt();
}
}  // namespace

int main(int argc,char** argv)try {
  if (argc<4) throw std::invalid_argument(
      "segment_check TARGET OUT_DIR SEGMENT_CG [SEGMENT_CG ...]");
  analysis::IslContext isl;
  mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  auto target=TargetSpec::FromJson(argv[1]);
  solver::HopCurve hop;std::string why;
  if (!solver::HopCurve::FromTsv("docs/experiments/SIMULATOR/hop_ns.tsv",&hop,&why))
    throw std::runtime_error(why);
  std::string const out=argv[2];

  std::vector<Segment> segments;
  long long macro_begin=0,macro_end=0,kappa=0,residency=0;
  std::vector<int> stage_kappa;
  for (int a=3;a<argc;++a) {
    Segment segment;segment.path=argv[a];
    segment.module=mlir::parseSourceFile<mlir::ModuleOp>(argv[a],&context);
    if (!segment.module) throw std::invalid_argument("invalid segment CG: "+segment.path);
    std::string error;
    if (!dialect::ReadPlacementIntervalTables(*segment.module,&segment.tables,&error))
      throw std::runtime_error(segment.path+": "+error);
    if (segment.tables.empty()) throw std::runtime_error(segment.path+": empty interval");
    segment.begin=segment.tables.front().seq;segment.end=segment.tables.back().seq;
    // Every variant of a binary compiles against one set of launch macros, so
    // the segments must agree on them even though their tables do not overlap.
    long long const begin=Integer(*segment.module,"tilemega.solved_seq_begin");
    long long const end=Integer(*segment.module,"tilemega.solved_seq_end");
    long long const k=Integer(*segment.module,"tilemega.solved_kappa");
    long long const r=Integer(*segment.module,"tilemega.solved_residency");
    std::vector<int> table;
    if (auto per_stage=(*segment.module)->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "tilemega.solved_stage_kappa"))
      for (long long value:per_stage.asArrayRef()) table.push_back(int(value));
    if (segments.empty()) {macro_begin=begin;macro_end=end;kappa=k;residency=r;stage_kappa=table;}
    else if (begin!=macro_begin || end!=macro_end || k!=kappa || r!=residency ||
             table!=stage_kappa)
      throw std::runtime_error(segment.path+": segments disagree on the launch macros");
    segments.push_back(std::move(segment));
  }
  std::stable_sort(segments.begin(),segments.end(),
      [](Segment const& a,Segment const& b){return a.begin<b.begin;});
  long long expect=macro_begin;
  for (auto const& segment:segments) {
    if (segment.begin!=expect)
      throw std::runtime_error("segments do not partition the interval at seq "+
          std::to_string(expect));
    if (static_cast<long long>(segment.tables.size())!=segment.end-segment.begin+1)
      throw std::runtime_error(segment.path+": interval table has a hole");
    expect=segment.end+1;
  }
  if (expect!=macro_end+1)
    throw std::runtime_error("segments stop short of the advertised interval");

  std::ofstream rows(out+"/material.tsv");
  rows<<"segment\tseq\trole\tnodes\tgrid\tworker_diff\tslot_diff\tpipeline_diff\n";
  int points=0,endpoints=0,interior=0;
  for (std::size_t s=0;s<segments.size();++s) {
    auto const& segment=segments[s];
    for (auto const& table:segment.tables) {
      auto copy=mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::cast<mlir::ModuleOp>((*segment.module)->clone()));
      dialect::PlacementSolveOptions options;options.target=target;options.hop=hop;
      options.dims={int(table.seq),int(table.past),int(table.seq+table.past)};
      options.residency=int(residency);
      // Only a legality gate, not a cost input: the point solve is asked for
      // the same residency the binary launches at, so the same bound admits it.
      options.verified_resident_limit=int(residency);
      options.requested_grid=int(table.grid);options.kappa=int(kappa);
      options.stage_kappa=stage_kappa;
      auto point=dialect::SolveAndWritePlacement(*copy,options);
      // The interval writer materializes whatever family wins as an EFT table,
      // so the comparison is made against a table produced the same way rather
      // than against the winner's own parameters.
      auto materialized=point.candidates.front();
      materialized.mode=dialect::PlacementMode::kEft;materialized.params.clear();
      dialect::WriteSolvedPlacement(*copy,materialized,options);
      dialect::PlacementTable solved;std::string error;
      if (!dialect::ReadPlacementTable(*copy,&solved,&error))
        throw std::runtime_error("point solve wrote no table: "+error);
      auto const& worker=solved.worker;auto const& slot=solved.slot;
      auto const& pipeline=solved.pipeline;
      auto differ=[](auto const& a,auto const& b) {
        if (a.size()!=b.size()) return int(std::max(a.size(),b.size()));
        int count=0;
        for (std::size_t i=0;i<a.size();++i) count+=a[i]!=b[i];
        return count;
      };
      int const worker_diff=differ(worker,table.worker);
      int const slot_diff=differ(slot,table.slot);
      // A plan that pipelines nothing writes no field, so an absent flag vector
      // and an all-zero one are the same materialization.
      auto flags=[&](std::vector<unsigned char> const& v) {
        return v.empty() ? std::vector<unsigned char>(worker.size(),0) : v;
      };
      int const pipeline_diff=differ(flags(pipeline),flags(table.pipeline));
      bool const endpoint=table.seq==segment.begin || table.seq==segment.end;
      rows<<s<<'\t'<<table.seq<<'\t'<<(endpoint ? "endpoint" : "interior")<<'\t'
          <<worker.size()<<'\t'<<table.grid<<'\t'<<worker_diff<<'\t'<<slot_diff
          <<'\t'<<pipeline_diff<<'\n';
      rows.flush();
      std::cout<<"SEGMENT_MATERIAL segment="<<s<<" seq="<<table.seq
          <<" role="<<(endpoint ? "endpoint" : "interior")<<" nodes="<<worker.size()
          <<" diff="<<worker_diff+slot_diff+pipeline_diff<<'\n';
      if (worker_diff || slot_diff || pipeline_diff)
        throw std::runtime_error("segment "+std::to_string(s)+" seq "+
            std::to_string(table.seq)+" re-solves to a different table: worker "+
            std::to_string(worker_diff)+", slot "+std::to_string(slot_diff)+
            ", pipeline "+std::to_string(pipeline_diff));
      ++points;(endpoint ? endpoints : interior)+=1;
    }
  }

  // The variants are what the binary carries, so the round trip is checked on
  // the multi-variant lowering rather than on a segment in isolation.
  std::vector<codegen::RuntimeVariantModule> variants;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> reparsed;
  for (std::size_t s=0;s<segments.size();++s)
    variants.push_back({*segments[s].module,
        static_cast<std::uint32_t>(s ? segments[s].begin : 1),
        static_cast<std::uint32_t>(segments[s].end)});
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants(variants);
  std::vector<codegen::RuntimeVariantModule> again;
  for (std::size_t s=0;s<segments.size();++s) {
    std::string text;llvm::raw_string_ostream stream(text);
    (*segments[s].module).print(stream);stream.flush();
    reparsed.push_back(mlir::parseSourceString<mlir::ModuleOp>(text,&context));
    if (!reparsed.back()) throw std::runtime_error("segment does not round trip");
    again.push_back({*reparsed.back(),
        static_cast<std::uint32_t>(s ? segments[s].begin : 1),
        static_cast<std::uint32_t>(segments[s].end)});
  }
  if (codegen::CouplingGraphToCUDA{}.LowerVariants(again)!=source)
    throw std::runtime_error("segment serialization changed emitted CUDA");
  std::ofstream(out+"/variants.cu")<<source;

  std::cout<<"SEGMENT_CHECK segments="<<segments.size()<<" points="<<points
      <<" endpoints="<<endpoints<<" interior="<<interior
      <<" interval="<<macro_begin<<".."<<macro_end<<" kappa="<<kappa
      <<" residency="<<residency<<" serialization=byte_identical\n";
  return 0;
}catch(std::exception const& e){std::cerr<<"segment_check: "<<e.what()<<'\n';return 1;}
