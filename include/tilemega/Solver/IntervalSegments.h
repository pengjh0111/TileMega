// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/CompilerSearch.h>

namespace tilemega::solver {
/// One geometry and the contiguous block of theta it is materialized for.
struct IntervalSegment {
  GemmConfig config;
  int begin=0,end=0;
  double predicted_ns=0;  ///< summed over the segment's integer points
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
/// §5.4 B3. Every candidate geometry is priced at every integer point of the
/// interval and the interval is then cut. Two variants per binary (F-66) is
/// the whole bound on the cut, so the choice is one pass over
/// (geometry, geometry, cut) rather than a search.
struct IntervalSegmentation {
  std::vector<IntervalSegment> segments;
  std::vector<GemmConfig> candidates;
  /// Predicted makespan per candidate per point; infinite where the point has
  /// no legal placement under the pinned launch.
  std::vector<std::vector<double>> predicted;
  std::vector<std::pair<GemmConfig,std::string>> refused;
  GemmConfig fixed_config;
  double winner_ns=0,fixed_ns=0,segmented_ns=0;
  int cut=0;  ///< first seq of the second segment; zero when one geometry serves it all
};

inline std::string GeometryKey(GemmConfig const& g) {
  return std::to_string(g.tile_m)+"x"+std::to_string(g.tile_n)+"x"+
      std::to_string(g.tile_k)+"s"+std::to_string(g.stages)+"split"+
      std::to_string(g.split_k);
}

/// The launch stays the winner's: residency, kappa and resident grid are what
/// the binary was compiled for, so a segment may only change the geometry
/// behind the same launch. A candidate whose kernel compiled to fewer resident
/// CTAs than the winner cannot serve any segment and is recorded as refused.
inline IntervalSegmentation SolveIntervalSegments(std::string const& path,
    mlir::MLIRContext& context,CompilerSearchOptions const& options,
    CompilerSearchResult const& search,mlir::ModuleOp solved,int begin,int end,
    int max_segments,std::size_t max_candidates,std::ostream& evidence) {
  if (begin<1 || end<begin) throw std::invalid_argument("invalid segmentation interval");
  if (max_segments<1 || max_segments>2)
    throw std::invalid_argument("a binary carries at most two runtime variants (F-66)");
  if (!max_candidates) throw std::invalid_argument("segmentation needs a candidate budget");
  if (!solved) throw std::invalid_argument("segmentation needs a solved winner");
  double const infinite=std::numeric_limits<double>::infinity();
  IntervalSegmentation result;
  auto key=[](GemmConfig const& g) {
    return std::tie(g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k);
  };
  struct Observed { GemmConfig config;double floor_ns,makespan_ns;int residency; };
  std::vector<Observed> observed;
  for (auto const& e:search.ranking) {
    if (e.candidate.key.empty()) continue;
    auto found=std::find_if(observed.begin(),observed.end(),[&](Observed const& o) {
      return key(o.config)==key(e.candidate.config);
    });
    if (found==observed.end()) {
      observed.push_back({e.candidate.config,infinite,infinite,0});
      found=std::prev(observed.end());
    }
    // The inner loop walks residency 1..limit, so the largest residency the
    // search ever evaluated for a geometry is that geometry's compiled limit.
    found->residency=std::max(found->residency,e.candidate.ctas_per_sm);
    if (e.simulated && e.status=="ok" &&
        std::tie(e.floor_ns,e.makespan_ns)<std::tie(found->floor_ns,found->makespan_ns)) {
      found->floor_ns=e.floor_ns;found->makespan_ns=e.makespan_ns;
    }
  }
  std::stable_sort(observed.begin(),observed.end(),[](Observed const& a,Observed const& b) {
    return std::tie(a.floor_ns,a.makespan_ns)<std::tie(b.floor_ns,b.makespan_ns);
  });
  int const residency=search.winner.ctas_per_sm;
  std::vector<int> limits;
  auto admit=[&](Observed const& o) {
    bool const winner=key(o.config)==key(search.winner.config);
    if (!winner && !search.stage_kappa.empty()) {
      // The table names the winner's projected stages, and a geometry with a
      // different split-K projects a different number of them.
      result.refused.push_back({o.config,"per-stage kappa table is the winner's"});
      return;
    }
    if (!winner && o.residency<residency) {
      result.refused.push_back({o.config,"compiled residency "+std::to_string(o.residency)+
          " below the pinned "+std::to_string(residency)});
      return;
    }
    if (!std::isfinite(o.makespan_ns)) {
      result.refused.push_back({o.config,"no legal placement in the search"});
      return;
    }
    result.candidates.push_back(o.config);limits.push_back(o.residency);
  };
  auto winner=std::find_if(observed.begin(),observed.end(),[&](Observed const& o) {
    return key(o.config)==key(search.winner.config);
  });
  if (winner==observed.end()) throw std::runtime_error("the winner is not in the ranking");
  admit(*winner);
  if (result.candidates.empty()) throw std::runtime_error("the winner cannot serve a segment");
  for (auto const& candidate:observed) {
    if (result.candidates.size()>=max_candidates) break;
    if (key(candidate.config)!=key(search.winner.config)) admit(candidate);
  }

  auto bridge=frontend::ReadExportBridge(path);
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  auto seed=frontend::TorchExportImporter{}.ImportPlan(path,plan,context);
  auto model=ModelDescription::FromCouplingGraph(*seed,options.placement.dims,"interval-segments");
  auto pinned=options.placement;
  pinned.kappa=search.winner.kappa;pinned.stage_kappa=search.stage_kappa;
  pinned.residency=residency;
  pinned.requested_grid=int(solved->getAttrOfType<mlir::IntegerAttr>(
      "tilemega.solved_grid").getInt());
  pinned.task_price_cache=std::make_shared<dialect::PlacementTaskPriceCache>();
  int const past=options.placement.dims.past;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> imported;
  evidence<<"geometry\tseq\tplacement\tfloor_ns\tpredicted_ns\terror\n";
  for (std::size_t c=0;c<result.candidates.size();++c) {
    auto const& g=result.candidates[c];
    frontend::ImportOptions import;
    import.gemms.assign(model.gemms.size(),{g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
    import.rope_tile_per_block=import.kv_tile_per_block=true;
    import.activation_tile_per_block=import.combiner_tile_per_block=true;
    imported.push_back(frontend::TorchExportImporter{}.ImportPlan(path,plan,context,nullptr,import));
    std::vector<double> row;
    for (int seq=begin;seq<=end;++seq) {
      auto point=mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::cast<mlir::ModuleOp>(imported.back()->clone()));
      auto at=pinned;at.dims={seq,past,seq+past};at.verified_resident_limit=limits[c];
      double predicted=infinite,floor=0;std::string name,error;
      try {
        auto solved=dialect::SolveAndWritePlacement(*point,at);
        auto const& choice=solved.candidates.front();
        name=choice.name;error=choice.error;floor=choice.bounds.lower_bound_ns;
        if (choice.error.empty()) predicted=choice.predicted_ns;
      } catch (std::exception const& failure) {error=failure.what();}
      evidence<<GeometryKey(g)<<'\t'<<seq<<'\t'<<name<<'\t'<<floor<<'\t'
          <<predicted<<'\t'<<error<<'\n';
      row.push_back(predicted);
    }
    evidence.flush();
    result.predicted.push_back(std::move(row));
  }

  std::size_t const points=std::size_t(end-begin+1),count=result.candidates.size();
  auto sum=[&](std::size_t c,std::size_t lo,std::size_t hi) {
    double total=0;for (std::size_t i=lo;i<hi;++i) total+=result.predicted[c][i];return total;
  };
  result.winner_ns=sum(0,0,points);
  result.fixed_ns=infinite;result.fixed_config=result.candidates.front();
  std::size_t fixed_index=0;
  for (std::size_t a=0;a<count;++a) {
    double const whole=sum(a,0,points);
    if (whole<result.fixed_ns)
      {result.fixed_ns=whole;result.fixed_config=result.candidates[a];fixed_index=a;}
  }
  if (!std::isfinite(result.fixed_ns))
    throw std::runtime_error("no candidate geometry covers the interval");
  // The two arms are built from the same solve so that the benefit is a
  // comparison and not two invocations: `segments` is how many geometries the
  // caller wants in the binary, not a budget the search may spend below. A cut
  // that costs more than the fixed geometry is kept and reported as such.
  double best=infinite;std::size_t first=fixed_index,second=fixed_index,cut=points;
  // Two geometries can price identically at every point (split-K clamped by
  // the K tiles), and then the two partial sums differ only in rounding, so a
  // tie is decided by order and not by the last bit of a double.
  double const tie=1e-9*result.fixed_ns;
  for (std::size_t a=0;a<count && max_segments>1;++a)
    for (std::size_t b=0;b<count;++b) {
      if (b==a) continue;
      for (std::size_t k=1;k<points;++k) {
        double const total=sum(a,0,k)+sum(b,k,points);
        if (total+tie<best) {best=total;first=a;second=b;cut=k;}
      }
    }
  // One admitted geometry, or none whose pair is legal at every point: the
  // segmented arm degrades to the fixed one and says so through cut=0.
  if (!std::isfinite(best)) {best=result.fixed_ns;first=second=fixed_index;cut=points;}
  result.segmented_ns=best;
  result.cut=cut<points ? begin+int(cut) : 0;

  auto materialize=[&](std::size_t c,int lo,int hi,double predicted) {
    auto segment=mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::cast<mlir::ModuleOp>(imported[c]->clone()));
    auto at=pinned;at.dims={hi,past,hi+past};at.verified_resident_limit=limits[c];
    dialect::SolveAndWritePlacementInterval(*segment,at,lo,hi);
    result.segments.push_back({result.candidates[c],lo,hi,predicted,std::move(segment)});
  };
  if (result.cut) {
    materialize(first,begin,result.cut-1,sum(first,0,cut));
    materialize(second,result.cut,end,sum(second,cut,points));
  } else {
    materialize(first,begin,end,best);
  }
  // Every variant of a binary is compiled against one set of launch macros, so
  // the segments advertise the whole interval; the sub-range each one actually
  // holds travels in its own runtime plan table.
  mlir::OpBuilder b(&context);
  for (auto& segment:result.segments) {
    (*segment.module)->setAttr("tilemega.solved_seq_begin",b.getI64IntegerAttr(begin));
    (*segment.module)->setAttr("tilemega.solved_seq_end",b.getI64IntegerAttr(end));
  }
  return result;
}
}  // namespace tilemega::solver
