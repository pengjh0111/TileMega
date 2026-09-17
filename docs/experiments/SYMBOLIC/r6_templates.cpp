// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/ParametricTemplates.h>
#include <tilemega/Solver/VariantSchedule.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/BuiltinOps.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <iostream>
#include <optional>
using namespace tilemega;
using namespace tilemega::solver;
using analysis::CouplingRelation;
static CouplingRelation map(std::string const& s){return CouplingRelation::FromIslText(s);}
static std::string serial(MaterializedPlan const& p) {
  std::ostringstream out;out << "stage\ttask\tworker\tslot\n";
  for (std::size_t s=0;s<p.owner.size();++s) for (std::size_t t=0;t<p.owner[s].size();++t)
    out << s << '\t' << t << '\t' << p.owner[s][t] << '\t' << p.slot[s][t] << '\n';
  return out.str();
}
int main(int argc,char** argv) try {
  if (argc!=3 && argc!=5 && argc!=7) throw std::invalid_argument("r6_templates SOLVED_CG OUT_DIR [FAMILY GRID]");
  analysis::IslContext ctx;mlir::MLIRContext mc;mc.getOrLoadDialect<dialect::CGDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&mc);
  if (!module) throw std::invalid_argument("invalid CG");
  std::filesystem::path out=argv[2];std::filesystem::create_directories(out);
  auto runtime=codegen::ReadRuntimePlan(*module);
  auto model=ModelDescription::FromCouplingGraph(*module,ModelDims::Symbolic("S",3),"symbolic-r6");
  int threads=model.dtype==ScalarType::kBF16 ? 128 : 256;
  int kappa=1;
  if (auto attr=module->getOperation()->getAttrOfType<mlir::IntegerAttr>("tilemega.solved_kappa")) kappa=attr.getInt();
  int projection_grid=argc>=5 ? std::stoi(argv[4]) : 256;
  if(projection_grid<=0)throw std::invalid_argument("grid must be positive");
  RuntimeProjectionOptions po{projection_grid,threads,kappa};po.count_wait_entries=false;
  auto projection=ProjectRuntimeQueues(model,runtime,po);
  auto event_members=map("{ [w,s,kind,g] -> [s,t] : kind=0 or (kind=1 and g*"+
      std::to_string(kappa)+"<=t<(g+1)*"+std::to_string(kappa)+") or (kind=2 and t=g) }");
  auto grouped=projection.dependencies.Union(projection.requested_events.ApplyRange(event_members)
      .ApplyRange(projection.tasks.ImageIdentity()));
  int n=projection.stages.size();
  std::vector<std::uint32_t> order;
  auto edges=runtime.dependencies;
  std::stable_sort(edges.begin(),edges.end(),[](auto const&a,auto const&b){return a.consumer<b.consumer;});
  for (auto const& e:BuildVariantStageSchedule(edges,model.stages.size()).schedule)
    for (int s=0;s<n;++s) if (projection.stages[s].logical_stage==int(e.stage)) order.push_back(s);
  std::vector<int> first(model.stages.size(),-1),last(first),level(n,0);
  for (int s=0;s<n;++s) {int l=projection.stages[s].logical_stage;if(first[l]<0) first[l]=s;last[l]=s;}
  std::vector<std::vector<int>> stage_preds(n);
  for (auto const& e:edges) stage_preds[first[e.consumer]].push_back(last[e.producer]);
  for (int s=1;s<n;++s) if (projection.stages[s].logical_stage==projection.stages[s-1].logical_stage) stage_preds[s].push_back(s-1);
  for (int s:order) for (int p:stage_preds[s]) level[s]=std::max(level[s],level[p]+1);
  int begin=argc==7 ? std::stoi(argv[5]) : 1,end=argc==7 ? std::stoi(argv[6]) : 128;
  if(begin<1 || begin>end || end>128)throw std::invalid_argument("invalid proof interval");
  auto resident_attr=module->getOperation()->getAttrOfType<mlir::IntegerAttr>("tilemega.solved_grid");
  if (!resident_attr) throw std::invalid_argument("need a compiled resident-grid witness on input CG");
  int resident=resident_attr.getInt();
  std::vector<std::vector<analysis::QuasiPolynomial::PolynomialInterval>> intervals;
  std::set<int> cuts{begin,end+1};
  for (auto const& stage:projection.stages) {
    intervals.push_back(stage.task_count.QuadraticIntervals("S",begin,end));
    for (auto const& p:intervals.back()) {cuts.insert(p.begin);cuts.insert(p.end+1);}
  }
  std::ofstream proofs(out/"proofs.tsv"),samples(out/"samples.tsv"),counts_out(out/"counts.tsv");
  proofs << "family\tgrid\tbegin\tend\ttotal\tbijective\tdense\tacyclic\tresident\tlevel_exact\tspan_bound\tstatus\n";
  samples << "family\tgrid\tseq\tnodes\tidentical\ttemplate\tnative\n";
  counts_out << "stage\tcount\n";
  for (int s=0;s<n;++s) counts_out << s << '\t' << projection.stages[s].task_count.ToString() << '\n';
  int failures=0;
  std::vector<int> grids=argc>=5 ? std::vector<int>{projection_grid} : std::vector<int>{256,340};
  for (int grid:grids) {
    auto local_cuts=cuts;
    if(argc>=5){int step=std::getenv("SYMBOLIC_PIECE_STEP") ? std::stoi(std::getenv("SYMBOLIC_PIECE_STEP")) : 4;
      if(step<1)throw std::invalid_argument("invalid proof piece width");for(int s=begin;s<=end;s+=step)local_cuts.insert(s);}
    for (int s=0;s<n;++s) {
      int previous=-1;
      for (int seq=begin;seq<=end;++seq) {
        analysis::ParamBinding theta;theta.Bind("S",seq);
        int count=projection.stages[s].task_count.Eval(theta),width=std::max(1,(count+grid-1)/grid);
        if (width!=previous) local_cuts.insert(seq);
        previous=width;
      }
    }
    std::vector<int> boundaries(local_cuts.begin(),local_cuts.end());
    std::vector<AffineCountPiece> pieces;
    long max_nodes=0;
    for (std::size_t i=1;i<boundaries.size();++i) {
      AffineCountPiece p;p.begin=boundaries[i-1];p.end=boundaries[i]-1;
      if (p.begin<begin || p.end>end || p.begin>p.end) continue;
      long piece_nodes=0;
      for (int s=0;s<n;++s) {
        auto const& is=intervals[s];auto it=std::find_if(is.begin(),is.end(),[&](auto const& c){return c.begin<=p.begin && c.end>=p.end;});
        if (it==is.end() || it->coefficients[2]!="0") throw std::runtime_error("count not affine on exact ISL piece");
        p.counts.push_back(it->coefficients[0]+"+"+it->coefficients[1]+"*S");
        analysis::ParamBinding theta;theta.Bind("S",p.begin);
        long a=projection.stages[s].task_count.Eval(theta);theta.Bind("S",p.end);
        long b=projection.stages[s].task_count.Eval(theta);
        p.band_width.push_back(std::max(1L,(a+grid-1)/grid));piece_nodes+=std::max(a,b);
      }
      max_nodes=std::max(max_nodes,piece_nodes);pieces.push_back(std::move(p));
    }
    for (std::string family:{"legacy_grid_stride","rotate","band","wavefront"}) {
      if(argc>=5 && family!=argv[3])continue;
      std::string stem=family+"_g"+std::to_string(grid);
      try {
        std::optional<ParametricPlacement> joined;
        bool all_pass=true;
        auto proof_pieces=pieces;
        if(family=="wavefront" && (grid&(grid-1))!=0) {
          // Exhaust the COMPLETE finite integer interval, rather than sample
          // it. Wide-parameter rank subsets at a non-power-of-two modulus
          // make ISL's redundant-inequality elimination explode. The template
          // formula is unchanged; these are certificate pieces, not binaries.
          proof_pieces.clear();
          for(auto const& p:pieces)for(int seq=p.begin;seq<=p.end;++seq){
            auto single=p;single.begin=single.end=seq;proof_pieces.push_back(std::move(single));
          }
        }
        for (auto const& piece:proof_pieces) {
          std::cout << "PROVE " << stem << " S=" << piece.begin << ".." << piece.end << '\n' << std::flush;
          auto p=BuildParametricTemplate(projection.tasks,grouped,order,{piece},{grid},resident,family,level);
          auto check=p;
          if(piece.begin==piece.end) {
            analysis::ParamBinding theta;theta.Bind("S",piece.begin).Bind("G",grid);
            check.tasks=check.tasks.BindParams(theta);check.dependencies=check.dependencies.BindParams(theta);
            check.pi_sigma=check.pi_sigma.BindParams(theta);check.rank=check.rank.BindParams(theta);check.grid_limit=check.grid_limit.BindParams(theta);
          }
          auto proof=ProveParametricPlacement(check,&std::cout,piece.begin==piece.end ? grid : 0);bool level_exact=true;
          if (family=="wavefront") {
            auto levels=check.rank.ApplyRange(map("{ [l,t] -> [l] }"));
            auto permitted=levels.ApplyRange(map("{ [l] -> [l+1] }")).ApplyRange(levels.Reverse());
            auto forward=check.dependencies.Reverse();
            auto witnesses=forward.Subtract(forward.Subtract(permitted)).Image();
            auto needed=levels.IntersectRange("{ [l] : l>0 }").Reverse().Image();
            level_exact=needed.IsSubset(witnesses);
          }
          auto across=check.pi_sigma.Reverse().ApplyRange(check.dependencies.Reverse()).ApplyRange(check.pi_sigma);
          long lo=0,hi=max_nodes;
          while (lo<hi) {
            long mid=(lo+hi)/2;
            auto bound=map("{ [pw,ps] -> [cw,cs] : -"+std::to_string(mid)+"<=cs-ps<="+std::to_string(mid)+" }");
            if (across.IsSubset(bound)) hi=mid;else lo=mid+1;
          }
          bool pass=proof.passed()&&level_exact;all_pass&=pass;
          proofs << family << '\t' << grid << '\t' << piece.begin << '\t' << piece.end << '\t' << proof.total << '\t' << proof.bijective << '\t' << proof.dense << '\t' << proof.acyclic << '\t' << proof.resident << '\t' << level_exact << '\t' << lo << '\t' << (pass ? "PASS" : "FAIL") << '\n';proofs.flush();
          if (!joined) joined=p;
          else {
            joined->tasks=joined->tasks.Union(p.tasks);joined->dependencies=joined->dependencies.Union(p.dependencies);
            joined->pi_sigma=joined->pi_sigma.Union(p.pi_sigma);joined->rank=joined->rank.Union(p.rank);
            joined->grid_limit=joined->grid_limit.Union(p.grid_limit);
          }
        }
        auto p=*joined;
        std::ofstream(out/(stem+".pi_sigma.isl")) << p.pi_sigma.ToString() << '\n';
        std::ofstream(out/(stem+".rank.isl")) << p.rank.ToString() << '\n';
        std::ofstream(out/(stem+".dependencies.isl")) << p.dependencies.ToString() << '\n';
        std::ofstream(out/(stem+".tasks.isl")) << p.tasks.ToString() << '\n';
        if (!all_pass) {++failures;continue;}
        for (int seq:{1,32,64,96,128}) {
          if(seq<begin || seq>end)continue;
          analysis::ParamBinding theta;theta.Bind("S",seq).Bind("G",grid);
          std::vector<int> counts;for (auto const& s:projection.stages) counts.push_back(s.task_count.Eval(theta));
          auto symbolic=EvaluateParametricPlacement(p,theta,counts,grid);
          auto graph=codegen::MaterializeRuntimeTaskGraph(counts,{},grid);
          analysis::VisitFiniteRelation(ctx,projection.dependencies.BindParams(theta).ToString(),4,[&](long const* e) {
            graph.successors.at(graph.stage_offsets.at(e[2])+e[3]).push_back(graph.stage_offsets.at(e[0])+e[1]);
          });
          for (auto& row:graph.successors) {std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());}
          PlanRequest r;r.counts=counts;r.grid=grid;r.graph=&graph;r.stage_order=order;
          r.physical_worker.resize(grid);std::iota(r.physical_worker.begin(),r.physical_worker.end(),0);
          if (family=="rotate") r.mode=dialect::PlacementMode::kRotate;
          else if (family=="band" || family=="wavefront") {r.mode=dialect::PlacementMode::kTemplate;r.params={family=="band" ? 0 : 1};}
          MaterializedPlan native;std::string error;
          if (!MaterializePlanPlacement(r,&native,&error) || !CheckPlanLegality(graph,native,&error)) throw std::runtime_error(error);
          auto a=serial(symbolic),b=serial(native);bool same=a==b;
          auto prefix=stem+"_s"+std::to_string(seq);
          std::ofstream(out/(prefix+".template.tsv")) << a;std::ofstream(out/(prefix+".native.tsv")) << b;
          samples << family << '\t' << grid << '\t' << seq << '\t' << graph.successors.size() << '\t' << same << '\t' << prefix << ".template.tsv\t" << prefix << ".native.tsv\n";samples.flush();
          if (!same) ++failures;
        }
      } catch (std::exception const& e) {++failures;std::cerr << stem << ": " << e.what() << '\n';}
      std::cout << "SYMBOLIC_TEMPLATE " << stem << " done\n" << std::flush;
    }
  }
  return failures ? 1 : 0;
} catch (std::exception const& e) {std::cerr << e.what() << '\n';return 2;}
