// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Solver/JointPlacement.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/VariantSchedule.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <limits>

namespace tilemega::dialect {
struct PlacementSolveOptions {
  TargetSpec target;
  solver::ModelDims dims;
  int residency=1;
  int kappa=1;
  solver::HopCurve hop;
};
struct PlacementSolveResult {
  std::vector<solver::PlacementEvaluation> candidates;
  std::vector<solver::GemmConfig> geometry;
  int grid=0,kappa=1;
};
inline void WriteSolvedPlacement(mlir::ModuleOp module,
    solver::PlacementEvaluation const& selected,PlacementSolveOptions const& options) {
  mlir::OpBuilder b(module.getContext());
  int grid=int(selected.plan.queue.size());
  module->removeAttr(kPlacementTableAttr);
  if (selected.mode==PlacementMode::kEft) {
    std::vector<std::int64_t> worker,slot;
    for (std::size_t s=0;s<selected.plan.owner.size();++s)
      for (std::size_t t=0;t<selected.plan.owner[s].size();++t) {
        worker.push_back(selected.plan.owner[s][t]);slot.push_back(selected.plan.slot[s][t]);
      }
    module->setAttr(kPlacementTableAttr,b.getDictionaryAttr({
      b.getNamedAttr("worker",b.getDenseI64ArrayAttr(worker)),b.getNamedAttr("slot",b.getDenseI64ArrayAttr(slot)),
      b.getNamedAttr("seq",b.getI64IntegerAttr(options.dims.seq)),b.getNamedAttr("past",b.getI64IntegerAttr(options.dims.past)),
      b.getNamedAttr("grid",b.getI64IntegerAttr(grid))}));
  }
  for (auto placement:module.getOps<PlacementOp>()) {
    placement->removeAttr("mapping_mode");placement->removeAttr("params_map");
    placement->setAttr("mode",b.getStringAttr(PlacementModeName(selected.mode)));
    placement->setAttr("params",b.getDenseI64ArrayAttr(selected.params));
    placement->setAttr("window",b.getI64IntegerAttr(1));
    placement->setAttr("policy",b.getStringAttr(kPlacementPolicyAot));
    placement->setAttr("resident_only",b.getBoolAttr(true));
    auto function=analysis::CouplingRelation::FromIslText("[S,past] -> { [] -> ["+
        std::to_string(grid)+"] : S="+std::to_string(options.dims.seq)+
        " and past="+std::to_string(options.dims.past)+" }");
    placement->setAttr("grid_map",CouplingMapAttr::get(module.getContext(),function));
    placement->setAttr("resident_limit_map",CouplingMapAttr::get(module.getContext(),function));
  }
  module->setAttr("tilemega.solved_placement",b.getStringAttr(selected.name));
  module->setAttr("tilemega.solved_kappa",b.getI64IntegerAttr(options.kappa));
  module->setAttr("tilemega.solved_residency",b.getI64IntegerAttr(options.residency));
  module->setAttr("tilemega.solved_grid",b.getI64IntegerAttr(grid));
  module->setAttr("tilemega.solved_floor_ns",b.getF64FloatAttr(selected.bounds.lower_bound_ns));
  if (mlir::failed(mlir::verify(module))) throw std::invalid_argument("solved placement failed CG verification");
}

inline PlacementSolveResult SolveAndWritePlacement(mlir::ModuleOp module,
    PlacementSolveOptions const& options) {
  using namespace solver;
  if (!module || mlir::failed(mlir::verify(module)) || options.dims.seq<=0 ||
      options.dims.past<0 || options.residency<=0 || options.kappa!=1)
    throw std::invalid_argument("placement solve requires verified CG, bound theta and kappa=1 until grouped-event readiness is supplied");
  auto runtime=codegen::ReadRuntimePlan(module);
  auto model=ModelDescription::FromCouplingGraph(module,options.dims,"placement-pass");
  PlacementSolveResult result;result.grid=options.target.res.num_sms*options.residency;
  for (auto const& g:runtime.gemms) result.geometry.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  int threads=0,max_shared=0;
  for (std::size_t s=0;s<model.stages.size();++s) {
    auto const& stage=model.stages[s];
    auto const& g=result.geometry.at(stage.IsCollective() ? stage.gemm : 0);
    auto traits=ModelTaskTraits(model,int(s),g);
    threads=std::max(threads,traits.threads);max_shared=std::max(max_shared,traits.smem_bytes);
  }
  // Register residency still requires ptxas confirmation; one CTA is safe for
  // every legal body, whereas an unmeasured higher resident count is not.
  if (options.residency!=1 || max_shared>options.target.res.max_dynamic_smem_per_cta)
    throw std::invalid_argument("resident grid requires compiler-confirmed resource metadata");
  RuntimeProjectionOptions po{result.grid,threads,1};po.count_wait_entries=false;
  auto projection=ProjectRuntimeQueues(model,runtime,po);
  std::vector<int> counts;
  for (auto const& stage:projection.stages) counts.push_back(int(stage.task_count.Eval({})));
  auto tasks=projection.tasks.ToString(),dependencies=projection.dependencies.ToString();
  codegen::RuntimeExactDependencyDesc desc{tasks.c_str(),dependencies.c_str(),"S","past"};
  auto graph=codegen::MaterializeExactRuntimeTaskGraph(counts,desc,options.dims.seq,options.dims.past,result.grid);
  auto semantic_graph=InstantiateModelTasks(model,result.geometry);
  SimulatorInput input;input.graph=&graph;input.task_ns.resize(graph.successors.size());
  CostModel cost(options.target,model.dtype);
  auto theta=model.MetricBindings();
  for (std::size_t s=0;s<counts.size();++s) {
    auto const& projected=projection.stages[s];auto const& stage=model.stages[projected.logical_stage];
    auto const& g=result.geometry.at(stage.IsCollective() ? stage.gemm : 0);
    if (projected.combine) {
      int chunks=cost.Chunks(model.gemms.at(stage.gemm),g);
      double waves=std::ceil(double(counts[s])/result.grid);
      double ns=cost.CombineStageNs(model.gemms.at(stage.gemm),chunks,model.dims)/std::max(1.,waves);
      std::fill(input.task_ns.begin()+graph.stage_offsets[s],input.task_ns.begin()+graph.stage_offsets[s+1],ns);
      continue;
    }
    auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& x) {
      return x.stage==projected.logical_stage && (!stage.IsCollective() || x.op.kind==analysis::OperatorKind::kMatmul);
    });
    if (found==model.task_semantics.end()) throw std::invalid_argument("stage lacks derived semantic task costs");
    auto task=DeriveModelTaskInput(model,*found,semantic_graph,stage.IsCollective() ? &g : nullptr);
    auto traits=ModelTaskTraits(model,projected.logical_stage,g);
    int chunks=stage.IsCollective() ? cost.Chunks(model.gemms.at(stage.gemm),g) : 1;
    std::vector<analysis::ParamBinding> coordinates(counts[s]);
    if (task.scalar_access) {
      for (int t=0;t<counts[s];++t) coordinates[t].Bind("q",t);
    } else {
      auto ownership=ProjectTaskOwnership(*found,task.task,stage,threads).BindParams(theta);
      auto names=ownership.RangeDimNames();
      for (auto const& [physical,logical]:ownership.Points()) {
        if (physical.size()!=1 || physical[0]<0 || physical[0]>=counts[s])
          throw std::invalid_argument("task cost ownership disagrees with projected count");
        for (std::size_t i=0;i<names.size();++i) coordinates[physical[0]].Bind(names[i],logical[i]);
      }
    }
    for (int t=0;t<counts[s];++t)
      input.task_ns[graph.stage_offsets[s]+t]=cost.TaskInstanceNs(task,traits,{options.residency},model,chunks,coordinates[t],1.0);
  }
  PlanRequest request;request.graph=&graph;request.grid=result.grid;request.counts=counts;
  request.physical_worker.resize(result.grid);std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
  auto edges=runtime.dependencies;
  std::stable_sort(edges.begin(),edges.end(),[](auto const& a,auto const& b){return a.consumer<b.consumer;});
  for (auto const& entry:BuildVariantStageSchedule(edges,model.stages.size()).schedule)
    for (std::size_t s=0;s<projection.stages.size();++s)
      if (projection.stages[s].logical_stage==int(entry.stage)) request.stage_order.push_back(std::uint32_t(s));
  SimulatorOptions sim;sim.observed_task_times=true;sim.flat_hop=true;
  result.candidates=SolvePlacementCatalog(input,request,sim,options.hop);
  WriteSolvedPlacement(module,result.candidates.front(),options);
  return result;
}

struct PlacementSolvePass : mlir::PassWrapper<PlacementSolvePass,mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlacementSolvePass)
  PlacementSolvePass()=default;
  PlacementSolvePass(PlacementSolvePass const& other):PassWrapper(other) {}
  mlir::Pass::Option<std::string> target{*this,"target",llvm::cl::desc("Calibrated TargetSpec JSON")};
  mlir::Pass::Option<int> seq{*this,"seq",llvm::cl::init(4)};
  mlir::Pass::Option<int> past{*this,"past",llvm::cl::init(3)};
  llvm::StringRef getArgument() const final {return "tilemega-solve-placement";}
  llvm::StringRef getDescription() const final {return "Solve the placement catalog and write the selected Plan into CG";}
  void runOnOperation() override {
    try {
      PlacementSolveOptions options;options.target=TargetSpec::FromJson(target);
      options.dims={seq,past,seq+past};
      SolveAndWritePlacement(getOperation(),options);
    } catch (std::exception const& e) {getOperation().emitError(e.what());signalPassFailure();}
  }
};
inline void RegisterPlacementSolvePass() {mlir::PassRegistration<PlacementSolvePass>();}
} // namespace tilemega::dialect
