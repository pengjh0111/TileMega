// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/BuiltinOps.h>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace tilemega;
int main(int argc,char** argv)try {
  if(argc!=4)throw std::invalid_argument("audit_selected CG TARGET OUTPUT.tsv");
  analysis::IslContext isl;mlir::MLIRContext ctx;ctx.getOrLoadDialect<dialect::CGDialect>();
  auto m=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&ctx);if(!m)throw std::invalid_argument("invalid CG");
  auto value=[&](char const* key){auto a=m->getOperation()->getAttrOfType<mlir::IntegerAttr>(key);if(!a)throw std::invalid_argument(key);return int(a.getInt());};
  int seq=value("tilemega.solved_seq"),past=value("tilemega.solved_past"),grid=value("tilemega.solved_grid"),res=value("tilemega.solved_residency");
  auto model=solver::ModelDescription::FromCouplingGraph(*m,{seq,past,seq+past},"price-audit");
  auto runtime=codegen::ReadRuntimePlan(*m);std::vector<solver::GemmConfig> geometry;
  for(auto const& g:runtime.gemms)geometry.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  auto semantic=solver::InstantiateModelTasks(model,geometry);auto target=TargetSpec::FromJson(argv[2]);solver::CostModel cost(target,model.dtype);
  solver::RuntimeProjectionOptions options{grid,model.dtype==solver::ScalarType::kBF16 ? 128 : 256,value("tilemega.solved_kappa")};options.count_wait_entries=false;
  auto projection=solver::ProjectRuntimeQueues(model,runtime,options);
  std::ofstream out(argv[3]);out<<std::setprecision(12)<<"runtime_stage\tlogical_stage\tcombine\ttasks\tprice_task0_ns\tactive_ctas_per_sm\tpricing\n";
  for(std::size_t s=0;s<projection.stages.size();++s){
    auto const& p=projection.stages[s];auto const& stage=model.stages[p.logical_stage];int n=p.task_count.Eval({});
    auto const& g=geometry.at(stage.IsCollective() ? stage.gemm : 0);int chunks=stage.IsCollective() ? cost.Chunks(model.gemms.at(stage.gemm),g) : 1;double ns=0;
    if(p.combine)ns=cost.CombineStageNs(model.gemms.at(stage.gemm),chunks,model.dims)/std::max(1.,std::ceil(double(n)/grid));
    else {
      auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& x){return x.stage==p.logical_stage && (!stage.IsCollective() || x.op.kind==analysis::OperatorKind::kMatmul);});
      if(found==model.task_semantics.end())throw std::runtime_error("semantic input missing");
      auto input=solver::DeriveModelTaskInput(model,*found,semantic,stage.IsCollective() ? &g : nullptr);
      auto traits=solver::ModelTaskTraits(model,p.logical_stage,g);analysis::ParamBinding coordinates;
      // Physical task zero has all-zero local coordinates in these templates.
      for(auto const& name:input.cost_coordinates)coordinates.Bind(name,0);
      ns=solver::PriceTaskInstances(cost,input,traits,{res},model,chunks,{coordinates},1.).at(0);
    }
    out<<s<<'\t'<<p.logical_stage<<'\t'<<p.combine<<'\t'<<n<<'\t'<<ns<<"\t1\t"<<(p.combine ? "whole_stage_divided_by_waves" : "derived_access_task_instance")<<'\n';out.flush();
  }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
