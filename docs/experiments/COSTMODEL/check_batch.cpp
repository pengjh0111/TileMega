// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/TaskModel.h>
#include <iostream>
int main(int argc,char** argv) try {
  if(argc!=3) throw std::runtime_error("check_batch EXPORT TARGET");
  tilemega::analysis::IslContext isl;mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  auto module=tilemega::frontend::TorchExportImporter{}.Import(argv[1],context);
  using namespace tilemega;using namespace tilemega::solver;
  auto model=ModelDescription::FromCouplingGraph(*module,{128,3,131},"batch-traffic");
  std::vector<GemmConfig> configs(model.gemms.size(),{128,128,16,3,1});
  auto graph=InstantiateModelTasks(model,configs);auto target=TargetSpec::FromJson(argv[2]);
  CostModel cost(target,model.dtype);
  auto theta=model.MetricBindings();int checked=0;
  for(auto const& semantic:model.task_semantics) {
    auto const& stage=model.stages.at(semantic.stage);
    if(stage.IsCollective() && semantic.op.kind!=analysis::OperatorKind::kMatmul)continue;
    auto task=DeriveModelTaskInput(model,semantic,graph,stage.IsCollective()? &configs.at(stage.gemm):nullptr);
    auto n=task.work.task_count.SubstituteParams(theta).Eval({});
    std::vector<analysis::ParamBinding> points;
    for(long q:{0L,n/2,n-1}) if(q>=0 && q<n){analysis::ParamBinding p;p.Bind("q",q);points.push_back(p);}
    if(stage.IsCollective()) {
      points.clear();analysis::ParamBinding p;
      for(auto const& name:task.cost_coordinates)p.Bind(name,0);
      points.push_back(p);points.push_back(p);
    }
    auto batch=DeriveTaskMemoryTrafficBatch(task,theta,points,2,2);
    auto traits=ModelTaskTraits(model,semantic.stage,configs.front());
    auto prices=PriceTaskInstances(cost,task,traits,{1},model,1,points);
    for(std::size_t i=0;i<points.size();++i){auto one=DeriveTaskMemoryTraffic(task,theta,points[i],2,2);
      if(one.global_read_bytes!=batch[i].global_read_bytes || one.global_write_bytes!=batch[i].global_write_bytes)
        throw std::runtime_error("batch traffic differs from independent coordinate binding");
      if (prices[i]!=cost.TaskInstanceNs(task,traits,{1},model,1,points[i],1.0))
        throw std::runtime_error("equal-work price reuse differs from independent task pricing");++checked;}
  }
  std::cout << "BATCH_TRAFFIC PASS independently_bound_points=" << checked << '\n';
} catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
