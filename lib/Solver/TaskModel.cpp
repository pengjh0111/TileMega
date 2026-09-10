// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
analysis::OperatorGraph InstantiateModelTasks(ModelDescription const& model,
                                            std::vector<GemmConfig> const& configs) {
  analysis::IslReferenceAudit audit(__func__);
  if (model.task_semantics.empty() || configs.size()!=model.gemms.size())
    throw std::invalid_argument("task pricing requires CG semantics and every GEMM configuration");
  analysis::SemanticGraph semantics;
  analysis::Granularity granularity;
  std::set<std::string> names;
  for (auto const& input:model.task_semantics) {
    auto const& op=input.op;
    if (!names.insert(op.name).second) throw std::invalid_argument("duplicate semantic cost task");
    auto const& stage=model.stages.at(input.stage);
    semantics.ops.push_back(op);
    for (auto const& [dim,tile]:input.tiles) granularity.Tile(op.name,dim,tile);
    if (stage.gemm<0) continue;
    auto const& config=configs.at(stage.gemm);
    if (config.tile_m<=0 || config.tile_n<=0 || config.tile_k<=0 || config.split_k<=0)
      throw std::invalid_argument("invalid candidate task granularity");
    if (op.result.axes.size()!=2 || op.result_map.results.size()!=2)
      throw std::invalid_argument("collective output rank is not implemented");
    int tiles[]={config.tile_m,config.tile_n};
    for (int axis=0;axis<2;++axis) {
      auto const& index=op.result_map.results[axis];
      if (index.kind!=analysis::IndexResult::Kind::kAffine || index.terms.size()!=1 ||
          !index.terms[0].coefficient.IsLiteral(1) || !index.terms[0].group.IsLiteral(1))
        throw std::invalid_argument("collective output requires unit iteration indexing");
      granularity.Tile(op.name,index.terms[0].dim,analysis::ClosedForm::Constant(tiles[axis]));
    }
    if (!op.reduction.splittable) continue;
    auto const* reduction=op.Dim(op.reduction.dim);
    if (!reduction) throw std::invalid_argument("collective reduction axis is missing");
    auto extent=reduction->extent.Eval({},{});
    long k_tiles=(extent+config.tile_k-1)/config.tile_k;
    long chunks=std::min<long>(config.split_k,k_tiles);
    if (chunks>1)
      granularity.Split(op.name,reduction->extent.CeilDiv(analysis::ClosedForm::Constant(chunks)));
  }
  return analysis::Instantiate(semantics,granularity);
}

DerivedTaskInput DeriveModelTaskInput(ModelDescription const& model,
                                    ModelTaskSemantics const& semantic,
                                    analysis::OperatorGraph const& graph,
                                    GemmConfig const* config) {
  analysis::IslReferenceAudit audit(__func__);
  auto const* task=graph.Find(semantic.op.name);
  if (!task) throw std::invalid_argument("semantic cost task is absent from candidate graph");
  analysis::TaskWorkOptions options;
  if (config && semantic.op.reduction.splittable)
    options.reduction_tiles.emplace(semantic.op.reduction.dim,
                                    analysis::ClosedForm::Constant(config->tile_k));
  auto work=analysis::DeriveTaskWork(semantic.op,*task,{},options);
  analysis::ArithmeticInputs arithmetic;
  arithmetic.reduction=work.nominal_task_reduce_extent;
  arithmetic.total=work.reduce_extent;
  arithmetic.width=task->tile.at(semantic.op.result.axes.size()-1).Eval({},{});
  arithmetic.dtype=semantic.op.dtype;
  auto signature=analysis::InstantiateArithmetic(semantic.op.arithmetic,arithmetic);
  analysis::RequireArithmeticImplementation(signature);
  (void)model;
  return {*task,std::move(work),std::move(signature)};
}
}  // namespace tilemega::solver
