// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingDerivation.h>
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

std::vector<ModelCouplingMetrics> InstantiateModelCouplings(
    ModelDescription const& model,std::vector<GemmConfig> const& configs,
    std::optional<std::pair<int,int>> stage_pair) {
  analysis::IslReferenceAudit audit(__func__);
  auto graph=InstantiateModelTasks(model,configs);
  std::map<std::string,int> stages;
  for (auto const& input:model.task_semantics) {
    stages.emplace(input.op.name,input.stage);
    if (input.op.reduction.splittable) stages.emplace(input.op.reduction.combiner,input.stage);
  }
  auto known=model.metric_bindings;
  for (auto const& name:{std::string("S"),std::string("past"),std::string("P"),std::string("L_s"),
                         model.seq_metric_parameter,model.past_metric_parameter})
    known.values.erase(name);
  for (auto const& [alias,canonical]:model.metric_aliases)
    if (canonical==model.seq_metric_parameter || canonical==model.past_metric_parameter)
      known.values.erase(alias);
  if (stage_pair) {
    // Restrict only which edges are queried. Operand access maps and task
    // spaces are unchanged, so the ordinary derivation remains the authority.
    for (auto& node:graph.nodes) {
      auto consumer=stages.at(node.name);
      node.operands.erase(std::remove_if(node.operands.begin(),node.operands.end(),
          [&](auto const& operand) {
            auto producer=stages.find(operand.producer);
            return consumer!=stage_pair->second || producer==stages.end() ||
                   producer->second!=stage_pair->first;
          }),node.operands.end());
    }
  }
  auto derived=analysis::CouplingDerivation{}.Derive(graph,known);
  std::vector<ModelCouplingMetrics> edges;
  for (auto const& edge:derived)
    edges.push_back({stages.at(edge.src.name),stages.at(edge.dst.name),
      edge.metrics.wait,edge.metrics.fanout,edge.metrics.volume,edge.metrics.count,edge.C,
      edge.src.name,edge.dst.name});
  return edges;
}

analysis::TaskAccesses DeriveModelTaskAccesses(ModelTaskSemantics const& semantic,
                                             DerivedTaskInput const& input) {
  analysis::IslReferenceAudit audit(__func__);
  analysis::TaskAccesses accesses;
  auto const& task=input.task;
  if (task.output.name.empty())
    throw std::invalid_argument("fusion output tensor identity is missing");
  if (input.scalar_access) {
    accesses.reads=input.scalar_access->reads;
    accesses.writes.emplace(task.output.name,input.scalar_access->writes);
    return accesses;
  }
  accesses.writes.emplace(task.output.name,analysis::ElementAccess(task,
      analysis::BuildWriteMap(task),{},analysis::AccessDomain::kPhysicalTensor));
  std::map<std::string,std::string> layouts;
  auto append=[&](analysis::TensorSpace const& tensor, analysis::CouplingRelation relation) {
    if (tensor.name.empty()) throw std::invalid_argument("fusion input tensor identity is missing");
    auto [layout,inserted]=layouts.emplace(tensor.name,tensor.layout_id);
    if (!inserted && layout->second!=tensor.layout_id)
      throw std::invalid_argument("fusion read union requires a common tensor layout");
    auto found=accesses.reads.find(tensor.name);
    if (found==accesses.reads.end()) accesses.reads.emplace(tensor.name,std::move(relation));
    else found->second=found->second.Union(relation);
  };
  // Complete element reads replace rectangular coupling projections; using
  // issued/nominal work here would incorrectly retain predicated tail bytes.
  if (!semantic.op.element_reads.empty()) {
    for (auto const& read:semantic.op.element_reads)
      append(read.tensor,analysis::ExactElementRead(semantic.op,task,read,{}));
  } else {
    for (std::size_t operand=0;operand<task.operands.size();++operand) {
      auto read=analysis::BuildReadMap(task,operand);
      append(read.tensor,analysis::ElementAccess(task,read,{},analysis::AccessDomain::kPhysicalTensor));
    }
  }
  return accesses;
}

namespace {
ModelFusionCandidate ComposeModelCandidate(ModelDescription const& model,
    std::vector<GemmConfig> const& configs, ModelTaskSemantics const* producer,
    ModelTaskSemantics const* consumer, bool runtime_ownership) {
  analysis::IslReferenceAudit audit(__func__);
  std::set<std::string> incoming;
  for (auto const& operand:consumer->op.operands)
    if (!operand.producer.empty()) incoming.insert(operand.producer);
  if (incoming!=std::set<std::string>{producer->op.name})
    throw std::invalid_argument("fusion consumer is not a single-producer chain node");
  auto graph=InstantiateModelTasks(model,configs);
  auto input=[&](ModelTaskSemantics const& semantic) {
    auto const& stage=model.stages.at(semantic.stage);
    GemmConfig const* config=stage.gemm<0 || semantic.op.kind!=analysis::OperatorKind::kMatmul
        ? nullptr : &configs.at(stage.gemm);
    if (config && config->split_k!=1)
      throw std::invalid_argument("fusion partial stage requires explicit combine ownership");
    return DeriveModelTaskInput(model,semantic,graph,config,runtime_ownership);
  };
  auto p=input(*producer),c=input(*consumer);
  auto pa=DeriveModelTaskAccesses(*producer,p),ca=DeriveModelTaskAccesses(*consumer,c);
  if (runtime_ownership) {
    if (model.dims.IsSymbolic())
      throw std::invalid_argument("runtime fusion candidate requires bound dimensions");
    auto theta=model.MetricBindings();
    for (auto* accesses:{&pa,&ca})
      for (auto* maps:{&accesses->reads,&accesses->writes})
        for (auto& [tensor,map]:*maps) map=map.BindParams(theta);
  }
  std::set<std::string> internal,external;
  for (auto const& [name,write]:pa.writes) {
    if (ca.reads.count(name)) internal.insert(name);
    if (model.exported_tensors.count(name)) external.insert(name);
    for (auto const& other:model.task_semantics) {
      if (&other==consumer) continue;
      for (auto const& operand:other.op.operands)
        if (operand.tensor.name==name) external.insert(name);
    }
  }
  auto accesses=analysis::ComposeFusionAccesses(pa,ca,internal,external);
  // Arithmetic phases keep distinct output domains. The coupled producer
  // work is re-indexed by the consumer relation, not averaged over fanout.
  auto producer_outputs=accesses.intermediate_tiles.at(*internal.begin()).Card();
  auto arithmetic=analysis::ComposeArithmetic({{p.arithmetic,std::move(producer_outputs)},
                                              {c.arithmetic,c.work.write_elements}});
  return {std::move(p),std::move(c),std::move(accesses),std::move(arithmetic),std::move(pa),std::move(ca)};
}
}  // namespace

ModelFusionCandidate DeriveLogicalFusionCandidate(ModelDescription const& model,
    std::vector<GemmConfig> const& configs, std::string const& producer_name,
    std::string const& consumer_name) {
  analysis::IslReferenceAudit audit(__func__);
  auto find=[&](std::string const& name) {
    auto found=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),
        [&](auto const& semantic) { return semantic.op.name==name; });
    if (found==model.task_semantics.end())
      throw std::invalid_argument("fusion semantic task is missing: "+name);
    return found;
  };
  auto producer=find(producer_name),consumer=find(consumer_name);
  if (std::next(producer)!=consumer)
    throw std::invalid_argument("fusion requires adjacent logical tasks");
  return ComposeModelCandidate(model,configs,&*producer,&*consumer,false);
}

ModelFusionCandidate DeriveModelFusionCandidate(ModelDescription const& model,
    std::vector<GemmConfig> const& configs, int producer_stage, int consumer_stage) {
  analysis::IslReferenceAudit audit(__func__);
  if (producer_stage<0 || consumer_stage!=producer_stage+1 ||
      consumer_stage>=static_cast<int>(model.stages.size()))
    throw std::invalid_argument("fusion requires adjacent runtime stages");
  ModelTaskSemantics const* producer=nullptr;
  ModelTaskSemantics const* consumer=nullptr;
  for (auto const& semantic:model.task_semantics) {
    auto assign=[&](auto& chosen) {
      if (chosen) throw std::invalid_argument("fusion stage has multiple logical tasks");
      chosen=&semantic;
    };
    if (semantic.stage==producer_stage) assign(producer);
    if (semantic.stage==consumer_stage) assign(consumer);
  }
  if (!producer || !consumer) throw std::invalid_argument("fusion stage lacks semantic task");
  return ComposeModelCandidate(model,configs,producer,consumer,true);
}

DerivedTaskInput DeriveModelTaskInput(ModelDescription const& model,
                                    ModelTaskSemantics const& semantic,
                                    analysis::OperatorGraph const& graph,
                                    GemmConfig const* config, bool runtime_ownership) {
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
  DerivedTaskInput result{*task,std::move(work),std::move(signature),task->Coordinates(),std::nullopt,std::nullopt};
  if (!config && runtime_ownership) {
    int threads=model.dtype==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
    result.scalar_access.emplace();
    result.work=DeriveRuntimeScalarWork(model,semantic,*task,std::move(result.work),threads,&*result.scalar_access);
    result.cost_coordinates={"q"};
    result.scalar_flow=codegen::ScalarTaskDataflow(static_cast<codegen::TaskKind>(model.stages.at(semantic.stage).kind));
  }
  if (!config && !runtime_ownership) {
    auto kind=static_cast<codegen::TaskKind>(model.stages.at(semantic.stage).kind);
    if (kind==codegen::TaskKind::kGemm && task->kind==analysis::OperatorKind::kPointwise)
      kind=codegen::TaskKind::kElementwise;
    result.scalar_flow=codegen::ScalarTaskDataflow(kind);
  }
  return result;
}
}  // namespace tilemega::solver
