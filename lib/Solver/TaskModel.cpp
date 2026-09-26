// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
void BindTaskDramProvenance(DerivedTaskInput& input,
    ModelTaskSemantics const& semantic,analysis::DramFloor const& floor,
    analysis::ParamBinding const& theta,bool serving) {
  auto accesses=DeriveModelTaskAccesses(semantic,input);
  std::vector<analysis::QuasiPolynomial> external_reads,external_writes,typed_reads;
  bool mixed_width=false;
  input.stream_bytes=floor.no_producer_bytes.Eval(theta);input.produced_live_bytes=0;
  for(auto const& [name,read]:accesses.reads) {
    // Serving plans are priced at a bound (B, past) point. Eliminating those
    // parameters before Barvinok cardinality avoids a very large parametric
    // polyhedron for indirect tokens and split vocabulary reductions.
    auto concrete=serving ? read.BindParams(theta) : read;
    auto found=floor.tensors.find(name);
    if(found==floor.tensors.end()){typed_reads.push_back(concrete.Card().Scale(2));continue;} // Split partials have a producer.
    auto const& tensor=found->second;
    typed_reads.push_back(concrete.Card().Scale(tensor.element_bytes));mixed_width|=tensor.element_bytes!=2;
    // Gather work uses a row-zero cardinality placeholder. For an entirely
    // read-only tensor every physical read is external, independent of the
    // actual index values bound to the unique-image floor.
    auto no_producer=serving ? tensor.no_producer.BindParams(theta) : tensor.no_producer;
    auto external=tensor.writes.ImageCard().Eval(theta)==0 ? concrete : concrete.ApplyRange(no_producer.ImageIdentity());
    external_reads.push_back(external.Card().Scale(tensor.element_bytes));
    auto produced=concrete.Subtract(external);
    if(produced.ImageCard().Eval(theta)>0)
      input.produced_live_bytes+=tensor.writes.ImageCard().Eval(theta)*tensor.element_bytes;
  }
  for(auto const& [name,write]:accesses.writes) {
    auto found=floor.tensors.find(name);if(found==floor.tensors.end())continue;
    auto const& tensor=found->second;
    auto concrete=serving ? write.BindParams(theta) : write;
    auto external=serving ? tensor.external_writes.BindParams(theta) : tensor.external_writes;
    external_writes.push_back(concrete.ApplyRange(external.ImageIdentity()).Card().Scale(tensor.element_bytes));
  }
  if(mixed_width && !input.physical_read_bytes)input.physical_read_bytes=analysis::QuasiPolynomial::Sum(typed_reads);
  input.no_producer_read_bytes=analysis::QuasiPolynomial::Sum(external_reads);
  input.external_write_bytes=analysis::QuasiPolynomial::Sum(external_writes);
}
TaskMemoryTraffic DeriveTaskMemoryTraffic(DerivedTaskInput const& input,
    analysis::ParamBinding const& theta, analysis::ParamBinding const& coordinates,
    int read_element_bytes, int write_element_bytes, analysis::AccessDomain domain) {
  if (read_element_bytes <= 0 || write_element_bytes <= 0)
    throw std::invalid_argument("task traffic needs positive element byte widths");
  auto count = [&](analysis::QuasiPolynomial const& work) {
    auto elements = work.BindCoordinates(coordinates).SubstituteParams(theta).Eval({});
    if (elements < 0) throw std::invalid_argument("negative access-derived task footprint");
    return double(elements);
  };
  bool physical = domain == analysis::AccessDomain::kPhysicalTensor;
  TaskMemoryTraffic traffic;
  traffic.global_read_bytes = read_element_bytes * count(physical
      ? input.work.read_elements : input.work.nominal_read_elements);
  if (physical && input.physical_read_bytes)
    traffic.global_read_bytes = count(*input.physical_read_bytes);
  traffic.global_write_bytes = write_element_bytes * count(physical
      ? input.work.write_elements : input.work.nominal_write_elements);
  if(physical && input.no_producer_read_bytes) {
    traffic.no_producer_read_bytes=count(*input.no_producer_read_bytes);
    traffic.external_write_bytes=count(*input.external_write_bytes);
  }
  traffic.produced_read_bytes=traffic.global_read_bytes-traffic.no_producer_read_bytes;
  if(traffic.produced_read_bytes<0)throw std::runtime_error("external reads exceed physical traffic");
  return traffic;
}

std::vector<TaskMemoryTraffic> DeriveTaskMemoryTrafficBatch(DerivedTaskInput const& input,
    analysis::ParamBinding const& theta,std::vector<analysis::ParamBinding> const& coordinates,
    int read_element_bytes,int write_element_bytes,analysis::AccessDomain domain) {
  if (read_element_bytes<=0 || write_element_bytes<=0)
    throw std::invalid_argument("task traffic needs positive element byte widths");
  bool physical=domain==analysis::AccessDomain::kPhysicalTensor;
  auto reads=(physical ? input.work.read_elements : input.work.nominal_read_elements).EvalPoints(theta,coordinates);
  if (physical && input.physical_read_bytes) {
    reads=input.physical_read_bytes->EvalPoints(theta,coordinates);
    read_element_bytes=1;
  }
  auto writes=(physical ? input.work.write_elements : input.work.nominal_write_elements).EvalPoints(theta,coordinates);
  std::vector<long> np(coordinates.size(),0),ew(coordinates.size(),0);
  if(physical && input.no_producer_read_bytes) {
    np=input.no_producer_read_bytes->EvalPoints(theta,coordinates);
    ew=input.external_write_bytes->EvalPoints(theta,coordinates);
  }
  std::vector<TaskMemoryTraffic> result(coordinates.size());
  for (std::size_t i=0;i<result.size();++i) {
    if (reads[i]<0 || writes[i]<0) throw std::invalid_argument("negative access-derived task footprint");
    result[i].global_read_bytes=double(reads[i])*read_element_bytes;
    result[i].global_write_bytes=double(writes[i])*write_element_bytes;
    result[i].no_producer_read_bytes=np[i];result[i].external_write_bytes=ew[i];
    result[i].produced_read_bytes=result[i].global_read_bytes-np[i];
    if(result[i].produced_read_bytes<0)throw std::runtime_error("external reads exceed physical batch traffic");
  }
  return result;
}

std::vector<double> PriceTaskInstances(CostModel const& cost,DerivedTaskInput const& input,
    BackendTraits const& traits,Residency residency,ModelDescription const& model,int chunks,
    std::vector<analysis::ParamBinding> const& coordinates,double active_ctas_per_sm,
    PrefetchPricing const* prefetch) {
  auto theta=model.MetricBindings();bool collective=traits.stages>0;
  bool regime_a=cost.options().regime_a && model.dtype==ScalarType::kBF16;
  int bytes=model.dtype==ScalarType::kBF16 ? 2 : 4;
  auto traffic=DeriveTaskMemoryTrafficBatch(input,theta,coordinates,bytes,bytes,
      collective && !(regime_a && cost.options().physical_traffic) ? analysis::AccessDomain::kNominalTile : analysis::AccessDomain::kPhysicalTensor);
  std::vector<long> reduction(coordinates.size(),0);
  if (collective) reduction=input.work.nominal_task_reduce_extent.EvalPoints(theta,coordinates);
  bool prefetchable=prefetch && prefetch->ns && !collective && input.scalar_access &&
      input.prefetch_operand>=0;
  std::vector<long> page(coordinates.size(),0);
  if (prefetchable) {
    auto found=input.scalar_access->reads.find(input.task.operands.at(input.prefetch_operand).tensor.name);
    if (found==input.scalar_access->reads.end())
      throw std::invalid_argument("prefetch operand has no runtime read relation");
    page=found->second.Card().EvalPoints(theta,coordinates);
  }
  // Within one immutable task signature these are every coordinate-dependent
  // quantity consumed by TaskCostImpl. Equal work classes have exactly equal
  // prices; no averaging, sampling, stage-kind rule or fitted shortcut occurs.
  std::map<std::tuple<double,double,long,double,double>,double> classes;
  std::map<std::tuple<double,double,long>,double> prefetch_classes;
  std::vector<double> result;result.reserve(coordinates.size());
  if (prefetch && prefetch->ns) prefetch->ns->assign(coordinates.size(),0.0);
  for (std::size_t i=0;i<coordinates.size();++i) {
    auto key=std::make_tuple(traffic[i].global_read_bytes,traffic[i].global_write_bytes,reduction[i],
        regime_a?traffic[i].no_producer_read_bytes:0.0,regime_a?traffic[i].external_write_bytes:0.0);
    auto found=classes.find(key);
    if (found==classes.end()) found=classes.emplace(key,cost.TaskInstanceNs(
        input,traits,residency,model,chunks,coordinates[i],active_ctas_per_sm,nullptr,
        collective ? nullptr : &traffic[i])).first;
    result.push_back(found->second);
    double const fetched=double(page[i])*bytes;
    if (!prefetchable || fetched<=0 || page[i]*bytes%16 || fetched>prefetch->page_bytes) continue;
    // The same task with that operand served from shared memory, which is how
    // the model already prices a fused input; no coefficient is introduced for
    // the mechanism.  A load phase that still fetches another operand from
    // global keeps its latency, so a body whose prefetched operand shares its
    // phase with a re-read is credited nothing: that is the model's answer,
    // recorded rather than adjusted (PIPELINE/summary.md).
    auto local=traffic[i];
    local.global_read_bytes=std::max(0.0,local.global_read_bytes-fetched);
    local.local_read_bytes+=fetched;
    local.local_read_operands.insert(input.prefetch_operand);
    auto local_key=std::make_tuple(local.global_read_bytes,local.global_write_bytes,reduction[i]);
    auto cheaper=prefetch_classes.find(local_key);
    if (cheaper==prefetch_classes.end()) cheaper=prefetch_classes.emplace(local_key,
        cost.TaskInstanceNs(input,traits,residency,model,chunks,coordinates[i],
                            active_ctas_per_sm,&local,nullptr)).first;
    (*prefetch->ns)[i]=std::max(0.0,found->second-cheaper->second);
  }
  return result;
}


DerivedTaskInput DeriveCombineTaskInput(ModelDescription const& model,int stage,
    GemmConfig const& config,analysis::OperatorGraph const& graph,
    int threads,bool tile_ownership,bool fp32_partials) {
  using namespace analysis;
  auto semantic=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),
      [&](auto const& s){return s.stage==stage && s.op.reduction.splittable;});
  if (semantic==model.task_semantics.end() || threads<=0)
    throw std::invalid_argument("combine lacks split semantics or launch width");
  auto const* declared=graph.Find(semantic->op.reduction.combiner);
  if (!declared || declared->output.axes.size()<2 || declared->operands.size()!=1)
    throw std::invalid_argument("combine requires the instantiated partial tensor");
  auto task=*declared;
  if (!tile_ownership)
    task.tile.assign(task.output.axes.size(),ClosedForm::Constant(1));
  SemanticOp reduction;
  reduction.name=task.name;reduction.kind=OperatorKind::kReduction;
  reduction.dtype=semantic->op.dtype;reduction.result=task.output;
  for (auto const& a:task.output.axes) {
    reduction.domain.push_back({a.name,a.extent,a.origin});
    reduction.result_map.results.push_back(IndexResult::Dim(a.name));
  }
  auto const& partial=task.operands.front().tensor;
  auto const& chunk=partial.axes.back();
  reduction.domain.push_back({chunk.name,chunk.extent,chunk.origin,IteratorType::kReduction});
  SemanticOperand read;read.tensor=partial;read.map=reduction.result_map;
  read.map.results.push_back(IndexResult::Dim(chunk.name));reduction.operands.push_back(read);
  // Residual addition is an explicit semantic phase on the same runtime
  // stage. Only the external operand is read here; the GEMM output is the
  // rounded register result of reducing the partial tensor.
  for (auto const& phase:model.task_semantics) {
    if (!fp32_partials || model.dtype!=ScalarType::kBF16 ||
        phase.stage!=stage || phase.op.kind!=OperatorKind::kPointwise) continue;
    if (phase.op.arithmetic!="add")
      throw std::invalid_argument("combine epilogue has no declared residual implementation");
    for (auto const& operand:phase.op.operands) {
      if (operand.tensor.name==semantic->op.result.name) continue;
      if (operand.tensor.axes.size()!=task.output.axes.size())
        throw std::invalid_argument("combine residual rank mismatch");
      Operand r;r.tensor=operand.tensor;r.producer=operand.producer;
      for (std::size_t axis=0;axis<task.output.axes.size();++axis)
        r.axes.push_back(OperandAxisMap::Indexed(axis));
      task.operands.push_back(r);
      auto mapped=operand;mapped.map=reduction.result_map;reduction.operands.push_back(mapped);
    }
  }
  auto work=DeriveTaskWork(reduction,task,{});
  auto ownership_semantic=*semantic;ownership_semantic.element_chunk=!tile_ownership;
  auto ownership=ProjectTaskOwnership(ownership_semantic,task,model.stages.at(stage),threads);
  RuntimeScalarAccess accesses;accesses.ownership=ownership;
  accesses.writes=ownership.ApplyRange(ElementAccess(task,BuildWriteMap(task),{},AccessDomain::kPhysicalTensor));
  std::vector<QuasiPolynomial> read_elements,read_bytes;
  int element_bytes=model.dtype==ScalarType::kBF16 ? 2 : 4;
  for (std::size_t i=0;i<task.operands.size();++i) {
    auto relation=ownership.ApplyRange(ElementAccess(task,BuildReadMap(task,i),{},AccessDomain::kPhysicalTensor));
    accesses.reads.emplace(task.operands[i].tensor.name,relation);
    auto count=relation.Card();read_elements.push_back(count);
    read_bytes.push_back(count.Scale(i==0 && fp32_partials ? 4 : element_bytes));
  }
  work.read_elements=QuasiPolynomial::Sum(read_elements);
  work.write_elements=accesses.writes.Card();work.task_count=accesses.writes.Reverse().ImageCard();
  ArithmeticInputs arithmetic;arithmetic.reduction=work.task_reduce_extent;
  auto signature=InstantiateArithmetic("sum",arithmetic);
  for (std::size_t i=1;i<task.operands.size();++i) {
    auto add=InstantiateArithmetic("add",{});
    signature.flops_per_output_element.numerator=signature.flops_per_output_element.numerator.Add(
        add.flops_per_output_element.numerator);
  }
  DerivedTaskInput result{task,work,signature,{"q"},
      codegen::ScalarTaskDataflow(codegen::TaskKind::kGemmCombine),accesses};
  result.physical_read_bytes=QuasiPolynomial::Sum(read_bytes);
  return result;
}

BackendTraits ModelTaskTraits(ModelDescription const& model, int index,
                              GemmConfig const& config) {
  auto collective = model.dtype == ScalarType::kBF16
      ? (model.serving
             ? ServingBF16Traits(config.tile_m, config.tile_n, config.tile_k,
                                 config.stages)
             : TensorBF16Traits(config.tile_m, config.tile_n, config.tile_k,
                                config.stages))
      : SimtF32Traits(config.tile_m, config.tile_n, config.tile_k, config.stages);
  auto const& stage = model.stages.at(index);
  bool uses_collective = false;
  for (auto const& semantic : model.task_semantics)
    if (semantic.stage == index)
      uses_collective |= semantic.op.kind == analysis::OperatorKind::kMatmul;
  if (uses_collective) return collective;
  if (stage.kind == StageKind::kFusedAttention) {
    BackendTraits traits;
    traits.threads = 128;
    traits.smem_bytes = codegen::ServingAttentionSharedBytes(stage.width);
    traits.shape_legal = stage.width == 64 || stage.width == 128;
    return traits;
  }
  auto resources = codegen::ReadSimtTaskResources(
      static_cast<codegen::TaskKind>(stage.kind), collective.threads);
  BackendTraits traits;
  traits.threads = resources.threads;
  traits.smem_bytes = resources.shared_bytes;
  traits.shape_legal = true;
  return traits;
}

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
    if (!model.serving) {
      for(auto const& [dim,tile]:input.tiles)granularity.Tile(op.name,dim,tile);
    } else for (std::size_t axis=0;axis<op.result.axes.size();++axis) {
      auto found=input.tiles.find(op.result.axes[axis].name);
      if(found==input.tiles.end())continue;
      auto const& index=op.result_map.results.at(axis);
      if(index.kind==analysis::IndexResult::Kind::kAffine &&
         index.terms.size()==1 && index.terms[0].coefficient.IsLiteral(1) &&
         index.terms[0].group.IsLiteral(1))
        granularity.Tile(op.name,index.terms[0].dim,found->second);
    }
    if (stage.gemm<0) continue;
    auto const& config=configs.at(stage.gemm);
    if (config.tile_m<=0 || config.tile_n<=0 || config.tile_k<=0 || config.split_k<=0)
      throw std::invalid_argument("invalid candidate task granularity");
    bool grouped = model.serving && op.result.axes.size()==3 &&
                   op.result.axes[1].name=="g" &&
                   op.result.axes[2].name=="u";
    if ((!grouped && op.result.axes.size()!=2) ||
        op.result_map.results.size()!=op.result.axes.size())
      throw std::invalid_argument("collective output rank is not implemented: "+op.name);
    if (grouped && op.result.axes[2].extent.Eval({}, {}) % config.tile_n)
      throw std::invalid_argument("packed group width must divide the serving N tile: "+op.name);
    for (int axis=0;axis<int(op.result.axes.size());++axis) {
      auto const& index=op.result_map.results[axis];
      if (index.kind!=analysis::IndexResult::Kind::kAffine || index.terms.size()!=1 ||
          !index.terms[0].coefficient.IsLiteral(1) || !index.terms[0].group.IsLiteral(1))
        throw std::invalid_argument("collective output requires unit iteration indexing");
      int tile = axis==0 ? config.tile_m :
                 ((grouped && axis==1) ||
                  (model.serving && op.result.axes[axis].name=="tile")
                     ? 1 : (model.serving && op.result.axes[axis].name=="i"
                                ? config.tile_n/2 : config.tile_n));
      granularity.Tile(op.name,index.terms[0].dim,
                       analysis::ClosedForm::Constant(tile));
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
  auto const& stage=model.stages.at(semantic.stage);
  if(model.serving) {
    switch(stage.kind) {
      case StageKind::kGemm:
        if(semantic.op.arithmetic=="swiglu_gemm")result.serving_body_kind="gemm_swiglu";
        else if(semantic.op.arithmetic=="argmax_gemm")result.serving_body_kind="gemm_argmax_partial";
        else result.serving_body_kind=semantic.op.operands.size()>2?"gemm_residual":"gemm_store";
        break;
      case StageKind::kFusedAttention:
        result.serving_body_kind=std::string("fused_attention_")+
            (model.dims.seq==1?"decode_d":"prefill_d")+std::to_string(stage.width);
        break;
      case StageKind::kAttentionMerge: result.serving_body_kind="attention_merge";break;
      case StageKind::kRMSNorm: result.serving_body_kind="rmsnorm";break;
      case StageKind::kEmbedding: result.serving_body_kind="embedding";break;
      case StageKind::kArgmaxReduce: result.serving_body_kind="argmax_reduce";break;
      default: break;
    }
  }
  if(config && model.serving && stage.kind==StageKind::kGemm &&
     semantic.op.arithmetic=="argmax_gemm")
    result.collective_k_extent=model.gemms.at(stage.gemm).k;
  if(model.serving && stage.kind==StageKind::kFusedAttention &&
     stage.attention_kv_block>0) {
    bool prefill=model.dims.seq>1;
    result.serving_attention=DerivedTaskInput::ServingAttention{
        prefill?1:(model.serving_capacity+stage.attention_kv_block-1)/stage.attention_kv_block,
        prefill?model.dims.seq:stage.attention_kv_block,model.dims.total,64,
        int(stage.width),prefill?int(stage.attention_query_rows):int(stage.group),
        prefill};
  }
  if (!config && runtime_ownership) {
    int threads=ModelTaskTraits(model,semantic.stage,{}).threads;
    result.scalar_access.emplace();
    result.work=DeriveRuntimeScalarWork(model,semantic,*task,std::move(result.work),threads,&*result.scalar_access);
    result.cost_coordinates={"q"};
    auto kind=static_cast<codegen::TaskKind>(model.stages.at(semantic.stage).kind);
    result.scalar_flow=codegen::ScalarTaskDataflow(kind);
    // The body's declaration is honoured only on the derived frontier, exactly
    // as the executor honours it (`PrefetchBytes` in ModelHarness.cuh).
    int const declared=codegen::ScalarPrefetchOperand(kind);
    if (declared>=0 && declared<int(task->operands.size()) && task->operands[declared].producer.empty())
      result.prefetch_operand=declared;
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
