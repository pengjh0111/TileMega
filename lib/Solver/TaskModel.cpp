// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/DmConvReductionPartition.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <tilemega/Solver/BindingRequestTraffic.h>
#include <tilemega/Solver/DmVirtualGemmPartition.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskOwnershipGeometry.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tilemega::solver {
DerivedTaskInput RestrictVirtualTaskRows(DerivedTaskInput const& input,
    std::uint32_t live_rows,BackendTraits const& traits,
    analysis::ParamBinding const& theta) {
  using namespace analysis;
  if(!input.task.element_access || traits.tile_k<=0 || traits.stages<2)
    throw std::invalid_argument("live binding rows require an exact collective task");
  auto access=*input.task.element_access;
  auto const& original=access.semantic;
  if(original.kind!=OperatorKind::kMatmul || original.task_space.axes.size()!=3 ||
     access.partition.ownership.results.size()!=3)
    throw std::invalid_argument("live binding rows require virtual, row and column ownership");
  auto const& virtual_name=UnitTaskOwnershipDimension(original,0);
  auto const& row_name=UnitTaskOwnershipDimension(original,1);
  auto const* virtual_dim=original.Dim(virtual_name);
  auto const* row_dim=original.Dim(row_name);
  if(!virtual_dim || !virtual_dim->runtime || !virtual_dim->capacity ||
     virtual_dim->binding_source.empty() || !row_dim || row_dim->runtime ||
     row_dim->type!=IteratorType::kParallel || !row_dim->origin.IsLiteral(0) ||
     live_rows>row_dim->BoundExtent().Eval(theta,{}))
    throw std::invalid_argument("live binding rows differ from the capacity semantics");
  for(auto& dim:access.semantic.domain)if(dim.name==row_name) {
    dim.extent=ClosedForm::Constant(live_rows);dim.capacity.reset();
  }
  TaskWorkOptions options;
  if(original.reduction.splittable)
    options.reduction_tiles.emplace(original.reduction.dim,ClosedForm::Constant(traits.tile_k));
  auto result=input;
  result.task.element_access=std::make_shared<TaskElementAccess const>(std::move(access));
  result.work=DeriveTaskWork(result.task.element_access->semantic,result.task,theta,options);
  // A nonempty tail executes the same padded MMA tile. Empty subtiles are
  // handled by the binding-aware price path, but still occupy static task IDs.
  result.work.task_count=input.work.task_count;
  result.work.nominal_read_elements=input.work.nominal_read_elements;
  result.work.nominal_write_elements=input.work.nominal_write_elements;
  result.work.nominal_task_reduce_extent=input.work.nominal_task_reduce_extent;
  result.physical_read_bytes.reset();result.physical_write_bytes.reset();
  result.no_producer_read_bytes.reset();result.external_write_bytes.reset();
  result.produced_live_bytes=0;
  return result;
}
void BindTaskDramProvenance(DerivedTaskInput& input,
    ModelTaskSemantics const& semantic,analysis::DramFloor const& floor,
    analysis::ParamBinding const& theta,bool serving) {
  if(input.task.element_access) {
    auto const& op=input.task.element_access->semantic;
    bool requests=analysis::HasBindingRequests(op.result_map);
    for(auto const& read:op.operands)requests|=analysis::HasBindingRequests(read.map);
    for(auto const& read:op.element_reads)requests|=analysis::HasBindingRequests(read.map);
    for(auto const& write:op.additional_writes)requests|=analysis::HasBindingRequests(write.map);
    // Exact affine tasks also have typed main/side stores. Retain the
    // legacy scalar-runtime projection unless binding requests require this path.
    if(requests || !input.scalar_access) {
      auto typed=floor;
      auto const& original=semantic.op;
      // The geometry-independent DRAM floor omits split temporaries. Their
      // declaration, not a missing-name fallback, proves internal FP32 storage.
      if(original.reduction.splittable && !original.reduction.partial_tensor.empty() &&
          !typed.tensors.count(original.reduction.partial_tensor)) {
        auto const& name=original.reduction.partial_tensor;
        analysis::CouplingRelation writes;
        auto const& partition=input.task.element_access->partition;
        if(op.result.name==name)
          writes=analysis::ProjectTaskWrite(op,input.task,partition,
              op.result,op.result_map,{},theta).Image();
        for(auto const& read:op.operands)if(read.tensor.name==name)
          writes=writes.Union(analysis::ProjectTaskRead(op,input.task,partition,
              read.tensor,read.map,{},theta).Image());
        if(!writes.empty()) {
          auto& partial=typed.tensors[name];partial.element_bytes=4;
          partial.writes=std::move(writes);
        }
      }
      auto traffic=DeriveBindingRequestTraffic(input.task,typed,theta);
      input.physical_read_bytes=std::move(traffic.read_bytes);
      input.physical_write_bytes=std::move(traffic.write_bytes);
      input.no_producer_read_bytes=std::move(traffic.no_producer_read_bytes);
      input.external_write_bytes=std::move(traffic.external_write_bytes);
      input.produced_live_bytes=traffic.produced_live_bytes;
      input.stream_bytes=floor.no_producer_bytes.Eval(theta);
      return;
    }
  }
  auto accesses=DeriveModelTaskAccesses(semantic,input);
  std::vector<analysis::QuasiPolynomial> external_reads,external_writes,typed_reads,typed_writes;
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
    typed_writes.push_back(concrete.Card().Scale(tensor.element_bytes));
    auto external=serving ? tensor.external_writes.BindParams(theta) : tensor.external_writes;
    external_writes.push_back(concrete.ApplyRange(external.ImageIdentity()).Card().Scale(tensor.element_bytes));
  }
  if(mixed_width && !input.physical_read_bytes)input.physical_read_bytes=analysis::QuasiPolynomial::Sum(typed_reads);
  if(input.task.element_access && input.scalar_access)
    input.physical_write_bytes=analysis::QuasiPolynomial::Sum(typed_writes);
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
  if(physical && input.physical_write_bytes)
    traffic.global_write_bytes=count(*input.physical_write_bytes);
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
  if(physical && input.physical_write_bytes) {
    writes=input.physical_write_bytes->EvalPoints(theta,coordinates);
    write_element_bytes=1;
  }
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
  std::map<std::tuple<double,double,long,double,double,double,double,double>,double> classes;
  std::map<std::tuple<double,double,long,double,double>,double> prefetch_classes;
  std::vector<double> result;result.reserve(coordinates.size());
  if (prefetch && prefetch->ns) prefetch->ns->assign(coordinates.size(),0.0);
  for (std::size_t i=0;i<coordinates.size();++i) {
    auto arithmetic=[&](analysis::ArithmeticRatio const& ratio) {
      return model.dm && input.task.element_access
          ? double(ratio.numerator.BindCoordinates(coordinates[i]).Eval(theta))/ratio.denominator:0.0;
    };
    double flops=arithmetic(input.arithmetic.flops_per_output_element);
    double transcendental=arithmetic(input.arithmetic.transcendental_per_output_element);
    auto key=std::make_tuple(traffic[i].global_read_bytes,traffic[i].global_write_bytes,reduction[i],
        regime_a?traffic[i].no_producer_read_bytes:0.0,regime_a?traffic[i].external_write_bytes:0.0,
        flops,transcendental,cost.PrivateComputeNs(input,theta,coordinates[i],active_ctas_per_sm));
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
    auto local_key=std::make_tuple(local.global_read_bytes,local.global_write_bytes,reduction[i],flops,transcendental);
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
  if (!declared || declared->output.axes.size()<2 || declared->operands.empty() ||
      (!model.dm && declared->operands.size()!=1))
    throw std::invalid_argument("combine requires the instantiated partial tensor");
  auto task=*declared;
  if(model.dm && task.element_access) {
    if(!tile_ownership)
      throw std::invalid_argument("mapped DM split combine requires tile ownership");
    auto theta=model.MetricBindings();
    auto work=DeriveTaskWork(task.element_access->semantic,task,theta);
    auto ownership_semantic=*semantic;ownership_semantic.element_chunk=false;
    auto ownership=ProjectTaskOwnership(ownership_semantic,task,model.stages.at(stage),threads)
        .BindParams(theta);
    auto const& exact=*task.element_access;
    RuntimeScalarAccess accesses;accesses.ownership=ownership;
    accesses.writes=ownership.ApplyRange(ProjectTaskElements(exact.semantic,task,
        exact.partition,exact.semantic.result,exact.semantic.result_map,{},theta));
    std::vector<QuasiPolynomial> read_counts,read_bytes;
    int element_bytes=model.dtype==ScalarType::kBF16?2:4;
    for(unsigned i=0;i<exact.semantic.operands.size();++i) {
      auto const& operand=exact.semantic.operands[i];
      auto relation=ownership.ApplyRange(ProjectTaskRead(exact.semantic,task,exact.partition,
          operand.tensor,operand.map,{},theta));
      accesses.reads[operand.tensor.name]=accesses.reads[operand.tensor.name].Union(relation);
      auto count=relation.BoundTaskCard();read_counts.push_back(count);
      int bytes=i==0 && fp32_partials?4:element_bytes;
      if(i>0) {
        auto found=model.buffer_element_bytes.find(operand.tensor.name);
        if(found==model.buffer_element_bytes.end())
          throw std::invalid_argument("DM combine epilogue input lacks typed storage metadata");
        bytes=found->second;
      }
      read_bytes.push_back(count.Scale(bytes));
    }
    work.read_elements=QuasiPolynomial::Sum(read_counts);
    work.nominal_read_elements=work.read_elements;
    work.write_elements=accesses.writes.BoundTaskCard();
    work.nominal_write_elements=work.nominal_write_elements.SumAlong(ownership);
    work.task_count=accesses.writes.Reverse().Image().BoundTaskCard();
    work.task_reduce_extent=work.task_reduce_extent.SumAlong(ownership);
    work.nominal_task_reduce_extent=work.nominal_task_reduce_extent.SumAlong(ownership);
    ArithmeticInputs arithmetic;arithmetic.reduction=work.task_reduce_extent;
    auto signature=InstantiateTaskArithmetic("sum",arithmetic,
        accesses.writes.Reverse().ImageIdentity());
    DerivedTaskInput result{task,work,signature,{"q"},
        codegen::ScalarTaskDataflow(codegen::TaskKind::kGemmCombine),accesses};
    result.physical_read_bytes=QuasiPolynomial::Sum(read_bytes);
    return result;
  }
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
      ? (model.dm
             ? DmServingBF16Traits(config.tile_m, config.tile_n, config.tile_k,config.stages)
             : (model.serving
                    ? ServingBF16Traits(config.tile_m, config.tile_n, config.tile_k,
                                        config.stages)
             : TensorBF16Traits(config.tile_m, config.tile_n, config.tile_k,
                                config.stages)))
      : SimtF32Traits(config.tile_m, config.tile_n, config.tile_k, config.stages);
  auto const& stage = model.stages.at(index);
  bool uses_collective = false;
  for (auto const& semantic : model.task_semantics)
    if (semantic.stage == index)
      uses_collective |= semantic.op.kind == analysis::OperatorKind::kMatmul;
  if (uses_collective) return collective;
  if(model.dm && stage.dm_workspace_bytes) {
    BackendTraits traits;traits.threads=128;traits.smem_bytes=stage.dm_workspace_bytes;
    traits.shape_legal=true;return traits;
  }
  if (stage.kind == StageKind::kFusedAttention) {
    BackendTraits traits;
    traits.threads = 128;
    traits.smem_bytes = codegen::ServingAttentionSharedBytes(stage.width);
    traits.shape_legal = stage.width == 64 || stage.width == 128;
    return traits;
  }
  auto resources = codegen::ReadSimtTaskResources(
      static_cast<codegen::TaskKind>(stage.kind),
      model.dm && stage.kind>=StageKind::kDepthwiseConv?128:collective.threads);
  BackendTraits traits;
  traits.threads = resources.threads;
  traits.smem_bytes = resources.shared_bytes;
  traits.shape_legal = true;
  return traits;
}

analysis::OperatorGraph InstantiateModelTasks(ModelDescription const& model,
                                            std::vector<GemmConfig> const& configs) {
  return InstantiateModelTasks(model,configs,nullptr);
}
analysis::OperatorGraph InstantiateModelTasks(ModelDescription const& model,
    std::vector<GemmConfig> const& configs,analysis::Granularity* partition) {
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
    bool owned=model.dm && op.exact_task_access;
    if(owned) {
      for(unsigned axis=0;axis<op.task_space.axes.size();++axis) {
        auto found=input.tiles.find(op.task_space.axes[axis].name);
        if(found==input.tiles.end())throw std::invalid_argument("DM task is missing its ownership tile");
        granularity.Tile(op.name,stage.gemm<0?op.task_space.axes[axis].name:
            analysis::UnitTaskOwnershipDimension(op,axis),found->second);
      }
    } else if (!model.serving) {
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
    bool convolution=owned && !model.gemm_access.empty() && model.gemm_access.at(stage.gemm).a==
        codegen::DmAAccess::kIm2Col;
    if(convolution) {
      auto const& access=model.gemm_access.at(stage.gemm);
      auto const& conv=model.convolutions.at(access.conv);
      PartitionDmConvGemm(op,conv,model.buffer_layouts.at(conv.input_layout),
                          config,granularity,model.MetricBindings());
      continue;
    }
    bool virtual_binding=owned && !model.gemm_access.empty() && model.gemm_access.at(stage.gemm).b==
        codegen::DmBAccess::kExpertIndirect;
    if(virtual_binding) {
      PartitionDmVirtualGemm(op,model.gemm_access.at(stage.gemm),config,
          granularity,model.MetricBindings());
    } else {
      auto const& output=owned?op.task_space:op.result;
      auto const& output_map=owned?op.task_map:op.result_map;
      bool grouped = !owned && model.serving && op.result.axes.size()==3 &&
                     op.result.axes[1].name=="g" &&
                     op.result.axes[2].name=="u";
      if ((!grouped && output.axes.size()!=2) ||
          output_map.results.size()!=output.axes.size())
        throw std::invalid_argument("collective output rank is not implemented: "+op.name);
      if (grouped && op.result.axes[2].extent.Eval({}, {}) % config.tile_n)
        throw std::invalid_argument("packed group width must divide the serving N tile: "+op.name);
      for (int axis=0;axis<int(output.axes.size());++axis) {
        auto const& index=output_map.results[axis];
        if (index.kind!=analysis::IndexResult::Kind::kAffine || index.terms.size()!=1 ||
            !index.terms[0].coefficient.IsLiteral(1) || !index.terms[0].group.IsLiteral(1) ||
            !index.outer_divisor.IsLiteral(1))
          throw std::invalid_argument("collective output requires unit iteration indexing");
        int tile = axis==0 ? config.tile_m :
                   ((grouped && axis==1) ||
                    (!owned && model.serving && output.axes[axis].name=="tile")
                       ? 1 : (!owned && model.serving && output.axes[axis].name=="i"
                                  ? config.tile_n/2 : config.tile_n));
        if(model.dm && owned && axis==1 && op.arithmetic=="simple_gate_gemm") {
          if(config.tile_n%2)throw std::invalid_argument("DM GEMM N tile splits a channel pair");
          tile=config.tile_n/2;
        }
        granularity.Tile(op.name,index.terms[0].dim,
                         analysis::ClosedForm::Constant(tile));
      }
    }
    if (!op.reduction.splittable) continue;
    auto const* reduction=op.Dim(op.reduction.dim);
    if (!reduction) throw std::invalid_argument("collective reduction axis is missing");
    auto extent=reduction->extent.Eval({},{});
    long k_tiles=(extent+config.tile_k-1)/config.tile_k;
    long chunks=std::min<long>(config.split_k,k_tiles);
    if(model.dm && owned) {
      analysis::TaskReductionIndex index;
      index.index=analysis::IndexResult::Dim(op.reduction.dim,analysis::ClosedForm::Constant(1),
          analysis::ClosedForm::Constant(config.tile_k));
      index.capacity=analysis::ClosedForm::Constant(k_tiles);
      index.issued_width=analysis::ClosedForm::Constant(config.tile_k);
      index.chunks=chunks>1?chunks:0;
      granularity.IndexReduction(op.name,std::move(index));
      if(chunks>1)granularity.Split(op.name,analysis::ClosedForm::Constant(1));
    }else if (chunks>1)
      granularity.Split(op.name,reduction->extent.CeilDiv(analysis::ClosedForm::Constant(chunks)));
  }
  auto graph=analysis::Instantiate(semantics,granularity);
  if(partition)*partition=granularity;
  return graph;
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
      edge.src.name,edge.dst.name,edge.interface_elements});
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
    if(task.element_access) {
      auto const& exact=*task.element_access;
      accesses.writes.emplace(exact.semantic.result.name,input.scalar_access->writes);
      for(auto const& side:exact.semantic.additional_writes)
        accesses.writes[side.tensor.name]=accesses.writes[side.tensor.name].Union(
            input.scalar_access->ownership.ApplyRange(analysis::ProjectTaskWrite(
                exact.semantic,task,exact.partition,side.tensor,side.map,side.nonnegative,{})));
    }else accesses.writes.emplace(task.output.name,input.scalar_access->writes);
    return accesses;
  }
  if (task.element_access) {
    auto const& exact = *task.element_access;
    auto const& sem = exact.semantic;
    accesses.writes.emplace(sem.result.name, analysis::ProjectTaskWrite(sem, task,
        exact.partition, sem.result, sem.result_map, {}, {}));
    for (auto const& side : sem.additional_writes)
      accesses.writes[side.tensor.name] = accesses.writes[side.tensor.name].Union(
          analysis::ProjectTaskWrite(sem, task, exact.partition, side.tensor, side.map, side.nonnegative, {}));
    auto append = [&](analysis::TensorSpace const& tensor, analysis::IndexingMap const& map,
                      std::vector<analysis::IndexResult> const& predicates) {
      accesses.reads[tensor.name] = accesses.reads[tensor.name].Union(
          analysis::ProjectTaskRead(sem, task, exact.partition, tensor, map, predicates, {}));
    };
    if (sem.element_reads.empty()) {
      for (auto const& operand : sem.operands) append(operand.tensor, operand.map, {});
    } else {
      for (auto const& read : sem.element_reads) append(read.tensor, read.map, read.nonnegative);
    }
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
  auto accesses=analysis::ComposeFusionAccesses(pa,ca,internal,external,
      model.dm?model.MetricBindings():analysis::ParamBinding{});
  // Arithmetic phases keep distinct output domains. The coupled producer
  // work is re-indexed by the consumer relation, not averaged over fanout.
  auto producer_outputs=accesses.intermediate_tiles.at(*internal.begin()).Card();
  auto producer_arithmetic=p.arithmetic;
  if (p.task.element_access) {
    producer_arithmetic.flops_per_output_element.numerator=
        producer_arithmetic.flops_per_output_element.numerator.SumAlong(accesses.consumer_to_producer);
    producer_arithmetic.transcendental_per_output_element.numerator=
        producer_arithmetic.transcendental_per_output_element.numerator.SumAlong(accesses.consumer_to_producer);
  }
  auto arithmetic=analysis::ComposeArithmetic({{std::move(producer_arithmetic),std::move(producer_outputs)},
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
  auto known=model.dm && model.forward && semantic.op.exact_task_access?
      model.MetricBindings():analysis::ParamBinding{};
  auto work=analysis::DeriveTaskWork(semantic.op,*task,known,options);
  analysis::ArithmeticInputs arithmetic;
  arithmetic.reduction=work.nominal_task_reduce_extent;
  arithmetic.total=work.reduce_extent;
  auto width_axis=model.dm && task->element_access?
      task->element_access->partition.ownership.results.size():semantic.op.result.axes.size();
  arithmetic.width=task->tile.at(width_axis-1).Eval({},{});
  arithmetic.dtype=semantic.op.dtype;
  analysis::OpArithmetic signature;
  if(model.dm && task->element_access) {
    auto const& access=*task->element_access;
    auto domain=analysis::ProjectTaskElements(access.semantic,*task,access.partition,
        access.semantic.task_space,access.semantic.task_map,{},known).Reverse().ImageIdentity();
    signature=analysis::InstantiateTaskArithmetic(semantic.op.arithmetic,arithmetic,domain);
  } else signature=analysis::InstantiateArithmetic(semantic.op.arithmetic,arithmetic);
  analysis::RequireArithmeticImplementation(signature);
  DerivedTaskInput result{*task,std::move(work),std::move(signature),task->Coordinates(),std::nullopt,std::nullopt};
  for(auto const& phase:semantic.op.compute_prologue) {
    if(!task->element_access)
      throw std::invalid_argument("private arithmetic requires exact runtime ownership");
    auto const& access=*task->element_access;
    auto image=analysis::ProjectTaskRead(access.semantic,*task,access.partition,
        phase.output,phase.map,{},known);
    analysis::ArithmeticInputs inputs;
    inputs.reduction=analysis::QuasiPolynomial::FromClosedForm(phase.reduction);
    inputs.dtype=semantic.op.dtype;
    auto declared=analysis::InstantiateArithmetic(phase.arithmetic,inputs);
    analysis::RequireArithmeticImplementation(declared);
    result.compute_prologue.push_back({std::move(declared),image.BoundTaskCard()});
  }
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
  if(model.dm) {
    switch(stage.kind) {
      case StageKind::kGemm:
        if(!model.gemm_access.empty()) {
          auto const& access=model.gemm_access.at(stage.gemm);
          if(access.a==codegen::DmAAccess::kIm2Col)result.serving_body_kind="gemm_im2col";
          else if(access.b==codegen::DmBAccess::kExpertIndirect)result.serving_body_kind="gemm_expert_indirect";
          else if(access.a==codegen::DmAAccess::kRowGather)result.serving_body_kind="gemm_rowgather";
          else if(access.a_scale!=codegen::kDmNoIndex)result.serving_body_kind="gemm_a_scale";
        }
        break;
      case StageKind::kDepthwiseConv: result.serving_body_kind="depthwise_conv";break;
      case StageKind::kPool: result.serving_body_kind="pool";break;
      case StageKind::kGlobalPoolReduce: result.serving_body_kind="global_pool_reduce";break;
      case StageKind::kLayerNorm: result.serving_body_kind="layernorm";break;
      case StageKind::kEncoderAttention: result.serving_body_kind="encoder_attention";break;
      case StageKind::kEmbeddingSum: result.serving_body_kind="embedding_sum";break;
      case StageKind::kDwPwFused: result.serving_body_kind="dwpw_fused";break;
      case StageKind::kMoETopK: result.serving_body_kind="moe_topk_dispatch";break;
      case StageKind::kMoECombine: result.serving_body_kind="moe_combine";break;
      case StageKind::kLayoutConvert: result.serving_body_kind="layout_convert";break;
      default: break;
    }
  }
  if(model.serving && stage.kind==StageKind::kFusedAttention &&
     stage.attention_kv_block>0) {
    bool prefill=model.dims.seq>1;
    result.serving_attention=DerivedTaskInput::ServingAttention{
        prefill?1:(model.serving_capacity+stage.attention_kv_block-1)/stage.attention_kv_block,
        prefill?model.dims.seq:stage.attention_kv_block,model.dims.total,64,
        int(stage.width),prefill?int(stage.attention_query_rows):int(stage.group),
        prefill,int(stage.group)*model.dims.seq,int(stage.group)};
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
