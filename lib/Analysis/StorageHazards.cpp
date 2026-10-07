// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/StorageHazards.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <stdexcept>

namespace tilemega::analysis {
namespace {
void Validate(StorageTaskAccess const& access) {
  if (!access.task || access.elements.empty() ||
      access.elements.DomainDimNames().size() != access.task->Coordinates().size())
    throw std::invalid_argument("invalid physical storage task access");
}
bool Symbolic(OperatorNode const& task) {
  for (auto const& axis : task.output.axes)
    if (!axis.extent.FreeSymbols().empty() || !axis.origin.FreeSymbols().empty()) return true;
  for (auto const& tile : task.tile) if (!tile.FreeSymbols().empty()) return true;
  return false;
}
bool NoPairs(CouplingRelation const& relation) {
  auto tuple = [](unsigned size, char prefix) {
    std::string text = "[";
    for (unsigned axis = 0; axis < size; ++axis) {
      if (axis) text += ',';
      text += prefix + std::to_string(axis);
    }
    return text + ']';
  };
  return relation.IsSubset(CouplingRelation::FromIslText("{ " +
      tuple(relation.DomainDimNames().size(), 'c') + " -> " +
      tuple(relation.RangeDimNames().size(), 'p') + " : false }"));
}
CouplingEdge Edge(StorageTaskAccess const& source, CouplingRelation const& source_elements,
    StorageTaskAccess const& target, ParamBinding const& known) {
  auto consumer = target.elements.BindParams(known);
  auto exact = DeriveExactTaskCoupling(source_elements, consumer, *target.task, known);
  CouplingEdge edge;
  edge.src = {source.task->name}; edge.dst = {target.task->name};
  edge.C = exact.relation; edge.metrics = exact.metrics;
  edge.shared_elements = exact.shared_elements;
  edge.consumer_elements = consumer;
  edge.read_box = DescribeTaskElementBox(consumer).relation;
  edge.coupled_reads = consumer.ApplyRange(source_elements.ImageIdentity());
  edge.interface_elements = edge.coupled_reads->BoundTaskCard().SumDomain().Add(
      edge.coupled_reads->Image().BoundTaskCard().Scale(-1));
  edge.event_shape = ComputeEventShape(*target.task, target.task->Coordinates());
  bool dependent = source.attributes.relation_kind == RelationKind::kDataDependent ||
      target.attributes.relation_kind == RelationKind::kDataDependent;
  edge.attributes.relation_kind = dependent ? RelationKind::kDataDependent : RelationKind::kAffine;
  edge.attributes.extent_kind = source.task->HasRuntimeTaskSpace() || target.task->HasRuntimeTaskSpace()
      ? ExtentKind::kRuntimeDynamic : Symbolic(*source.task) || Symbolic(*target.task)
          ? ExtentKind::kSymbolicStatic : ExtentKind::kStaticLiteral;
  edge.exact = !dependent && source.attributes.exactness == Exactness::kExact &&
      target.attributes.exactness == Exactness::kExact;
  edge.attributes.exactness = edge.exact ? Exactness::kExact : Exactness::kRelaxed;
  edge.attributes.runtime_requirement = dependent ? RuntimeRequirement::kTensorValues
      : edge.attributes.extent_kind == ExtentKind::kRuntimeDynamic
          ? RuntimeRequirement::kPrefixSum : RuntimeRequirement::kNone;
  edge.attributes.countability = dependent ? Countability::kUncountable : Countability::kPiecewiseQuasiPolynomial;
  if (!dependent) {
    try { (void)edge.metrics.wait.Eval(known); edge.attributes.countability = Countability::kConstant; }
    catch (std::exception const&) {}
  }
  if (!edge.exact) edge.relaxation = "I2: physical storage access uses the declared conservative relation";
  edge.tier = DeriveTier(edge.attributes);
  return edge;
}
}
std::vector<StorageHazard> DeriveStorageReuseHazards(
    StorageTaskAccess const& previous_write,
    std::vector<StorageTaskAccess> const& previous_reads,
    StorageTaskAccess const& next_write, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
  Validate(previous_write); Validate(next_write);
  auto old_write = previous_write.elements.BindParams(known);
  auto new_write = next_write.elements.BindParams(known);
  if (old_write.RangeDimNames().size() != new_write.RangeDimNames().size())
    throw std::invalid_argument("storage reuse coordinate rank mismatch");
  if (!new_write.Reverse().IsSingleValued())
    throw std::invalid_argument("reused storage has overlapping writer ownership");
  std::vector<StorageHazard> hazards;
  CouplingRelation read_elements;
  for (auto const& read : previous_reads) {
    Validate(read);
    if (read.elements.RangeDimNames().size() != old_write.RangeDimNames().size())
      throw std::invalid_argument("storage reuse coordinate rank mismatch");
    auto physical = read.elements.BindParams(known);
    read_elements = read_elements.Union(physical.Image());
    if (!NoPairs(new_write.ApplyRange(physical.Reverse())))
      hazards.push_back({StorageHazardKind::kWAR, Edge(read, physical, next_write, known)});
  }
  // A reader protects its observed elements through WAR and its existing RAW
  // edge. Elements with no reader need a direct old-writer -> new-writer edge.
  auto unread = read_elements.empty() ? old_write
      : old_write.Subtract(old_write.ApplyRange(read_elements.ImageIdentity()));
  if (!NoPairs(new_write.ApplyRange(unread.Reverse())))
    hazards.push_back({StorageHazardKind::kWAW, Edge(previous_write, unread, next_write, known)});
  return hazards;
}
}  // namespace tilemega::analysis
