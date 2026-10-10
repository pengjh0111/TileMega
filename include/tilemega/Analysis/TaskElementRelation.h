// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/Semantics.h>
#include <tilemega/Analysis/DerivedMetrics.h>
#include <tilemega/Analysis/TaskReductionIndex.h>
#include <optional>

namespace tilemega::analysis {

// Ownership coordinates need not be the stored tensor's coordinates: a
// flattened convolution tile and pixel-shuffle stores use the same domain.
struct TaskElementPartition {
  IndexingMap ownership;
  ClosedForm reduction_chunk = ClosedForm::Constant(0);
  std::optional<TaskReductionIndex> reduction_index;
};

CouplingRelation TaskElementBoxEnvelope(CouplingRelation const& exact);
struct TaskElementEnvelope {
  CouplingRelation relation;
  static constexpr char const* exactness = "over";
};
TaskElementEnvelope DescribeTaskElementBox(CouplingRelation const& exact);

struct TaskElementAccess {
  SemanticOp semantic;
  TaskElementPartition partition;
};

// Every original iteration belongs to exactly one owner/chunk. Reject an
// index capacity that clips semantic work before deriving any dependencies.
void ValidateTaskReductionIndex(OperatorNode const& task);

CouplingRelation ProjectTaskElements(SemanticOp const& semantic,
    OperatorNode const& task, TaskElementPartition const& partition,
    TensorSpace const& tensor, IndexingMap const& indexing,
    std::vector<IndexResult> const& nonnegative,
    ParamBinding const& known = {});

CouplingRelation ProjectTaskRead(SemanticOp const& semantic, OperatorNode const& task,
    TaskElementPartition const& partition, TensorSpace const& tensor,
    IndexingMap const& indexing, std::vector<IndexResult> const& nonnegative,
    ParamBinding const& known = {});

// Data-dependent stores retain every affine axis and widen only the bound
// indexed axes. This is an I2 relation, not an issued-write byte count.
CouplingRelation ProjectTaskWrite(SemanticOp const& semantic, OperatorNode const& task,
    TaskElementPartition const& partition, TensorSpace const& tensor,
    IndexingMap const& indexing, std::vector<IndexResult> const& nonnegative,
    ParamBinding const& known = {});

// Abstract binding keys count issued accesses, including repeated physical
// rows from different requests. This relation never substitutes for I2 R/W.
bool HasBindingRequests(IndexingMap const& indexing);
CouplingRelation ProjectTaskRequests(SemanticOp const& semantic, OperatorNode const& task,
    TaskElementPartition const& partition, TensorSpace const& tensor,
    IndexingMap const& indexing, std::vector<IndexResult> const& nonnegative,
    ParamBinding const& known = {});

struct ExactTaskCoupling {
  CouplingRelation relation;
  // (consumer, producer) -> shared tensor elements, before cardinality.
  CouplingRelation shared_elements;
  DerivedMetrics metrics;
  QuasiPolynomial consumer_read_elements;
};

ExactTaskCoupling DeriveExactTaskCoupling(CouplingRelation const& writes,
    CouplingRelation const& reads, OperatorNode const& consumer,
    ParamBinding const& known = {});

}  // namespace tilemega::analysis
