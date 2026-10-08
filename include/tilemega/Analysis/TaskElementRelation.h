// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/Semantics.h>
#include <tilemega/Analysis/DerivedMetrics.h>

namespace tilemega::analysis {

// Ownership coordinates need not be the stored tensor's coordinates: a
// flattened convolution tile and pixel-shuffle stores use the same domain.
struct TaskElementPartition {
  IndexingMap ownership;
  ClosedForm reduction_chunk = ClosedForm::Constant(0);
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
