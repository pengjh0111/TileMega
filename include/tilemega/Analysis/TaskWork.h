// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/Semantics.h>
#include <map>

#ifndef TILEMEGA_EXACT_ELEMENT_WORK
#define TILEMEGA_EXACT_ELEMENT_WORK 1
#endif

namespace tilemega::analysis {
enum class AccessDomain { kPhysicalTensor, kNominalTile };

// Distinct domains must stay distinct: predicated tail loads touch fewer
// tensor elements than the allocated/issued collective tile contains.
CouplingRelation ElementAccess(OperatorNode const& task, AccessRelation const& access,
                               ParamBinding const& known, AccessDomain domain);
CouplingRelation ExactElementRead(SemanticOp const& semantic, OperatorNode const& task,
                                 ElementRead const& read, ParamBinding const& known);
struct TaskWork {
  QuasiPolynomial task_count;
  QuasiPolynomial read_elements, write_elements;
  QuasiPolynomial nominal_read_elements, nominal_write_elements;
  /// The part of `read_elements` that comes from operands with no in-edge --
  /// the read-only frontier §5.3.1's Prefetch phase may fetch before the task's
  /// dependencies resolve.  It is `Operand::producer` being empty, i.e. the
  /// same write relation the coupling edges come from, and never an annotation.
  QuasiPolynomial frontier_read_elements;
  QuasiPolynomial reduce_extent, parallel_extent;
  // Local reduction span after L-task splitting; the semantic reduction
  // above still describes the complete operator, not one partial.
  QuasiPolynomial task_reduce_extent;
  QuasiPolynomial nominal_task_reduce_extent;
};
struct TaskWorkOptions {
  // Inner collective tiles, supplied by implementation traits rather than
  // operator-kind formulas. They pad issued work only, never physical R/W.
  std::map<std::string, ClosedForm> reduction_tiles;
};
TaskWork DeriveTaskWork(SemanticOp const& semantic, OperatorNode const& task,
                       ParamBinding const& known, TaskWorkOptions const& options = {});
}  // namespace tilemega::analysis
