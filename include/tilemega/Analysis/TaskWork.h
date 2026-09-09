// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/AccessRelation.h>
#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/Semantics.h>
#include <map>

namespace tilemega::analysis {
enum class AccessDomain { kPhysicalTensor, kNominalTile };

// Distinct domains must stay distinct: predicated tail loads touch fewer
// tensor elements than the allocated/issued collective tile contains.
CouplingRelation ElementAccess(OperatorNode const& task, AccessRelation const& access,
                               ParamBinding const& known, AccessDomain domain);
struct TaskWork {
  QuasiPolynomial task_count;
  QuasiPolynomial read_elements, write_elements;
  QuasiPolynomial nominal_read_elements, nominal_write_elements;
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
