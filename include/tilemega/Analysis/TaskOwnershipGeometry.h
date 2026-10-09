// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/Semantics.h>
#include <stdexcept>

namespace tilemega::analysis {
inline TensorSpace const& TaskOwnershipSpace(SemanticOp const& op) {
  return op.exact_task_access?op.task_space:op.result;
}
inline IndexingMap const& TaskOwnershipMap(SemanticOp const& op) {
  return op.exact_task_access?op.task_map:op.result_map;
}
inline std::string const& UnitTaskOwnershipDimension(SemanticOp const& op,unsigned axis) {
  auto const& index=TaskOwnershipMap(op).results.at(axis);
  if(index.kind!=IndexResult::Kind::kAffine || index.terms.size()!=1 ||
     !index.outer_divisor.IsLiteral(1) ||
     !index.offset.IsLiteral(0) || !index.terms[0].coefficient.IsLiteral(1) ||
     !index.terms[0].group.IsLiteral(1) || !index.terms[0].shift.IsLiteral(0) ||
     !op.Dim(index.terms[0].dim))
    throw std::invalid_argument("candidate requires unit task ownership indexing: "+op.name);
  return index.terms[0].dim;
}
} // namespace tilemega::analysis
