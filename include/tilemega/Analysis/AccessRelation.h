// SPDX-License-Identifier: BSD-3-Clause
// Skeleton ref: §2 Definition 2 tensor access relations.
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/AffineRelation.h>
#include <tilemega/Analysis/TensorSpace.h>

namespace tilemega::analysis {

/// The element interval touched along one tensor axis: [base, base + span).
/// `base` is affine in the task coordinates, `span` is closed form in (theta,g).
struct ElementInterval {
  AffineExpr base;
  ClosedForm span = ClosedForm::Constant(1);
};

/// W_op or R_op evaluated symbolically: task coordinates -> element set.
/// A rectangular element set is enough for every operator class §2 Definition 2
/// enumerates; anything that is not rectangular sets `data_dependent`, which
/// forces the Tier 3 relaxation instead of a fabricated affine form.
struct AccessRelation {
  std::string op;                       ///< operator that performs the access
  std::string producer;                 ///< operator that writes the tensor
  TensorSpace tensor;
  std::vector<std::string> coordinates; ///< task coordinates of `op`
  std::vector<ElementInterval> index;   ///< one per tensor axis
  bool data_dependent = false;
  bool is_write = false;

  std::string ToString() const;
};

/// W_op: the output tiling, read straight off `zipped_divide(OutLayout, g)`.
AccessRelation BuildWriteMap(OperatorNode const& op);

/// R_op for operand `operand_index`, built per operator class.
AccessRelation BuildReadMap(OperatorNode const& op, std::size_t operand_index);

}  // namespace tilemega::analysis
