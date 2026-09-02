// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definition 1 (task space) and Definition 2 (W / R).
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/ClosedForm.h>

namespace tilemega::analysis {

/// One tensor axis.  `runtime` marks an extent that is only known when theta is
/// instantiated (sequence length, indptr-derived chunk counts).  It is what
/// separates a Tier 0 edge from a Tier 2 one: the index map stays closed form,
/// but the extent is filled at run time (§2.4).
struct TensorAxis {
  std::string name;
  ClosedForm extent = ClosedForm::Constant(1);
  /// First element index this axis covers.  A non-zero origin is how an
  /// in-place append is modelled: KVappend writes rows [past, past + S) of a
  /// cache whose consumers address absolute rows.  It keeps W honest -- the
  /// operator does not claim to write rows it never touched.
  ClosedForm origin = ClosedForm::Constant(0);
  bool runtime = false;
};

/// A tensor together with the identity of the layout function through which it
/// is addressed.  Two accesses that quote the same non-empty `layout_id` share
/// an injective layout `L`, which cancels in `W^-1 o L^-1 o L o R`; that is the
/// Tier 1 criterion (§2.4).  An empty id means the identity layout.
struct TensorSpace {
  std::string name;
  std::vector<TensorAxis> axes;
  std::string layout_id;

  ClosedForm Volume() const;
};

/// How one operand axis is addressed from the consumer task coordinates.
struct OperandAxisMap {
  enum class Kind {
    kIndexed,       ///< follows output axis `output_axis`, scaled and offset
    kFullRange,     ///< the whole operand axis is read (matmul K, reduction)
    kBroadcast,     ///< a single element is reused across the output axis
    kDataDependent  ///< index comes from a runtime tensor (gather / routing)
  };

  /// One contribution to the operand index: scale * floordiv(output coordinate
  /// of `output_axis`, group).  Several terms compose the head packing that GQA
  /// and multi-head attention need.  Terms are given coarsest first; the span
  /// of a kIndexed access is the scale of the last term times that axis' tile.
  struct Term {
    int output_axis = -1;
    ClosedForm scale = ClosedForm::Constant(1);
    ClosedForm group = ClosedForm::Constant(1);
  };

  Kind kind = Kind::kIndexed;
  std::vector<Term> terms;
  /// Affine base offset: the `past` boundary introduced by `cat`, a slice base.
  ClosedForm offset = ClosedForm::Constant(0);
  /// Number of elements read, for kFullRange / kBroadcast / kDataDependent.
  /// Ignored for kIndexed, whose span follows the finest term.
  ClosedForm span = ClosedForm::Constant(1);

  static OperandAxisMap Indexed(int output_axis,
                                ClosedForm scale = ClosedForm::Constant(1),
                                ClosedForm group = ClosedForm::Constant(1));
  static OperandAxisMap Packed(std::vector<Term> terms);
  static OperandAxisMap FullRange(ClosedForm offset = ClosedForm::Constant(0));
  static OperandAxisMap Broadcast(ClosedForm span = ClosedForm::Constant(1));
  static OperandAxisMap DataDependent();
};

/// One read operand of an operator.  `producer` is the operator that writes the
/// tensor, or empty for a graph input / weight.
struct Operand {
  std::string producer;
  TensorSpace tensor;
  std::vector<OperandAxisMap> axes;
};

enum class OperatorKind {
  kPointwise,
  kReduction,
  kMatmul,
  kBroadcast,
  kConcat,
  kSlice,
  kTranspose,
  kView,
  kGather
};

std::string ToString(OperatorKind kind);

/// An operator at the granularity §2.7 talks about: one node per tensor
/// operation with its own output tiling, not one node per FX call_function.
struct OperatorNode {
  std::string name;
  OperatorKind kind = OperatorKind::kPointwise;
  TensorSpace output;
  /// Tile extent per output axis.  An axis whose tile equals its extent is
  /// "whole" and contributes no task coordinate, which is what makes the
  /// RMSNorm task space one-dimensional in §2.7.
  std::vector<ClosedForm> tile;
  std::vector<Operand> operands;

  bool IsTiled(std::size_t axis) const;
  /// Task coordinate names, in order; one per tiled output axis.
  std::vector<std::string> Coordinates() const;
  /// Extent of the task coordinate at `axis`, i.e. ceildiv(extent, tile).
  ClosedForm CoordinateExtent(std::size_t axis) const;
  /// count(T_op) = |T_op(theta, g)| (§2 Definition 4).
  ClosedForm Count() const;
  /// True when the task space itself has a run-time extent.  Operand extents
  /// are deliberately not consulted: a consumer that reads a ragged tensor but
  /// whose own task space is static (attention combine -> Wo) is still Tier 0.
  bool HasRuntimeTaskSpace() const;
};

/// A whole model at operator granularity.
struct OperatorGraph {
  std::vector<OperatorNode> nodes;

  OperatorNode const* Find(std::string const& name) const;
};

}  // namespace tilemega::analysis
