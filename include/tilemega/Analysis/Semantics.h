// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 (three-layer IR), invariant I1.
//
// L-sem: the operator semantics layer. It is modelled on MLIR Linalg's
// structured ops -- an iteration domain, one indexing map per operand, and an
// iterator type per dimension -- because those indexing maps *are* affine maps
// and therefore convert directly to isl. W and R are not "generated from"
// semantics: W = tiling(g) o result_map and R = tiling(g) o operand_map, so
// the semantics is the access relation, up to the tiling.
//
// The layer is g-independent by construction: nothing here names a tile size,
// and no expression divides by one. That is the machine-checkable form of I1
// (Serialize() is byte-identical under two different g), and it is also why
// isl's literal-divisor restriction never binds here -- indexing maps are
// element level, so the only floordivs are semantic ones (GQA head grouping)
// whose divisors are theta symbols with fixed values, not tile sizes.
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/ClosedForm.h>
#include <tilemega/Analysis/TensorSpace.h>

namespace tilemega::analysis {

enum class IteratorType { kParallel, kReduction };

std::string ToString(IteratorType type);

/// One dimension of the iteration domain. Reduction dimensions are part of the
/// domain here even though they never become task coordinates: whether a
/// reduction is split is a partitioning decision that belongs to L-task.
struct IterationDim {
  std::string name;
  ClosedForm extent = ClosedForm::Constant(1);
  ClosedForm origin = ClosedForm::Constant(0);
  IteratorType type = IteratorType::kParallel;
  bool runtime = false;
};

/// One result of an indexing map: the element index along one tensor axis as a
/// function of the iteration coordinates.
struct IndexResult {
  enum class Kind {
    kAffine,        ///< sum(coefficient * floordiv(dim, group)) + offset
    kFullRange,     ///< the whole axis, independent of the iteration point
    kBroadcast,     ///< one element reused across the domain
    kDataDependent  ///< index read from a tensor
  };

  struct Term {
    std::string dim;
    ClosedForm coefficient = ClosedForm::Constant(1);
    ClosedForm group = ClosedForm::Constant(1);
  };

  Kind kind = Kind::kAffine;
  std::vector<Term> terms;
  ClosedForm offset = ClosedForm::Constant(0);
  ClosedForm span = ClosedForm::Constant(1);  ///< non-affine kinds only

  static IndexResult Dim(std::string name,
                         ClosedForm coefficient = ClosedForm::Constant(1),
                         ClosedForm group = ClosedForm::Constant(1));
  static IndexResult Affine(std::vector<Term> terms,
                            ClosedForm offset = ClosedForm::Constant(0));
  static IndexResult FullRange(ClosedForm offset = ClosedForm::Constant(0));
  static IndexResult Broadcast(ClosedForm span = ClosedForm::Constant(1));
  static IndexResult DataDependent();

  std::string Serialize() const;
};

/// domain -> tensor element index, one result per tensor axis.
struct IndexingMap {
  std::vector<IndexResult> results;
  std::string Serialize() const;
};

enum class EffectKind { kRead, kWrite, kReadWrite };

std::string ToString(EffectKind kind);

/// §2.3 memory effects. `alias_set` names the storage two tensors share; the
/// layer records the sharing, it does not analyse it. `state_object` marks
/// storage that outlives one forward step, which is what makes a KV cache
/// different from scratch: the append is a kReadWrite on a kv_cache state
/// object, not a fresh allocation.
struct MemoryEffect {
  EffectKind kind = EffectKind::kRead;
  std::string alias_set;
  std::string state_object;
  std::string Serialize() const;
};

struct SemanticOperand {
  std::string producer;   ///< empty for a graph input or a weight
  TensorSpace tensor;
  IndexingMap map;
  MemoryEffect effect;
};

/// §2.4 split-K semantics, declared rather than hidden in a TaskBody. An op
/// whose reduction dimension can be partitioned says so here; splitting it is
/// then an L-task transform (SplitReduction) that materializes a partial
/// tensor and an explicit combiner task space.
struct ReductionSemantics {
  std::string dim;                   ///< reduction dim that may be split
  std::string reduction_operator;    ///< "add", "max", "flash_combine"
  std::string partial_tensor;        ///< name for the materialized partials
  std::string combiner;              ///< name for the combine task space
  bool splittable = false;
  /// Which iteration dims own one partial contribution. Empty means every
  /// parallel dim, which is the ordinary case.
  std::vector<std::string> ownership;
  std::string Serialize() const;
};

/// One structured operator. Everything about it is g-independent.
struct SemanticOp {
  std::string name;
  OperatorKind kind = OperatorKind::kPointwise;
  std::vector<IterationDim> domain;
  TensorSpace result;
  IndexingMap result_map;
  MemoryEffect result_effect;
  std::vector<SemanticOperand> operands;
  ReductionSemantics reduction;
  /// Set when the op fell through every declarative pattern and was given the
  /// conservative generic semantics (identity result map, full-range reads).
  bool generic = false;

  IterationDim const* Dim(std::string const& name) const;
  std::string Serialize() const;
};

struct SemanticGraph {
  std::vector<SemanticOp> ops;
  SemanticOp const* Find(std::string const& name) const;
  /// Stable textual form. Byte-identical for two different granularities by
  /// construction; test/unit/semantics_test.cpp asserts it.
  std::string Serialize() const;
};

/// Conservative semantics for an operator no pattern recognized: the result is
/// written elementwise, every operand is read in full. It is I2-safe (the read
/// set contains the true one) and degrades to one task space per operator
/// rather than failing.
SemanticOp GenericSemantics(std::string name, TensorSpace result,
                            std::vector<SemanticOperand> operands);

}  // namespace tilemega::analysis
