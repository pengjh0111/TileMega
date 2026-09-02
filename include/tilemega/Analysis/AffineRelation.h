// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Definitions 1-3 and invariant I1.
#pragma once

#include <string>
#include <vector>

#include <tilemega/Analysis/ClosedForm.h>

namespace tilemega::analysis {

struct TaskSpaceId {
  std::string name;
  friend bool operator==(TaskSpaceId const& a, TaskSpaceId const& b) {
    return a.name == b.name;
  }
};

/// sum_i(coeff_i(theta,g) * coordinate_i) plus a parameter-only offset.
struct AffineExpr {
  /// coefficient * floordiv(coordinate, group).  `group` defaults to 1, which
  /// is the plain affine term.  A group greater than one is what a GQA head
  /// map needs (kv_head = floor(query_head / (n_h / n_kv))); it is kept as a
  /// first-class term because folding it into a coefficient would be wrong.
  struct Term {
    std::string coordinate;
    ClosedForm coefficient = ClosedForm::Constant(1);
    ClosedForm group = ClosedForm::Constant(1);
  };

  std::vector<Term> terms;
  ClosedForm offset = ClosedForm::Constant(0);
  /// Floor divisor applied to the whole sum.  A value of 1 (the default) is a
  /// plain affine expression and is not printed.  Anything else denotes
  /// floordiv(sum, divisor), which §2.7 edge 5 (`(s,hh,j) -> (floor(s/Tm), hh)`)
  /// requires; it is exact only because the projected extent is one element.
  ClosedForm divisor = ClosedForm::Constant(1);

  static AffineExpr Constant(ClosedForm value);
  static AffineExpr Variable(std::string coordinate,
                             ClosedForm coefficient = ClosedForm::Constant(1),
                             ClosedForm group = ClosedForm::Constant(1));
  AffineExpr operator+(AffineExpr const& rhs) const;
  AffineExpr operator*(ClosedForm const& scale) const;
  /// Divide every coefficient and the offset exactly.  Returns false when any
  /// part does not cancel; the caller must then relax rather than guess.
  bool TryExactDivide(ClosedForm const& d, AffineExpr* quotient) const;
  /// Coordinates that occur in this expression.
  std::vector<std::string> Coordinates() const;
  bool IsZero() const;
  std::string ToString() const;
};

/// A quantified producer coordinate: begin <= name < begin + extent.
struct AffineRange {
  std::string name;
  AffineExpr begin;
  ClosedForm extent = ClosedForm::Constant(0);
};

/// One member of the relation image. Multiple members represent a producer
/// set such as {gate(m,n), up(m,n)}.
struct ProducerMap {
  TaskSpaceId source;
  std::vector<AffineExpr> coordinates;
  std::vector<AffineRange> quantified;
};

/// Structured consumer-coordinate -> producer-coordinate-set relation.
class AffineRelation {
 public:
  AffineRelation() = default;
  AffineRelation(std::vector<std::string> consumer_coordinates,
                 std::vector<ProducerMap> producers,
                 std::vector<std::string> parameters = {});

  /// Reparameterize a quantified range into `pieces_parameter` segments.
  /// This preserves StructureKey: no W^-1 o R derivation is repeated.
  AffineRelation PartitionRange(std::string const& range_name,
                                std::string const& split_coordinate,
                                std::string const& pieces_parameter) const;

  bool SameStructure(AffineRelation const& other) const;
  std::string const& StructureKey() const { return structure_key_; }
  std::vector<std::string> const& ConsumerCoordinates() const {
    return consumer_coordinates_;
  }
  std::vector<ProducerMap> const& Producers() const { return producers_; }
  std::vector<std::string> const& Parameters() const { return parameters_; }
  std::string ToString() const;
  bool empty() const { return producers_.empty(); }

 private:
  std::vector<std::string> consumer_coordinates_;
  std::vector<ProducerMap> producers_;
  std::vector<std::string> parameters_;
  std::string structure_key_;
};

}  // namespace tilemega::analysis
