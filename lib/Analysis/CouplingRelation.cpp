// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingRelation.h>

#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/QuasiPolynomial.h>

#include "IslUtil.h"

#include <sstream>

namespace tilemega::analysis {

namespace {
isl_ctx* Ctx() { return SharedIslContext().raw(); }
}  // namespace

CouplingRelation CouplingRelation::FromIslText(std::string const& text) {
  isl_util::Map map = isl_util::ReadMap(Ctx(), text);
  return CouplingRelation(isl_util::ToString(map.get()));
}

CouplingRelation CouplingRelation::Reverse() const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Map reversed(isl_map_reverse(map.release()));
  return CouplingRelation(isl_util::ToString(reversed.get()));
}

CouplingRelation CouplingRelation::ApplyRange(
    CouplingRelation const& other) const {
  if (empty() || other.empty()) return {};
  isl_util::Map lhs = isl_util::ReadMap(Ctx(), text_);
  isl_util::Map rhs = isl_util::ReadMap(Ctx(), other.text_);
  isl_util::Map composed(isl_map_apply_range(lhs.release(), rhs.release()));
  return CouplingRelation(isl_util::ToString(composed.get()));
}

CouplingRelation CouplingRelation::IntersectDomain(
    std::string const& domain_set_text) const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set domain = isl_util::ReadSet(Ctx(), domain_set_text);
  isl_util::Map restricted(
      isl_map_intersect_domain(map.release(), domain.release()));
  return CouplingRelation(isl_util::ToString(restricted.get()));
}

CouplingRelation CouplingRelation::IntersectRange(
    std::string const& range_set_text) const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set range = isl_util::ReadSet(Ctx(), range_set_text);
  isl_util::Map restricted(
      isl_map_intersect_range(map.release(), range.release()));
  return CouplingRelation(isl_util::ToString(restricted.get()));
}

CouplingRelation CouplingRelation::Coarsen(
    std::vector<long> const& kappa) const {
  if (empty()) return {};
  std::vector<std::string> range = RangeDimNames();
  if (kappa.size() != range.size())
    throw std::invalid_argument(
        "Coarsen needs one kappa per range (producer-coordinate) dimension");
  // The floor map's output names must not collide with its input names --
  // the input names are this relation's *current* range names, which after
  // one Coarsen are already the names a naive fresh-name scheme would pick
  // again. A collision is silently destructive rather than an error: isl
  // reads `q1 = floord(q1, 2)` as a constraint on one variable, whose only
  // solution is 0, so a second Coarsen would collapse that coordinate to a
  // point instead of halving it (caught by the "floor(floor(./2)/2) ==
  // floor(./4)" check in docs/experiments/P3_ISL/coarsen_probe.cpp). Pick a
  // prefix no input name shares, then rename the result's range dims back to
  // the original names so repeated coarsening is textually stable and
  // kappa = 1 is literally the identity.
  std::string prefix = "c";
  for (bool collides = true; collides;) {
    collides = false;
    for (auto const& name : range)
      if (name.rfind(prefix, 0) == 0) collides = true;
    if (collides) prefix += "_";
  }
  std::ostringstream domain, out, constraints;
  for (std::size_t i = 0; i < range.size(); ++i) {
    if (i) { domain << ","; out << ","; }
    domain << range[i];
    out << prefix << i;
  }
  bool first = true;
  for (std::size_t i = 0; i < range.size(); ++i) {
    if (!first) constraints << " and ";
    first = false;
    if (kappa[i] == 1)
      constraints << prefix << i << " = " << range[i];
    else
      constraints << prefix << i << " = floord(" << range[i] << ", " << kappa[i]
                  << ")";
  }
  std::ostringstream floor_map;
  floor_map << "{ [" << domain.str() << "] -> [" << out.str() << "] : "
            << constraints.str() << " }";
  CouplingRelation coarsened = ApplyRange(CouplingRelation::FromIslText(floor_map.str()));
  // Re-intersect with this relation's own domain: apply_range's composition
  // can express the result's domain condition through the *floor map's*
  // output dims (an indirect chain implying the original bound, e.g.
  // `128m <= q0 <= 127+128m and q0 < S` rather than a direct `128m < S`)
  // instead of restating it directly. That is a legitimate simplification
  // for isl_map operations generally, but isl_map_card's own piecewise
  // decomposition does not always re-derive the implied direct bound, which
  // then breaks "is this the same value everywhere" detection the same way
  // an unbound range dimension did (see FanoutCard's comment) -- confirmed
  // empirically for Coarsen composed with Card() on this codebase's models.
  isl_util::Map original_map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set original_domain(isl_map_domain(original_map.release()));
  isl_util::Map coarse_map = isl_util::ReadMap(Ctx(), coarsened.text_);
  isl_util::Map restricted(isl_map_intersect_domain(
      coarse_map.release(), original_domain.release()));
  // Restore the original range names (see the prefix comment above).
  for (std::size_t i = 0; i < range.size(); ++i)
    restricted = isl_util::Map(isl_map_set_dim_name(
        restricted.release(), isl_dim_out, static_cast<unsigned>(i),
        range[i].c_str()));
  return CouplingRelation(isl_util::ToString(restricted.get()));
}

bool CouplingRelation::IsSubset(CouplingRelation const& wide) const {
  if (empty()) return true;
  if (wide.empty()) return false;
  isl_util::Map narrow_map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Map wide_map = isl_util::ReadMap(Ctx(), wide.text_);
  isl_bool result = isl_map_is_subset(narrow_map.get(), wide_map.get());
  if (result == isl_bool_error)
    throw std::runtime_error("isl: is_subset query failed");
  return result == isl_bool_true;
}

bool CouplingRelation::IsSingleValued() const {
  if (empty()) return true;
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_bool result = isl_map_is_single_valued(map.get());
  if (result == isl_bool_error)
    throw std::runtime_error("isl: is_single_valued query failed");
  return result == isl_bool_true;
}

QuasiPolynomial CouplingRelation::Card() const {
  if (empty()) return QuasiPolynomial::Constant(0);
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::PwQPolynomial card(isl_map_card(map.release()));
  return QuasiPolynomial::FromIslText(isl_util::ToString(card.get()));
}

QuasiPolynomial CouplingRelation::FanoutCard() const {
  if (empty()) return QuasiPolynomial::Constant(0);
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set range(isl_map_range(isl_map_copy(map.get())));
  isl_util::Map reversed(isl_map_reverse(map.release()));
  isl_util::Map restricted(
      isl_map_intersect_domain(reversed.release(), range.release()));
  isl_util::PwQPolynomial card(isl_map_card(restricted.release()));
  return QuasiPolynomial::FromIslText(isl_util::ToString(card.get()));
}

namespace {
std::vector<std::string> DimNames(isl_map* map /* borrowed */,
                                  isl_dim_type type) {
  std::vector<std::string> names;
  isl_size count = isl_map_dim(map, type);
  for (int i = 0; i < count; ++i) {
    char const* name = isl_map_get_dim_name(map, type, i);
    names.push_back(name ? name : ("d" + std::to_string(i)));
  }
  return names;
}
}  // namespace

std::vector<std::string> CouplingRelation::DomainDimNames() const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  return DimNames(map.get(), isl_dim_in);
}

std::vector<std::string> CouplingRelation::RangeDimNames() const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  return DimNames(map.get(), isl_dim_out);
}

bool CouplingRelation::CouplesEveryDomainPointTo(
    std::string const& set_text) const {
  if (empty()) return false;
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set target = isl_util::ReadSet(Ctx(), set_text);
  isl_util::Set domain(isl_map_domain(isl_map_copy(map.get())));
  isl_util::Map product(
      isl_map_from_domain_and_range(domain.release(), target.release()));
  isl_bool result = isl_map_is_subset(product.get(), map.get());
  if (result == isl_bool_error)
    throw std::runtime_error("isl: relaxation coverage query failed");
  return result == isl_bool_true;
}

llvm::hash_code hash_value(CouplingRelation const& value) {
  return llvm::hash_value(value.text_);
}

}  // namespace tilemega::analysis
