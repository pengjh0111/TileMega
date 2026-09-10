// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingRelation.h>

#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/QuasiPolynomial.h>

#include "IslUtil.h"

#include <algorithm>
#include <sstream>

#ifndef TILEMEGA_ISL_COMPONENT_ENUMERATION
#define TILEMEGA_ISL_COMPONENT_ENUMERATION 1
#endif

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

CouplingRelation CouplingRelation::Union(CouplingRelation const& other) const {
  IslReferenceAudit audit(__func__);
  if (empty()) return other;
  if (other.empty()) return *this;
  auto lhs = isl_util::ReadMap(Ctx(), text_);
  auto rhs = isl_util::ReadMap(Ctx(), other.text_);
  isl_util::Map result(isl_map_union(lhs.release(), rhs.release()));
  if (!result) throw std::invalid_argument("union requires matching task/tensor spaces");
  return CouplingRelation(isl_util::ToString(result.get()));
}

CouplingRelation CouplingRelation::Subtract(CouplingRelation const& other) const {
  IslReferenceAudit audit(__func__);
  if (empty() || other.empty()) return *this;
  auto lhs=isl_util::ReadMap(Ctx(),text_);
  auto rhs=isl_util::ReadMap(Ctx(),other.text_);
  isl_util::Map result(isl_map_subtract(lhs.release(),rhs.release()));
  if (!result) throw std::invalid_argument("difference requires matching relation spaces");
  return CouplingRelation(isl_util::ToString(result.get()));
}

CouplingRelation CouplingRelation::ImageIdentity() const {
  IslReferenceAudit audit(__func__);
  if (empty()) return {};
  auto map=isl_util::ReadMap(Ctx(),text_);
  isl_util::Map result(isl_set_identity(isl_map_range(map.release())));
  if (!result) throw std::invalid_argument("cannot construct image identity");
  return CouplingRelation(isl_util::ToString(result.get()));
}

CouplingRelation CouplingRelation::RangeProduct(CouplingRelation const& other) const {
  IslReferenceAudit audit(__func__);
  if (empty() || other.empty()) return {};
  auto lhs=isl_util::ReadMap(Ctx(),text_);
  auto rhs=isl_util::ReadMap(Ctx(),other.text_);
  isl_util::Map result(isl_map_flat_range_product(lhs.release(),rhs.release()));
  if (!result) throw std::invalid_argument("range product requires common domains");
  return CouplingRelation(isl_util::ToString(result.get()));
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

CouplingRelation CouplingRelation::BindParams(ParamBinding const& known) const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  for (auto const& [name, value] : known.values) {
    int const position =
        isl_map_find_dim_by_name(map.get(), isl_dim_param, name.c_str());
    if (position >= 0) {
      map = isl_util::Map(
          isl_map_fix_si(map.release(), isl_dim_param, position, value));
      map = isl_util::Map(
          isl_map_project_out(map.release(), isl_dim_param, position, 1));
    }
  }
  return CouplingRelation(isl_util::ToString(map.get()));
}

CouplingRelation CouplingRelation::LexMin() const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Map result(isl_map_lexmin(map.release()));
  return CouplingRelation(isl_util::ToString(result.get()));
}

CouplingRelation CouplingRelation::LexMax() const {
  if (empty()) return {};
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Map result(isl_map_lexmax(map.release()));
  return CouplingRelation(isl_util::ToString(result.get()));
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

namespace {

struct PointSink {
  int arity = 0;
  std::vector<std::pair<std::vector<long>, std::vector<long>>>* out = nullptr;
  bool overflow = false;
};

isl_stat CollectPoint(isl_point* point, void* user) {
  isl_util::Point owned(point);
  isl_util::Space space(isl_point_get_space(point));
  if (!space) return isl_stat_error;
  auto* sink = static_cast<PointSink*>(user);
  std::vector<long> flat;
  isl_size dims = isl_space_dim(space.get(), isl_dim_set);
  for (isl_size i = 0; i < dims; ++i) {
    isl_util::Val value(isl_point_get_coordinate_val(point, isl_dim_set, i));
    if (!isl_val_is_int(value.get())) {
      sink->overflow = true;
      break;
    }
    flat.push_back(isl_val_get_num_si(value.get()));
  }
  if (sink->overflow) return isl_stat_error;
  sink->out->push_back({{flat.begin(), flat.begin() + sink->arity},
                        {flat.begin() + sink->arity, flat.end()}});
  return isl_stat_ok;
}

isl_stat EnumerateBasicSet(isl_basic_set* basic, void* user) {
  isl_util::Set part(isl_set_from_basic_set(basic));
  if (!part) return isl_stat_error;
  return isl_set_foreach_point(part.get(), CollectPoint, user);
}

}  // namespace

std::vector<std::pair<std::vector<long>, std::vector<long>>>
CouplingRelation::Points() const {
  IslReferenceAudit audit(__func__);
  std::vector<std::pair<std::vector<long>, std::vector<long>>> points;
  if (empty()) return points;
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  if (isl_map_dim(map.get(), isl_dim_param) != 0)
    throw std::invalid_argument(
        "isl: Points() needs every parameter bound: " + text_);
  int const arity = isl_map_dim(map.get(), isl_dim_in);
  isl_util::Set wrapped(isl_map_wrap(map.release()));
  if (isl_set_is_bounded(wrapped.get()) != isl_bool_true)
    throw std::invalid_argument("isl: Points() needs a bounded relation: " +
                                text_);
  PointSink sink{arity, &points, false};
#if TILEMEGA_ISL_COMPONENT_ENUMERATION
  // Disjointizing a many-piece endpoint relation can dominate the entire
  // import. Enumerate each convex component, then remove overlaps exactly.
  if (isl_set_foreach_basic_set(wrapped.get(), EnumerateBasicSet, &sink) != isl_stat_ok)
    throw std::runtime_error("isl: point enumeration failed");
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
#else
  if (isl_set_foreach_point(wrapped.get(), CollectPoint, &sink) != isl_stat_ok)
    throw std::runtime_error("isl: point enumeration failed");
#endif
  return points;
}

QuasiPolynomial CouplingRelation::Card() const {
  IslReferenceAudit audit(__func__);
  if (empty()) return QuasiPolynomial::Constant(0);
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::PwQPolynomial card(isl_map_card(map.release()));
  if (!card) throw std::runtime_error("isl: relation cardinality failed");
  return QuasiPolynomial::FromIslText(isl_util::ToString(card.get()));
}

QuasiPolynomial CouplingRelation::ImageCard() const {
  IslReferenceAudit audit(__func__);
  if (empty()) return QuasiPolynomial::Constant(0);
  auto map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set image(isl_map_range(map.release()));
  isl_util::PwQPolynomial count(isl_set_card(image.release()));
  if (!count) throw std::runtime_error("isl: event image is not countable");
  return QuasiPolynomial::FromIslText(isl_util::ToString(count.get()));
}

CouplingRelation CouplingRelation::Image() const {
  IslReferenceAudit audit(__func__);
  if (empty()) return {};
  auto map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set image(isl_map_range(map.release()));
  image = isl_util::Set(isl_set_coalesce(image.release()));
  isl_util::Map result(isl_map_from_range(image.release()));
  if (!result) throw std::runtime_error("isl: image projection failed");
  return CouplingRelation(isl_util::ToString(result.get()));
}

CouplingRelation CouplingRelation::AggregateImage() const {
  IslReferenceAudit audit(__func__);
  if (empty()) return {};
  auto names = RangeDimNames();
  std::ostringstream projection;
  projection << "{ [";
  for (std::size_t i=0; i<names.size(); ++i) {
    if (i) projection << ',';
    projection << names[i];
  }
  projection << "] -> [";
  for (std::size_t i=0; i<names.size(); ++i) {
    if (i) projection << ',';
    projection << '0';
  }
  projection << "] }";
  return ApplyRange(FromIslText(projection.str()));
}

QuasiPolynomial CouplingRelation::FanoutCard() const {
  IslReferenceAudit audit(__func__);
  if (empty()) return QuasiPolynomial::Constant(0);
  isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Set range(isl_map_range(isl_map_copy(map.get())));
  isl_util::Map reversed(isl_map_reverse(map.release()));
  isl_util::Map restricted(
      isl_map_intersect_domain(reversed.release(), range.release()));
  isl_util::PwQPolynomial card(isl_map_card(restricted.release()));
  if (!card) throw std::runtime_error("isl: reversed relation cardinality failed");
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
