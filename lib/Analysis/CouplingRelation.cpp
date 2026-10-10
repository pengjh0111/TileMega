// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/CouplingRelation.h>

#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <tilemega/Analysis/QuasiPolynomial.h>

#include "IslUtil.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <isl/ilp.h>
#include <isl/constraint.h>
#include <optional>

#ifndef TILEMEGA_ISL_COMPONENT_ENUMERATION
#define TILEMEGA_ISL_COMPONENT_ENUMERATION 1
#endif

namespace tilemega::analysis {

namespace {
isl_ctx* Ctx() { return SharedIslContext().raw(); }

bool HasDirectOutputCoupling(isl_map* map,int axis) {
  struct State {int axis;bool coupled=false;} state{axis};
  auto component=[](isl_basic_map* raw,void* user)->isl_stat {
    isl_util::Obj<isl_basic_map,isl_basic_map_copy,isl_basic_map_free> part(raw);
    auto constraint=[](isl_constraint* raw,void* user)->isl_stat {
      isl_util::Obj<isl_constraint,isl_constraint_copy,isl_constraint_free> c(raw);
      auto& state=*static_cast<State*>(user);
      auto coefficient=isl_util::Val(isl_constraint_get_coefficient_val(c.get(),isl_dim_in,state.axis));
      if(!coefficient)return isl_stat_error;
      if(isl_val_is_zero(coefficient.get())==isl_bool_true)return isl_stat_ok;
      for(auto type:{isl_dim_out,isl_dim_div})
        for(int i=0;i<isl_constraint_dim(c.get(),type);++i) {
          auto value=isl_util::Val(isl_constraint_get_coefficient_val(c.get(),type,i));
          if(!value)return isl_stat_error;
          if(isl_val_is_zero(value.get())!=isl_bool_true)state.coupled=true;
        }
      return isl_stat_ok;
    };
    return isl_basic_map_foreach_constraint(part.get(),constraint,user);
  };
  if(isl_map_foreach_basic_map(map,component,&state)!=isl_stat_ok)
    throw std::runtime_error("cannot inspect task coordinate coupling");
  return state.coupled;
}

isl_util::Val CountFiniteFiber(isl_set* elements) {
  if(isl_set_is_empty(elements)==isl_bool_true)
    return isl_util::Val(isl_val_zero(Ctx()));
  // Eliminating local divisions gives a rational cover without constructing
  // the convex hull of a union of halo fragments. Exact equality below, not
  // the cover, decides whether a proposed box can be counted as a product.
  auto cover=isl_util::Set(isl_set_remove_divs(isl_set_copy(elements)));
  if(!cover)throw std::runtime_error("finite task fiber cover failed");
  auto box=isl_util::Set(isl_set_universe(isl_set_get_space(elements)));
  auto product=isl_util::Val(isl_val_one(Ctx()));
  std::vector<isl_util::Val> lower,upper,widths;
  for(int axis=0;axis<isl_set_dim(elements,isl_dim_set);++axis) {
    isl_util::Val lo(isl_set_dim_min_val(isl_set_copy(cover.get()),axis));
    isl_util::Val hi(isl_set_dim_max_val(isl_set_copy(cover.get()),axis));
    if(lo && hi && (isl_val_is_int(lo.get())!=isl_bool_true ||
                    isl_val_is_int(hi.get())!=isl_bool_true)) {
      lo=isl_util::Val(isl_set_dim_min_val(isl_set_copy(elements),axis));
      hi=isl_util::Val(isl_set_dim_max_val(isl_set_copy(elements),axis));
    }
    if(!lo || !hi || isl_val_is_int(lo.get())!=isl_bool_true ||
        isl_val_is_int(hi.get())!=isl_bool_true)
      throw std::invalid_argument("finite task fiber is not bounded");
    box=isl_util::Set(isl_set_lower_bound_val(box.release(),isl_dim_set,axis,isl_val_copy(lo.get())));
    box=isl_util::Set(isl_set_upper_bound_val(box.release(),isl_dim_set,axis,isl_val_copy(hi.get())));
    lower.emplace_back(isl_val_copy(lo.get()));upper.emplace_back(isl_val_copy(hi.get()));
    auto span=isl_util::Val(isl_val_add_ui(isl_val_sub(hi.release(),lo.release()),1));
    widths.emplace_back(isl_val_copy(span.get()));
    product=isl_util::Val(isl_val_mul(product.release(),span.release()));
  }
  // Bounding boxes are used only after an exact equality proof. A generic
  // scan would enumerate millions of physical pixels for each reused tile.
  if(isl_set_is_equal(elements,box.get())==isl_bool_true)return product;
  auto reduced=isl_util::Set(isl_set_copy(elements));
  auto factor=isl_util::Val(isl_val_one(Ctx()));
  for(int axis=lower.size()-1;axis>=0;--axis) {
    auto projection=isl_util::Set(isl_set_project_out(isl_set_copy(reduced.get()),isl_dim_set,axis,1));
    auto cylinder=isl_util::Set(isl_set_insert_dims(isl_set_copy(projection.get()),isl_dim_set,axis,1));
    cylinder=isl_util::Set(isl_set_reset_space(cylinder.release(),isl_set_get_space(reduced.get())));
    cylinder=isl_util::Set(isl_set_lower_bound_val(cylinder.release(),isl_dim_set,axis,isl_val_copy(lower[axis].get())));
    cylinder=isl_util::Set(isl_set_upper_bound_val(cylinder.release(),isl_dim_set,axis,isl_val_copy(upper[axis].get())));
    if(isl_set_is_equal(reduced.get(),cylinder.get())!=isl_bool_true)continue;
    reduced=std::move(projection);
    factor=isl_util::Val(isl_val_mul(factor.release(),isl_val_copy(widths[axis].get())));
    product=isl_util::Val(isl_val_div(product.release(),isl_val_copy(widths[axis].get())));
  }
  // Factoring a Cartesian channel interval leaves a small spatial boundary
  // set. Scan only a proved small cover; large tensors still use barvinok.
  if(isl_val_cmp_si(product.get(),4096)<=0)
    return isl_util::Val(isl_val_mul(factor.release(),isl_set_count_val(reduced.get())));
  isl_util::PwQPolynomial count(isl_set_card(reduced.release()));
  if(!count)throw std::runtime_error("finite task fiber cardinality failed");
  isl_util::Point point(isl_point_zero(isl_pw_qpolynomial_get_domain_space(count.get())));
  return isl_util::Val(isl_val_mul(factor.release(),isl_pw_qpolynomial_eval(count.release(),point.release())));
}
}  // namespace

CouplingRelation CouplingRelation::FromIslText(std::string const& text) {
  return MemoExact({"parse_relation",text},[&] {
    isl_util::Map map = isl_util::ReadMap(Ctx(), text);
    return CouplingRelation(isl_util::ToString(map.get()));
  });
}

CouplingRelation CouplingRelation::Reverse() const {
  if (empty()) return {};
  return MemoExact({"reverse_relation",text_},[&] {
    isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
    isl_util::Map reversed(isl_map_reverse(map.release()));
    return CouplingRelation(isl_util::ToString(reversed.get()));
  });
}

CouplingRelation CouplingRelation::ApplyRange(
    CouplingRelation const& other) const {
  if (empty() || other.empty()) return {};
  return MemoExact({"compose_relation",text_,other.text_},[&] {
    isl_util::Map lhs = isl_util::ReadMap(Ctx(), text_);
    isl_util::Map rhs = isl_util::ReadMap(Ctx(), other.text_);
    isl_util::Map composed(isl_map_apply_range(lhs.release(), rhs.release()));
    return CouplingRelation(isl_util::ToString(composed.get()));
  });
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

CouplingRelation CouplingRelation::UnionAll(std::vector<CouplingRelation> const& relations) {
  IslReferenceAudit audit(__func__);
  std::vector<isl_util::Map> maps;
  for(auto const& relation:relations)if(!relation.empty())
    maps.push_back(isl_util::ReadMap(Ctx(),relation.text_));
  while(maps.size()>1) {
    std::vector<isl_util::Map> next;next.reserve((maps.size()+1)/2);
    for(std::size_t i=0;i<maps.size();i+=2) {
      auto merged=i+1<maps.size()?isl_util::Map(isl_map_union(maps[i].release(),maps[i+1].release())):std::move(maps[i]);
      if(!merged)throw std::invalid_argument("union requires matching task/tensor spaces");
      next.push_back(std::move(merged));
    }
    maps=std::move(next);
  }
  return maps.empty()?CouplingRelation{}:CouplingRelation(isl_util::ToString(maps.front().get()));
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

CouplingRelation CouplingRelation::ProjectRange(unsigned first, unsigned count) const {
  IslReferenceAudit audit(__func__);
  if (empty()) return {};
  auto map = isl_util::ReadMap(Ctx(), text_);
  auto rank = isl_map_dim(map.get(), isl_dim_out);
  if (first > unsigned(rank) || count > unsigned(rank) - first)
    throw std::invalid_argument("range projection outside relation rank");
  isl_util::Map projected(isl_map_project_out(map.release(), isl_dim_out, first, count));
  if (!projected) throw std::runtime_error("range projection failed");
  return CouplingRelation(isl_util::ToString(projected.get()));
}

CouplingRelation CouplingRelation::FlatProduct(CouplingRelation const& other) const {
  IslReferenceAudit audit(__func__);
  if (empty() || other.empty()) return {};
  auto lhs=isl_util::ReadMap(Ctx(),text_);
  auto rhs=isl_util::ReadMap(Ctx(),other.text_);
  isl_util::Map product(isl_map_flat_product(lhs.release(),rhs.release()));
  if (!product) throw std::invalid_argument("cannot form independent relation product");
  return CouplingRelation(isl_util::ToString(product.get()));
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
  if(text_==wide.text_)return true;
  return MemoExact({"relation_subset",text_,wide.text_},[&] {
  isl_util::Map narrow_map = isl_util::ReadMap(Ctx(), text_);
  isl_util::Map wide_map = isl_util::ReadMap(Ctx(), wide.text_);
  isl_bool result = isl_map_is_subset(narrow_map.get(), wide_map.get());
  if (result == isl_bool_error)
    throw std::runtime_error("isl: is_subset query failed");
  return result == isl_bool_true;
  });
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
  return MemoExact({"relation_cardinality",text_},[&] {
    IslReferenceAudit audit(__func__);
    if (empty()) return QuasiPolynomial::Constant(0);
    isl_util::Map map = isl_util::ReadMap(Ctx(), text_);
    isl_util::PwQPolynomial card(isl_map_card(map.release()));
    if (!card) throw std::runtime_error("isl: relation cardinality failed");
    return QuasiPolynomial::FromIslText(isl_util::ToString(card.get()));

  });
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

QuasiPolynomial CouplingRelation::BoundTaskCard(unsigned max_domain_points) const {
  auto limit = std::to_string(max_domain_points);
  return MemoExact({"bound_task_cardinality", text_, limit}, [&] {
    IslReferenceAudit audit(__func__);
    if (empty()) return QuasiPolynomial::Constant(0);
    auto map = isl_util::ReadMap(Ctx(), text_);
    if (isl_map_dim(map.get(), isl_dim_param)) {
      // A parameter-only image count must share the scalar space of a
      // sum over task fibers; map_card would retain an empty input tuple.
      if (isl_map_dim(map.get(), isl_dim_in) == 0) return ImageCard();
      return Card();
    }
    isl_util::Set domain(isl_map_domain(isl_map_copy(map.get())));
    if (isl_set_is_empty(domain.get()) == isl_bool_true)
      return isl_map_dim(map.get(), isl_dim_in) == 0 ? QuasiPolynomial::Constant(0) : Card();
    auto factor=[&]() -> std::optional<QuasiPolynomial> {
      // Many N/chunk coordinates repeat the same element fiber. Remove one
      // only after proving that its pullback reproduces the original map.
      // This keeps bound forward graphs out of huge symbolic floor sums.
      for(int axis=isl_map_dim(map.get(),isl_dim_in)-1;axis>=0;--axis) {
        // Reject visibly coupled coordinates before attempting a potentially
        // expensive equality. This screen never authorizes a projection;
        // every accepted coordinate still needs the full pullback proof.
        if(HasDirectOutputCoupling(map.get(),axis))continue;
        auto reduced=isl_util::Map(isl_map_project_out(isl_map_copy(map.get()),isl_dim_in,axis,1));
        auto projection=isl_util::Map(isl_map_identity(isl_space_map_from_set(isl_set_get_space(domain.get()))));
        projection=isl_util::Map(isl_map_project_out(projection.release(),isl_dim_out,axis,1));
        projection=isl_util::Map(isl_map_intersect_domain(projection.release(),isl_set_copy(domain.get())));
        auto lifted=isl_util::Map(isl_map_apply_range(isl_map_copy(projection.get()),isl_map_copy(reduced.get())));
        if(isl_map_is_equal(map.get(),lifted.get())==isl_bool_true) {
          auto count=CouplingRelation(isl_util::ToString(reduced.get())).BoundTaskCard(max_domain_points);
          auto value=isl_util::ReadPwQPolynomial(Ctx(),count.ToString());
          // This proved projection is an identity with one coordinate
          // removed. Inserting that coordinate and restricting the original
          // domain is its exact pullback, with no general fiber summation.
          value=isl_util::PwQPolynomial(isl_pw_qpolynomial_insert_dims(
              value.release(),isl_dim_in,axis,1));
          value=isl_util::PwQPolynomial(isl_pw_qpolynomial_reset_domain_space(
              value.release(),isl_set_get_space(domain.get())));
          value=isl_util::PwQPolynomial(isl_pw_qpolynomial_intersect_domain(
              value.release(),isl_set_copy(domain.get())));
          if(!value)throw std::runtime_error("finite task coordinate pullback failed");
          return QuasiPolynomial(isl_util::ToString(value.get()));
        }
      }
      return std::nullopt;
    };
    // Repeated channel/chunk coordinates dominate even small task domains.
    // Prove independence before scanning fibers, not only after a size cap.
    if(auto factored=factor())return *factored;
    std::uint64_t box = 1;bool box_exceeds_limit=false;
    for (int axis = 0; axis < isl_set_dim(domain.get(), isl_dim_set); ++axis) {
      isl_util::Val lo(isl_set_dim_min_val(isl_set_copy(domain.get()), axis));
      isl_util::Val hi(isl_set_dim_max_val(isl_set_copy(domain.get()), axis));
      if (!lo || !hi || isl_val_is_int(lo.get()) != isl_bool_true ||
          isl_val_is_int(hi.get()) != isl_bool_true) return Card();
      isl_util::Val width(isl_val_add_ui(isl_val_sub(hi.release(), lo.release()), 1));
      if (isl_val_is_pos(width.get()) != isl_bool_true ||
          isl_val_cmp_si(width.get(), max_domain_points) > 0) {box_exceeds_limit=true;break;}
      auto span = isl_val_get_num_si(width.get());
      if (box > max_domain_points / std::uint64_t(span)) {box_exceeds_limit=true;break;}
      box *= span;
    }
    if(box_exceeds_limit) {
      // Consumer/producer coordinates are often correlated by halo edges.
      // Their Cartesian cover can be huge while the actual domain is small.
      struct Count {std::uint64_t size=0;unsigned limit;bool capped=false;} actual{0,max_domain_points};
      auto count=[](isl_point* point,void* data)->isl_stat {
        isl_point_free(point);auto& c=*static_cast<Count*>(data);
        if(++c.size>c.limit){c.capped=true;return isl_stat_error;}
        return isl_stat_ok;
      };
      auto status=isl_set_foreach_point(domain.get(),count,&actual);
      if(actual.capped){isl_ctx_reset_error(Ctx());return Card();}
      if(status!=isl_stat_ok)throw std::runtime_error("finite task domain scan failed");
    }
    struct Group {
      std::vector<isl_util::Set> points;
      std::vector<long> lower,upper;
    };
    struct Fibers {
      isl_map* map;
      std::map<long, Group> groups;
      std::string error;
    } fibers{map.get(), {}, {}};
    auto collect = [](isl_point* raw, void* data) -> isl_stat {
      auto& fibers = *static_cast<Fibers*>(data);
      isl_util::Point coordinate(raw);
      std::vector<long> values;
      for(int axis=0;axis<isl_map_dim(fibers.map,isl_dim_in);++axis) {
        isl_util::Val value(isl_point_get_coordinate_val(coordinate.get(),isl_dim_set,axis));
        if(!value || isl_val_is_int(value.get())!=isl_bool_true ||
            isl_val_cmp_si(value.get(),std::numeric_limits<long>::min())<0 ||
            isl_val_cmp_si(value.get(),std::numeric_limits<long>::max())>0) {
          fibers.error="finite task coordinate exceeds host range";return isl_stat_error;
        }
        values.push_back(isl_val_get_num_si(value.get()));
      }
      isl_util::Set point(isl_set_from_point(coordinate.release()));
      isl_util::Map restricted(isl_map_intersect_domain(isl_map_copy(fibers.map), isl_set_copy(point.get())));
      isl_util::Set image(isl_map_range(restricted.release()));
      auto count=CountFiniteFiber(image.get());
      if (!count || isl_val_is_int(count.get()) != isl_bool_true ||
          isl_val_is_nonneg(count.get()) != isl_bool_true ||
          isl_val_cmp_si(count.get(), std::numeric_limits<long>::max()) > 0) {
        fibers.error = "exact finite task fiber is not countable"; return isl_stat_error;
      }
      auto value = isl_val_get_num_si(count.get());
      auto& group = fibers.groups[value];
      if(group.points.empty())group.lower=group.upper=values;
      else for(unsigned axis=0;axis<values.size();++axis) {
        group.lower[axis]=std::min(group.lower[axis],values[axis]);
        group.upper[axis]=std::max(group.upper[axis],values[axis]);
      }
      group.points.push_back(std::move(point));
      return isl_stat_ok;
    };
    if (isl_set_foreach_point(domain.get(), collect, &fibers) != isl_stat_ok)
      throw std::runtime_error(fibers.error.empty() ? "finite task fiber enumeration failed" : fibers.error);
    // An image count is parameter-only, like isl_set_card. Keeping an empty
    // tuple here would prevent adding it to a sum over task coordinates.
    if (isl_map_dim(map.get(), isl_dim_in) == 0)
      return QuasiPolynomial::Constant(fibers.groups.begin()->first);
    isl_util::PwQPolynomial count;
    bool compact=true;
    for (auto& [value, group] : fibers.groups) {
      auto points=isl_util::Set(isl_set_copy(domain.get()));
      for(unsigned axis=0;axis<group.lower.size();++axis) {
        points=isl_util::Set(isl_set_lower_bound_val(points.release(),isl_dim_set,axis,
            isl_val_int_from_si(Ctx(),group.lower[axis])));
        points=isl_util::Set(isl_set_upper_bound_val(points.release(),isl_dim_set,axis,
            isl_val_int_from_si(Ctx(),group.upper[axis])));
      }
      isl_util::Val size(isl_set_count_val(points.get()));
      if(!size)throw std::runtime_error("finite task domain count failed");
      // The original domain intersected with this box contains every group
      // point. Equal finite cardinalities prove equality, including holes.
      if(isl_val_cmp_si(size.get(),group.points.size())!=0) {
        compact=false;
        while(group.points.size()>1) {
          std::vector<isl_util::Set> next;
          for(std::size_t index=0;index<group.points.size();index+=2) {
            auto merged=index+1<group.points.size()?isl_util::Set(isl_set_union(
                group.points[index].release(),group.points[index+1].release())):std::move(group.points[index]);
            if(!merged)throw std::runtime_error("finite task domain union failed");
            next.push_back(std::move(merged));
          }
          group.points=std::move(next);
        }
        points=std::move(group.points.front());
        auto lattice=isl_util::Set(isl_set_from_basic_set(isl_set_affine_hull(isl_set_copy(points.get()))));
        lattice=isl_util::Set(isl_set_intersect(lattice.release(),isl_set_copy(domain.get())));
        for(unsigned axis=0;axis<group.lower.size();++axis) {
          lattice=isl_util::Set(isl_set_lower_bound_val(lattice.release(),isl_dim_set,axis,
              isl_val_int_from_si(Ctx(),group.lower[axis])));
          lattice=isl_util::Set(isl_set_upper_bound_val(lattice.release(),isl_dim_set,axis,
              isl_val_int_from_si(Ctx(),group.upper[axis])));
        }
        // Periodic fiber sizes often occupy a lattice rather than a box.
        // The hull is a proposal only; equality retains holes and boundaries.
        if(isl_set_is_equal(lattice.get(),points.get())==isl_bool_true)points=std::move(lattice);
      }
      auto* polynomial = isl_qpolynomial_val_on_domain(isl_set_get_space(points.get()), isl_val_int_from_si(Ctx(), value));
      isl_util::PwQPolynomial piece(isl_pw_qpolynomial_alloc(points.release(), polynomial));
      // Enumeration partitions domain points by their fiber size. Each box
      // was proved equal to its group, so these pieces are disjoint by
      // construction; general addition needlessly intersects every pair.
      count = count ? isl_util::PwQPolynomial(isl_pw_qpolynomial_add_disjoint(count.release(), piece.release())) : std::move(piece);
    }
    if (!count) throw std::runtime_error("finite task fiber count failed");
    if(compact)count = isl_util::PwQPolynomial(isl_pw_qpolynomial_coalesce(count.release()));
    return QuasiPolynomial::FromIslText(isl_util::ToString(count.get()));
  });
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
