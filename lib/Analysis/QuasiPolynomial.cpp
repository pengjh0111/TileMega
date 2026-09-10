// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/QuasiPolynomial.h>

#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/ISLContext.h>

#include "IslUtil.h"
#include <isl/ilp.h>

#include <sstream>
#include <stdexcept>
#include <cctype>
#include <map>
#include <limits>

#ifndef TILEMEGA_EARLY_QP_BINDING
#define TILEMEGA_EARLY_QP_BINDING 1
#endif
#ifndef TILEMEGA_BALANCED_QP_SUM
#define TILEMEGA_BALANCED_QP_SUM 1
#endif

namespace tilemega::analysis {

std::vector<QuasiPolynomial::PolynomialInterval> QuasiPolynomial::QuadraticIntervals(
    std::string const& parameter,long begin,long end) const {
  IslReferenceAudit audit(__func__);
  if (begin>end || end==std::numeric_limits<long>::max() ||
      static_cast<unsigned long>(end)-static_cast<unsigned long>(begin)>=
          static_cast<unsigned long>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("invalid bounded polynomial interval");
  auto* ctx=SharedIslContext().raw();
  auto polynomial=isl_util::ReadPwQPolynomial(ctx,text_);
  if (isl_pw_qpolynomial_dim(polynomial.get(),isl_dim_param)==0) {
    polynomial=isl_util::PwQPolynomial(isl_pw_qpolynomial_add_dims(polynomial.release(),isl_dim_param,1));
    polynomial=isl_util::PwQPolynomial(isl_pw_qpolynomial_set_dim_name(polynomial.release(),isl_dim_param,0,parameter.c_str()));
  }
  auto space=isl_util::Space(isl_pw_qpolynomial_get_space(polynomial.get()));
  int param=isl_space_find_dim_by_name(space.get(),isl_dim_param,parameter.c_str());
  if (param<0 || isl_space_dim(space.get(),isl_dim_param)!=1 ||
      isl_pw_qpolynomial_dim(polynomial.get(),isl_dim_in)!=0)
    throw std::invalid_argument("polynomial interval requires one parameter and no task coordinates");
  auto domain=isl_util::Set(isl_set_universe(isl_space_params(space.release())));
  domain=isl_util::Set(isl_set_lower_bound_val(domain.release(),isl_dim_param,param,isl_val_int_from_si(ctx,begin)));
  domain=isl_util::Set(isl_set_upper_bound_val(domain.release(),isl_dim_param,param,isl_val_int_from_si(ctx,end)));
  polynomial=isl_util::PwQPolynomial(isl_pw_qpolynomial_intersect_params(polynomial.release(),domain.release()));
  if (!polynomial) throw std::invalid_argument("failed polynomial interval restriction");
  auto bounded=QuasiPolynomial(isl_util::ToString(polynomial.get())).SplitPeriods(
      static_cast<int>(static_cast<unsigned long>(end)-static_cast<unsigned long>(begin)+1));
  auto pieces=bounded.QuadraticPieces(parameter);
  std::vector<PolynomialInterval> intervals;
  for (auto const& piece:pieces) {
    auto set=isl_util::ReadSet(ctx,piece.domain);
    set=isl_util::Set(isl_set_move_dims(set.release(),isl_dim_set,0,isl_dim_param,0,1));
    auto min=isl_util::Val(isl_set_dim_min_val(isl_set_copy(set.get()),0));
    auto max=isl_util::Val(isl_set_dim_max_val(isl_set_copy(set.get()),0));
    if (!min || !max || isl_val_is_int(min.get())!=isl_bool_true ||
        isl_val_is_int(max.get())!=isl_bool_true)
      throw std::invalid_argument("unbounded polynomial piece after interval restriction");
    auto hull=isl_util::Set(isl_set_universe(isl_set_get_space(set.get())));
    hull=isl_util::Set(isl_set_lower_bound_val(hull.release(),isl_dim_set,0,isl_val_copy(min.get())));
    hull=isl_util::Set(isl_set_upper_bound_val(hull.release(),isl_dim_set,0,isl_val_copy(max.get())));
    if (isl_set_is_equal(set.get(),hull.get())!=isl_bool_true)
      throw std::invalid_argument("polynomial piece has non-interval congruence constraints");
    intervals.push_back({isl_val_get_num_si(min.get()),isl_val_get_num_si(max.get()),piece.coefficients});
  }
  return intervals;
}

std::vector<QuasiPolynomial::PolynomialPiece> QuasiPolynomial::QuadraticPieces(
    std::string const& parameter) const {
  IslReferenceAudit audit(__func__);
  auto* ctx=SharedIslContext().raw();
  auto polynomial=isl_util::ReadPwQPolynomial(ctx,text_);
  auto space=isl_util::Space(isl_pw_qpolynomial_get_space(polynomial.get()));
  int param=isl_space_find_dim_by_name(space.get(),isl_dim_param,parameter.c_str());
  if (param<0 || isl_pw_qpolynomial_dim(polynomial.get(),isl_dim_param)!=1 ||
      isl_pw_qpolynomial_dim(polynomial.get(),isl_dim_in)!=0)
    throw std::invalid_argument("quadratic extraction requires exactly one bound-role parameter");
  struct State {
    std::vector<PolynomialPiece> pieces;
    std::string error;
    int param;
  } state{{},{},param};
  auto piece=[](isl_set* raw_set,isl_qpolynomial* raw_qp,void* user)->isl_stat {
    isl_util::Set domain(raw_set);
    isl_util::Obj<isl_qpolynomial,isl_qpolynomial_copy,isl_qpolynomial_free> qp(raw_qp);
    auto& state=*static_cast<State*>(user);
    struct Terms {
      std::array<isl_util::Val,3> coefficients;
      std::string error;
      int param;
    } terms{{isl_util::Val(isl_val_zero(isl_set_get_ctx(raw_set))),
             isl_util::Val(isl_val_zero(isl_set_get_ctx(raw_set))),
             isl_util::Val(isl_val_zero(isl_set_get_ctx(raw_set)))},{},state.param};
    auto term=[](isl_term* raw,void* data)->isl_stat {
      isl_util::Obj<isl_term,isl_term_copy,isl_term_free> value(raw);
      auto& terms=*static_cast<Terms*>(data);
      if (isl_term_dim(raw,isl_dim_div)>0) {
        terms.error="quadratic extraction requires exact floor-domain partition first";
        return isl_stat_error;
      }
      int degree=isl_term_get_exp(raw,isl_dim_param,terms.param);
      if (degree<0 || degree>2) {
        terms.error="lane intersection degree exceeds two; numerical search is forbidden";
        return isl_stat_error;
      }
      terms.coefficients[degree]=isl_util::Val(isl_val_add(terms.coefficients[degree].release(),
                                                         isl_term_get_coefficient_val(raw)));
      return terms.coefficients[degree] ? isl_stat_ok : isl_stat_error;
    };
    if (isl_qpolynomial_foreach_term(qp.get(),term,&terms)!=isl_stat_ok) {
      state.error=terms.error.empty() ? "failed polynomial term extraction" : terms.error;
      return isl_stat_error;
    }
    PolynomialPiece output;
    char* text=isl_set_to_str(domain.get());
    if (!text) { state.error="failed polynomial domain printing"; return isl_stat_error; }
    output.domain=text; free(text);
    for (int i=0;i<3;++i) {
      text=isl_val_to_str(terms.coefficients[i].get());
      if (!text) { state.error="failed polynomial coefficient printing"; return isl_stat_error; }
      output.coefficients[i]=text; free(text);
    }
    state.pieces.push_back(std::move(output));
    return isl_stat_ok;
  };
  if (isl_pw_qpolynomial_foreach_piece(polynomial.get(),piece,&state)!=isl_stat_ok)
    throw std::invalid_argument(state.error.empty() ? "failed polynomial piece extraction" : state.error);
  return state.pieces;
}

namespace {
isl_ctx* Ctx() { return SharedIslContext().raw(); }

// Specialize parameter tokens before parsing the polynomial body. This does
// no arithmetic: isl still parses/evaluates every operator and rational. A
// late fix otherwise first normalizes thousands of unnecessary symbolic
// floor terms, even when the caller has already supplied every parameter.
std::string BindParameterTokens(std::string const& text, ParamBinding const& known) {
  IslReferenceAudit audit(__func__);
#if !TILEMEGA_EARLY_QP_BINDING
  return text;
#endif
  auto begin = text.find_first_not_of(" \n\t");
  if (known.values.empty() || begin == std::string::npos || text[begin] != '[') return text;
  auto close = text.find(']',begin), body = text.find('{',close);
  if (close == std::string::npos || body == std::string::npos)
    throw std::invalid_argument("malformed canonical quasi-polynomial parameters");
  auto parameters = isl_util::ReadSet(Ctx(),text.substr(begin,close-begin+1)+" -> { : }");
  std::map<std::string,long> replace;
  std::vector<std::string> remaining;
  for (int i=0; i<isl_set_dim(parameters.get(),isl_dim_param); ++i) {
    char const* raw = isl_set_get_dim_name(parameters.get(),isl_dim_param,i);
    if (!raw) throw std::invalid_argument("unnamed quasi-polynomial parameter");
    std::string name(raw);
    if (known.Contains(name)) replace.emplace(name,known.At(name));
    else remaining.push_back(name);
  }
  if (replace.empty()) return text;
  std::string result;
  if (!remaining.empty()) {
    result = "[";
    for (std::size_t i=0; i<remaining.size(); ++i) result += (i ? "," : "")+remaining[i];
    result += "] -> ";
  }
  auto identifier_start = [](unsigned char c) { return std::isalpha(c) || c == '_'; };
  auto identifier_part = [&](unsigned char c) {
    return identifier_start(c) || std::isdigit(c) || c == '\'';
  };
  for (std::size_t i=body; i<text.size();) {
    if (!identifier_start(text[i])) { result += text[i++]; continue; }
    std::size_t end = i+1;
    while (end<text.size() && identifier_part(text[end])) ++end;
    auto token = text.substr(i,end-i);
    auto found = replace.find(token);
    if (found == replace.end()) result += token;
    else {
      // isl's printer can emit an implicit integer coefficient, e.g. 31S.
      if (i>body && std::isdigit(static_cast<unsigned char>(text[i-1]))) result += '*';
      result += "("+std::to_string(found->second)+")";
    }
    i = end;
  }
  return result;
}

/// isl_pw_qpolynomial has no direct get_dim_name; go through its space,
/// which does (isl_space_get_dim_name).
std::string PwQPolynomialDimName(isl_pw_qpolynomial* value, isl_dim_type type,
                                 int pos) {
  isl_space* space = isl_pw_qpolynomial_get_space(value);
  char const* raw_name = isl_space_get_dim_name(space, type, pos);
  std::string name = raw_name ? raw_name : "";
  isl_space_free(space);
  return name;
}

/// Fix every isl_dim_param entry named in `known` to its literal value.
/// Leaves set/in dims (task coordinates) untouched -- `known` only ever
/// names theta/g symbols, never a task coordinate.
///
/// Implemented as isl_pw_qpolynomial_intersect_params with a param-equality
/// set, not isl_pw_qpolynomial_fix_val + drop_dims: that pair looked right
/// but produces a contradictory (`1 = 0`) domain whenever the fixed
/// parameter also appears inside an internal div/floor representation (as
/// it does for practically every quantity this migration builds, e.g.
/// `heads * ceild(S,3)` -- confirmed empirically, see the fix_val+drop_dims
/// trace this comment replaces). intersect_params does not remove the
/// dimension from the tuple (a fixed param still prints as `name = value`
/// in the domain condition), so "is anything still unbound" is answered by
/// name membership in `known`, not by dimension count -- see Eval.
isl_util::PwQPolynomial FixParams(isl_util::PwQPolynomial value,
                                  ParamBinding const& known) {
  isl_size count = isl_pw_qpolynomial_dim(value.get(), isl_dim_param);
  std::vector<std::string> names;
  std::vector<std::string> equalities;
  for (int pos = 0; pos < count; ++pos) {
    std::string name = PwQPolynomialDimName(value.get(), isl_dim_param, pos);
    if (name.empty() || !known.Contains(name)) continue;
    names.push_back(name);
    equalities.push_back(name + " = " + std::to_string(known.At(name)));
  }
  if (equalities.empty()) return value;
  std::ostringstream text;
  text << "[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i) text << ",";
    text << names[i];
  }
  text << "] -> { : ";
  for (std::size_t i = 0; i < equalities.size(); ++i) {
    if (i) text << " and ";
    text << equalities[i];
  }
  text << " }";
  isl_util::Set fix = isl_util::ReadSet(Ctx(), text.str());
  value = isl_util::PwQPolynomial(
      isl_pw_qpolynomial_intersect_params(value.release(), fix.release()));
  // Cosmetic only (does not affect Eval/SemanticallyEqual, which check name
  // membership): drop parameters intersect_params fully pinned so printed
  // results (error messages, ToString) do not carry `name = value` noise.
  return isl_util::PwQPolynomial(
      isl_pw_qpolynomial_drop_unused_params(value.release()));
}
}  // namespace

QuasiPolynomial::QuasiPolynomial() : text_("{ 0 }") {}

QuasiPolynomial QuasiPolynomial::Constant(long value) {
  return FromIslText("{ " + std::to_string(value) + " }");
}

QuasiPolynomial QuasiPolynomial::FromIslText(std::string const& text) {
  IslReferenceAudit audit(__func__);
  isl_util::PwQPolynomial value = isl_util::ReadPwQPolynomial(Ctx(), text);
  return QuasiPolynomial(isl_util::ToString(value.get()));
}

QuasiPolynomial QuasiPolynomial::Card(CouplingRelation const& relation) {
  return relation.Card();
}

QuasiPolynomial QuasiPolynomial::Add(QuasiPolynomial const& other) const {
  return Sum({*this,other});
}

QuasiPolynomial QuasiPolynomial::ScaleRational(std::string const& factor) const {
  IslReferenceAudit audit(__func__);
  auto value=isl_util::ReadPwQPolynomial(Ctx(),text_);
  auto scale=isl_util::Val(isl_val_read_from_str(Ctx(),factor.c_str()));
  if (!scale || isl_val_is_rat(scale.get())!=isl_bool_true)
    throw std::invalid_argument("polynomial scale must be finite rational");
  value=isl_util::PwQPolynomial(isl_pw_qpolynomial_scale_val(value.release(),scale.release()));
  if (!value) throw std::invalid_argument("invalid rational polynomial scale");
  return QuasiPolynomial(isl_util::ToString(value.get()));
}

QuasiPolynomial QuasiPolynomial::Multiply(QuasiPolynomial const& other) const {
  IslReferenceAudit audit(__func__);
  auto first=isl_util::ReadPwQPolynomial(Ctx(),text_);
  auto second=isl_util::ReadPwQPolynomial(Ctx(),other.text_);
  // This isl version aligns parameters for addition, but not multiplication.
  auto parameters=isl_util::Space(isl_space_align_params(
      isl_space_params(isl_pw_qpolynomial_get_domain_space(first.get())),
      isl_space_params(isl_pw_qpolynomial_get_domain_space(second.get()))));
  if (!parameters) throw std::invalid_argument("incompatible polynomial parameters");
  auto align=[&](isl_util::PwQPolynomial value) {
    auto space=isl_util::Space(isl_space_align_params(
        isl_pw_qpolynomial_get_space(value.get()),isl_space_copy(parameters.get())));
    std::unique_ptr<isl_union_pw_qpolynomial,decltype(&isl_union_pw_qpolynomial_free)> united(
        isl_union_pw_qpolynomial_align_params(
          isl_union_pw_qpolynomial_from_pw_qpolynomial(value.release()),isl_space_copy(parameters.get())),
        &isl_union_pw_qpolynomial_free);
    if (!united || !space) throw std::invalid_argument("cannot align polynomial parameters");
    auto aligned=isl_util::PwQPolynomial(isl_union_pw_qpolynomial_extract_pw_qpolynomial(united.get(),space.release()));
    if (!aligned) throw std::invalid_argument("cannot extract aligned polynomial");
    return aligned;
  };
  first=align(std::move(first)); second=align(std::move(second));
  auto product=isl_util::PwQPolynomial(isl_pw_qpolynomial_mul(first.release(),second.release()));
  if (!product) throw std::invalid_argument("incompatible polynomial product domains: "+text_+" * "+other.text_);
  return QuasiPolynomial(isl_util::ToString(product.get()));
}

QuasiPolynomial QuasiPolynomial::Scale(long factor) const {
  IslReferenceAudit audit(__func__);
  auto value = isl_util::ReadPwQPolynomial(Ctx(),text_);
  isl_util::PwQPolynomial scaled(isl_pw_qpolynomial_scale_val(
      value.release(),isl_val_int_from_si(Ctx(),factor)));
  if (!scaled) throw std::runtime_error("quasi-polynomial scaling failed");
  return QuasiPolynomial(isl_util::ToString(scaled.get()));
}

QuasiPolynomial QuasiPolynomial::Sum(std::vector<QuasiPolynomial> const& terms) {
  IslReferenceAudit audit(__func__);
  if (terms.empty()) return Constant(0);
#if TILEMEGA_BALANCED_QP_SUM
  // Repeated stage/worker counts otherwise repeatedly gist a growing domain.
  // This only reorders exact rational/integer additions, never FP64 prices.
  std::map<std::string,long> repeats;
  for (auto const& term : terms) ++repeats[term.text_];
  std::vector<isl_util::PwQPolynomial> level;
  for (auto const& [text, count] : repeats) {
    auto value = isl_util::ReadPwQPolynomial(Ctx(),text);
    if (count>1) value = isl_util::PwQPolynomial(isl_pw_qpolynomial_scale_val(
        value.release(),isl_val_int_from_si(Ctx(),count)));
    if (!value) throw std::invalid_argument("quasi-polynomial repeated sum failed");
    level.push_back(std::move(value));
  }
  while (level.size()>1) {
    std::vector<isl_util::PwQPolynomial> next;
    for (std::size_t i=0;i<level.size();i+=2) {
      auto value = std::move(level[i]);
      if (i+1<level.size()) value = isl_util::PwQPolynomial(
          isl_pw_qpolynomial_add(value.release(),level[i+1].release()));
      if (!value) throw std::invalid_argument("incompatible quasi-polynomial sum");
      next.push_back(std::move(value));
    }
    level = std::move(next);
  }
  auto sum = std::move(level.front());
#else
  // The additive identity must live in the task-coordinate space. A scalar
  // {0} seed cannot be added to a per-task polynomial [m,n] -> work.
  auto sum = isl_util::ReadPwQPolynomial(Ctx(), terms.front().text_);
  for (std::size_t i=1; i<terms.size(); ++i) {
    auto rhs = isl_util::ReadPwQPolynomial(Ctx(), terms[i].text_);
    sum = isl_util::PwQPolynomial(isl_pw_qpolynomial_add(sum.release(),rhs.release()));
    if (!sum) throw std::invalid_argument("incompatible quasi-polynomial sum");
  }
#endif
  sum = isl_util::PwQPolynomial(isl_pw_qpolynomial_coalesce(sum.release()));
  return QuasiPolynomial(isl_util::ToString(sum.get()));
}

QuasiPolynomial QuasiPolynomial::SplitPeriods(int max_periods) const {
  IslReferenceAudit audit(__func__);
  if (max_periods<=0) throw std::invalid_argument("period split limit must be positive");
  auto value=isl_util::ReadPwQPolynomial(Ctx(),text_);
  isl_util::PwQPolynomial split(isl_pw_qpolynomial_split_periods(value.release(),max_periods));
  if (!split) throw std::runtime_error("exact quasi-polynomial period splitting failed");
  return QuasiPolynomial(isl_util::ToString(split.get()));
}

QuasiPolynomial QuasiPolynomial::SubstituteParams(
    ParamBinding const& known) const {
  IslReferenceAudit audit(__func__);
  isl_util::PwQPolynomial value = isl_util::ReadPwQPolynomial(Ctx(), BindParameterTokens(text_,known));
  value = FixParams(std::move(value), known);
  return QuasiPolynomial(isl_util::ToString(value.get()));
}

QuasiPolynomial QuasiPolynomial::BindCoordinates(ParamBinding const& point) const {
  IslReferenceAudit audit(__func__);
  auto value=isl_util::ReadPwQPolynomial(Ctx(),text_);
  isl_util::Set domain(isl_set_universe(isl_pw_qpolynomial_get_domain_space(value.get())));
  int dimensions=isl_set_dim(domain.get(),isl_dim_set);
  for (int i=0;i<dimensions;++i) {
    char const* raw=isl_set_get_dim_name(domain.get(),isl_dim_set,i);
    if (!raw || !point.Contains(raw))
      throw std::invalid_argument("missing task coordinate: "+std::string(raw ? raw : "unnamed"));
    domain=isl_util::Set(isl_set_fix_val(domain.release(),isl_dim_set,i,
                                       isl_val_int_from_si(Ctx(),point.At(raw))));
    if (!domain) throw std::runtime_error("cannot restrict task coordinate");
  }
  value=isl_util::PwQPolynomial(isl_pw_qpolynomial_intersect_domain(value.release(),domain.release()));
  if (!value) throw std::runtime_error("cannot bind quasi-polynomial task point");
  return QuasiPolynomial(isl_util::ToString(value.get()));
}

long QuasiPolynomial::Eval(ParamBinding const& known) const {
  IslReferenceAudit audit(__func__);
  isl_util::PwQPolynomial value = isl_util::ReadPwQPolynomial(Ctx(), BindParameterTokens(text_,known));
  value = FixParams(std::move(value), known);
  // isl_pw_qpolynomial_max/_min range over the *whole* remaining domain --
  // both task coordinates (e.g. wait's own consumer coordinate) and any
  // parameter `known` did not name -- so comparing them is exactly "is this
  // quantity the same value everywhere that remains free" (Definition 4's
  // wait(x)/fanout(y) collapsed to one scalar exactly when that is true).
  // No separate "is everything bound" pre-check is needed: a value that
  // genuinely still depends on an unresolved parameter is *unbounded* over
  // that parameter's own unrestricted range (confirmed empirically: max on
  // `[S] -> { S : S > 0 }` returns null, not a symbolic result), so the null
  // check below already catches it, and a value merely *validity-gated* by
  // an unresolved parameter but not depending on it (e.g. `wait` = 1 for
  // every m with 128m < S, regardless of what S is) correctly still
  // resolves without needing S bound at all.
  isl_util::Val max_value(isl_pw_qpolynomial_max(isl_pw_qpolynomial_copy(value.get())));
  isl_util::Val min_value(isl_pw_qpolynomial_min(value.release()));
  if (!max_value || !min_value)
    throw std::out_of_range(
        "quasi-polynomial has no finite max/min -- it depends on an unbound "
        "parameter, or its domain is empty");
  if (!isl_val_is_int(max_value.get()) || !isl_val_is_int(min_value.get()))
    throw std::runtime_error("isl: quasi-polynomial did not reduce to an integer");
  if (!isl_val_eq(max_value.get(), min_value.get()))
    throw std::out_of_range(
        "quasi-polynomial is genuinely position-dependent (max " +
        std::to_string(isl_val_get_num_si(max_value.get())) + " != min " +
        std::to_string(isl_val_get_num_si(min_value.get())) +
        "); it is not a single scalar");
  return isl_val_get_num_si(max_value.get());
}

std::vector<long> QuasiPolynomial::EvalPoints(ParamBinding const& known,
    std::vector<ParamBinding> const& coordinates) const {
  IslReferenceAudit audit(__func__);
  auto value=isl_util::ReadPwQPolynomial(Ctx(),BindParameterTokens(text_,known));
  value=FixParams(std::move(value),known);
  auto space=isl_util::Space(isl_pw_qpolynomial_get_domain_space(value.get()));
  int parameters=isl_space_dim(space.get(),isl_dim_param),dims=isl_space_dim(space.get(),isl_dim_set);
  std::vector<long> result;
  result.reserve(coordinates.size());
  for (auto const& coordinate:coordinates) {
    auto point=isl_util::Point(isl_point_zero(isl_space_copy(space.get())));
    for (auto type:{isl_dim_param,isl_dim_set}) {
      int count=type==isl_dim_param ? parameters : dims;
      auto const& binding=type==isl_dim_param ? known : coordinate;
      for (int i=0;i<count;++i) {
        auto name=isl_space_get_dim_name(space.get(),type,i);
        if (!name || !binding.Contains(name))
          throw std::invalid_argument("batch QP evaluation requires every parameter and task coordinate");
        point=isl_util::Point(isl_point_set_coordinate_val(point.release(),type,i,isl_val_int_from_si(Ctx(),binding.At(name))));
      }
    }
    auto evaluated=isl_util::Val(isl_pw_qpolynomial_eval(isl_pw_qpolynomial_copy(value.get()),point.release()));
    if (!evaluated || isl_val_is_int(evaluated.get())!=isl_bool_true ||
        isl_val_cmp_si(evaluated.get(),std::numeric_limits<long>::min())<0 ||
        isl_val_cmp_si(evaluated.get(),std::numeric_limits<long>::max())>0)
      throw std::invalid_argument("batch QP value is not a representable integer");
    result.push_back(isl_val_get_num_si(evaluated.get()));
  }
  return result;
}

QuasiPolynomial QuasiPolynomial::SumDomain() const {
  IslReferenceAudit audit(__func__);
  auto value = isl_util::ReadPwQPolynomial(Ctx(), text_);
  if (isl_pw_qpolynomial_dim(value.get(),isl_dim_in)==0) return *this;
  isl_util::PwQPolynomial sum(isl_pw_qpolynomial_sum(value.release()));
  if (!sum) throw std::runtime_error("isl: cannot sum quasi-polynomial domain");
  return FromIslText(isl_util::ToString(sum.get()));
}

QuasiPolynomial QuasiPolynomial::SupportIndicator() const {
  IslReferenceAudit audit(__func__);
  auto value=isl_util::ReadPwQPolynomial(Ctx(),text_);
  auto domain=isl_util::Set(isl_pw_qpolynomial_domain(value.release()));
  if (!domain) throw std::runtime_error("cannot obtain polynomial support");
  auto* scalar=isl_qpolynomial_val_on_domain(isl_set_get_space(domain.get()),isl_val_one(Ctx()));
  auto indicator=isl_util::PwQPolynomial(isl_pw_qpolynomial_from_qpolynomial(scalar));
  indicator=isl_util::PwQPolynomial(isl_pw_qpolynomial_intersect_domain(indicator.release(),domain.release()));
  if (!indicator) throw std::runtime_error("cannot construct polynomial support indicator");
  return FromIslText(isl_util::ToString(indicator.get()));
}

bool QuasiPolynomial::SemanticallyEqual(QuasiPolynomial const& other,
                                        ParamBinding const& known) const {
  // Try the constant-vs-constant shortcut first: if both sides reduce to a
  // single scalar under `known` (Eval, regardless of how many "in"/task-
  // coordinate dims either side has), compare the scalars directly. This
  // sidesteps a real isl limitation: isl_pw_qpolynomial_sub requires
  // matching spaces (same "in" dim count), but a stored placeholder
  // constant (0 "in" dims, e.g. a not-yet-derived wait = 1) and a value
  // computed from a domain-bound relation (N "in" dims, though constant
  // across all of them, e.g. card() of a single-point relation) legitimately
  // describe the same number with different dimensionality -- confirmed by
  // a genuine isl error ("spaces don't match") when this path is skipped.
  try {
    return Eval(known) == other.Eval(known);
  } catch (std::out_of_range const&) {
    // At least one side is genuinely non-constant, or has a parameter
    // `known` does not name: fall through to the exact structural check.
    // That check's space-matching requirement is not a practical problem
    // here, because a genuinely position-dependent wait/fanout pair is
    // always derived from the same relation on both sides (once real
    // coupling derivation feeds the dialect, not the Frontend placeholder
    // this shortcut exists for), so their domain spaces already match.
  }
  isl_util::PwQPolynomial lhs = isl_util::ReadPwQPolynomial(Ctx(), text_);
  isl_util::PwQPolynomial rhs = isl_util::ReadPwQPolynomial(Ctx(), other.text_);
  lhs = FixParams(std::move(lhs), known);
  rhs = FixParams(std::move(rhs), known);
  isl_util::PwQPolynomial diff(
      isl_pw_qpolynomial_sub(lhs.release(), rhs.release()));
  isl_bool zero = isl_pw_qpolynomial_is_zero(diff.get());
  if (zero == isl_bool_error)
    throw std::runtime_error("isl: quasi-polynomial equality check failed");
  return zero == isl_bool_true;
}

bool QuasiPolynomial::IsZero() const {
  isl_util::PwQPolynomial value = isl_util::ReadPwQPolynomial(Ctx(), text_);
  isl_bool zero = isl_pw_qpolynomial_is_zero(value.get());
  return zero == isl_bool_true;
}

llvm::hash_code hash_value(QuasiPolynomial const& value) {
  return llvm::hash_value(value.text_);
}

}  // namespace tilemega::analysis
