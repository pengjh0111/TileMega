// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/QuasiPolynomial.h>

#include <tilemega/Analysis/CouplingRelation.h>
#include <tilemega/Analysis/ISLContext.h>

#include "IslUtil.h"

#include <sstream>
#include <stdexcept>

namespace tilemega::analysis {

namespace {
isl_ctx* Ctx() { return SharedIslContext().raw(); }

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
  isl_util::PwQPolynomial value = isl_util::ReadPwQPolynomial(Ctx(), text);
  return QuasiPolynomial(isl_util::ToString(value.get()));
}

QuasiPolynomial QuasiPolynomial::Card(CouplingRelation const& relation) {
  return relation.Card();
}

QuasiPolynomial QuasiPolynomial::SubstituteParams(
    ParamBinding const& known) const {
  isl_util::PwQPolynomial value = isl_util::ReadPwQPolynomial(Ctx(), text_);
  value = FixParams(std::move(value), known);
  return QuasiPolynomial(isl_util::ToString(value.get()));
}

long QuasiPolynomial::Eval(ParamBinding const& known) const {
  isl_util::PwQPolynomial value = isl_util::ReadPwQPolynomial(Ctx(), text_);
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
