// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/AffineRelation.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tilemega::analysis {

AffineExpr AffineExpr::Constant(ClosedForm value) {
  AffineExpr result;
  result.offset = std::move(value);
  return result;
}

AffineExpr AffineExpr::Variable(std::string coordinate,
                                ClosedForm coefficient, ClosedForm group) {
  if (coordinate.empty()) throw std::invalid_argument("empty coordinate");
  AffineExpr result;
  result.terms.push_back(
      {std::move(coordinate), std::move(coefficient), std::move(group)});
  return result;
}

AffineExpr AffineExpr::operator+(AffineExpr const& rhs) const {
  AffineExpr result = *this;
  result.terms.insert(result.terms.end(), rhs.terms.begin(), rhs.terms.end());
  result.offset = result.offset + rhs.offset;
  return result;
}

AffineExpr AffineExpr::operator*(ClosedForm const& scale) const {
  AffineExpr result = *this;
  for (auto& term : result.terms) term.coefficient = term.coefficient * scale;
  result.offset = result.offset * scale;
  return result;
}

bool AffineExpr::TryExactDivide(ClosedForm const& d, AffineExpr* quotient) const {
  if (!quotient) return false;
  if (!divisor.IsLiteral(1)) return false;  // nested floordiv: not established
  AffineExpr result;
  for (auto const& term : terms) {
    ClosedForm scaled = ClosedForm::Constant(0);
    if (!term.coefficient.TryExactDivide(d, &scaled)) return false;
    result.terms.push_back({term.coordinate, scaled, term.group});
  }
  if (!offset.TryExactDivide(d, &result.offset)) return false;
  *quotient = result;
  return true;
}

std::vector<std::string> AffineExpr::Coordinates() const {
  std::vector<std::string> result;
  for (auto const& term : terms) result.push_back(term.coordinate);
  return result;
}

std::vector<std::string> AffineExpr::FreeSymbols() const {
  std::vector<std::string> result;
  auto merge = [&](std::vector<std::string> const& more) {
    for (auto const& name : more)
      if (std::find(result.begin(), result.end(), name) == result.end())
        result.push_back(name);
  };
  for (auto const& term : terms) {
    merge(term.coefficient.FreeSymbols());
    merge(term.group.FreeSymbols());
  }
  merge(offset.FreeSymbols());
  merge(divisor.FreeSymbols());
  return result;
}

bool AffineExpr::IsZero() const {
  return terms.empty() && offset.IsConstant() && offset.Eval({}, {}) == 0 &&
         divisor.IsLiteral(1);
}

std::string AffineExpr::ToString() const {
  std::ostringstream out;
  bool emitted = false;
  for (auto const& term : terms) {
    if (emitted) out << " + ";
    if (!term.coefficient.IsLiteral(1)) out << term.coefficient.ToString() << "*";
    if (term.group.IsLiteral(1))
      out << term.coordinate;
    else
      out << "floordiv(" << term.coordinate << ", " << term.group.ToString()
          << ")";
    emitted = true;
  }
  if (!offset.IsConstant() || offset.Eval({}, {}) != 0 || !emitted) {
    if (emitted) out << " + ";
    out << offset.ToString();
  }
  if (!divisor.IsLiteral(1))
    return "floordiv(" + out.str() + ", " + divisor.ToString() + ")";
  return out.str();
}

std::string AffineExpr::ToIslText(ParamBinding const& known) const {
  std::ostringstream out;
  bool emitted = false;
  for (auto const& term : terms) {
    if (emitted) out << " + ";
    ClosedForm coefficient = term.coefficient.Substitute(known);
    ClosedForm group = term.group.Substitute(known);
    std::string factor;
    if (!group.IsLiteral(1))
      factor = "floord(" + term.coordinate + ", " + group.ToIslText() + ")";
    else
      factor = term.coordinate;
    if (!coefficient.IsLiteral(1))
      out << "(" << coefficient.ToIslText() << " * " << factor << ")";
    else
      out << factor;
    emitted = true;
  }
  ClosedForm subst_offset = offset.Substitute(known);
  if (!subst_offset.IsConstant() || subst_offset.Eval({}, {}) != 0 || !emitted) {
    if (emitted) out << " + ";
    out << subst_offset.ToIslText();
  }
  ClosedForm subst_divisor = divisor.Substitute(known);
  if (!subst_divisor.IsLiteral(1))
    return "floord(" + out.str() + ", " + subst_divisor.ToIslText() + ")";
  return out.str();
}

}  // namespace tilemega::analysis
