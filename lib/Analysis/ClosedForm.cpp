// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ClosedForm.h>

#include <algorithm>
#include <functional>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tilemega::analysis {

struct ClosedForm::Node {
  enum class Kind { kConstant, kSymbol, kAdd, kMultiply, kCeilDiv, kFloorDiv };
  Kind kind = Kind::kConstant;
  long value = 0;
  std::string symbol;
  std::shared_ptr<Node const> lhs;
  std::shared_ptr<Node const> rhs;
};

ParamBinding& ParamBinding::Bind(std::string name, long value) {
  values[std::move(name)] = value;
  return *this;
}

bool ParamBinding::Contains(std::string const& name) const {
  return values.find(name) != values.end();
}

long ParamBinding::At(std::string const& name) const {
  auto found = values.find(name);
  if (found == values.end()) {
    throw std::out_of_range("unbound parameter: " + name);
  }
  return found->second;
}

ClosedForm::ClosedForm() : ClosedForm(Constant(0)) {}

ClosedForm::ClosedForm(std::shared_ptr<Node const> node)
    : node_(std::move(node)) {}

ClosedForm ClosedForm::Constant(long value) {
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::kConstant;
  node->value = value;
  return ClosedForm(std::move(node));
}

ClosedForm ClosedForm::Symbol(std::string const& name) {
  if (name.empty()) throw std::invalid_argument("empty closed-form symbol");
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::kSymbol;
  node->symbol = name;
  return ClosedForm(std::move(node));
}

namespace {
class ClosedFormParser {
 public:
  explicit ClosedFormParser(std::string const& text) : text_(text) {}

  ClosedForm Parse() {
    auto value = ParseAdd();
    Skip();
    if (position_ != text_.size()) Fail("unexpected trailing input");
    return value;
  }

 private:
  ClosedForm ParseAdd() {
    auto lhs = ParseMultiply();
    while (Consume('+')) lhs = lhs + ParseMultiply();
    return lhs;
  }

  ClosedForm ParseMultiply() {
    auto lhs = ParsePrimary();
    for (;;) {
      if (Consume('*')) {
        lhs = lhs * ParsePrimary();
      } else if (ConsumeString("//")) {
        lhs = lhs.FloorDiv(ParsePrimary());
      } else {
        return lhs;
      }
    }
  }

  ClosedForm ParsePrimary() {
    Skip();
    if (Consume('(')) {
      auto value = ParseAdd();
      if (!Consume(')')) Fail("expected ')'");
      return value;
    }
    if (PeekIdentifier("ceildiv")) {
      position_ += 7;
      if (!Consume('(')) Fail("expected '(' after ceildiv");
      auto numerator = ParseAdd();
      if (!Consume(',')) Fail("expected ',' in ceildiv");
      auto denominator = ParseAdd();
      if (!Consume(')')) Fail("expected ')' after ceildiv");
      return numerator.CeilDiv(denominator);
    }
    if (PeekIdentifier("floordiv")) {
      position_ += 8;
      if (!Consume('(')) Fail("expected '(' after floordiv");
      auto numerator = ParseAdd();
      if (!Consume(',')) Fail("expected ',' in floordiv");
      auto denominator = ParseAdd();
      if (!Consume(')')) Fail("expected ')' after floordiv");
      return numerator.FloorDiv(denominator);
    }
    Skip();
    if (position_ < text_.size() &&
        (std::isdigit(static_cast<unsigned char>(text_[position_])) ||
         (text_[position_] == '-' && position_ + 1 < text_.size() &&
          std::isdigit(static_cast<unsigned char>(text_[position_ + 1]))))) {
      char* end = nullptr;
      long value = std::strtol(text_.c_str() + position_, &end, 10);
      position_ = static_cast<std::size_t>(end - text_.c_str());
      return ClosedForm::Constant(value);
    }
    if (position_ < text_.size() &&
        (std::isalpha(static_cast<unsigned char>(text_[position_])) ||
         text_[position_] == '_')) {
      std::size_t begin = position_++;
      while (position_ < text_.size() &&
             (std::isalnum(static_cast<unsigned char>(text_[position_])) ||
              text_[position_] == '_')) ++position_;
      return ClosedForm::Symbol(text_.substr(begin, position_ - begin));
    }
    Fail("expected integer, symbol, or parenthesized expression");
    return ClosedForm::Constant(0);
  }

  bool PeekIdentifier(char const* word) {
    Skip();
    std::size_t length = std::char_traits<char>::length(word);
    return text_.compare(position_, length, word) == 0 &&
           (position_ + length == text_.size() ||
            !std::isalnum(static_cast<unsigned char>(text_[position_ + length])));
  }
  void Skip() {
    while (position_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_;
  }
  bool Consume(char value) {
    Skip();
    if (position_ == text_.size() || text_[position_] != value) return false;
    ++position_;
    return true;
  }
  bool ConsumeString(char const* value) {
    Skip();
    std::size_t length = std::char_traits<char>::length(value);
    if (text_.compare(position_, length, value) != 0) return false;
    position_ += length;
    return true;
  }
  [[noreturn]] void Fail(char const* message) const {
    throw std::invalid_argument(std::string("invalid ClosedForm at byte ") +
                                std::to_string(position_) + ": " + message);
  }

  std::string const& text_;
  std::size_t position_ = 0;
};
}  // namespace

ClosedForm ClosedForm::Parse(std::string const& expression) {
  return ClosedFormParser(expression).Parse();
}

ClosedForm ClosedForm::operator+(ClosedForm const& rhs) const {
  if (IsConstant() && rhs.IsConstant()) {
    return Constant(Eval({}, {}) + rhs.Eval({}, {}));
  }
  // Identity folding.  Not an optimization: an unfolded `0 + x` makes every
  // derived quantity print as a tree, and the §2.7 comparison is textual.
  if (IsLiteral(0)) return rhs;
  if (rhs.IsLiteral(0)) return *this;
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::kAdd;
  node->lhs = node_;
  node->rhs = rhs.node_;
  return ClosedForm(std::move(node));
}

ClosedForm ClosedForm::operator*(ClosedForm const& rhs) const {
  if (IsConstant() && rhs.IsConstant()) {
    return Constant(Eval({}, {}) * rhs.Eval({}, {}));
  }
  if (IsLiteral(0) || rhs.IsLiteral(0)) return Constant(0);
  if (IsLiteral(1)) return rhs;
  if (rhs.IsLiteral(1)) return *this;
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::kMultiply;
  node->lhs = node_;
  node->rhs = rhs.node_;
  return ClosedForm(std::move(node));
}

ClosedForm ClosedForm::CeilDiv(ClosedForm const& divisor) const {
  if (IsConstant() && divisor.IsConstant()) {
    long numerator = Eval({}, {});
    long denominator = divisor.Eval({}, {});
    if (denominator <= 0) {
      throw std::domain_error("ceildiv divisor must be positive");
    }
    long quotient = numerator / denominator;
    long remainder = numerator % denominator;
    return Constant(quotient + (remainder > 0 ? 1 : 0));
  }
  if (divisor.IsLiteral(1)) return *this;
  // ceildiv(1, k) == 1 for every k >= 1, and every tile/extent is >= 1.
  if (IsLiteral(1)) return Constant(1);
  // An established exact division has no remainder, so ceil is the quotient.
  // This is what turns ceildiv(n_h * d, d) into n_h.
  ClosedForm quotient = Constant(0);
  if (TryExactDivide(divisor, &quotient)) return quotient;
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::kCeilDiv;
  node->lhs = node_;
  node->rhs = divisor.node_;
  return ClosedForm(std::move(node));
}

ClosedForm ClosedForm::FloorDiv(ClosedForm const& divisor) const {
  if (IsConstant() && divisor.IsConstant()) {
    long numerator = Eval({}, {});
    long denominator = divisor.Eval({}, {});
    if (denominator <= 0)
      throw std::domain_error("floordiv divisor must be positive");
    long quotient = numerator / denominator;
    long remainder = numerator % denominator;
    if (remainder < 0) --quotient;
    return Constant(quotient);
  }
  if (divisor.IsLiteral(1)) return *this;
  auto node = std::make_shared<Node>();
  node->kind = Node::Kind::kFloorDiv;
  node->lhs = node_;
  node->rhs = divisor.node_;
  return ClosedForm(std::move(node));
}

long ClosedForm::Eval(ParamBinding const& theta, ParamBinding const& g) const {
  std::function<long(std::shared_ptr<Node const> const&)> eval =
      [&](std::shared_ptr<Node const> const& node) -> long {
    switch (node->kind) {
      case Node::Kind::kConstant:
        return node->value;
      case Node::Kind::kSymbol:
        if (theta.Contains(node->symbol)) return theta.At(node->symbol);
        if (g.Contains(node->symbol)) return g.At(node->symbol);
        throw std::out_of_range("unbound closed-form symbol: " + node->symbol);
      case Node::Kind::kAdd:
        return eval(node->lhs) + eval(node->rhs);
      case Node::Kind::kMultiply:
        return eval(node->lhs) * eval(node->rhs);
      case Node::Kind::kCeilDiv: {
        long numerator = eval(node->lhs);
        long denominator = eval(node->rhs);
        if (denominator <= 0) {
          throw std::domain_error("ceildiv divisor must be positive");
        }
        long quotient = numerator / denominator;
        long remainder = numerator % denominator;
        return quotient + (remainder > 0 ? 1 : 0);
      }
      case Node::Kind::kFloorDiv: {
        long numerator = eval(node->lhs);
        long denominator = eval(node->rhs);
        if (denominator <= 0)
          throw std::domain_error("floordiv divisor must be positive");
        long quotient = numerator / denominator;
        if (numerator % denominator < 0) --quotient;
        return quotient;
      }
    }
    throw std::logic_error("unknown closed-form node");
  };
  return eval(node_);
}

std::string ClosedForm::ToString() const {
  std::function<std::string(std::shared_ptr<Node const> const&)> print =
      [&](std::shared_ptr<Node const> const& node) -> std::string {
    switch (node->kind) {
      case Node::Kind::kConstant:
        return std::to_string(node->value);
      case Node::Kind::kSymbol:
        return node->symbol;
      case Node::Kind::kAdd:
        return "(" + print(node->lhs) + " + " + print(node->rhs) + ")";
      case Node::Kind::kMultiply:
        return "(" + print(node->lhs) + " * " + print(node->rhs) + ")";
      case Node::Kind::kCeilDiv:
        return "ceildiv(" + print(node->lhs) + ", " + print(node->rhs) + ")";
      case Node::Kind::kFloorDiv:
        return "floordiv(" + print(node->lhs) + ", " + print(node->rhs) + ")";
    }
    throw std::logic_error("unknown closed-form node");
  };
  return print(node_);
}

bool ClosedForm::IsLiteral(long value) const {
  return node_->kind == Node::Kind::kConstant && node_->value == value;
}

std::vector<std::string> ClosedForm::FactorStrings() const {
  std::vector<std::string> result;
  std::function<void(std::shared_ptr<Node const> const&)> walk =
      [&](std::shared_ptr<Node const> const& node) {
        if (node->kind == Node::Kind::kMultiply) {
          walk(node->lhs);
          walk(node->rhs);
          return;
        }
        result.push_back(ClosedForm(node).ToString());
      };
  walk(node_);
  std::sort(result.begin(), result.end());
  return result;
}

bool ClosedForm::TryExactDivide(ClosedForm const& divisor,
                                ClosedForm* quotient) const {
  if (!quotient) return false;
  if (divisor.IsLiteral(1)) { *quotient = *this; return true; }
  if (IsLiteral(0)) { *quotient = Constant(0); return true; }
  if (ToString() == divisor.ToString()) { *quotient = Constant(1); return true; }
  // Distribute over a sum: (a + b) / d is exact iff both halves are.
  if (node_->kind == Node::Kind::kAdd) {
    ClosedForm left, right;
    if (ClosedForm(node_->lhs).TryExactDivide(divisor, &left) &&
        ClosedForm(node_->rhs).TryExactDivide(divisor, &right)) {
      *quotient = left + right;
      return true;
    }
    return false;
  }
  if (IsConstant() && divisor.IsConstant()) {
    long numerator = Eval({}, {});
    long denominator = divisor.Eval({}, {});
    if (denominator == 0 || numerator % denominator != 0) return false;
    *quotient = Constant(numerator / denominator);
    return true;
  }
  // Cancel matching multiplicative factors.  Every factor of the divisor must
  // be matched exactly; a partial match is reported as "not established".
  std::function<void(std::shared_ptr<Node const> const&,
                     std::vector<std::shared_ptr<Node const>>&)> factors =
      [&](std::shared_ptr<Node const> const& node,
          std::vector<std::shared_ptr<Node const>>& out) {
        if (node->kind == Node::Kind::kMultiply) {
          factors(node->lhs, out);
          factors(node->rhs, out);
          return;
        }
        out.push_back(node);
      };
  std::vector<std::shared_ptr<Node const>> mine, theirs;
  factors(node_, mine);
  factors(divisor.node_, theirs);
  std::vector<bool> used(mine.size(), false);
  for (auto const& want : theirs) {
    std::string key = ClosedForm(want).ToString();
    bool matched = false;
    for (std::size_t i = 0; i < mine.size(); ++i) {
      if (used[i] || ClosedForm(mine[i]).ToString() != key) continue;
      used[i] = true;
      matched = true;
      break;
    }
    if (!matched) return false;
  }
  ClosedForm result = Constant(1);
  for (std::size_t i = 0; i < mine.size(); ++i)
    if (!used[i]) result = result * ClosedForm(mine[i]);
  *quotient = result;
  return true;
}

std::vector<std::string> ClosedForm::FreeSymbols() const {
  std::vector<std::string> result;
  std::function<void(std::shared_ptr<Node const> const&)> walk =
      [&](std::shared_ptr<Node const> const& node) {
        switch (node->kind) {
          case Node::Kind::kConstant:
            return;
          case Node::Kind::kSymbol:
            if (std::find(result.begin(), result.end(), node->symbol) ==
                result.end())
              result.push_back(node->symbol);
            return;
          case Node::Kind::kAdd:
          case Node::Kind::kMultiply:
          case Node::Kind::kCeilDiv:
          case Node::Kind::kFloorDiv:
            walk(node->lhs);
            walk(node->rhs);
            return;
        }
      };
  walk(node_);
  return result;
}

ClosedForm ClosedForm::Substitute(ParamBinding const& known) const {
  std::function<ClosedForm(std::shared_ptr<Node const> const&)> walk =
      [&](std::shared_ptr<Node const> const& node) -> ClosedForm {
    switch (node->kind) {
      case Node::Kind::kConstant:
        return Constant(node->value);
      case Node::Kind::kSymbol:
        if (known.Contains(node->symbol)) return Constant(known.At(node->symbol));
        return Symbol(node->symbol);
      case Node::Kind::kAdd:
        return walk(node->lhs) + walk(node->rhs);
      case Node::Kind::kMultiply:
        return walk(node->lhs) * walk(node->rhs);
      case Node::Kind::kCeilDiv:
        return walk(node->lhs).CeilDiv(walk(node->rhs));
      case Node::Kind::kFloorDiv:
        return walk(node->lhs).FloorDiv(walk(node->rhs));
    }
    throw std::logic_error("unknown closed-form node");
  };
  return walk(node_);
}

std::string ClosedForm::ToIslText() const {
  std::function<std::string(std::shared_ptr<Node const> const&)> emit =
      [&](std::shared_ptr<Node const> const& node) -> std::string {
    switch (node->kind) {
      case Node::Kind::kConstant:
        return std::to_string(node->value);
      case Node::Kind::kSymbol:
        return node->symbol;
      case Node::Kind::kAdd:
        return "(" + emit(node->lhs) + " + " + emit(node->rhs) + ")";
      case Node::Kind::kMultiply:
        return "(" + emit(node->lhs) + " * " + emit(node->rhs) + ")";
      case Node::Kind::kCeilDiv: {
        if (!ClosedForm(node->rhs).IsConstant())
          throw std::domain_error(
              "ToIslText: ceildiv divisor is not a literal after "
              "Substitute (isl requires a constant divisor): " +
              ClosedForm(node->rhs).ToString());
        return "ceild(" + emit(node->lhs) + ", " + emit(node->rhs) + ")";
      }
      case Node::Kind::kFloorDiv: {
        if (!ClosedForm(node->rhs).IsConstant())
          throw std::domain_error(
              "ToIslText: floordiv divisor is not a literal after "
              "Substitute (isl requires a constant divisor): " +
              ClosedForm(node->rhs).ToString());
        return "floord(" + emit(node->lhs) + ", " + emit(node->rhs) + ")";
      }
    }
    throw std::logic_error("unknown closed-form node");
  };
  return emit(node_);
}

bool ClosedForm::IsConstant() const {
  std::function<bool(std::shared_ptr<Node const> const&)> constant =
      [&](std::shared_ptr<Node const> const& node) -> bool {
    if (node->kind == Node::Kind::kSymbol) return false;
    if (node->kind == Node::Kind::kConstant) return true;
    return constant(node->lhs) && constant(node->rhs);
  };
  return constant(node_);
}

}  // namespace tilemega::analysis
