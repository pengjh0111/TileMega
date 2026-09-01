// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ClosedForm.h>

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
