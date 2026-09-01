// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ClosedForm.h>

#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tilemega::analysis {

struct ClosedForm::Node {
  enum class Kind { kConstant, kSymbol, kAdd, kMultiply, kCeilDiv };
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
