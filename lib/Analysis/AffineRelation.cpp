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
                                ClosedForm coefficient) {
  if (coordinate.empty()) throw std::invalid_argument("empty coordinate");
  AffineExpr result;
  result.terms.push_back({std::move(coordinate), std::move(coefficient)});
  return result;
}

AffineExpr AffineExpr::operator+(AffineExpr const& rhs) const {
  AffineExpr result = *this;
  result.terms.insert(result.terms.end(), rhs.terms.begin(), rhs.terms.end());
  result.offset = result.offset + rhs.offset;
  return result;
}

std::string AffineExpr::ToString() const {
  std::ostringstream out;
  bool emitted = false;
  for (auto const& term : terms) {
    if (emitted) out << " + ";
    out << term.coefficient.ToString() << "*" << term.coordinate;
    emitted = true;
  }
  if (!offset.IsConstant() || offset.Eval({}, {}) != 0 || !emitted) {
    if (emitted) out << " + ";
    out << offset.ToString();
  }
  return out.str();
}

AffineRelation::AffineRelation(
    std::vector<std::string> consumer_coordinates,
    std::vector<ProducerMap> producers,
    std::vector<std::string> parameters)
    : consumer_coordinates_(std::move(consumer_coordinates)),
      producers_(std::move(producers)),
      parameters_(std::move(parameters)) {
  if (producers_.empty()) return;
  std::ostringstream key;
  key << "image=" << producers_.size();
  for (auto const& producer : producers_) {
    key << "|" << producer.source.name << ":coords="
        << producer.coordinates.size() << ":ranges=";
    for (auto const& range : producer.quantified) key << range.name << ",";
  }
  structure_key_ = key.str();
}

AffineRelation AffineRelation::PartitionRange(
    std::string const& range_name, std::string const& split_coordinate,
    std::string const& pieces_parameter) const {
  if (split_coordinate.empty() || pieces_parameter.empty()) {
    throw std::invalid_argument("empty range partition name");
  }
  AffineRelation result = *this;
  bool found = false;
  ClosedForm pieces = ClosedForm::Symbol(pieces_parameter);
  for (auto& producer : result.producers_) {
    for (auto& range : producer.quantified) {
      if (range.name != range_name) continue;
      ClosedForm segment = range.extent.CeilDiv(pieces);
      range.begin = range.begin + AffineExpr::Variable(split_coordinate, segment);
      range.extent = segment;
      found = true;
    }
  }
  if (!found) throw std::invalid_argument("unknown quantified range: " + range_name);
  if (std::find(result.consumer_coordinates_.begin(),
                result.consumer_coordinates_.end(), split_coordinate) ==
      result.consumer_coordinates_.end()) {
    result.consumer_coordinates_.push_back(split_coordinate);
  }
  if (std::find(result.parameters_.begin(), result.parameters_.end(),
                pieces_parameter) == result.parameters_.end()) {
    result.parameters_.push_back(pieces_parameter);
  }
  // Deliberately preserve structure_key_: this operation is Reparam, not
  // coupling derivation.
  return result;
}

bool AffineRelation::SameStructure(AffineRelation const& other) const {
  return !structure_key_.empty() && structure_key_ == other.structure_key_;
}

std::string AffineRelation::ToString() const {
  std::ostringstream out;
  out << "(";
  for (std::size_t i = 0; i < consumer_coordinates_.size(); ++i) {
    if (i) out << ",";
    out << consumer_coordinates_[i];
  }
  out << ") -> {";
  for (std::size_t p = 0; p < producers_.size(); ++p) {
    if (p) out << ", ";
    auto const& producer = producers_[p];
    out << producer.source.name << "(";
    for (std::size_t i = 0; i < producer.coordinates.size(); ++i) {
      if (i) out << ",";
      out << producer.coordinates[i].ToString();
    }
    for (auto const& range : producer.quantified) {
      if (!producer.coordinates.empty() || &range != &producer.quantified.front())
        out << ",";
      out << range.name;
    }
    out << ")";
    if (!producer.quantified.empty()) {
      out << " : ";
      for (std::size_t i = 0; i < producer.quantified.size(); ++i) {
        if (i) out << " and ";
        auto const& range = producer.quantified[i];
        out << range.begin.ToString() << " <= " << range.name << " < "
            << range.begin.ToString() << " + " << range.extent.ToString();
      }
    }
  }
  out << "}";
  return out.str();
}

}  // namespace tilemega::analysis
