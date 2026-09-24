// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/ClosedForm.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tilemega::analysis {
enum class OracleKind { Unique,Rectangular,General };
enum class EdgeStructure { OneToOne,ManyToOne,OneToMany,ManyToMany };
char const* ToString(OracleKind kind);
char const* ToString(EdgeStructure structure);
struct OracleImage {
  std::vector<std::vector<long>> points;
  std::vector<std::pair<long,long>> box;
  bool rectangular=false,empty=true;
  long Count() const;
  void ForEach(std::function<void(std::vector<long> const&)> const& visitor) const;
};
/// Immutable per-edge symbolic fiber, with source coordinates moved to parameters.
class SymbolicOracle {
 public:
  explicit SymbolicOracle(std::string const& directed_relation);
  OracleKind kind() const;
  OracleImage Query(std::vector<long> const& source,ParamBinding const& theta={}) const;
  /// Establishes that every source in the relation domain reaches the same box.
  bool IsAllBox(std::vector<std::pair<long,long>> const& box,ParamBinding const& theta={}) const;
  std::string const& relation() const;
  std::uint64_t queries() const;
  double query_ms() const;
 private:
  struct Impl;std::shared_ptr<Impl> impl_;
};
struct OraclePair {
  explicit OraclePair(std::string const& producer_to_consumer);
  EdgeStructure structure;
  SymbolicOracle forward,reverse;
};
}
