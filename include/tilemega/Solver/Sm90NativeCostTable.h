// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Solver/Sm90DenseWorkload.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace tilemega::solver {

struct Sm90StageCalibration {
  std::string id;
  double body_ns = 0;
  double head_ns = 0;
  std::optional<double> steady_ns;
  double merge_ns = 0;
  double epilogue_ns = 0;
};

class Sm90NativeCostTable {
 public:
  explicit Sm90NativeCostTable(std::string const& path);
  Sm90StageCalibration const& Gemm(Sm90DenseWorkload const&, unsigned family,
                                    Sm90DenseGemm const&, int residency) const;
  Sm90StageCalibration const& Attention(Sm90DenseWorkload const&,
                                         Sm90DenseAttention const&, int residency) const;
  std::string const& Variant() const { return variant_; }
  std::optional<double> Scale(std::string const& id) const;
  void SaveScale(std::string const& id, double value) const;

 private:
  std::string variant_;
  std::unordered_map<std::string, Sm90StageCalibration> gemms_, attention_;
  mutable std::unordered_map<std::string, double> scales_;
};
}  // namespace tilemega::solver
