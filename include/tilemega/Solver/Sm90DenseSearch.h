// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>

namespace tilemega::solver {
struct Sm90DenseSearchOptions {
  int passes=12,beam_width=8,budget_ms=600000,retained=64;
};
struct Sm90DenseCandidate {
  Sm90DenseCoordinates coordinates;
  Sm90DensePrediction prediction;
  std::string origin;
  std::string schedule_key;
};
struct Sm90DenseSearchResult {
  std::vector<Sm90DenseCandidate> ranked;
  std::size_t evaluated=0;
  double elapsed_ms=0;
  bool converged=false,budget_exhausted=false;
};
struct Sm90DenseCalibrationPending : std::invalid_argument {
  explicit Sm90DenseCalibrationPending(json::Value report)
      :std::invalid_argument("native calibration_pending: missing exact attention measurements"),
       report(std::move(report)) {}
  json::Value report;
};
std::string Sm90DenseScheduleKey(Sm90DenseCoordinates const&,Sm90DensePrediction const&);
Sm90DenseSearchResult SearchSm90Dense(Sm90DenseSearchSpace const&,
    Sm90DenseWorkload const&,Sm90DenseCostModel const&,Sm90DenseSearchOptions const&);
}  // namespace tilemega::solver
