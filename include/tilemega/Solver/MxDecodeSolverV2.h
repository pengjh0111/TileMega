// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/MxDecodeProgramV2.h>
#include <tilemega/Support/Json.h>

#include <array>
#include <map>
#include <string>
#include <vector>

namespace tilemega::solver {

struct MxDecodeResourcesV2 {
  int mtreg = 0, local_bytes = 0, static_smem_bytes = 0, resident_ctas = 0;
};

struct MxDecodeCandidateV2 {
  codegen::MxDecodeClassV2 k;
  double ns = 0, p90_ns = 0;
  int b_min = 0, attn_prefetch = 0, r_next = -1;
  MxDecodeResourcesV2 resources;
  // Full aggregate rows preserve per-process and zero-prefetch build evidence.
  std::vector<json::Value> evidence;
};

struct MxDecodePredictionV2 {
  codegen::MxDecodeConfigV2 config;
  double tpot_ns = 0;
  std::map<std::string, double> stage_ns;
  std::array<std::size_t, 6> ranks{};  // qkv, o, gate_up, down, head, attention
};

struct MxDecodeSelectionV2 {
  std::string status = "selected_top4";
  json::Value domain_report;
  std::vector<MxDecodePredictionV2> top;
  std::map<std::string, std::vector<MxDecodeCandidateV2>> candidates;
  double fit_r2 = 0;
  int fit_rows = 0;
};

struct MxDecodeMeasuredSelectionV2 {
  std::size_t selected = 0;
  std::vector<double> measured_ns, pred_over_meas;
};

/// Exact additive k-best merge. Input lists must be nonempty, finite,
/// nonnegative and sorted. Equal totals use lexicographic rank order.
std::vector<std::array<std::size_t, 6>> MxDecodeKBestV2(
    std::array<std::vector<double>, 6> const& costs, std::size_t count = 4);

std::vector<codegen::MxDecodeClassV2> MxDecodeDomainV2(
    codegen::MxDecodeModel const& model, std::string const& cls, int batch, int grid = 104);

/// Median of each pair of fresh-process TPOT measurements; use a challenger
/// only when it improves the predicted winner by at least one percent.
MxDecodeMeasuredSelectionV2 SelectMxDecodeMeasuredV2(
    MxDecodeSelectionV2 const& selection, std::vector<std::array<double, 2>> const& measurements);

class MxDecodeSolverV2 {
 public:
  explicit MxDecodeSolverV2(std::string const& table);
  explicit MxDecodeSolverV2(char const* table) : MxDecodeSolverV2(std::string(table)) {}
  explicit MxDecodeSolverV2(json::Value const& table);

  /// Every legal coordinate of a covered shape must have a qualified row.
  /// Attention averages interpolated costs for L=past+1, inclusive.
  MxDecodeSelectionV2 Solve(codegen::MxDecodeModel const& model, int batch,
                           int threads = 576, int grid = 104,
                           int past_lo = 64, int past_hi = 1086) const;
  /// Explicit schema3 diagnostic route. Never authorizes execution or 512.
  /// Empty classes return status=no_executable_plan with retained rejection evidence.
  MxDecodeSelectionV2 SolveQualifiedDomain(codegen::MxDecodeModel const& model, int batch,
                           int threads = 576, int grid = 104,
                           int past_lo = 64, int past_hi = 1086) const;
  MxDecodePredictionV2 Predict(codegen::MxDecodeModel const& model, int batch,
                             codegen::MxDecodeConfigV2 const& config, int grid = 104,
                             int past_lo = 64, int past_hi = 1086) const;

  /// MX-4 §4.3 least-squares diagnostic for a wholly absent GEMM/head shape.
  /// No configuration or R_next is fabricated from a time estimate. Covered
  /// shapes, including incomplete covered domains, cannot use this interface.
  double EstimateUncovered(codegen::MxDecodeModel const& model, std::string const& cls,
                           int batch, codegen::MxDecodeClassV2 const& candidate,
                           int threads = 576, int grid = 104) const;
  double FitR2() const { return fit_r2_; }
  int FitRows() const { return fit_rows_; }

 private:
  struct Row {
    std::string kind;
    std::map<std::string, int> key;
    double ns = 0, p90 = 0;
    int r_next = -1;
    MxDecodeResourcesV2 resources;
    json::Value evidence;
  };
  struct Rejection {
    std::string kind;
    std::map<std::string,int> key;
    int r_next = -1;
    MxDecodeResourcesV2 resources;
    json::Value evidence;
  };
  std::vector<Row> rows_;
  std::vector<Rejection> rejected_;
  bool outcomes_v3_ = false;
  json::Value scope_, coverage_;
  bool Rejected(std::string const&, std::map<std::string,int> const&) const;
  json::Value ValidateScope(codegen::MxDecodeModel const&, int, int, int, int, int) const;
  MxDecodeSelectionV2 SolveCore(codegen::MxDecodeModel const&, int, int, int, int, int) const;
  std::array<std::vector<double>, 2> coefficients_;
  double fit_r2_ = 0;
  int fit_rows_ = 0;

  std::vector<MxDecodeCandidateV2> Candidates(codegen::MxDecodeModel const&, std::string const&,
                                             int batch, int threads, int grid) const;
  std::vector<MxDecodeCandidateV2> Attention(codegen::MxDecodeModel const&, int batch, int threads,
                                            int past_lo, int past_hi) const;
  void Fit();
};

}  // namespace tilemega::solver
