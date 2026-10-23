// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Solver/MxDecodeSolver.h
//
// The MX-4 decode solver (MX4 prompt §4). Its execution model is the static
// program itself: stages are separated by full barriers and reduce inside
// the stage, so a step is the sum of its stages and each GEMM class is
// chosen on its own. Costs are the measured stage table
// (configs/targets/xcore1000_mx_decode.json, `mx_decode_costs` of the
// target); a shape the table lacks is priced by the least-squares model of
// §4.3, fitted to the table.
#pragma once

#include <tilemega/Codegen/MxDecodeProgram.h>

#include <map>
#include <string>
#include <vector>

namespace tilemega::solver {

/// One GEMM class's candidate and its stage envelope (ns).
struct MxDecodeCandidate {
  codegen::MxDecodeClass k;
  double ns = 0;
  bool measured = false;  // from the table; false: §4.3 model
  int spill_bytes = -1;   // stage_bench kernel's stack frame (-1: unknown)
};

struct MxDecodeSelection {
  codegen::MxDecodeConfig first, second;
  double first_ns = 0, second_ns = 0;  // predicted step (TPOT) ns
  std::string swapped;                 // the class plan #2 changes
  /// Per class: every candidate, cheapest first.
  std::map<std::string, std::vector<MxDecodeCandidate>> candidates;
  /// Predicted envelope per stage class of plan #1 (attention: mean over the
  /// plan's decode steps), and the stages per step it counts.
  std::map<std::string, double> stage_ns;
  double fit_r2 = 0, fit_max_rel = 0;  // §4.3 model on the table
  int fit_rows = 0;
};

class MxDecodeSolver {
 public:
  /// `table`: the measured stage table (JSON).
  explicit MxDecodeSolver(std::string const& table);

  /// §4.1 domain x §4.2 costs -> §4.4 items 1: plan #1 and #2.
  MxDecodeSelection Solve(codegen::MxDecodeModel const& model, int batch, int tmin, int grid, int past_lo,
                          int past_hi) const;

  /// The predicted step of a given configuration (ns) and its stage classes.
  double Predict(codegen::MxDecodeModel const& model, int batch, codegen::MxDecodeConfig const& config, int grid,
                 int past_lo, int past_hi, std::map<std::string, double>* stages = nullptr) const;

 private:
  struct GemmRow {
    int n, k, b, tn, d;
    std::string op;
    int norm;
    bool stream_k;
    double ns;
    int spill;
  };
  struct AttnRow {
    int dim, q, hkv, b, length, tmin;
    double ns;
  };
  std::vector<GemmRow> gemm_;
  std::vector<AttnRow> attn_;
  std::vector<std::pair<std::pair<int, int>, double>> embed_;  // (hidden, b) -> ns
  // §4.3 coefficients: see MxDecodeSolver.cpp Features().
  std::vector<double> coef_;
  double r2_ = 0, max_rel_ = 0;

  void Fit();
  MxDecodeCandidate Price(int n, int k, int b, std::string const& op, int norm, codegen::MxDecodeClass const& c,
                          int grid) const;
  double Attention(codegen::MxDecodeModel const& model, int batch, int tmin, int grid, int past_lo,
                   int past_hi) const;
  double Embed(int hidden, int batch) const;
};

/// The §4.1 candidates of one class: TN per §3.5, D in 1..4 with rule 6,
/// Part (head: kTileRange), reduction legality, stream-K only with U >= G.
std::vector<codegen::MxDecodeClass> MxDecodeDomain(codegen::MxDecodeModel const& model, std::string const& cls,
                                                   int batch, int grid);

}  // namespace tilemega::solver
