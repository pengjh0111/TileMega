// SPDX-License-Identifier: BSD-3-Clause
// P4.5: the chain DP is only trustworthy if its decomposition is the cost
// model it claims to minimize, so every check here is paired with the mutation
// it must catch.
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Target/TargetSpec.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tilemega;
using namespace tilemega::solver;

#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

namespace {

/// Two GEMMs of different widths with an RMSNorm between them and an
/// elementwise tail, which is the smallest model that exercises a prefix, an
/// interior interface and a suffix at once.
ModelDescription TinyModel() {
  ModelDescription model;
  model.name = "tiny";
  model.dims = {4, 3, 7};
  model.gemms = {{512, 512}, {256, 512}};
  model.stages = {
      {StageKind::kRMSNorm, -1, 1, 512, 0},
      {StageKind::kGemm, 0, 0, 0, 0},
      {StageKind::kRMSNorm, -1, 1, 512, 0},
      {StageKind::kGemm, 1, 0, 0, 0},
      {StageKind::kElementwise, -1, 256, 0, 0},
  };
  return model;
}

int SmemBytes(GemmConfig const& c) {
  return 4 * c.stages * c.tile_k * ((c.tile_m + 1) + (c.tile_n + 1));
}

std::vector<DpCandidate> Candidates() {
  std::vector<DpCandidate> out;
  // Registers are not predictable from the tile shape (F-40); these are the
  // ptxas counts the oracle recorded for these four shapes on sm_89.
  struct Row { int m, n, k, stages, regs; };
  Row const rows[] = {{16, 64, 16, 2, 168}, {16, 32, 16, 2, 122},
                      {32, 32, 16, 3, 200}, {64, 64, 16, 3, 248}};
  for (auto const& row : rows) {
    for (int split : {1, 2, 4, 8, 16}) {
      DpCandidate candidate;
      candidate.config = {row.m, row.n, row.k, row.stages, split};
      candidate.registers = row.regs;
      candidate.smem_bytes = SmemBytes(candidate.config);
      out.push_back(candidate);
    }
  }
  return out;
}

}  // namespace

int main() {
  TargetSpec const target =
      TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR) +
                           "/configs/targets/sm_89.json");
  CostModel const cost(target);
  ModelDescription const model = TinyModel();
  ChainDP const dp(cost, Candidates());

  ChainDpOptions general_opts;
  ChainDpStats general_stats;
  ChainDpSolution const general = dp.Solve(model, general_opts, &general_stats);
  REQUIRE(general.feasible);
  REQUIRE(general.configs.size() == model.gemms.size());
  REQUIRE(general_stats.operators == 2);
  REQUIRE(general_stats.transitions > 0);

  // The decomposition has to reproduce the whole-model evaluation exactly, not
  // approximately: a barrier charged twice, or an interior stage charged to
  // nobody, shows up here and nowhere else.
  REQUIRE(general_stats.decomposition_error_ns < 1e-6);

  // The general transition loop and the separable shortcut are two independent
  // ways to the same optimum whenever the interface term does not depend on
  // the pair, which is what interface_spread_ns reports.
  REQUIRE(general_stats.interface_spread_ns == 0.0);
  ChainDpOptions separable_opts;
  separable_opts.general_transitions = false;
  ChainDpStats separable_stats;
  ChainDpSolution const separable =
      dp.Solve(model, separable_opts, &separable_stats);
  REQUIRE(separable.feasible);
  REQUIRE(std::fabs(separable.cost.total_ns - general.cost.total_ns) < 1e-6);
  REQUIRE(separable_stats.transitions == 0);
  REQUIRE(general_stats.transitions > separable_stats.transitions);

  // Widening the state can only lower the objective, and each mode has to be
  // reachable by the one above it.
  ChainDpOptions uniform_opts;
  uniform_opts.mode = DpMode::kUniform;
  ChainDpSolution const uniform = dp.Solve(model, uniform_opts, nullptr);
  ChainDpOptions split_opts;
  split_opts.mode = DpMode::kPerOperatorSplit;
  ChainDpSolution const split = dp.Solve(model, split_opts, nullptr);
  REQUIRE(uniform.feasible && split.feasible);
  REQUIRE(split.cost.total_ns <= uniform.cost.total_ns + 1e-6);
  REQUIRE(general.cost.total_ns <= split.cost.total_ns + 1e-6);

  // kUniform means one configuration everywhere; kPerOperatorSplit means one
  // tile shape everywhere.  Both are contracts the megakernel depends on.
  std::set<std::string> uniform_keys, split_shapes;
  for (auto const& c : uniform.configs) {
    uniform_keys.insert(std::to_string(c.tile_m) + "x" +
                        std::to_string(c.tile_n) + "x" +
                        std::to_string(c.tile_k) + "s" +
                        std::to_string(c.stages) + "k" +
                        std::to_string(c.split_k));
  }
  for (auto const& c : split.configs) {
    split_shapes.insert(std::to_string(c.tile_m) + "x" +
                        std::to_string(c.tile_n) + "x" +
                        std::to_string(c.tile_k) + "s" +
                        std::to_string(c.stages));
  }
  REQUIRE(uniform_keys.size() == 1);
  REQUIRE(split_shapes.size() == 1);

  // The residency the solver reports has to be the one F-40 gives for the
  // configurations it actually chose -- the minimum over them, not a bound.
  // Relaxing the solver's pin to a bound does not change the answer on this
  // model or on either reference model, so this assertion is a guard on the
  // formulation rather than a caught bug; it is kept because §4.3's residency
  // is a minimum, and a bound would silently score a plan at an occupancy the
  // hardware would never give it.
  int expected = 1 << 20;
  for (auto const& c : general.configs) {
    for (auto const& candidate : dp.candidates()) {
      if (candidate.config.tile_m != c.tile_m ||
          candidate.config.tile_n != c.tile_n ||
          candidate.config.tile_k != c.tile_k ||
          candidate.config.stages != c.stages) {
        continue;
      }
      int const ctas =
          dp.CtasPerSm(candidate.smem_bytes, candidate.registers);
      expected = expected < ctas ? expected : ctas;
      break;
    }
  }
  REQUIRE(general.residency.ctas_per_sm == expected);

  // A GEMM executed by two stages is a different recurrence, and answering it
  // with this one would be silently wrong.
  ModelDescription duplicated = model;
  duplicated.stages.push_back({StageKind::kGemm, 0, 0, 0, 0});
  bool threw = false;
  try {
    dp.Solve(duplicated, general_opts, nullptr);
  } catch (std::exception const&) {
    threw = true;
  }
  REQUIRE(threw);

  std::printf("chain_dp: ok\n");
  return 0;
}
