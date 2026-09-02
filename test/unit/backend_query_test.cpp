// SPDX-License-Identifier: BSD-3-Clause
// The backend query channel of §1.2 principle 2: what a candidate answers
// before anything is compiled, what only ptxas can answer, and that the
// enumeration is the legal region rather than a filtered Cartesian product.
#include <tilemega/Solver/CandidateGenerator.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

using namespace tilemega::solver;
using tilemega::TargetSpec;

namespace {

void Require(bool condition, char const* expression, int line) {
  if (condition) return;
  std::cerr << "requirement failed at line " << line << ": " << expression
            << '\n';
  std::exit(1);
}
#define REQUIRE(expr) Require((expr), #expr, __LINE__)

TargetSpec Load(char const* name) {
  return TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR) +
                              "/configs/targets/" + name + ".json");
}

std::string ReadFile(std::string const& path) {
  std::ifstream in(path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

/// The two-layer GQA acceptance model: 4 tokens, hidden 512, intermediate
/// 1024, 2 kv heads of 128 (docs/experiments/E2E/fixture/manifest.json).
std::vector<GemmProblem> GqaProblems() {
  std::vector<GemmProblem> problems;
  for (int layer = 0; layer < 2; ++layer) {
    problems.push_back({4, 512, 512});    // q_proj
    problems.push_back({4, 256, 512});    // k_proj
    problems.push_back({4, 256, 512});    // v_proj
    problems.push_back({4, 512, 512});    // o_proj
    problems.push_back({4, 1024, 512});   // gate_proj
    problems.push_back({4, 1024, 512});   // up_proj
    problems.push_back({4, 512, 1024});   // down_proj
  }
  return problems;
}

}  // namespace

int main() {
  TargetSpec sm89 = Load("sm_89");
  TargetSpec sm90 = Load("sm_90");
  CandidateGenerator generator(sm89);

  CandidateGenerator::Stats stats;
  auto enumerate_start = std::chrono::steady_clock::now();
  std::vector<BackendCandidate> candidates = generator.Enumerate(&stats);
  auto enumerate_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - enumerate_start).count();
  std::cerr << "enumeration: touched=" << stats.touched
            << " wall_pruned=" << stats.wall_pruned
            << " rejected=" << stats.shape_rejected << " legal=" << stats.legal
            << " cartesian=" << stats.cartesian << '\n';
  // The same envelope the CUTLASS probe walks (docs/experiments/ORACLE), whose
  // own instantiation of every point reports 224 candidates that fit sm_89's
  // 101376-byte opt-in budget.
  REQUIRE(stats.cartesian == 300);
  REQUIRE(stats.legal == 224);
  REQUIRE(static_cast<int>(candidates.size()) == stats.legal);
  REQUIRE(stats.touched < stats.cartesian);
  std::set<std::string> unique;
  for (auto const& candidate : candidates) {
    REQUIRE(unique.insert(candidate.Describe()).second);
    REQUIRE(candidate.isLegal(sm89));
    REQUIRE(candidate.smemBytes() <= sm89.res.max_dynamic_smem_per_cta);
    REQUIRE(!candidate.estimatedRegisters());
  }

  // The six queries on the candidate the fixed-`g` control uses. 49536 bytes
  // is what `sizeof(Mainloop::SharedStorage)` reports for this shape.
  BackendCandidate control(SimtF32Traits(128, 128, 16, 3));
  REQUIRE(control.Describe() == "128x128x16s3");
  REQUIRE(control.isLegal(sm89));
  REQUIRE(control.threads() == 256);
  REQUIRE(control.smemBytes() == 49536);
  REQUIRE(control.clusterShape().size() == 1);
  REQUIRE(control.alignmentRequirement().a == 1 &&
          control.alignmentRequirement().b == 1);
  REQUIRE(control.architectureRequirement() == 80);
  REQUIRE(control.CoResidentPerSM(sm89) == 2);

  // Legality is answered against the target, never against a version literal.
  BackendTraits hopper = SimtF32Traits(128, 128, 16, 3);
  hopper.arch_sm = 90;
  REQUIRE(!BackendCandidate(hopper).isLegal(sm89));
  REQUIRE(BackendCandidate(hopper).isLegal(sm90));
  BackendTraits clustered = SimtF32Traits(128, 128, 16, 3);
  clustered.cluster = {2, 1, 1};
  REQUIRE(!BackendCandidate(clustered).isLegal(sm89));
  REQUIRE(BackendCandidate(clustered).isLegal(sm90));
  BackendTraits huge = SimtF32Traits(256, 256, 32, 5);
  REQUIRE(huge.smem_bytes == 328960);
  REQUIRE(!BackendCandidate(huge).isLegal(sm89));

  // Registers exist only after a real compile.
  BackendCandidate measured(SimtF32Traits(128, 128, 16, 3));
  REQUIRE(!measured.estimatedRegisters());
  REQUIRE(!measured.RecordPtxas("ptxas info : nothing useful here\n"));
  REQUIRE(!measured.estimatedRegisters());
  // A real ptxas log of the control configuration, kept as a fixture so the
  // parser is tested without the experiment tree.
  std::string log = ReadFile(std::string(TILEMEGA_SOURCE_DIR) +
                             "/test/fixtures/ptxas_gqa2_128x128x16s3.txt");
  REQUIRE(!log.empty());
  REQUIRE(measured.RecordPtxas(log));
  REQUIRE(measured.estimatedRegisters() && *measured.estimatedRegisters() == 168);
  BackendCandidate l1(SimtF32Traits(128, 128, 16, 3));
  REQUIRE(l1.RecordPtxas(log, "tilemega_l1_kernel"));
  REQUIRE(*l1.estimatedRegisters() == 167);
  std::cerr << "ptxas: l1 kernel uses " << *l1.estimatedRegisters()
            << " registers\n";

  // Tier 2 ranks the survivors without compiling any of them.
  std::vector<GemmProblem> problems = GqaProblems();
  auto rank_start = std::chrono::steady_clock::now();
  std::vector<BackendCandidate> top = generator.TopK(candidates, problems, 8);
  auto rank_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - rank_start).count();
  std::cerr << "cost: enumerate " << enumerate_us << " us for " << stats.legal
            << " candidates, rank " << rank_us << " us over "
            << problems.size() << " GEMMs\n";
  REQUIRE(static_cast<int>(top.size()) == 8);
  std::cerr << "tier-2 top-8:";
  for (auto const& candidate : top) std::cerr << ' ' << candidate.Describe();
  std::cerr << '\n';
  double best = generator.RankKey(top.front(), problems);
  double control_cost = generator.RankKey(control, problems);
  std::cerr << "tier-2 cycles: best=" << best << " control(128x128x16s3)="
            << control_cost << '\n';
  REQUIRE(best < control_cost);
  int rank = 0, measured_best = -1;
  for (auto const& candidate : generator.TopK(candidates, problems,
                                              static_cast<int>(candidates.size()))) {
    ++rank;
    if (candidate.Describe() == "32x32x32s3") measured_best = rank;
  }
  std::cerr << "tier-2 rank of the measured best (32x32x32s3): " << measured_best
            << " of " << candidates.size() << '\n';
  REQUIRE(measured_best > 0);
  return 0;
}
