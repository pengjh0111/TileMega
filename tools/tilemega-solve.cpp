// SPDX-License-Identifier: BSD-3-Clause
//
// P4.5 driver: solve the chain DP over the same candidate set the oracle
// measured, and check §4.4's acceptance.
//
//   (a) the uniform-g solution has to land in the oracle's measured top 3% on
//       both reference models,
//   (b) the per-operator solution's predicted gain is reported here and
//       measured end to end separately,
//   (c) the transition count and wall clock are reported so the O(L*|C|^2)
//       claim is a measurement rather than an assertion.
//
// It also checks two invariants that would otherwise be assumptions: the
// decomposition prefix + sum(Cost_i) + sum(Interface) + suffix must equal
// CostModel::Evaluate on the chosen configuration, and the general transition
// loop must agree with the separable shortcut.

#include <tilemega/Solver/AlignmentPropagation.h>
#include <tilemega/Solver/ChainDP.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/ListScheduler.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Target/TargetSpec.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tilemega::TargetSpec;
using namespace tilemega::solver;

struct Point {
  GemmConfig config;
  double measured_ms = 0.0;
};

std::vector<std::string> Split(std::string const& line, char sep) {
  std::vector<std::string> out;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, sep)) out.push_back(field);
  return out;
}

int SmemBytes(GemmConfig const& c) {
  return 4 * c.stages * c.tile_k * ((c.tile_m + 1) + (c.tile_n + 1));
}

std::string ShapeKey(GemmConfig const& c) {
  std::ostringstream out;
  out << c.tile_m << 'x' << c.tile_n << 'x' << c.tile_k << 's' << c.stages;
  return out.str();
}

std::string ConfigKey(GemmConfig const& c) {
  return ShapeKey(c) + 'k' + std::to_string(c.split_k);
}

std::map<std::string, int> ReadRegisters(std::string const& path) {
  std::map<std::string, int> out;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open register table: " + path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto fields = Split(line, '\t');
    if (fields.size() >= 2) out[fields[0]] = std::stoi(fields[1]);
  }
  return out;
}

std::vector<Point> ReadScreen(std::string const& path) {
  std::vector<Point> out;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open oracle sweep: " + path);
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    auto f = Split(line, '\t');
    if (f.size() < 13 || f[8] != "PASS") continue;
    Point point;
    point.config = {std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2]),
                    std::stoi(f[3]), std::stoi(f[4])};
    point.measured_ms = std::stod(f[6]);
    out.push_back(point);
  }
  return out;
}

/// The DP's search space is exactly the set of configurations that have a
/// ptxas register count, because occupancy is not predictable from the tile
/// shape (F-40) and a candidate whose occupancy is unknown cannot be placed.
std::vector<DpCandidate> BuildCandidates(
    std::vector<Point> const& points,
    std::map<std::string, int> const& registers) {
  std::vector<DpCandidate> out;
  std::set<std::string> seen;
  for (auto const& point : points) {
    auto const it = registers.find(ShapeKey(point.config));
    if (it == registers.end()) continue;
    if (!seen.insert(ConfigKey(point.config)).second) continue;
    DpCandidate candidate;
    candidate.config = point.config;
    candidate.registers = it->second;
    candidate.smem_bytes = SmemBytes(point.config);
    out.push_back(candidate);
  }
  return out;
}

/// 1-based rank of `config` in the measured ordering, and the size of the
/// measured top-3% band.
struct MeasuredRank {
  int rank = 0;
  int band = 0;
  double ms = 0.0;
  double best_ms = 0.0;
};

MeasuredRank RankOf(std::vector<Point> const& points, GemmConfig const& config) {
  std::vector<std::size_t> order(points.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return points[a].measured_ms < points[b].measured_ms;
  });
  MeasuredRank out;
  out.band = std::max<int>(1, int(0.03 * points.size()));
  out.best_ms = points.empty() ? 0.0 : points[order.front()].measured_ms;
  std::string const key = ConfigKey(config);
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (ConfigKey(points[order[i]].config) != key) continue;
    out.rank = int(i) + 1;
    out.ms = points[order[i]].measured_ms;
    break;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) try {
  std::string repo = ".";
  std::string cost_dir = "docs/experiments/COST_MODEL/raw";
  std::string out_dir = "docs/experiments/SOLVER/raw";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--repo" && i + 1 < argc) repo = argv[++i];
    else if (arg == "--cost-dir" && i + 1 < argc) cost_dir = argv[++i];
    else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    else {
      std::cerr << "usage: tilemega-solve [--repo DIR] [--cost-dir DIR]"
                   " [--out DIR]\n";
      return 2;
    }
  }

  TargetSpec const target =
      TargetSpec::FromJson(repo + "/configs/targets/sm_89.json");
  CostModel const cost(target);

  struct ModelSource { char const* name; char const* cu; };
  ModelSource const sources[] = {
      {"gqa2", "/docs/experiments/E2E_GEN/raw/generated_e2e.cu"},
      {"mha4", "/docs/experiments/P3_GENERALIZATION/raw/generated.cu"},
  };

  std::ofstream summary(out_dir + "/summary.tsv");
  if (!summary) throw std::runtime_error("cannot write " + out_dir + "/summary.tsv");
  summary << "model\tmode\tconfigs\tpredicted_ms\tctas_per_sm\tmeasured_rank"
             "\tband\tmeasured_ms\tbest_ms\tsolve_ms\ttransitions\n";

  bool accepted = true;
  for (auto const& source : sources) {
    auto const registers =
        ReadRegisters(cost_dir + "/registers_" + source.name + ".tsv");
    auto const points = ReadScreen(
        repo + "/docs/experiments/ORACLE/raw/screen_" + source.name + ".tsv");
    auto const candidates = BuildCandidates(points, registers);
    ModelDescription const model = ModelDescription::FromGeneratedCuda(
        repo + source.cu, ModelDims{4, 3, 7}, source.name);
    ChainDP const dp(cost, candidates);

    std::cout << "\n== " << source.name << ": " << model.gemms.size()
              << " GEMMs, " << candidates.size() << " candidates, "
              << points.size() << " measured configurations\n";

    // Tier 2 (§P4.3): prune each operator's tile shapes against its
    // neighbours' granularities before the DP ever scores them.  The
    // granularities come out of the generated stage table, so the "QKV columns
    // align to d" rule is derived rather than written down.
    std::vector<TileAxes> axes;
    axes.reserve(candidates.size());
    for (auto const& c : candidates) {
      axes.push_back({c.config.tile_n, c.config.tile_k});
    }
    AlignmentStats align;
    auto const kept = PropagateAlignment(model, axes, &align);
    {
      std::ofstream trace(out_dir + "/alignment_" + source.name + ".txt");
      if (!trace) throw std::runtime_error("cannot write the alignment trace");
      trace << ExplainAlignment(model, axes, kept);
    }

    // P4.8: the schedule the megakernel could execute, against the one it does.
    // `levels` is the DAG's critical path in stages, so it is the floor on the
    // barrier count; `nodes - levels` is everything list scheduling can win.
    {
      ScheduleStats sched;
      auto const order = ListScheduler().Schedule(model.stage_successors, &sched);
      auto const levels = ListScheduler().Levels(model.stage_successors);
      std::ofstream report(out_dir + "/schedule_" + source.name + ".txt");
      if (!report) throw std::runtime_error("cannot write the schedule report");
      report << "stages " << sched.nodes << "\nlevels " << sched.levels
             << "\nwidest_level " << sched.widest_level << "\nbarriers_saved "
             << sched.barriers_saved << "\n\nposition\tstage\tlevel\n";
      for (std::size_t i = 0; i < order.size(); ++i)
        report << i << '\t' << order[i] << '\t' << levels[order[i]] << '\n';
      std::cout << "schedule   = " << sched.nodes << " stages -> "
                << sched.levels << " barrier intervals (widest "
                << sched.widest_level << "), " << sched.barriers_saved
                << " barriers saved = "
                << (sched.nodes ? 100.0 * sched.barriers_saved / sched.nodes : 0.0)
                << "% of the barrier count\n";
    }
    // Tier 2 turns out to be non-binding on the tier-1 candidate set (below),
    // so the counterfactual is reported next to it: the same derivation, run
    // over a tile_n axis stepped by 16 instead of by powers of two, which is
    // what the constraint would face if the candidate generator ever emitted
    // one.  Without this line "prunes nothing" is indistinguishable from
    // "computes nothing".
    AlignmentStats dense_stats;
    {
      std::vector<TileAxes> dense;
      for (int tile_n = 16; tile_n <= 256; tile_n += 16) {
        for (int tile_k : {8, 16, 32}) dense.push_back({tile_n, tile_k});
      }
      PropagateAlignment(model, dense, &dense_stats);
    }
    std::cout << "tier 2     = " << align.before << " -> ["
              << align.after_min << ", " << align.after_max
              << "] candidates per operator (mean " << align.after_mean
              << "), joint space 10^" << align.joint_log10_before << " -> 10^"
              << align.joint_log10_after << ", "
              << align.unconstrained << " operators unconstrained\n"
              << "           counterfactual on a tile_n axis stepped by 16: "
              << dense_stats.before << " -> [" << dense_stats.after_min << ", "
              << dense_stats.after_max << "] per operator, joint space 10^"
              << dense_stats.joint_log10_before << " -> 10^"
              << dense_stats.joint_log10_after << "\n";

    // The tier-1-only answer is solved as a control.  Tier 2 is only allowed
    // to shrink the space, so it must not move the winner; that is a claim,
    // and it needs the other solve to exist before it can be checked.
    ChainDpOptions tier1_opts;
    tier1_opts.mode = DpMode::kUniform;
    ChainDpSolution const tier1 = dp.Solve(model, tier1_opts);

    // (a) uniform g -- what the oracle actually measured.
    ChainDpOptions uniform_opts;
    uniform_opts.mode = DpMode::kUniform;
    uniform_opts.per_operator_candidates = kept;
    ChainDpStats uniform_stats;
    ChainDpSolution const uniform = dp.Solve(model, uniform_opts, &uniform_stats);
    if (!uniform.feasible) throw std::runtime_error("no feasible uniform g");
    MeasuredRank const rank = RankOf(points, uniform.configs.front());
    bool const in_band = rank.rank > 0 && rank.rank <= rank.band;
    accepted = accepted && in_band;
    std::cout << "uniform g  = " << ConfigKey(uniform.configs.front())
              << "  ctas/sm " << uniform.residency.ctas_per_sm
              << "  predicted " << uniform.cost.total_ns / 1e6 << " ms"
              << "  measured rank " << rank.rank << "/" << points.size()
              << " (top-3% band = " << rank.band << ") "
              << (in_band ? "ACCEPT" : "REJECT") << "\n"
              << "           measured " << rank.ms << " ms vs best "
              << rank.best_ms << " ms ("
              << 100.0 * (rank.ms / rank.best_ms - 1.0) << "% off)\n";
    std::cout << "           tier-1-only control picks "
              << ConfigKey(tier1.configs.front())
              << (ConfigKey(tier1.configs.front()) ==
                          ConfigKey(uniform.configs.front())
                      ? " (same answer)"
                      : " (TIER 2 MOVED THE WINNER)")
              << "\n";
    summary << source.name << "\tuniform\t1\t" << uniform.cost.total_ns / 1e6
            << '\t' << uniform.residency.ctas_per_sm << '\t' << rank.rank
            << '\t' << rank.band << '\t' << rank.ms << '\t' << rank.best_ms
            << '\t' << uniform_stats.solve_ms << '\t'
            << uniform_stats.transitions << '\n';

    // (b) per-GEMM split factor, one tile shape.  This is the half of a
    // per-operator plan that the megakernel can already express: the split is
    // a host-side loop bound, not a template argument.
    ChainDpOptions split_opts;
    split_opts.mode = DpMode::kPerOperatorSplit;
    split_opts.per_operator_candidates = kept;
    ChainDpStats split_stats;
    ChainDpSolution const split_only = dp.Solve(model, split_opts, &split_stats);
    if (!split_only.feasible) throw std::runtime_error("no feasible split plan");
    {
      std::set<int> splits;
      for (auto const& c : split_only.configs) splits.insert(c.split_k);
      std::cout << "split-only = " << ShapeKey(split_only.configs.front())
                << " with " << splits.size() << " distinct split factors"
                << "  predicted " << split_only.cost.total_ns / 1e6 << " ms ("
                << 100.0 * (1.0 - split_only.cost.total_ns /
                                      uniform.cost.total_ns)
                << "% below uniform)\n";
      summary << source.name << "\tper_gemm_split\t" << splits.size() << '\t'
              << split_only.cost.total_ns / 1e6 << '\t'
              << split_only.residency.ctas_per_sm << "\t0\t" << rank.band
              << "\t0\t" << rank.best_ms << '\t' << split_stats.solve_ms
              << '\t' << split_stats.transitions << '\n';
    }

    // (b) the full per-operator state, both ways round.
    ChainDpStats general_stats, separable_stats;
    ChainDpOptions general_opts;
    general_opts.per_operator_candidates = kept;
    ChainDpSolution const general = dp.Solve(model, general_opts, &general_stats);
    ChainDpOptions separable_opts;
    separable_opts.per_operator_candidates = kept;
    separable_opts.general_transitions = false;
    ChainDpSolution const separable =
        dp.Solve(model, separable_opts, &separable_stats);
    if (!general.feasible) throw std::runtime_error("no feasible per-operator g");

    double const disagreement =
        std::abs(general.cost.total_ns - separable.cost.total_ns);
    std::size_t distinct = 0;
    {
      std::set<std::string> keys;
      for (auto const& c : general.configs) keys.insert(ConfigKey(c));
      distinct = keys.size();
    }
    std::cout << "per-op g   = " << distinct << " distinct configurations"
              << "  ctas/sm " << general.residency.ctas_per_sm
              << "  predicted " << general.cost.total_ns / 1e6 << " ms ("
              << 100.0 * (1.0 - general.cost.total_ns / uniform.cost.total_ns)
              << "% below uniform)\n"
              << "           general " << general_stats.transitions
              << " transitions in " << general_stats.solve_ms << " ms; "
              << "separable " << separable_stats.solve_ms << " ms; "
              << "objectives differ by " << disagreement << " ns\n"
              << "           interface spread " << general_stats.interface_spread_ns
              << " ns, decomposition error " << general_stats.decomposition_error_ns
              << " ns, " << general_stats.residency_levels
              << " residency levels admitted\n";
    summary << source.name << "\tper_operator\t" << distinct << '\t'
            << general.cost.total_ns / 1e6 << '\t'
            << general.residency.ctas_per_sm << "\t0\t" << rank.band
            << "\t0\t" << rank.best_ms << '\t' << general_stats.solve_ms << '\t'
            << general_stats.transitions << '\n';
    summary << source.name << "\tper_operator_separable\t" << distinct << '\t'
            << separable.cost.total_ns / 1e6 << '\t'
            << separable.residency.ctas_per_sm << "\t0\t" << rank.band
            << "\t0\t" << rank.best_ms << '\t' << separable_stats.solve_ms
            << '\t' << separable_stats.transitions << '\n';

    // Every arm is written out per operator, including the two that are
    // constant, so run_per_operator.sh compiles what the DP said instead of
    // re-deriving it from the winning column.
    std::ofstream plan(out_dir + "/plan_" + source.name + ".tsv");
    plan << "gemm\tn\tk";
    for (char const* arm : {"uniform", "split_only", "per_op"})
      plan << '\t' << arm << "_tile_m\t" << arm << "_tile_n\t" << arm
           << "_tile_k\t" << arm << "_stages\t" << arm << "_split_k";
    plan << '\n';
    for (std::size_t i = 0; i < general.configs.size(); ++i) {
      plan << i << '\t' << model.gemms[i].n << '\t' << model.gemms[i].k;
      for (auto const* arm : {&uniform, &split_only, &general}) {
        auto const& c = arm->configs[i];
        plan << '\t' << c.tile_m << '\t' << c.tile_n << '\t' << c.tile_k
             << '\t' << c.stages << '\t' << c.split_k;
      }
      plan << '\n';
    }
  }

  std::cout << "\nacceptance (a) uniform g inside the measured top 3% on both"
               " models: " << (accepted ? "PASS" : "FAIL") << '\n';
  return accepted ? 0 : 1;
} catch (std::exception const& error) {
  std::cerr << "tilemega-solve: " << error.what() << '\n';
  return 1;
}
