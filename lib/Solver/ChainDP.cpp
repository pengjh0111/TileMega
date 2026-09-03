// SPDX-License-Identifier: BSD-3-Clause

#include <tilemega/Solver/ChainDP.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace tilemega::solver {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kThreads = 256;

/// (tile_m, tile_n, tile_k, stages) -- everything a compiled GemmImpl variant
/// is parameterised by.  The split factor is deliberately not part of it.
using ShapeKey = std::array<int, 4>;

}  // namespace

ChainDP::ChainDP(CostModel const& model, std::vector<DpCandidate> candidates)
    : cost_(&model), candidates_(std::move(candidates)) {}

int ChainDP::CtasPerSm(int smem_bytes, int registers) const {
  // F-40, verified on 1075 of the oracle's 1077 measured shapes.
  auto const& res = cost_->target().res;
  int const per_cta_regs = 8 * ((registers * 32 + 255) / 256) * kThreads;
  int const by_regs = per_cta_regs > 0 ? res.regs_per_sm / per_cta_regs : 1;
  int const by_smem =
      smem_bytes > 0 ? res.max_dynamic_smem_per_cta / smem_bytes : 1;
  int const by_threads = res.max_threads_per_sm / kThreads;
  return std::max(1, std::min({by_regs, by_smem, by_threads}));
}

std::vector<int> ChainDP::GemmStages(ModelDescription const& model) const {
  std::vector<int> out;
  std::vector<bool> seen(model.gemms.size(), false);
  for (std::size_t i = 0; i < model.stages.size(); ++i) {
    auto const& stage = model.stages[i];
    if (stage.kind != StageKind::kGemm) continue;
    if (stage.gemm < 0 ||
        static_cast<std::size_t>(stage.gemm) >= model.gemms.size()) {
      throw std::runtime_error("stage names a GEMM the model does not have");
    }
    // The chain carries one state per GEMM operator.  A GEMM executed twice
    // would need both stages to share that state, which is a different
    // recurrence; the reference models never do it, so reject rather than
    // silently solve the wrong problem.
    if (seen[stage.gemm]) {
      throw std::runtime_error("a GEMM appears in more than one stage");
    }
    seen[stage.gemm] = true;
    out.push_back(static_cast<int>(i));
  }
  return out;
}

double ChainDP::BetweenNs(ModelDescription const& model,
                          std::vector<int> const& gemm_stages, int from,
                          int to, Residency residency) const {
  int const begin = from < 0 ? 0 : gemm_stages[from] + 1;
  int const end =
      to < 0 ? static_cast<int>(model.stages.size()) : gemm_stages[to];
  double const barrier = cost_->BarrierNs(residency);
  double ns = 0.0;
  for (int i = begin; i < end; ++i) {
    ns += cost_->NonGemmStageNs(model.stages[i], model.dims, residency) +
          barrier;
  }
  return ns;
}

double ChainDP::CarryNs(ModelDescription const& model,
                        std::vector<int> const& gemm_stages, int from, int to,
                        double miss, GemmConfig const& from_config,
                        GemmConfig const& to_config) const {
  (void)from_config;
  (void)to_config;
  if (from < 0 || to < 0 || miss <= 0.0) return 0.0;
  auto const& calib = cost_->target().calib;
  if (calib.dram_gbps <= 0.0 || calib.l2_gbps <= 0.0) return 0.0;
  // An upper bound: it charges the producer's whole output, and it is
  // identically zero whenever the live footprint sits below the measured L2
  // knee -- the regime both reference models are in.  The tile shapes are in
  // the signature because a tighter form is a function of them, not because
  // this one uses them; ChainDpStats::interface_spread_ns reports the
  // consequence instead of asserting it.
  GemmOp const& producer = model.gemms[model.stages[gemm_stages[from]].gemm];
  double const bytes = 4.0 * model.dims.seq * producer.n;
  return miss * bytes * (1.0 / calib.dram_gbps - 1.0 / calib.l2_gbps);
}

double ChainDP::Interface(ModelDescription const& model, int from, int to,
                          GemmConfig const& from_config,
                          GemmConfig const& to_config,
                          Residency residency) const {
  std::vector<int> const gemm_stages = GemmStages(model);
  double const miss =
      1.0 - cost_->CacheHitProbability(model.LiveFootprintBytes());
  return BetweenNs(model, gemm_stages, from, to, residency) +
         CarryNs(model, gemm_stages, from, to, miss, from_config, to_config);
}

ChainDpSolution ChainDP::Solve(ModelDescription const& model,
                               ChainDpOptions options,
                               ChainDpStats* stats) const {
  auto const started = std::chrono::steady_clock::now();
  std::vector<int> const gemm_stages = GemmStages(model);
  int const layers = static_cast<int>(gemm_stages.size());
  ChainDpSolution best;
  double best_total = kInf;
  ChainDpStats local;
  local.candidates = static_cast<int>(candidates_.size());
  local.operators = layers;
  if (layers == 0) {
    if (stats) *stats = local;
    return best;
  }

  double const miss =
      1.0 - cost_->CacheHitProbability(model.LiveFootprintBytes());

  std::vector<int> ctas(candidates_.size());
  for (std::size_t c = 0; c < candidates_.size(); ++c) {
    ctas[c] = CtasPerSm(candidates_[c].smem_bytes, candidates_[c].registers);
  }

  for (int r = 1; r <= options.max_ctas_per_sm; ++r) {
    // §4.3: residency is one number for the whole kernel, and it is the
    // *minimum* over the chosen configurations, not a free parameter.  Pinning
    // it in an outer loop and admitting only states that reach it keeps the
    // inner problem exact; requiring one chosen state to sit exactly at `r` is
    // what stops a solution whose true residency is higher from being scored
    // at the wrong occupancy here instead of at its own level.
    std::vector<int> admissible;
    for (std::size_t c = 0; c < candidates_.size(); ++c) {
      if (ctas[c] < r) continue;
      admissible.push_back(static_cast<int>(c));
    }
    if (admissible.empty()) continue;
    Residency const residency{r};
    std::size_t const n = admissible.size();

    // Tier-2 (§3.2) admissibility, lifted from global candidate indices onto
    // this level's admitted set.  `allow_all` is the subset a *uniform* answer
    // may use: one configuration for the whole model has to satisfy every
    // operator's alignment constraint, not just its own.
    std::vector<std::vector<char>> allow(layers, std::vector<char>(n, 1));
    std::vector<char> allow_all(n, 1);
    if (!options.per_operator_candidates.empty()) {
      if (static_cast<int>(options.per_operator_candidates.size()) != layers) {
        throw std::invalid_argument(
            "chain dp: tier-2 plan does not have one entry per GEMM");
      }
      std::vector<int> slot(candidates_.size(), -1);
      for (std::size_t j = 0; j < n; ++j) slot[admissible[j]] = static_cast<int>(j);
      for (int i = 0; i < layers; ++i) {
        std::fill(allow[i].begin(), allow[i].end(), 0);
        for (int c : options.per_operator_candidates[i]) {
          if (c >= 0 && c < static_cast<int>(slot.size()) && slot[c] >= 0) {
            allow[i][slot[c]] = 1;
          }
        }
      }
      for (std::size_t j = 0; j < n; ++j) {
        for (int i = 0; i < layers; ++i) {
          allow_all[j] = allow_all[j] && allow[i][j];
        }
      }
    }
    // A level survives only if every operator keeps a state and some kept
    // state sits exactly at `r`; otherwise the pin would score a plan at an
    // occupancy no admissible choice actually reaches.
    bool has_exact = false;
    bool level_ok = true;
    for (int i = 0; i < layers && level_ok; ++i) {
      bool any = false;
      for (std::size_t j = 0; j < n; ++j) {
        if (!allow[i][j]) continue;
        any = true;
        has_exact = has_exact || ctas[admissible[j]] == r;
      }
      level_ok = any;
    }
    if (!level_ok || !has_exact) continue;
    ++local.residency_levels;
    double const barrier = cost_->BarrierNs(residency);

    // Cost_i(s): the GEMM stage, the combiner the split rewrite appends, and
    // the one grid barrier each of those stages owns.
    std::vector<std::vector<double>> op_cost(layers, std::vector<double>(n));
    for (int i = 0; i < layers; ++i) {
      GemmOp const& gemm = model.gemms[model.stages[gemm_stages[i]].gemm];
      for (std::size_t j = 0; j < n; ++j) {
        GemmConfig const& cfg = candidates_[admissible[j]].config;
        int chunks = 1;
        double ns = cost_->GemmStageNs(gemm, cfg, residency, model, &chunks);
        if (chunks > 1) {
          ns += cost_->CombineStageNs(gemm, chunks, model.dims) + barrier;
        }
        op_cost[i][j] = ns + barrier;
      }
    }

    // The configuration-independent part of each interface, hoisted; and the
    // spread of the pair-dependent part, measured over every predecessor.
    std::vector<double> between(layers + 1);
    between[0] = BetweenNs(model, gemm_stages, -1, 0, residency);
    between[layers] = BetweenNs(model, gemm_stages, layers - 1, -1, residency);
    for (int i = 1; i < layers; ++i) {
      between[i] = BetweenNs(model, gemm_stages, i - 1, i, residency);
      if (!options.interface_term) continue;
      double lo = kInf, hi = -kInf;
      for (std::size_t p = 0; p < n; ++p) {
        double const v =
            CarryNs(model, gemm_stages, i - 1, i, miss,
                    candidates_[admissible[p]].config,
                    candidates_[admissible[0]].config);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      local.interface_spread_ns = std::max(local.interface_spread_ns, hi - lo);
    }

    double const fixed = between[0] + between[layers];
    std::vector<int> choice(layers, -1);
    double total = kInf;

    if (options.mode == DpMode::kUniform) {
      for (std::size_t j = 0; j < n; ++j) {
        if (ctas[admissible[j]] != r || !allow_all[j]) continue;
        double sum = fixed;
        for (int i = 0; i < layers; ++i) {
          sum += op_cost[i][j];
          if (i) {
            sum += between[i];
            if (options.interface_term) {
              sum += CarryNs(model, gemm_stages, i - 1, i, miss,
                             candidates_[admissible[j]].config,
                             candidates_[admissible[j]].config);
            }
          }
        }
        if (sum < total) {
          total = sum;
          std::fill(choice.begin(), choice.end(), static_cast<int>(j));
        }
      }
    } else if (options.mode == DpMode::kPerOperatorSplit) {
      // One tile shape, chosen for the whole model; the split factor is then
      // free per GEMM.  Candidates that share a shape share their smem and
      // register counts, so a shape is admissible at exactly one residency and
      // the outer loop already selected it.
      std::map<ShapeKey, std::vector<std::size_t>> by_shape;
      for (std::size_t j = 0; j < n; ++j) {
        if (ctas[admissible[j]] != r) continue;
        GemmConfig const& c = candidates_[admissible[j]].config;
        // A shape is only a candidate if every operator can take *some* split
        // of it; the split itself is then free per operator below.

        by_shape[{c.tile_m, c.tile_n, c.tile_k, c.stages}].push_back(j);
      }
      for (auto const& [shape, group] : by_shape) {
        (void)shape;
        double sum = fixed;
        std::vector<int> pick(layers, -1);
        for (int i = 0; i < layers; ++i) {
          if (i) {
            sum += between[i];
            if (options.interface_term) {
              sum += CarryNs(model, gemm_stages, i - 1, i, miss,
                             candidates_[admissible[group.front()]].config,
                             candidates_[admissible[group.front()]].config);
            }
          }
          double best_op = kInf;
          for (std::size_t j : group) {
            if (!allow[i][j]) continue;
            if (op_cost[i][j] < best_op) {
              best_op = op_cost[i][j];
              pick[i] = static_cast<int>(j);
            }
          }
          if (pick[i] < 0) { sum = kInf; break; }
          sum += best_op;
        }
        if (sum < total) {
          total = sum;
          choice = pick;
        }
      }
    } else if (options.general_transitions) {
      // O(layers * |C|^2), the recurrence written out in full.  The second
      // index of the state is "some operator already sits exactly at r",
      // which is what makes the pinned residency exact rather than a bound.
      std::vector<std::array<double, 2>> prev(n), cur(n);
      std::vector<std::vector<std::array<int, 2>>> back(
          layers, std::vector<std::array<int, 2>>(n, {-1, -1}));
      for (std::size_t j = 0; j < n; ++j) {
        if (!allow[0][j]) { prev[j] = {kInf, kInf}; continue; }
        double const v = between[0] + op_cost[0][j];
        bool const exact = ctas[admissible[j]] == r;
        prev[j][0] = exact ? kInf : v;
        prev[j][1] = exact ? v : kInf;
      }
      for (int i = 1; i < layers; ++i) {
        for (std::size_t j = 0; j < n; ++j) cur[j] = {kInf, kInf};
        for (std::size_t j = 0; j < n; ++j) {
          if (!allow[i][j]) continue;
          bool const exact = ctas[admissible[j]] == r;
          for (std::size_t p = 0; p < n; ++p) {
            ++local.transitions;
            double step = between[i] + op_cost[i][j];
            if (options.interface_term) {
              step += CarryNs(model, gemm_stages, i - 1, i, miss,
                              candidates_[admissible[p]].config,
                              candidates_[admissible[j]].config);
            }
            for (int f = 0; f < 2; ++f) {
              if (prev[p][f] == kInf) continue;
              int const nf = (f == 1 || exact) ? 1 : 0;
              double const v = prev[p][f] + step;
              if (v < cur[j][nf]) {
                cur[j][nf] = v;
                back[i][j][nf] = static_cast<int>(p) * 2 + f;
              }
            }
          }
        }
        prev.swap(cur);
      }
      int end_j = -1;
      for (std::size_t j = 0; j < n; ++j) {
        if (prev[j][1] < total) {
          total = prev[j][1];
          end_j = static_cast<int>(j);
        }
      }
      if (end_j >= 0) {
        total += between[layers];
        int j = end_j, f = 1;
        for (int i = layers - 1; i > 0; --i) {
          choice[i] = j;
          int const link = back[i][j][f];
          j = link / 2;
          f = link % 2;
        }
        choice[0] = j;
      }
    } else {
      // The separable shortcut, valid only while the carry term does not
      // depend on the pair.  Kept so the general loop above has something to
      // agree with.
      if (local.interface_spread_ns > 0.0) {
        throw std::runtime_error(
            "separable shortcut requested but Interface is pair-dependent");
      }
      std::vector<int> arg_any(layers, -1), arg_exact(layers, -1);
      double sum = fixed;
      for (int i = 0; i < layers; ++i) {
        if (i) {
          sum += between[i];
          if (options.interface_term) {
            sum += CarryNs(model, gemm_stages, i - 1, i, miss,
                           candidates_[admissible[0]].config,
                           candidates_[admissible[0]].config);
          }
        }
        double best_any = kInf, best_exact = kInf;
        for (std::size_t j = 0; j < n; ++j) {
          if (!allow[i][j]) continue;
          if (op_cost[i][j] < best_any) {
            best_any = op_cost[i][j];
            arg_any[i] = static_cast<int>(j);
          }
          if (ctas[admissible[j]] == r && op_cost[i][j] < best_exact) {
            best_exact = op_cost[i][j];
            arg_exact[i] = static_cast<int>(j);
          }
        }
        sum += best_any;
      }
      choice = arg_any;
      double penalty = kInf;
      int swap_at = -1;
      for (int i = 0; i < layers; ++i) {
        if (arg_exact[i] < 0) continue;
        double const delta = op_cost[i][arg_exact[i]] - op_cost[i][arg_any[i]];
        if (delta < penalty) {
          penalty = delta;
          swap_at = i;
        }
      }
      if (swap_at >= 0) {
        choice[swap_at] = arg_exact[swap_at];
        total = sum + penalty;
      }
    }

    if (choice.front() < 0 || total >= best_total) continue;
    ChainDpSolution candidate;
    candidate.feasible = true;
    candidate.residency = residency;
    candidate.configs.assign(model.gemms.size(), GemmConfig{});
    for (int i = 0; i < layers; ++i) {
      DpCandidate const& picked = candidates_[admissible[choice[i]]];
      candidate.configs[model.stages[gemm_stages[i]].gemm] = picked.config;
      candidate.max_smem_bytes =
          std::max(candidate.max_smem_bytes, picked.smem_bytes);
      candidate.max_registers =
          std::max(candidate.max_registers, picked.registers);
    }
    candidate.cost = cost_->Evaluate(model, candidate.configs, residency);
    // Prefix + sum of per-operator costs + interfaces + suffix has to
    // reproduce the whole-model evaluation exactly, or the DP is minimizing
    // something the validated cost model never said.  The carry is the one
    // term the DP adds on top of that model, so it comes back out first.
    double carry = 0.0;
    if (options.interface_term) {
      for (int i = 1; i < layers; ++i) {
        carry += CarryNs(model, gemm_stages, i - 1, i, miss,
                         candidates_[admissible[choice[i - 1]]].config,
                         candidates_[admissible[choice[i]]].config);
      }
    }
    local.decomposition_error_ns =
        std::max(local.decomposition_error_ns,
                 std::fabs(total - carry - candidate.cost.total_ns));
    best = candidate;
    best_total = total;
  }

  local.solve_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - started)
                       .count();
  if (stats) *stats = local;
  return best;
}

}  // namespace tilemega::solver
