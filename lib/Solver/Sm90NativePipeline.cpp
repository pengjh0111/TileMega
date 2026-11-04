// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePipeline.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tilemega::solver {
namespace {
std::string CoordinateKey(Sm90DenseCoordinates const& c) {
  return c.ToJson().Dump();
}

// Eq. (10): the replay. The sm90 typed-stage skeleton's contract gives each
// task exactly one legal owner - the template position - so |W_x| = 1 and
// Eq. (9)'s arg min has a single argument. Binding that placement and
// re-evaluating is therefore the replay, and the test asserts the placement
// the EFT loop would choose is the template's.
double Replay(Sm90DenseWorkload const& work, Sm90DenseCoordinates const& c,
              Sm90DenseCostModel const& cost) {
  auto flow = Sm90NativePoolScore(work, c, cost);
  return flow.step_ns;
}
}  // namespace

std::string Sm90NativeStructuralClass(Sm90DenseCoordinates const& c) {
  // Program level, attention structure and the handoff choices: everything
  // except each family's (TN, TK, wn, wk, multicast, slots).
  std::string key;
  key += "res" + std::to_string(c.residency);
  key += "/cl" + std::to_string(c.cluster);
  key += c.group_wait ? "/gw1" : "/gw0";
  key += c.embed_la ? "/la1" : "/la0";
  key += c.expert_fuse ? "/ef1" : "/ef0";
  key += "/ch" + std::to_string(c.attention.chunks);
  key += c.attention.dsmem ? "/ds1" : "/ds0";
  return key;
}

Sm90NativePipelineResult SolveSm90NativePipeline(
    Sm90DenseSearchSpace const& space, Sm90DenseWorkload const& work,
    Sm90DenseCostModel const& cost, Sm90NativePipelineOptions const& options,
    std::vector<Sm90DenseCoordinates> const& seeds) {
  using Clock = std::chrono::steady_clock;
  auto started = Clock::now();
  auto spent_ms = [&] {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  };

  Sm90NativePipelineResult result;
  std::map<std::string, double> memo;  // Algorithm 1 line 11's E

  auto score = [&](Sm90DenseCoordinates const& c) -> double {
    auto key = CoordinateKey(c);
    auto found = memo.find(key);
    if (found != memo.end()) {
      ++result.memo_hits;
      return found->second;
    }
    double value;
    if (options.operator_sum) {
      // AB-2: the same descent and shortlist, scored by the sum of the
      // operators' own times rather than by the program's completion.
      value = cost.Predict(work, c).step_ns;
    } else {
      value = Sm90NativePoolScore(work, c, cost).step_ns;
    }
    memo.emplace(key, value);
    ++result.evaluations;
    Sm90NativePipelineCandidate candidate;
    candidate.coordinates = c;
    candidate.pool_score_ns = value;
    candidate.structural_class = Sm90NativeStructuralClass(c);
    result.evaluated.push_back(std::move(candidate));
    return value;
  };

  // Algorithm 1 line 4: the seeds. (1) the caller's current final coordinates,
  // (2) the additive model's top-1 and (3) its per-family optima, which the
  // existing search already produces.
  std::vector<Sm90DenseCoordinates> starts;
  for (auto const& seed : seeds)
    if (space.Legal(seed)) starts.push_back(seed);
  Sm90DenseSearchOptions additive;
  additive.top = 4;
  auto seeded = SearchSm90Dense(space, work, cost, additive);
  for (auto const& candidate : seeded.ranked)
    if (space.Legal(candidate.coordinates)) starts.push_back(candidate.coordinates);
  if (starts.empty())
    throw std::invalid_argument("native pipeline search found no legal seed");

  // Lines 5-13: coordinate descent. A move is accepted only when it improves
  // the whole-program score, and thederived coordinates are re-derived after each
  // move exactly as the additive search does.
  for (auto const& start : starts) {
    if (spent_ms() > options.budget_ms) break;
    auto current = start;
    double best = score(current);
    result.descent.emplace_back();
    result.descent.back().push_back(best);
    for (bool changed = true; changed && spent_ms() <= options.budget_ms;) {
      changed = false;
      // The per-family geometry.
      for (unsigned family : current.Active()) {
        for (auto const& geometry : space.FamilyCandidates(family, current.residency)) {
          if (spent_ms() > options.budget_ms) break;
          auto trial = current;
          trial.gemms[family] = geometry;
          Sm90DenseSearchSpace::Structure(trial);
          if (!space.Legal(trial)) continue;
          double value = score(trial);
          if (value < best) {
            best = value; current = trial; changed = true;
            result.descent.back().push_back(value);
          }
        }
      }
      // The attention structure, unless AB-4a forbids changing it.
      if (!options.retile_only) {
        for (auto const& attention : space.AttentionCandidates(current.residency)) {
          if (spent_ms() > options.budget_ms) break;
          auto trial = current;
          trial.attention = attention;
          Sm90DenseSearchSpace::Structure(trial);
          if (!space.Legal(trial)) continue;
          double value = score(trial);
          if (value < best) {
            best = value; current = trial; changed = true;
            result.descent.back().push_back(value);
          }
        }
      }
      // The program-level coordinates, which are the structural class.
      for (int residency : {1, 2}) {
        for (bool wait : {false, true}) {
          if (spent_ms() > options.budget_ms) break;
          auto trial = current;
          trial.residency = residency;
          trial.group_wait = options.realize_aggregate ? false : wait;
          Sm90DenseSearchSpace::Structure(trial);
          if (!space.Legal(trial)) continue;
          double value = score(trial);
          if (value < best) {
            best = value; current = trial; changed = true;
            result.descent.back().push_back(value);
          }
        }
      }
    }
  }

  // Line 14: the shortlist. The best `shortlist` by score, then the best of
  // each structural class not yet represented, up to `structural_extra`.
  auto ranked = result.evaluated;
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](auto const& a, auto const& b) {return a.pool_score_ns < b.pool_score_ns;});
  std::set<std::string> classes;
  for (auto const& candidate : ranked) {
    if (int(result.shortlist.size()) >= options.shortlist) break;
    result.shortlist.push_back(candidate);
    classes.insert(candidate.structural_class);
  }
  int extra = 0;
  for (auto const& candidate : ranked) {
    if (extra >= options.structural_extra) break;
    if (classes.insert(candidate.structural_class).second) {
      result.shortlist.push_back(candidate);
      ++extra;
    }
  }

  // Lines 15-19: materialize, replay, and keep at most `per_class` finalists
  // from any one structural class so the measured set stays varied.
  for (auto& candidate : result.shortlist)
    candidate.replay_ns = Replay(work, candidate.coordinates, cost);
  auto by_replay = result.shortlist;
  std::stable_sort(by_replay.begin(), by_replay.end(),
                   [](auto const& a, auto const& b) {return a.replay_ns < b.replay_ns;});
  std::map<std::string, int> taken;
  for (auto const& candidate : by_replay) {
    if (int(result.finalists.size()) >= options.finalists) break;
    if (taken[candidate.structural_class] >= options.per_class) continue;
    ++taken[candidate.structural_class];
    result.finalists.push_back(candidate);
  }
  // Short of the quota, fill by replay time regardless of class.
  for (auto const& candidate : by_replay) {
    if (int(result.finalists.size()) >= options.finalists) break;
    bool present = std::any_of(result.finalists.begin(), result.finalists.end(),
        [&](auto const& f) {return CoordinateKey(f.coordinates) ==
                                   CoordinateKey(candidate.coordinates);});
    if (!present) result.finalists.push_back(candidate);
  }
  result.seconds = spent_ms() / 1000.0;
  return result;
}
}  // namespace tilemega::solver
