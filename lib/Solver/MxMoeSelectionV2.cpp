// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxMoeSelectionV2.h>
#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace tilemega::solver {
std::vector<MxMoeRanksV2> MxMoeKBestV2(std::array<std::vector<double>, 7> const& costs,
    std::size_t count, std::function<bool(MxMoeRanksV2 const&)> const& legal) {
  for (auto const& list : costs) {
    if (list.empty() || !std::is_sorted(list.begin(), list.end()) ||
        std::any_of(list.begin(), list.end(), [](double value) { return !std::isfinite(value) || value < 0; }))
      throw std::invalid_argument("MoE k-best requires seven nonempty sorted finite nonnegative lists");
  }
  if (!count) return {};
  auto total = [&](MxMoeRanksV2 const& ranks) {
    double sum = 0;
    for (std::size_t i = 0; i < ranks.size(); ++i) sum += costs[i][ranks[i]];
    if (!std::isfinite(sum)) throw std::invalid_argument("MoE k-best cost sum overflows");
    return sum;
  };
  using Node = std::pair<double, MxMoeRanksV2>;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pending;
  std::set<MxMoeRanksV2> seen;
  MxMoeRanksV2 first{};
  pending.emplace(total(first), first);
  seen.insert(first);
  std::vector<MxMoeRanksV2> result;
  while (!pending.empty() && result.size() < count) {
    auto ranks = pending.top().second;
    pending.pop();
    if (!legal || legal(ranks)) result.push_back(ranks);
    if (result.size() == count) break;
    for (std::size_t i = 0; i < ranks.size(); ++i) {
      auto next = ranks;
      if (++next[i] < costs[i].size() && seen.insert(next).second) pending.emplace(total(next), next);
    }
  }
  return result;
}

MxMoeMeasuredSelectionV2 SelectMxMoeMeasuredV2(std::array<double, 4> const& predicted_ns,
    std::array<std::array<double, 2>, 4> const& fresh_process_ns) {
  if (!std::is_sorted(predicted_ns.begin(), predicted_ns.end()))
    throw std::invalid_argument("MoE selection requires prediction order");
  MxMoeMeasuredSelectionV2 result;
  for (std::size_t i = 0; i < 4; ++i) {
    if (!std::isfinite(predicted_ns[i]) || predicted_ns[i] <= 0)
      throw std::invalid_argument("MoE prediction must be finite and positive");
    for (double value : fresh_process_ns[i])
      if (!std::isfinite(value) || value <= 0)
        throw std::invalid_argument("MoE selection needs two finite positive fresh-process samples");
    result.measured_ns[i] = fresh_process_ns[i][0] * .5 + fresh_process_ns[i][1] * .5;
    result.pred_over_meas[i] = predicted_ns[i] / result.measured_ns[i];
    if (!std::isfinite(result.pred_over_meas[i]))
      throw std::invalid_argument("MoE prediction/measurement ratio overflows");
  }
  auto best = std::size_t(std::min_element(result.measured_ns.begin(), result.measured_ns.end()) - result.measured_ns.begin());
  if (result.measured_ns[best] <= result.measured_ns[0] * .99) result.selected = best;
  return result;
}
}  // namespace tilemega::solver
