// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxMoeSelectionV2.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
using namespace tilemega::solver;

template<class F> void Reject(F f) {
  bool rejected = false;
  try { f(); } catch (std::invalid_argument const&) { rejected = true; }
  assert(rejected);
}
int main() {
  std::mt19937 random(2030);
  for (int trial = 0; trial < 120; ++trial) {
    std::array<std::vector<double>, 7> costs;
    for (auto& list : costs) {
      for (unsigned i = 0, n = 1 + random() % 3; i < n; ++i) list.push_back(random() % 8);
      std::sort(list.begin(), list.end());
    }
    auto legal = [](MxMoeRanksV2 const& r) { return (r[0] + r[2] + r[4]) % 3 != 1; };
    std::vector<std::pair<double, MxMoeRanksV2>> all;
    MxMoeRanksV2 ranks{};
    auto enumerate = [&](auto&& self, std::size_t dimension, double sum) -> void {
      if (dimension == 7) { all.emplace_back(sum, ranks); return; }
      for (std::size_t i = 0; i < costs[dimension].size(); ++i) {
        ranks[dimension] = i;
        self(self, dimension + 1, sum + costs[dimension][i]);
      }
    };
    enumerate(enumerate, 0, 0);
    std::sort(all.begin(), all.end());
    for (bool filtered : {false, true}) {
      for (std::size_t count : {0, 4, 17}) {
        std::vector<MxMoeRanksV2> expected;
        for (auto const& [cost, r] : all) {
          (void)cost;
          if ((!filtered || legal(r)) && expected.size() < count) expected.push_back(r);
        }
        auto actual = filtered ? MxMoeKBestV2(costs, count, legal) : MxMoeKBestV2(costs, count);
        assert(actual == expected);
      }
    }
  }
  std::array<std::vector<double>, 7> costs;
  for (auto& list : costs) list = {0, 1};
  for (auto invalid : {std::vector<double>{}, {2, 1}, {-1, 2}, {std::numeric_limits<double>::infinity()}}) {
    auto bad = costs; bad[3] = invalid;
    Reject([&] { MxMoeKBestV2(bad); });
  }
  auto overflow = costs;
  for (auto& list : overflow) list = {std::numeric_limits<double>::max()};
  Reject([&] { MxMoeKBestV2(overflow); });
  auto distant_overflow = costs;
  for (auto& list : distant_overflow) list = {0, std::numeric_limits<double>::max()};
  assert(MxMoeKBestV2(distant_overflow, 1).front() == MxMoeRanksV2{});
  std::array<double, 4> predicted{100, 101, 102, 103};
  std::array<std::array<double, 2>, 4> samples{{{99, 101}, {98.2, 100}, {110, 110}, {120, 120}}};
  assert(SelectMxMoeMeasuredV2(predicted, samples).selected == 0);
  samples[1] = {98, 100};
  auto result = SelectMxMoeMeasuredV2(predicted, samples);
  assert(result.selected == 1 && result.measured_ns[1] == 99 && result.pred_over_meas[1] == 101.0 / 99);
  samples[2] = {99, 99};
  assert(SelectMxMoeMeasuredV2(predicted, samples).selected == 1);
  for (double invalid : {0., -1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    auto bad = samples; bad[2][1] = invalid;
    Reject([&] { SelectMxMoeMeasuredV2(predicted, bad); });
  }
  auto unordered = predicted; std::swap(unordered[0], unordered[1]);
  Reject([&] { SelectMxMoeMeasuredV2(unordered, samples); });
  std::cout << "PASS MoE seven-class exact k-best: 720 brute-force comparisons; median/1% boundary\n";
}
