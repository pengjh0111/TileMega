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
int main(int argc, char** argv) {
  bool dump = argc == 2 && std::string(argv[1]) == "--dump-domain";
  if (argc != 1 && !dump) return 2;
  tilemega::codegen::MxMoeModel model{};
  model.hidden = 2048; model.intermediate = 768; model.heads = 32;
  model.hkv = 4; model.dim = 128; model.queries = 8;
  model.experts = 128; model.top_k = 8; model.vocabulary = 151936;
  std::size_t regular = 0, grouped = 0, routers = 0, heads = 0;
  for (int batch : {1, 2, 4, 8, 16}) {
    model.batch = batch;
    for (std::string stage : {"qkv", "o", "router", "gate_up", "down", "head"}) {
      auto shape = MxMoeGeometryV2(model, stage, batch);
      auto domain = MxMoeGemmDomainV2(model, stage, batch);
      if (dump) for (int threads : {512, 576}) for (auto const& c : domain)
        std::cout << stage << ',' << threads << ',' << batch << ',' << shape.n << ',' << shape.k << ','
                  << shape.n_active << ',' << c.tn << ',' << c.wn << ',' << c.d << ',' << int(!c.stream_k)
                  << ',' << shape.op << ',' << shape.norm << '\n';
      if (stage == "qkv") assert(shape.n == 5120 && shape.k == 2048 && shape.norm == 1);
      if (stage == "o") assert(shape.n == 2048 && shape.k == 4096 && shape.op == 1 && !shape.norm);
      if (stage == "gate_up") assert(shape.n == 1536 && shape.k == 2048 && shape.op == 2 && shape.norm);
      if (stage == "down") assert(shape.n == 2048 && shape.k == 768 && shape.op == 0 && !shape.norm);
      if (shape.n_active) {
        grouped += domain.size();
        assert(shape.partition_n == shape.n * shape.n_active);
        int expected = batch == 1 ? 8 : batch == 2 ? 16 : batch == 4 ? 29 : batch == 8 ? 52 : 82;
        assert(shape.n_active == expected);
        for (auto const& c : domain) {
          assert(!c.stream_k);
          if (stage == "gate_up") assert(c.tn >= 32 && c.prefetch == 0);
        }
      } else if (stage == "router") {
        routers += domain.size();
        for (auto const& c : domain) assert(c.wn == 1 && !(c.tn == 128 && c.stream_k));
      } else if (stage == "head") {
        heads += domain.size();
        for (auto const& c : domain) assert(!c.stream_k && (c.wn == 4 || c.wn == 8));
      } else regular += domain.size();
    }
  }
  // Independent full CPU inventories enumerate both thread shapes.
  assert(2 * regular == 760 && 2 * grouped == 896 && 2 * routers == 270 && 2 * heads == 300);
  model.batch = 1;
  Reject([&] { MxMoeGemmDomainV2(model, "qkv", 3); });
  Reject([&] { MxMoeGemmDomainV2(model, "qkv", 16); });
  Reject([&] { MxMoeGemmDomainV2(model, "qkv", 1, 52); });
  Reject([&] { MxMoeGemmDomainV2(model, "attention", 1); });
  auto invalid_model = model; invalid_model.experts = 64;
  Reject([&] { MxMoeGemmDomainV2(invalid_model, "router", 1); });
  invalid_model = model; invalid_model.intermediate = std::numeric_limits<unsigned>::max();
  Reject([&] { MxMoeGemmDomainV2(invalid_model, "gate_up", 1); });
  invalid_model = model; invalid_model.queries = std::numeric_limits<unsigned>::max();
  Reject([&] { MxMoeGemmDomainV2(invalid_model, "qkv", 1); });
  assert(MxMoeAttentionMeanV2({100, 100, 100, 100, 100}) == 100);
  assert(std::abs(MxMoeAttentionMeanV2({137, 519, 1031, 1543, 2183}) - 1159) < 1e-9);
  std::array<double, 5> nonlinear{113, 271, 173, 419, 227};
  double expected_average = (96 * nonlinear[0] + 223.5 * nonlinear[1] + 256 * nonlinear[2] +
      288 * nonlinear[3] + 159.5 * nonlinear[4]) / 1023;
  assert(std::abs(MxMoeAttentionMeanV2(nonlinear) - expected_average) < 1e-9);
  for (double invalid : {0., -1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    auto costs = nonlinear; costs[2] = invalid;
    Reject([&] { MxMoeAttentionMeanV2(costs); });
  }
  auto huge_grid = nonlinear; huge_grid.fill(std::numeric_limits<double>::max());
  Reject([&] { MxMoeAttentionMeanV2(huge_grid); });
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
