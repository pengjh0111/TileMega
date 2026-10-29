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
  bool import = argc == 5 && std::string(argv[1]) == "--import-rows";
  if (argc != 1 && !dump && !import) return 2;
  tilemega::codegen::MxMoeModel model{};
  model.hidden = 2048; model.intermediate = 768; model.heads = 32;
  model.hkv = 4; model.dim = 128; model.queries = 8;
  model.experts = 128; model.top_k = 8; model.vocabulary = 151936;
  model.capacity = 1088;
  if (import) {
    model.batch = std::stoi(argv[3]);
    auto doc = tilemega::json::ParseFile(argv[2]);
    auto imported = ImportMxMoeCostRowsV2(model, std::stoi(argv[4]), doc.At("rows").AsArray("rows"));
    auto coverage = CheckMxMoeCostCoverageV2(model, std::stoi(argv[4]), imported,
        doc.Find("rejections") ? doc.At("rejections").AsArray("rejections") : tilemega::json::Array{});
    std::cout << "PASS imported aggregate rows; stage sizes";
    for (auto const& stage : imported.stages) std::cout << ' ' << stage.size();
    std::cout << "; geometry measured=" << coverage.measured << " rejected=" << coverage.rejected <<
        " missing=" << coverage.missing.size() << "; import does not qualify a full domain or authorize execution\n";
    return 0;
  }
  model.batch = 1;
  {
    using tilemega::json::Array;
    using tilemega::json::Object;
    using tilemega::json::Value;
    // Synthetic aggregate contracts test parsing, not measurement provenance.
    auto row = [](char const* kind, Value key, double ns, int r_next = 120) -> Value {
      std::string implementation = kind;
      if (implementation == "moe_gate_up" || implementation == "moe_down_combine") implementation = "moe_grouped";
      return Object{{"kind", kind}, {"key", key}, {"median_ns", ns}, {"p90_ns", ns + 1}, {"spread", .01},
          {"processes", 3}, {"r_next", r_next}, {"rounds", Array{Object{{"round", 1}}, Object{{"round", 2}}, Object{{"round", 3}}}},
          {"resources", Object{{"mtreg", 160}, {"local_bytes", 0}, {"static_smem_bytes", 0}, {"resident_ctas", 1}}},
          {"provenance", Object{{"executor", "mx_static_stage_bench"}, {"implementation", "mx5_" + implementation + "_v2"},
              {"source_sha256", std::string(64, 'a')}, {"binary_sha256", std::string(64, 'b')}, {"base_head", "synthetic_contract_test"}}}};
    };
    std::vector<Value> rows;
    std::array<char const*, 6> names{"qkv", "o", "router", "gate_up", "down", "head"};
    std::array<char const*, 6> kinds{"gemm", "gemm", "moe_router", "moe_gate_up", "moe_down_combine", "head"};
    for (unsigned stage = 0; stage < 6; ++stage) {
      auto shape = MxMoeGeometryV2(model, names[stage], 1);
      Value key = Object{{"n", shape.n}, {"k", shape.k}, {"b", 1}, {"threads", 512},
          {"tn", 64}, {"wn", stage == 5 ? 8 : 1}, {"depth", 1}, {"prefetch", stage == 3 ? 0 : 1}};
      if (stage != 5) { key.Set("op", shape.op); key.Set("norm", shape.norm); key.Set("part", 1); }
      if (shape.n_active) key.Set("n_active", shape.n_active);
      auto value = row(kinds[stage], key, 100 + stage);
      if (stage == 2) value.Set("dispatch_context", Object{{"gate_up_tn", 64}, {"down_tn", 64}});
      rows.push_back(value);
    }
    for (int length : {65, 256, 512, 768, 1088}) rows.push_back(row("attention",
        Object{{"dim", 128}, {"q", 8}, {"hkv", 4}, {"b", 1}, {"length", length},
            {"b_min", 4}, {"qk_norm", 1}, {"capacity", 1088}, {"threads", 512}, {"prefetch_kind", 0}}, 201, 240));
    auto costs = ImportMxMoeCostRowsV2(model, 512, rows);
    for (unsigned stage = 0; stage < 7; ++stage) {
      assert(costs.stages[stage].size() == 1);
      assert(costs.stages[stage][0].ns == (stage == 6 ? 201 : 100 + stage));
      assert(costs.source_rows[stage].size() == 1);
      if (stage < 6) assert(costs.source_rows[stage][0] == std::vector<std::size_t>{stage});
    }
    assert(costs.source_rows[6][0] == (std::vector<std::size_t>{6, 7, 8, 9, 10}));
    assert(costs.stages[2][0].dispatch_gate_tn == 64 && costs.stages[6][0].b_min == 4);
    auto partial = CheckMxMoeCostCoverageV2(model, 512, costs, {});
    assert(!partial.Complete() && partial.measured == 11 && !partial.missing.empty());
    MxMoeImportedCostsV2 filled;
    for (unsigned stage = 0; stage < 6; ++stage)
      for (auto const& c : MxMoeGemmDomainV2(model, names[stage], 1)) {
        MxMoeStageCostV2 value; value.gemm = c;
        if (stage == 2) {
          for (int gu : {32, 64, 128}) for (int dn : {16, 32, 64, 128}) {
            value.dispatch_gate_tn = gu; value.dispatch_down_tn = dn;
            filled.stages[stage].push_back(value);
          }
        } else filled.stages[stage].push_back(value);
      }
    for (int b_min : {1, 2, 4, 8}) {
      MxMoeStageCostV2 value; value.b_min = b_min;
      filled.stages[6].push_back(value);
    }
    auto complete = CheckMxMoeCostCoverageV2(model, 512, filled, {});
    assert(complete.Complete() && complete.rejected == 0 && complete.expected == partial.expected);
    auto measured = filled;
    auto& gu = measured.stages[3];
    auto remove = std::find_if(gu.begin(), gu.end(), [](auto const& value) {
      auto const& c = value.gemm; return c.tn == 64 && c.wn == 1 && c.d == 1;
    });
    assert(remove != gu.end()); gu.erase(remove);
    Value rejection = Object{{"kind", "moe_gate_up"}, {"key", rows[3].At("key")},
        {"resources", Object{{"mtreg", 256}, {"local_bytes", 16}, {"static_smem_bytes", 0}}}};
    auto mixed = CheckMxMoeCostCoverageV2(model, 512, measured, {rejection});
    assert(mixed.Complete() && mixed.rejected == 1 && mixed.measured == complete.measured - 1);
    Reject([&] { CheckMxMoeCostCoverageV2(model, 512, filled, {rejection}); });
    for (int mutation = 0; mutation < 4; ++mutation) {
      auto bad = rejection;
      if (mutation == 0) bad.Set("median_ns", 123);
      if (mutation == 1) bad.Set("resources", Object{{"mtreg", 120}, {"local_bytes", 0}, {"static_smem_bytes", 0}});
      if (mutation == 2) { auto k = bad.At("key"); k.Set("n_active", 9); bad.Set("key", k); }
      if (mutation == 3) { auto k = bad.At("key"); k.Set("threads", 576); bad.Set("key", k); }
      Reject([&] { CheckMxMoeCostCoverageV2(model, 512, measured, {bad}); });
    }
    for (int mutation = 0; mutation < 15; ++mutation) {
      auto bad = rows;
      auto key = bad[0].At("key");
      if (mutation == 0) { key.Set("b", 16); bad[0].Set("key", key); }
      if (mutation == 1) bad[0].Set("processes", 2);
      if (mutation == 2) bad[0].Set("rounds", Array{Object{{"round", 1}}, Object{{"round", 1}}, Object{{"round", 3}}});
      if (mutation == 3) { auto v = bad[0].At("resources"); v.Set("local_bytes", 4); bad[0].Set("resources", v); }
      if (mutation == 4) bad.pop_back();
      if (mutation == 5) bad.push_back(bad.back());
      if (mutation == 6) { auto v = bad[6].At("key"); v.Set("q", 4); bad[6].Set("key", v); }
      if (mutation == 7) { auto v = bad[6].At("key"); v.Set("prefetch_kind", 1); bad[6].Set("key", v); }
      if (mutation == 8) { auto v = bad[0].At("provenance"); v.Set("implementation", "legacy"); bad[0].Set("provenance", v); }
      if (mutation == 9) { auto v = bad[10].At("provenance"); v.Set("binary_sha256", std::string(64, 'c')); bad[10].Set("provenance", v); }
      if (mutation == 10) bad[1].Set("median_ns", -1);
      if (mutation == 11) { auto v = bad[3].At("key"); v.Set("n_active", 9); bad[3].Set("key", v); }
      if (mutation == 12) bad[0].Set("r_next", 0);
      if (mutation == 13) { key.Set("part", 2); bad[0].Set("key", key); }
      if (mutation == 14) bad[2].Set("dispatch_context", Object{{"gate_up_tn", 16}, {"down_tn", 64}});
      Reject([&] { ImportMxMoeCostRowsV2(model, 512, bad); });
    }
  }
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
