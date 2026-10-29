// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxMoeSelectionV2.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace tilemega::solver {
namespace {
int Int(json::Value const& value, char const* name) {
  double n = value.At(name).AsNumber(name);
  if (!std::isfinite(n) || n < 0 || n > std::numeric_limits<int>::max() || std::floor(n) != n)
    throw std::invalid_argument(std::string("MoE cost row invalid integer ") + name);
  return int(n);
}
void Need(bool value, char const* why) {
  if (!value) throw std::invalid_argument(std::string("MoE cost row: ") + why);
}
void Hash(json::Value const& value, char const* name) {
  auto const& s = value.At(name).AsString(name);
  Need(s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos,
      "invalid source or binary hash");
}
}

MxMoeImportedCostsV2 ImportMxMoeCostRowsV2(codegen::MxMoeModel const& model,
    int threads, std::vector<json::Value> const& rows) {
  Need(threads == 512 || threads == 576, "invalid threads");
  MxMoeImportedCostsV2 result;
  using AttentionKey = std::pair<int, int>;
  std::map<AttentionKey, std::map<int, std::size_t>> attention;
  constexpr std::array<char const*, 6> names{"qkv", "o", "router", "gate_up", "down", "head"};
  for (std::size_t index = 0; index < rows.size(); ++index) {
    auto const& row = rows[index];
    auto const& key = row.At("key");
    auto const& kind = row.At("kind").AsString("kind");
    Need(Int(key, "b") == int(model.batch) && Int(key, "threads") == threads,
        "mixed batch or thread requests");
    auto const& resource = row.At("resources");
    Need(Int(resource, "mtreg") > 0 && Int(resource, "mtreg") <= (threads == 576 ? 227 : 255) &&
        !Int(resource, "local_bytes") && !Int(resource, "static_smem_bytes") &&
        Int(resource, "resident_ctas") == 1, "rejected executable resources");
    double median = row.At("median_ns").AsNumber("median_ns");
    double p90 = row.At("p90_ns").AsNumber("p90_ns");
    double spread = row.At("spread").AsNumber("spread");
    Need(std::isfinite(median) && median > 0 && std::isfinite(p90) && p90 >= median &&
        std::isfinite(spread) && spread >= 0, "invalid cost quantiles/spread");
    Need(Int(row, "processes") == 3, "requires three process aggregates");
    std::set<int> rounds;
    for (auto const& round : row.At("rounds").AsArray("rounds"))
      Need(rounds.insert(Int(round, "round")).second, "duplicate aggregate round");
    Need(rounds == std::set<int>{1, 2, 3}, "requires rounds1/2/3");
    auto const& provenance = row.At("provenance");
    Need(provenance.At("executor").AsString("executor") == "mx_static_stage_bench", "wrong executor");
    std::string implementation = kind == "moe_gate_up" || kind == "moe_down_combine" ? "moe_grouped" : kind;
    Need(provenance.At("implementation").AsString("implementation") == "mx5_" + implementation + "_v2",
        "wrong body implementation");
    Hash(provenance, "source_sha256"); Hash(provenance, "binary_sha256");
    Need(!provenance.At("base_head").AsString("base_head").empty(), "missing build HEAD");
    int r_next = Int(row, "r_next");
    Need(r_next > 0, "missing measured R_next");
    if (kind == "attention") {
      Need(Int(key, "dim") == int(model.dim) && Int(key, "q") == int(model.queries) &&
          Int(key, "hkv") == int(model.hkv) && Int(key, "capacity") == int(model.capacity) &&
          Int(key, "qk_norm") == 1, "attention shape differs from model");
      int b_min = Int(key, "b_min"), pf = Int(key, "prefetch_kind");
      Need(b_min == 1 || b_min == 2 || b_min == 4 || b_min == 8, "invalid attention b_min");
      int budget = std::min(64, (threads == 576 ? 220 : 248) - r_next);
      Need(pf == (budget >= 64 ? 2 : budget >= 32 ? 1 : 0), "attention prefetch differs from R_next");
      int length = Int(key, "length");
      Need(length == 65 || length == 256 || length == 512 || length == 768 || length == 1088,
          "attention length is outside the fixed grid");
      Need(attention[{b_min, pf}].emplace(length, index).second, "duplicate attention grid point");
      continue;
    }
    int stage = kind == "moe_router" ? 2 : kind == "moe_gate_up" ? 3 :
        kind == "moe_down_combine" ? 4 : kind == "head" ? 5 : -1;
    if (kind == "gemm") {
      for (int s : {0, 1}) {
        auto shape = MxMoeGeometryV2(model, names[s], model.batch);
        if (Int(key, "n") == shape.n && Int(key, "k") == shape.k) stage = s;
      }
    }
    Need(stage >= 0, "unknown body or GEMM shape");
    auto shape = MxMoeGeometryV2(model, names[stage], model.batch);
    Need(Int(key, "n") == shape.n && Int(key, "k") == shape.k, "GEMM shape differs from model");
    if (stage != 5)
      Need(Int(key, "op") == shape.op && Int(key, "norm") == shape.norm, "wrong op or norm");
    if (shape.n_active) Need(Int(key, "n_active") == shape.n_active, "wrong grouped active-expert shape");
    int part = stage == 5 ? 1 : Int(key, "part");
    Need(part <= 1, "invalid partition");
    MxMoeStageCostV2 cost;
    cost.ns = median;
    cost.gemm = {Int(key, "tn"), Int(key, "wn"), Int(key, "depth"), part == 0,
        Int(key, "prefetch"), r_next};
    auto const& c = cost.gemm;
    auto domain = MxMoeGemmDomainV2(model, names[stage], model.batch);
    Need(std::any_of(domain.begin(), domain.end(), [&](auto const& legal) {
      return std::tie(c.tn, c.wn, c.d, c.stream_k) == std::tie(legal.tn, legal.wn, legal.d, legal.stream_k);
    }), "illegal GEMM geometry");
    Need(stage == 3 ? c.prefetch == 0 : c.prefetch >= c.d, "invalid GEMM prefetch");
    if (stage == 2) {
      auto const& dispatch = row.At("dispatch_context");
      cost.dispatch_gate_tn = Int(dispatch, "gate_up_tn");
      cost.dispatch_down_tn = Int(dispatch, "down_tn");
      Need((cost.dispatch_gate_tn == 32 || cost.dispatch_gate_tn == 64 || cost.dispatch_gate_tn == 128) &&
          (cost.dispatch_down_tn == 16 || cost.dispatch_down_tn == 32 ||
           cost.dispatch_down_tn == 64 || cost.dispatch_down_tn == 128), "invalid dispatch context");
    }
    result.stages[stage].push_back(cost);
    result.source_rows[stage].push_back({index});
  }
  constexpr std::array<int, 5> lengths{65, 256, 512, 768, 1088};
  for (auto const& [key, grid] : attention) {
    Need(grid.size() == 5, "incomplete attention grid");
    std::array<double, 5> costs{};
    std::vector<std::size_t> evidence;
    auto const& first = rows[grid.at(65)];
    for (std::size_t i = 0; i < lengths.size(); ++i) {
      std::size_t index = grid.at(lengths[i]);
      auto const& row = rows[index];
      for (char const* field : {"source_sha256", "binary_sha256", "base_head", "executor", "implementation"})
        Need(row.At("provenance").At(field).AsString(field) == first.At("provenance").At(field).AsString(field),
            "attention grid sources differ");
      for (char const* field : {"mtreg", "local_bytes", "static_smem_bytes", "resident_ctas"})
        Need(Int(row.At("resources"), field) == Int(first.At("resources"), field), "attention grid resources differ");
      Need(Int(row, "r_next") == Int(first, "r_next"), "attention grid R_next differs");
      costs[i] = row.At("median_ns").AsNumber("median_ns");
      evidence.push_back(index);
    }
    MxMoeStageCostV2 cost;
    cost.ns = MxMoeAttentionMeanV2(costs);
    cost.b_min = key.first; cost.attn_prefetch = key.second;
    result.stages[6].push_back(cost);
    result.source_rows[6].push_back(std::move(evidence));
  }
  return result;
}
}  // namespace tilemega::solver
