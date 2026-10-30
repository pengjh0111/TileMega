// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxMoeSelectionV2.h>
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace tilemega::solver {
MxMoeStageGeometryV2 MxMoeGeometryV2(codegen::MxMoeModel const& model,
    std::string const& stage, int batch) {
  int active=codegen::MxMoeExpectedActiveV2(model.experts,batch);
  if (model.batch != unsigned(batch) || model.top_k != 8)
    throw std::invalid_argument("MoE cost domain requires matching batch, E64/128 and top8");
  auto integer = [](std::uint64_t value) {
    if (!value || value > std::uint64_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("MoE cost dimension is zero or overflows int");
    return int(value);
  };
  auto product = [&](int a, int b) { return integer(std::uint64_t(a) * b); };
  int hidden = integer(model.hidden), intermediate = integer(model.intermediate);
  int dim = integer(model.dim), heads = integer(model.heads), hkv = integer(model.hkv);
  int queries = integer(model.queries), q = integer(std::uint64_t(queries) + 2);
  if (stage == "qkv") return {product(product(q, hkv), dim), hidden, product(product(q, hkv), dim), 0, 1, 0};
  if (stage == "o") return {hidden, product(heads, dim), hidden, 1, 0, 0};
  if (stage == "router") return {int(model.experts), hidden, int(model.experts), 0, 1, 0};
  if (stage == "gate_up") {
    int n = product(2, intermediate);
    return {n, hidden, product(active, n), 2, 1, active};
  }
  if (stage == "down") return {hidden, intermediate, product(active, hidden), 0, 0, active};
  if (stage == "head") {
    int vocabulary = integer(model.vocabulary);
    return {vocabulary, hidden, vocabulary, 0, 1, 0};
  }
  throw std::invalid_argument("unknown MoE GEMM stage: " + stage);
}

std::vector<codegen::MxDecodeClassV2> MxMoeGemmDomainV2(codegen::MxMoeModel const& model,
    std::string const& stage, int batch, int grid) {
  using namespace backend::mx;
  if (grid != 104) throw std::invalid_argument("MoE cost domain requires grid104");
  auto shape = MxMoeGeometryV2(model, stage, batch);
  std::vector<codegen::MxDecodeClassV2> result;
  for (int tn : {16, 32, 64, 128}) for (int wn : {1, 2, 4, 8}) for (int d = 1; d <= 4; ++d)
    for (bool stream : {false, true}) {
      if (shape.n_active && stream) continue;
      if (stage == "router" && (wn != 1 || shape.n % tn)) continue;
      if (stage == "head" && (stream || (wn != 4 && wn != 8))) continue;
      if (MxGemmLegalV2(shape.partition_n, shape.k, batch, tn, wn, d,
              static_cast<MxOpV2>(shape.op), stream ? MxPart::kStreamK : MxPart::kTileRange, grid))
        result.push_back({tn, wn, d, stream, stage == "gate_up" ? 0 : -1, -1});
    }
  if (result.empty()) throw std::invalid_argument("MoE stage has an empty legal domain");
  return result;
}

double MxMoeAttentionMeanV2(std::array<double, 5> const& costs) {
  for (double value : costs)
    if (!std::isfinite(value) || value <= 0)
      throw std::invalid_argument("MoE attention needs five finite positive measured grid costs");
  constexpr std::array<int, 5> lengths{65, 256, 512, 768, 1088};
  double total = 0;
  std::size_t point = 0;
  for (int length = 65; length <= 1087; ++length) {
    while (length > lengths[point + 1]) ++point;
    double fraction = double(length - lengths[point]) / (lengths[point + 1] - lengths[point]);
    total += costs[point] + fraction * (costs[point + 1] - costs[point]);
  }
  if (!std::isfinite(total)) throw std::invalid_argument("MoE attention grid average overflows");
  return total / 1023;
}

std::vector<MxMoeProgramChoiceV2> RankMxMoeProgramsV2(frontend::ModelPlan const& plan,
    int threads, std::array<std::vector<MxMoeStageCostV2>, 7> const& rows,
    std::size_t count) {
  if (threads != 512 && threads != 576)
    throw std::invalid_argument("MoE ranking requires threads512 or576");
  auto model = codegen::AnalyzeMxMoe(plan);
  constexpr std::array<char const*, 6> names{"qkv", "o", "router", "gate_up", "down", "head"};
  std::array<std::vector<std::size_t>, 7> order;
  std::array<std::vector<double>, 7> costs;
  for (std::size_t stage = 0; stage < rows.size(); ++stage) {
    auto domain = stage < 6 ? MxMoeGemmDomainV2(model, names[stage], model.batch)
                            : std::vector<codegen::MxDecodeClassV2>{};
    std::set<std::tuple<int, int, int, bool, int, int, int, int>> seen;
    for (std::size_t index = 0; index < rows[stage].size(); ++index) {
      auto const& row = rows[stage][index];
      if (!std::isfinite(row.ns) || row.ns <= 0)
        throw std::invalid_argument("MoE ranking needs finite positive measured costs");
      if (stage < 6) {
        auto const& c = row.gemm;
        if (std::none_of(domain.begin(), domain.end(), [&](auto const& legal) {
              return std::tie(c.tn, c.wn, c.d, c.stream_k) ==
                     std::tie(legal.tn, legal.wn, legal.d, legal.stream_k);
            }) || c.r_next < 1 || (stage == 3 ? c.prefetch != 0 : c.prefetch < c.d))
          throw std::invalid_argument("MoE cost row has invalid GEMM geometry/reference/prefetch");
        if (stage == 2) {
          if ((row.dispatch_gate_tn != 32 && row.dispatch_gate_tn != 64 && row.dispatch_gate_tn != 128) ||
              (row.dispatch_down_tn != 16 && row.dispatch_down_tn != 32 &&
               row.dispatch_down_tn != 64 && row.dispatch_down_tn != 128))
            throw std::invalid_argument("MoE router cost lacks its measured dispatch TN context");
        } else if (row.dispatch_gate_tn || row.dispatch_down_tn)
          throw std::invalid_argument("MoE dispatch context belongs only to router costs");
        if (row.b_min || row.attn_prefetch ||
            !seen.emplace(c.tn, c.wn, c.d, c.stream_k, c.prefetch, 0,
                          row.dispatch_gate_tn, row.dispatch_down_tn).second)
          throw std::invalid_argument("MoE duplicate GEMM cost or misplaced attention coordinates");
      } else {
        if ((row.b_min != 1 && row.b_min != 2 && row.b_min != 4 && row.b_min != 8) ||
            row.attn_prefetch < 0 || row.attn_prefetch > 2 || row.dispatch_gate_tn || row.dispatch_down_tn ||
            !seen.emplace(row.b_min, row.attn_prefetch, 0, false, 0, 0, 0, 0).second)
          throw std::invalid_argument("MoE duplicate/invalid attention cost coordinates");
      }
      order[stage].push_back(index);
    }
    std::stable_sort(order[stage].begin(), order[stage].end(), [&](auto a, auto b) {
      return rows[stage][a].ns < rows[stage][b].ns;
    });
    for (auto index : order[stage]) costs[stage].push_back(rows[stage][index].ns *
        (stage == 5 ? 1 : model.layers.size()));
  }
  auto choice = [&](MxMoeRanksV2 const& ranks) {
    MxMoeProgramChoiceV2 result;
    result.config.threads = threads;
    std::array<codegen::MxDecodeClassV2*, 6> gemms{&result.config.qkv, &result.config.o,
        &result.config.router, &result.config.gate_up, &result.config.down, &result.config.head};
    for (std::size_t stage = 0; stage < rows.size(); ++stage) {
      auto index = order[stage][ranks[stage]];
      result.source_rows[stage] = index;
      result.predicted_ns += costs[stage][ranks[stage]];
      if (stage < 6) *gemms[stage] = rows[stage][index].gemm;
      else {
        result.config.b_min = rows[stage][index].b_min;
        result.config.attn_prefetch = rows[stage][index].attn_prefetch;
      }
    }
    return result;
  };
  // Partition the Cartesian product by dispatch context. Filtering a single
  // seven-dimensional heap could exhaust billions of incompatible tuples.
  std::set<std::pair<int, int>> contexts;
  for (auto const& row : rows[2]) contexts.emplace(row.dispatch_gate_tn, row.dispatch_down_tn);
  std::vector<std::pair<double, MxMoeRanksV2>> merged;
  // Validate empty lists even when no context has compatible grouped rows.
  MxMoeKBestV2(costs, 0);
  for (auto const& [gate_tn, down_tn] : contexts) {
    std::array<std::vector<std::size_t>, 7> context_order;
    std::array<std::vector<double>, 7> context_costs;
    for (std::size_t stage = 0; stage < rows.size(); ++stage)
      for (std::size_t rank = 0; rank < order[stage].size(); ++rank) {
        auto const& row = rows[stage][order[stage][rank]];
        if ((stage == 2 && (row.dispatch_gate_tn != gate_tn || row.dispatch_down_tn != down_tn)) ||
            (stage == 3 && row.gemm.tn != gate_tn) || (stage == 4 && row.gemm.tn != down_tn)) continue;
        context_order[stage].push_back(rank);
        context_costs[stage].push_back(costs[stage][rank]);
      }
    if (std::any_of(context_order.begin(), context_order.end(), [](auto const& list) { return list.empty(); }))
      continue;
    auto global = [&](MxMoeRanksV2 const& local) {
      MxMoeRanksV2 ranks{};
      for (std::size_t stage = 0; stage < ranks.size(); ++stage) ranks[stage] = context_order[stage][local[stage]];
      return ranks;
    };
    auto legal = [&](MxMoeRanksV2 const& local) {
      try { codegen::CheckMxMoeConfigV2(plan, choice(global(local)).config); }
      catch (std::invalid_argument const&) { return false; }
      return true;
    };
    for (auto const& local : MxMoeKBestV2(context_costs, count, legal)) {
      auto ranks = global(local);
      merged.emplace_back(choice(ranks).predicted_ns, ranks);
    }
  }
  std::sort(merged.begin(), merged.end());
  if (merged.size() > count) merged.resize(count);
  std::vector<MxMoeProgramChoiceV2> result;
  for (auto const& [cost, ranks] : merged) {
    (void)cost;
    result.push_back(choice(ranks));
  }
  return result;
}

json::Value MxMoeCandidatesJsonV2(frontend::ModelPlan const& plan, int threads,
    std::vector<json::Value> const& rows, std::vector<json::Value> const& rejections,
    std::size_t diagnostic_count) {
  using json::Array;
  using json::Object;
  using json::Value;
  if (!diagnostic_count || diagnostic_count > 4096)
    throw std::invalid_argument("MoE diagnostic candidate count must be in 1..4096");
  auto model = codegen::AnalyzeMxMoe(plan);
  auto imported = ImportMxMoeCostRowsV2(model, threads, rows);
  auto coverage = CheckMxMoeCostCoverageV2(model, threads, imported, rejections);
  if (!coverage.Complete())
    throw std::invalid_argument("MoE candidate preview requires complete geometric outcomes");
  auto ranked = RankMxMoeProgramsV2(plan, threads, imported.stages, diagnostic_count);
  if (ranked.size() != diagnostic_count)
    throw std::invalid_argument("MoE candidate preview has fewer legal programs than requested");
  constexpr std::array<char const*, 7> names{"qkv", "o", "router", "gate_up", "down", "head", "attention"};
  auto class_json = [](codegen::MxDecodeClassV2 const& c) -> Value {
    return Object{{"tn", c.tn}, {"wn", c.wn}, {"d", c.d},
        {"part", c.stream_k ? "kStreamK" : "kTileRange"},
        {"prefetch", c.prefetch}, {"r_next", c.r_next}};
  };
  Array candidates;
  for (std::size_t i = 0; i < ranked.size(); ++i) {
    auto const& choice = ranked[i];
    auto const& c = choice.config;
    Value pin = Object{{"qkv", class_json(c.qkv)}, {"o", class_json(c.o)},
        {"router", class_json(c.router)}, {"gate_up", class_json(c.gate_up)},
        {"down", class_json(c.down)}, {"head", class_json(c.head)},
        {"threads", threads}, {"b_min", c.b_min}, {"attn_prefetch", c.attn_prefetch},
        {"purpose", "manual"}};
    Object evidence, costs;
    for (std::size_t stage = 0; stage < names.size(); ++stage) {
      auto source = choice.source_rows[stage];
      Array indices;
      for (auto index : imported.source_rows[stage][source]) indices.emplace_back(double(index));
      evidence.emplace_back(names[stage], std::move(indices));
      costs.emplace_back(names[stage], imported.stages[stage][source].ns);
    }
    candidates.emplace_back(Object{{"index", double(i)}, {"pin", Object{{"mx_moe_v2", pin}}},
        {"predicted_tpot_ns", choice.predicted_ns}, {"predicted_stage_ns", costs},
        {"evidence_row_indices", evidence}, {"prompt_pair_required", true}});
  }
  return Object{{"schema_version", 1}, {"kind", "mx5_moe_candidate_preview"},
      {"batch", int(model.batch)}, {"threads", threads}, {"layers", int(model.layers.size())},
      {"stages_per_step", int(model.layers.size() * 6 + 1)},
      {"coverage", Object{{"expected", double(coverage.expected)}, {"measured", double(coverage.measured)},
          {"rejected", double(coverage.rejected)}, {"missing", 0}}},
      {"candidates", candidates}, {"diagnostic", true}, {"file_qualification_implied", false},
      {"execution_authorized", false}, {"solver_selected", false}, {"final", false}, {"default_enabled", false}};
}

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
