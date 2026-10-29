// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MxMoeSelectionV2.h>
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace tilemega::solver {
MxMoeStageGeometryV2 MxMoeGeometryV2(codegen::MxMoeModel const& model,
    std::string const& stage, int batch) {
  int active;
  switch (batch) {
    case 1: active = 8; break;
    case 2: active = 16; break;
    case 4: active = 29; break;
    case 8: active = 52; break;
    case 16: active = 82; break;
    default: throw std::invalid_argument("MoE cost domain supports batches 1,2,4,8,16");
  }
  if (model.batch != unsigned(batch) || model.experts != 128 || model.top_k != 8)
    throw std::invalid_argument("MoE cost domain requires matching batch, 128 experts and top8");
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
  if (stage == "router") return {128, hidden, 128, 0, 1, 0};
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
      if (stage == "router" && wn != 1) continue;
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
