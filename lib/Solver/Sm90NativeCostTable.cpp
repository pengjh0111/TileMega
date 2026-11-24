// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeCostTable.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace tilemega::solver {
namespace {
template<class... Fields>
std::string Key(Fields... fields) {
  std::ostringstream out;
  ((out << fields << ':'), ...);
  return out.str();
}

double Number(json::Value const& row, char const* name) {
  double value = row.At(name).AsNumber(name);
  if (!std::isfinite(value) || value < 0) throw std::invalid_argument("invalid measured value: " + std::string(name));
  return value;
}

int Integer(json::Value const& row, char const* name) {
  double value = Number(row, name);
  if (value > 1e9 || std::floor(value) != value) throw std::invalid_argument("invalid integer: " + std::string(name));
  return int(value);
}

void DenseB1(Sm90DenseWorkload const& work) {
  if (work.moe || work.tp != 1 || work.batch != 1)
    throw std::invalid_argument("native stage cost table is restricted to dense TP1 B1");
}

Sm90StageCalibration const& Lookup(std::unordered_map<std::string, Sm90StageCalibration> const& entries,
                                    std::string const& key, char const* kind) {
  auto found = entries.find(key);
  if (found == entries.end()) throw std::invalid_argument("missing calibrated " + std::string(kind) + " key " + key);
  return found->second;
}
}  // namespace

Sm90NativeCostTable::Sm90NativeCostTable(std::string const& path) {
  auto data = json::ParseFile(path);
  if (Integer(data, "schema") != 1) throw std::invalid_argument("unknown native stage cost table schema");
  variant_ = data.At("variant").AsString("variant");
  if (variant_ != "overlap" && variant_ != "head-startup" && variant_ != "tail-publication")
    throw std::invalid_argument("unknown native stage cost table variant");
  for (auto const& row : data.At("gemms").AsArray("gemms")) {
    auto key = Key(row.At("family").AsString("family"), Integer(row, "n"), Integer(row, "k"),
        Integer(row, "residency"), Integer(row, "tile_n"), Integer(row, "tile_k"),
        Integer(row, "slots"), Integer(row, "warps_n"), Integer(row, "warps_k"),
        Integer(row, "register_stages"));
    Sm90StageCalibration sample;
    sample.id = row.At("geometry_id").AsString("geometry_id");
    sample.body_ns = Number(row, "stage_body_ns");
    sample.head_ns = Number(row, "head_body_ns");
    sample.epilogue_ns = Number(row, "epilogue_ns");
    if (auto const* steady = row.Find("steady_ns"))
      sample.steady_ns = steady->AsNumber("steady_ns");
    if (!(sample.body_ns > 0) || !(sample.head_ns > 0) ||
        (sample.steady_ns && !(*sample.steady_ns > 0)) ||
        !gemms_.emplace(key, std::move(sample)).second)
      throw std::invalid_argument("invalid or duplicate calibrated GEMM " + key);
  }
  for (auto const& row : data.At("attention").AsArray("attention")) {
    auto key = Key(Integer(row, "hkv"), Integer(row, "head_dim"), Integer(row, "queries"),
        Integer(row, "norm"), Integer(row, "residency"), Integer(row, "chunks"),
        Integer(row, "block_keys"), Integer(row, "slots"), Integer(row, "new_row_work"),
        row.At("merge").AsString("merge"));
    Sm90StageCalibration sample;
    sample.id = row.At("geometry_id").AsString("geometry_id");
    sample.body_ns = Number(row, "compute_ns");
    sample.head_ns = sample.body_ns;
    sample.merge_ns = Number(row, "reducer_ns");
    if (!(sample.body_ns > 0) || !attention_.emplace(key, std::move(sample)).second)
      throw std::invalid_argument("invalid or duplicate calibrated attention " + key);
  }
  if (gemms_.empty() || attention_.empty()) throw std::invalid_argument("empty native cost table");
}

Sm90StageCalibration const& Sm90NativeCostTable::Gemm(Sm90DenseWorkload const& work, unsigned family,
                                                       Sm90DenseGemm const& g, int residency) const {
  DenseB1(work);
  if (family >= 5 || g.multicast || g.mma != Sm90DenseMma::kMmaSync)
    throw std::invalid_argument("calibrated GEMM requires a dense MMA-sync non-multicast stage");
  return Lookup(gemms_, Key(kSm90DenseFamilies[family], work.shapes[family][0], work.shapes[family][1],
      residency, g.tile_n, g.TileK(), g.slots, g.warps_n, g.warps_k, g.register_stages), "GEMM");
}

Sm90StageCalibration const& Sm90NativeCostTable::Attention(Sm90DenseWorkload const& work,
    Sm90DenseAttention const& a, int residency) const {
  DenseB1(work);
  return Lookup(attention_, Key(work.hkv, work.head_dim, work.queries,
      work.qk_norm ? (work.qk_norm_row ? 2 : 1) : 0, residency, a.chunks, a.block_keys,
      a.slots, a.new_row_work, a.dsmem ? "dsmem" : a.merge_stage ? "stage" : "la"), "attention");
}

std::optional<double> Sm90NativeCostTable::Scale(std::string const& id) const {
  auto entry = scales_.find(id);
  return entry == scales_.end() ? std::nullopt : std::optional<double>(entry->second);
}

void Sm90NativeCostTable::SaveScale(std::string const& id, double scale) const {
  scales_.emplace(id, scale);
}
}  // namespace tilemega::solver
