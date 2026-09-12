// SPDX-License-Identifier: BSD-3-Clause
// The inputs of one measured cell, shared by the EX-S1 simulator driver and the
// EX-S2 placement driver.
//
// It is one file because the two must not price a cell two ways: EX-S2's
// ranking gate compares a prediction from `tilemega-simulate` against a
// measurement of a plan `tilemega-place-eft` chose, and if the task DAG or the
// node durations were built by two copies of this code the comparison would be
// between two models rather than between a model and the machine.
//
// Every input here is the runtime's own, not a re-derivation of it:
//
//   * the task DAG is `MaterializeRuntimeTaskGraph` over the `kDependencies`
//     table parsed out of the generated .cu -- literally the array the harness
//     feeds it (ModelHarness.cuh:1434);
//   * the per-stage active task counts and the stage order come from the
//     `E2E_PLACE_BASE` lines a mode 5 run dumps, so they are the counts the
//     kernel actually launched with rather than this tool's idea of them;
//   * worker -> SM comes from a trace dump's `smid`, i.e. where the hardware
//     put the CTA, not where a model would have put it.
//
// Node durations are `CostModel::TaskInstanceNs` at `active_ctas_per_sm = 1`,
// i.e. the solo duration of one task under the residency the kernel compiled
// to.  Concurrency is applied afterwards by the co-residency model rather than
// baked into the price, which is the point: a task that happens to run alone on
// its SM should not be charged for a co-resident that is idle.
#pragma once

#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Dialect/CouplingGraph/PlacementPlan.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/TaskModel.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::experiments {

using namespace tilemega::solver;

inline std::vector<std::string> Split(std::string const& line, char separator) {
  std::istringstream input(line);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(input, field, separator)) result.push_back(field);
  return result;
}

/// `key=value` tokens of one `E2E_*` line, whitespace separated.
inline std::map<std::string, std::string> Fields(std::string const& line) {
  std::map<std::string, std::string> result;
  std::istringstream input(line);
  std::string token;
  while (input >> token) {
    auto equals = token.find('=');
    if (equals != std::string::npos)
      result[token.substr(0, equals)] = token.substr(equals + 1);
  }
  return result;
}

inline long Field(std::map<std::string, std::string> const& fields, std::string const& key) {
  auto found = fields.find(key);
  if (found == fields.end()) throw std::runtime_error("missing " + key + " in dump line");
  return std::stol(found->second);
}

struct CellDump {
  std::vector<int> counts;                 ///< by stage
  std::vector<std::uint32_t> stage_order;  ///< by position
  int grid = 0, ctas_per_sm = 0, num_sms = 0;
};

/// The mode 5 run's own dump: `E2E_PLACE_BASE` covers every stage in order,
/// `E2E_RESOURCE` carries the launch geometry.
inline CellDump ReadCellDump(std::string const& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read dump " + path);
  CellDump cell;
  std::map<int, int> counts;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("E2E_PLACE_BASE", 0) == 0) {
      auto fields = Fields(line);
      int position = int(Field(fields, "pos"));
      int stage = int(Field(fields, "stage"));
      if (int(cell.stage_order.size()) <= position) cell.stage_order.resize(position + 1);
      cell.stage_order[position] = std::uint32_t(stage);
      counts[stage] = int(Field(fields, "active"));
    } else if (line.rfind("E2E_RESOURCE", 0) == 0) {
      auto fields = Fields(line);
      cell.grid = int(Field(fields, "grid"));
      cell.ctas_per_sm = int(Field(fields, "ctas_per_sm"));
      cell.num_sms = int(Field(fields, "num_sms"));
    }
  }
  if (cell.stage_order.empty())
    throw std::runtime_error("no E2E_PLACE_BASE lines in " + path +
                             " (a mode 5 run with TILEMEGA_PLACEMENT_BASE_DUMP=1 is required)");
  if (cell.grid <= 0) throw std::runtime_error("no E2E_RESOURCE line in " + path);
  cell.counts.assign(counts.size(), 0);
  for (auto const& [stage, count] : counts) {
    if (stage < 0 || stage >= int(cell.counts.size()))
      throw std::runtime_error("E2E_PLACE_BASE stages are not dense in " + path);
    cell.counts[stage] = count;
  }
  return cell;
}

struct PlaceStats {
  long max_queue = -1, same_worker_edges = -1, cross_worker_edges = -1;
  bool present = false;
};

inline PlaceStats ReadPlaceStats(std::string const& path) {
  std::ifstream input(path);
  PlaceStats stats;
  if (!input) return stats;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("E2E_PLACE_STATS", 0) != 0) continue;
    auto fields = Fields(line);
    stats.max_queue = Field(fields, "max_queue");
    stats.same_worker_edges = Field(fields, "same_worker_edges");
    stats.cross_worker_edges = Field(fields, "cross_worker_edges");
    stats.present = true;
  }
  return stats;
}

/// The `kDependencies<variant>` array of a generated .cu, in the shape
/// `MaterializeRuntimeTaskGraph` consumes.  Parsed rather than recomputed: this
/// table is the DAG the megakernel enforces, and any reconstruction of it would
/// be the thing under test instead of the input.
inline std::vector<codegen::RuntimeDependencyWindow> ReadDependencies(std::string const& path,
                                                               int variant) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read generated source " + path);
  std::string const opening =
      "constexpr StageDependency kDependencies" + std::to_string(variant) + "[] = {";
  std::string line;
  bool inside = false;
  std::vector<codegen::RuntimeDependencyWindow> windows;
  while (std::getline(input, line)) {
    if (!inside) {
      inside = line.rfind(opening, 0) == 0;
      continue;
    }
    if (line.rfind("};", 0) == 0) return windows;
    std::string cleaned;
    for (char character : line)
      cleaned += (character == '{' || character == '}' || character == 'u') ? ' ' : character;
    auto tokens = Split(cleaned, ',');
    if (tokens.size() < 7) throw std::runtime_error("malformed StageDependency row: " + line);
    auto map = tokens[2];
    bool all = map.find("kAll") != std::string::npos;
    if (!all && map.find("kWindow") == std::string::npos && map.find("kIdentity") == std::string::npos)
      throw std::runtime_error("unknown StageDependency::Map in " + line);
    windows.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]), all, std::stol(tokens[3]),
                       std::stol(tokens[4]), std::stol(tokens[5]), std::stol(tokens[6])});
  }
  throw std::runtime_error("no " + opening + " in " + path);
}

/// Worker -> SM as the hardware actually chose it, read off a trace v2 dump.
/// A worker appears on exactly one SM for the life of a persistent launch, so
/// the first row for a worker settles it; a disagreement is a hard error rather
/// than a majority vote, because it would mean the launch was not persistent.
inline std::vector<int> ReadWorkerSm(std::string const& path, int grid) {
  std::ifstream input(path);
  if (!input) return {};
  std::string header;
  if (!std::getline(input, header)) return {};
  auto columns = Split(header, '\t');
  auto column = [&](char const* name) {
    for (std::size_t i = 0; i < columns.size(); ++i)
      if (columns[i] == name) return int(i);
    throw std::runtime_error(std::string("trace dump has no ") + name + " column: " + path);
  };
  int worker_column = column("worker"), sm_column = column("smid");
  std::vector<int> worker_sm(grid, -1);
  std::string line;
  while (std::getline(input, line)) {
    auto row = Split(line, '\t');
    if (int(row.size()) <= std::max(worker_column, sm_column)) continue;
    int worker = std::stoi(row[worker_column]), sm = std::stoi(row[sm_column]);
    if (worker < 0 || worker >= grid) throw std::runtime_error("trace worker out of grid: " + line);
    if (worker_sm[worker] == -1) worker_sm[worker] = sm;
    else if (worker_sm[worker] != sm)
      throw std::runtime_error("trace shows worker " + std::to_string(worker) +
                               " on two SMs; the launch was not persistent");
  }
  for (int worker = 0; worker < grid; ++worker)
    if (worker_sm[worker] == -1) return {};  // an unobserved worker: fall back to w % sms
  return worker_sm;
}

struct StagePrice {
  double solo_ns = 0.0;
  bool from_instance = false;
  std::string op;
};

/// One task of `stage`, alone on its SM, under the compiled residency.
inline std::vector<StagePrice> PriceStages(CostModel const& cost, ModelDescription& model,
                                    analysis::OperatorGraph const& graph,
                                    std::vector<GemmConfig> const& configs, Residency residency,
                                    int threads, std::size_t stages) {
  std::vector<StagePrice> prices(stages);
  std::map<int, ModelTaskSemantics const*> semantics;
  for (auto const& semantic : model.task_semantics)
    if (semantic.stage >= 0 && semantic.stage < int(stages)) semantics.emplace(semantic.stage, &semantic);
  for (std::size_t stage = 0; stage < stages; ++stage) {
    auto const& description = model.stages.at(stage);
    auto found = semantics.find(int(stage));
    if (found == semantics.end()) {
      // No CG semantic for this runtime stage: fall back to the stage closed
      // form divided by its wave count.  Reported per stage so a cell that
      // leans on the fallback cannot be mistaken for one that does not.
      auto const& config = configs.at(description.gemm >= 0 ? description.gemm : 0);
      prices[stage] = {cost.TaskStageNs(model, int(stage), config, residency), false, "-"};
      continue;
    }
    auto const& semantic = *found->second;
    analysis::ParamBinding point;
    if (description.kind == StageKind::kGemm) {
      auto const& config = configs.at(description.gemm);
      auto input = DeriveModelTaskInput(model, semantic, graph, &config);
      for (auto const& coordinate : input.cost_coordinates) point.Bind(coordinate, 0);
      int chunks = 0;
      cost.GemmStageNs(model.gemms.at(description.gemm), config, residency, model, &chunks);
      auto traits = model.dtype == ScalarType::kBF16
                        ? TensorBF16Traits(config.tile_m, config.tile_n, config.tile_k, config.stages)
                        : SimtF32Traits(config.tile_m, config.tile_n, config.tile_k, config.stages);
      prices[stage] = {cost.TaskInstanceNs(input, traits, residency, model, chunks, point, 1.0),
                       true, semantic.op.name};
    } else {
      auto input = DeriveModelTaskInput(model, semantic, graph, nullptr);
      for (auto const& coordinate : input.cost_coordinates) point.Bind(coordinate, 0);
      BackendTraits traits;
      traits.threads = threads;
      traits.smem_bytes = sizeof(float) * codegen::SimtSharedElements(
          static_cast<codegen::TaskKind>(description.kind), threads, TILEMEGA_ATTENTION_MAX_TOTAL);
      prices[stage] = {cost.TaskInstanceNs(input, traits, residency, model, 1, point, 1.0), true,
                       semantic.op.name};
    }
  }
  return prices;
}

struct Candidate {
  char const* name;
  dialect::PlacementMode mode;
  std::vector<std::int64_t> params;
};

/// §6.2's six.  A caller that cannot materialize one records `unimplemented`
/// for it rather than dropping it, so a candidate the report has to account for
/// cannot go missing: the committed S1 evidence was produced when only the
/// first three could be materialized, and `eft` still needs `PlanRequest::eft`
/// or a table because it is the one mode with no closed form.
inline std::vector<Candidate> Candidates() {
  return {{"legacy_grid_stride", dialect::PlacementMode::kLegacyGridStride, {}},
          {"rotate", dialect::PlacementMode::kRotate, {}},
          {"balanced", dialect::PlacementMode::kBalanced, {}},
          {"eft", dialect::PlacementMode::kEft, {}},
          {"band", dialect::PlacementMode::kTemplate, {long(dialect::PlacementTemplate::kBand)}},
          {"wavefront", dialect::PlacementMode::kTemplate,
           {long(dialect::PlacementTemplate::kWavefront)}}};
}

inline int ModeOfCandidate(Candidate const& candidate) {
  // The TILEMEGA_PLACEMENT value whose dump this candidate must reproduce, or
  // -1 when no measured run of it exists yet.
  switch (candidate.mode) {
    case dialect::PlacementMode::kLegacyGridStride: return 0;
    case dialect::PlacementMode::kBalanced: return 4;
    case dialect::PlacementMode::kRotate: return 5;
    default: return -1;
  }
}

}  // namespace tilemega::experiments
