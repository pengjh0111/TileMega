// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePrices.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace tilemega;
using namespace tilemega::solver;

namespace {
int Int(json::Value const& value, char const* name) {
  return int(value.At(name).AsNumber(name));
}

bool Compare(Sm90NativeStagePrice const& price, Sm90DenseCostModel const& cost, int residency,
             json::Value const& row, double measured, double reducer, double additive) {
  auto problem = Sm90NativeSingleStageProblem(price, cost, residency);
  problem.hop_ns = 0;
  double simulated = EvaluateFlow(problem).makespan_ns + reducer;
  double error = simulated / measured - 1;
  if (price.head.dram_bytes <= 0 || price.middle.dram_bytes <= 0)
    throw std::runtime_error("calibration dropped real traffic");
  std::cout << row.At("geometry_id").AsString("geometry_id") << ','
            << measured << ',' << simulated << ',' << error << ',' << additive << '\n';
  return std::abs(error) <= 0.03 && std::abs(additive / measured - 1) < 1e-8;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::invalid_argument("usage: tilemega-sq2-check-cost target source-root cost-table");
    auto target = TargetSpec::FromJson(argv[1]);
    auto constants = Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]), argv[2]);
    if (!constants.StaleHeaders(argv[2]).empty()) throw std::runtime_error("stale calibration headers");
    Sm90DenseCostModel cost(target, constants, argv[3]);
    auto table = json::ParseFile(argv[3]);
    bool pass = true;
    std::cout << "geometry_id,measured_ns,simulated_ns,relative_error,additive_ns\n";
    for (auto const& row : table.At("gemms").AsArray("gemms")) {
      unsigned family = 0;
      while (family < 5 && kSm90DenseFamilies[family] != row.At("family").AsString("family")) ++family;
      if (family == 5) throw std::invalid_argument("unknown GEMM family");
      Sm90DenseWorkload work{};
      work.batch = work.tp = 1;
      work.shapes[family] = {Int(row, "n"), Int(row, "k")};
      Sm90DenseGemm g{};
      g.tile_n = Int(row, "tile_n");
      g.slot_bytes = 2 * g.tile_n * Int(row, "tile_k");
      g.slots = Int(row, "slots");
      g.warps_n = Int(row, "warps_n");
      g.warps_k = Int(row, "warps_k");
      g.register_stages = Int(row, "register_stages");
      int res = Int(row, "residency");
      double additive = cost.Gemm(work, family, g, res) - constants.Barrier(cost.Grid(res));
      auto price = Sm90NativeGemmPrice(work, family, g, res, cost, false, false);
      pass &= Compare(price, cost, res, row, row.At("stage_body_ns").AsNumber("stage_body_ns"), 0, additive);
    }
    for (auto const& row : table.At("attention").AsArray("attention")) {
      Sm90DenseWorkload work{};
      work.batch = work.tp = 1;
      work.hkv = Int(row, "hkv");
      work.head_dim = Int(row, "head_dim");
      work.queries = Int(row, "queries");
      work.qk_norm = Int(row, "norm") != 0;
      work.qk_norm_row = Int(row, "norm") == 2;
      Sm90DenseAttention a{};
      a.chunks = Int(row, "chunks");
      a.block_keys = Int(row, "block_keys");
      a.slots = Int(row, "slots");
      a.new_row_work = Int(row, "new_row_work");
      auto merge = row.At("merge").AsString("merge");
      a.dsmem = merge == "dsmem";
      a.merge_stage = merge == "stage";
      int res = Int(row, "residency");
      double additive = cost.Attention(work, a, res) - constants.Barrier(cost.Grid(res)) * (a.merge_stage ? 2 : 1);
      auto price = Sm90NativeAttentionPrice(work, a, res, cost);
      pass &= Compare(price, cost, res, row, row.At("combined_body_ns").AsNumber("combined_body_ns"),
                      row.At("reducer_ns").AsNumber("reducer_ns"), additive);
    }
    return pass ? 0 : 1;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
