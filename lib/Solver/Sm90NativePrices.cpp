// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePrices.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace tilemega::solver {
namespace {
// Sm90DenseCostModel.cpp:48 charges the ring one L_dram per ring refill, which
// is a rate ceiling of one ring capacity per L_dram; :49's waves term is the
// same ceiling expressed per task. Both are caps on how fast a CTA can pull
// its own bytes, which is what dram_rate_cap means.
double RateCap(Sm90NativeConstants const& c, double slot_capacity_bytes) {
  double ring = slot_capacity_bytes / c.dram_latency;  // bytes/ns
  return std::min(c.CtaDramGbps(), ring);
}
}  // namespace

Sm90NativeStagePrice Sm90NativeGemmPrice(Sm90DenseWorkload const& work, unsigned family,
                                         Sm90DenseGemm const& g, int residency,
                                         Sm90DenseCostModel const& cost, bool weight_stream,
                                         bool pre) {
  auto const& c = cost.Constants();
  int grid = cost.Grid(residency);
  int b = work.batch;
  // Shapes exactly as Sm90DenseCostModel.cpp:34-40 derives them, including
  // MR6's n_active = E (1 - (1 - k/E)^B) and the per-family row count.
  double n = work.shapes[family][0], k = work.shapes[family][1];
  double active = family >= 6
      ? work.experts * (1 - std::pow(1 - double(work.top_k) / work.experts, b)) : 1.0;
  double rows = family == 6 ? b * double(work.top_k) / active
                            : family == 7 ? (b <= 8 ? 8 : 16) : b;
  double tasks = n / g.tile_n * active;

  Sm90NativeStagePrice price;
  price.name = "gemm" + std::to_string(family);
  price.category = "gemm";
  price.tasks = int(std::llround(tasks));
  price.first_tasks = std::min(price.tasks, grid);
  price.hop_ns = c.Barrier(grid);

  // :47, per task: this task's own weight and activation bytes through the
  // CTA's async stream. The stage term divides the per-CTA sum by BW_sm, so a
  // single task's share is its own tile.
  double weights = g.tile_n * k * 2.0;
  double activation = rows * k * 2.0;
  if (g.multicast) activation /= 2.0;  // :45, L2 serves each A slot once per pair

  TaskPriceParts parts;
  parts.compute_ns = (weights + activation) / c.sm_gbps;
  parts.dram_bytes = weights;  // the stage's w_tot is tasks x this (:41)
  // :50 feeds w_tot through BW_dram only when the weights are not already in
  // flight from the previous stage; weight_stream is exactly that case (D6).
  parts.no_producer_dram_bytes = weight_stream ? weights : 0.0;
  parts.dram_rate_cap = RateCap(c, double(g.slots) * g.slot_bytes);

  // :58 epilogue, :54 the MoE per-task fixed cost, :61 the TP all-reduce.
  static constexpr char const* kEpilogue[] = {"qkv", "residual", "gate_up", "residual",
                                              "lm_head", "router", "expert_gate_up",
                                              "expert_down"};
  std::string geometry = family >= 5 && work.experts == 64 ? "_e64" : "";
  parts.fixed_ns = c.Epilogue(kEpilogue[family] + geometry, g.tile_n, b, g.warps_k);
  if (family >= 6)
    parts.fixed_ns += c.TaskExpert((family == 6 ? "expert_gate_up" : "expert_down") + geometry, b);
  if (work.tp > 1 && (family == 1 || family == 3))
    parts.fixed_ns += c.AllReduce(b, g.tile_n, work.tp);

  price.rest = parts;
  price.first = parts;
  // :59 the stage pays its first bytes once; a predecessor that streamed them
  // leaves only the L2 hit.
  price.first.fixed_ns += pre ? c.l2_latency : c.dram_latency;
  return price;
}

Sm90NativeStagePrice Sm90NativeAttentionPrice(Sm90DenseWorkload const& work,
                                              Sm90DenseAttention const& a, int residency,
                                              Sm90DenseCostModel const& cost) {
  auto const& c = cost.Constants();
  int grid = cost.Grid(residency), d = work.head_dim, q = work.queries;
  int norm = work.qk_norm ? (work.qk_norm_row ? 2 : 1) : 0;
  // :80-83, the same past and chunking the stage model uses.
  double past = 575, tasks = double(work.batch) * work.hkv * a.chunks;
  double keys = std::ceil((past + 1) / a.chunks), blocks = std::ceil(keys / 64);

  Sm90NativeStagePrice price;
  price.name = "attention";
  price.category = "attention";
  price.tasks = int(std::llround(tasks));
  price.first_tasks = std::min(price.tasks, grid);
  price.hop_ns = c.Barrier(grid);

  TaskPriceParts parts;
  // :84's per-CTA block term, per task. Appendix B puts the q preparation in
  // the task's compute rather than in its fixed cost, because every task
  // prepares its own queries.
  parts.compute_ns = c.AttentionPrepare(d, q, norm) + blocks * c.AttentionBlock(d, q, norm);
  parts.dram_bytes = keys * d * 4.0;              // :83 kv_task
  parts.no_producer_dram_bytes = keys * d * 4.0;  // history has no producer in this CTG
  // Attention streams the history in block_keys-sized slots (D10a), so its
  // ring capacity is slots x one block of keys for this head.
  parts.dram_rate_cap = RateCap(c, double(a.slots) * a.block_keys * d * 4.0);
  // :88 prepare is the stage's own entry cost; the merge is the handoff, so it
  // belongs to the reducer space rather than to these tasks.
  parts.fixed_ns = 0;
  price.rest = parts;
  price.first = parts;
  // :88 charges the stage's first bytes once. The stage model uses L_l2 here,
  // not L_dram: the KV history of the previous step is still in L2.
  price.first.fixed_ns += c.l2_latency;
  return price;
}

FlowProblem Sm90NativeSingleStageProblem(Sm90NativeStagePrice const& price,
                                         Sm90DenseCostModel const& cost, int residency) {
  auto const& c = cost.Constants();
  FlowProblem problem;
  problem.workers = cost.Grid(residency);
  problem.dram_gbps = c.dram_gbps;
  problem.hop_ns = price.hop_ns;
  problem.page_bytes = 0;
  problem.pages_per_worker = 0;

  // The evaluator needs a producer for the boundary, so the stage ahead of
  // this one is represented by a single zero-cost task: all_producer then
  // means "the whole previous stage finished", which is the V3 barrier, and
  // problem.hop_ns carries its constant.
  FlowSpace entry;
  entry.name = "entry";
  entry.category = "entry";
  entry.count = 1;
  entry.order = 0;
  entry.pieces.push_back(FlowPiece{1, TaskPriceParts{}});
  entry.piece_of_task.assign(1, 0);
  problem.spaces.push_back(entry);

  FlowSpace space;
  space.name = price.name;
  space.category = price.category;
  space.count = price.tasks;
  space.order = 1;
  // The first task a CTA runs carries the first-byte latency; the rest do not.
  // Tasks are handed to workers in index order, so the first `first_tasks`
  // indices are exactly one per busy CTA.
  space.pieces.push_back(FlowPiece{price.first_tasks, price.first});
  if (price.tasks > price.first_tasks)
    space.pieces.push_back(FlowPiece{price.tasks - price.first_tasks, price.rest});
  space.piece_of_task.assign(price.tasks, 0);
  for (int task = price.first_tasks; task < price.tasks; ++task) space.piece_of_task[task] = 1;
  problem.spaces.push_back(space);

  // The stage's in-edge is that boundary: every producer must finish, which
  // is what an all-producer release means (appendix A).
  FlowRelease edge;
  edge.producer = 0;
  edge.consumer = 1;
  edge.all_producer = true;
  edge.kappa = 1;
  // Each consumer waits for the whole entry space, whose single task is
  // producer position 0: the release table every consumer shares.
  std::vector<std::pair<int, int>> release;
  release.reserve(price.tasks);
  for (int task = 0; task < price.tasks; ++task) release.emplace_back(0, task);
  edge.sorted = std::make_shared<std::vector<std::pair<int, int>> const>(std::move(release));
  problem.edges.push_back(edge);
  return problem;
}
}  // namespace tilemega::solver
