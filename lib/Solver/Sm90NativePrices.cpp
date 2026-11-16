// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativePrices.h>
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
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
// Section 2.3: assemble the four prices. A middle task carries its epilogue
// in c_x, where the next task's transfer hides it; a track's tail has no next
// task, so its epilogue is exposed in o_x. A track's head additionally pays
// the first-byte latency and one step of pipeline fill, and for that fill the
// compute compared against the transfer excludes the epilogue.
void Finish(Sm90NativeStagePrice& price, TaskPriceParts const& base, double epilogue,
            double steps, double first_byte) {
  TaskPriceParts middle = base;
  middle.compute_ns = base.compute_ns + epilogue;

  TaskPriceParts tail = base;
  tail.compute_ns = base.compute_ns;
  tail.fixed_ns = base.fixed_ns + epilogue;

  TaskPriceParts head = middle;
  head.fixed_ns = base.fixed_ns + first_byte;
  head.pipeline_steps = steps;
  head.pipeline_compute_ns = base.compute_ns;

  TaskPriceParts both = tail;
  both.fixed_ns = base.fixed_ns + epilogue + first_byte;
  both.pipeline_steps = steps;
  both.pipeline_compute_ns = base.compute_ns;

  price.middle = middle;
  price.tail = tail;
  price.head = head;
  price.head_and_tail = both;
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
      ? Sm90NativeExpectedActiveExperts(work.experts,work.top_k,b) : 1.0;
  double rows = family == 6 ? b * double(work.top_k) / active
                            : family == 7 ? (b <= 8 ? 8 : 16) : b;
  double tasks = n / g.tile_n * active;

  Sm90NativeStagePrice price;
  price.name = "gemm" + std::to_string(family);
  price.category = "gemm";
  if(family>=6)price.approximation=kSm90NativeMoeDomainApproximation;
  price.tasks = family>=6?Sm90NativeExpectedExpertTasks(work.experts,work.top_k,b,
      work.shapes[family][0]/g.tile_n):int(std::llround(tasks));
  
  price.hop_ns = c.Barrier(grid);

  // :47, per task: this task's own weight and activation bytes through the
  // CTA's async stream. The stage term divides the per-CTA sum by BW_sm, so a
  // single task's share is its own tile.
  double weights = g.tile_n * k * 2.0;
  double activation = rows * k * 2.0;
  if (g.multicast) activation /= 2.0;  // :45, L2 serves each A slot once per pair

  TaskPriceParts parts;
  // Eq. (7) c_x: the body's arithmetic and on-chip accesses.
  double body_ns = (weights + activation) / c.sm_gbps;
  parts.compute_ns = body_ns;
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
  double epilogue = c.Epilogue(kEpilogue[family] + geometry, g.tile_n, b, g.warps_k);
  // o_x: per-task costs that nothing overlaps.
  parts.fixed_ns = 0;
  if (family >= 6)
    parts.fixed_ns += c.TaskExpert((family == 6 ? "expert_gate_up" : "expert_down") + geometry, b);
  if (work.tp > 1 && (family == 1 || family == 3))
    parts.fixed_ns += c.AllReduce(b, g.tile_n, work.tp);
  // n_x: the slots this task's weights pass through, so max{c,m} + min{c,m}/n
  // is the n-step pipeline. One slot means no overlap at all.
  double steps = std::max(1.0, std::ceil(weights / double(g.slot_bytes)));
  double first_byte = pre ? c.l2_latency : c.dram_latency;
  price.ends = std::min(price.tasks, grid);
  Finish(price, parts, epilogue, steps, first_byte);
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
  // n_x for attention is its KV block count (section 2.3 item 3).
  double steps = std::max(1.0, blocks);
  price.ends = std::min(price.tasks, grid);
  // :88 uses L_l2, not L_dram: last step's KV history is still in L2.
  Finish(price, parts, /*epilogue=*/0.0, steps, c.l2_latency);
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
  // Section 2.3 item 4: tasks go to tracks in index order, so the first
  // `ends` indices head a track and the last `ends` indices tail one. The two
  // sets overlap when the stage has no more tasks than the grid has tracks.
  enum {kBoth = 0, kHead = 1, kTail = 2, kMiddle = 3};
  space.pieces.push_back(FlowPiece{0, price.head_and_tail});
  space.pieces.push_back(FlowPiece{0, price.head});
  space.pieces.push_back(FlowPiece{0, price.tail});
  space.pieces.push_back(FlowPiece{0, price.middle});
  space.piece_of_task.assign(price.tasks, kMiddle);
  for (int task = 0; task < price.tasks; ++task) {
    bool head = task < price.ends;
    bool tail = task >= price.tasks - price.ends;
    int piece = head && tail ? kBoth : head ? kHead : tail ? kTail : kMiddle;
    space.piece_of_task[task] = piece;
    ++space.pieces[piece].count;
  }
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
