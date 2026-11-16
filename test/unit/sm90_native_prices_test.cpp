// SPDX-License-Identifier: BSD-3-Clause
// Continuation 1 section 2: check Eq. (7)'s pipeline arithmetic. Comparison
// with the additive model is diagnostic and imposes no acceptance threshold.
#include <tilemega/Solver/Sm90NativePrices.h>
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace tilemega;
using namespace tilemega::solver;

// Section 2.4's closed form for a stage that fits one wave. The tasks of a
// wave contend for the stage's DRAM, so a task's transfer runs at the smaller
// of its own per-CTA ceiling and its share of the stage bandwidth.
static double ClosedForm(Sm90NativeStagePrice const& price, double dram_gbps, int grid) {
  auto const& e = price.head_and_tail;
  int wave = std::min(price.tasks, grid);
  double rate = std::min(e.dram_rate_cap, dram_gbps / wave);
  double m = e.dram_bytes / rate;
  return price.hop_ns + e.fixed_ns + std::max(e.compute_ns, m) +
         std::min(e.compute_ns, m) / e.pipeline_steps;
}

int main(int argc, char** argv) {
  if (argc < 2) {std::fprintf(stderr, "usage: %s <target.json>\n", argv[0]); return 2;}
  auto target = TargetSpec::FromJson(argv[1]);
  auto constants = Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]), argc > 2 ? argv[2] : ".");
  Sm90DenseCostModel cost(target, constants);

  // Two tasks on one track: the first task has body=10, epilogue=6,
  // transfer=20 and two pipeline steps. Fill is 10/2, not (10+6)/2.
  FlowProblem fill;
  fill.workers=1;fill.dram_gbps=1;
  FlowSpace s;s.count=2;s.piece_of_task={0,1};
  TaskPriceParts first;first.compute_ns=16;first.dram_bytes=20;first.dram_rate_cap=1;
  first.pipeline_steps=2;first.pipeline_compute_ns=10;
  TaskPriceParts last;last.compute_ns=10;last.dram_bytes=20;last.fixed_ns=6;last.dram_rate_cap=1;
  s.pieces={{1,first},{1,last}};fill.spaces.push_back(s);
  assert(std::abs(EvaluateFlow(fill).makespan_ns-51)<1e-9);

  // Qwen3-1.7B decode, the shape table the dense cells are solved on.
  Sm90DenseWorkload work;
  work.shapes = {{{4096, 2048}, {2048, 2048}, {6144, 2048}, {2048, 3072}, {151936, 2048}}};
  work.layers = 28; work.head_dim = 128; work.queries = 2; work.hkv = 8;
  work.qk_norm = true;

  {
  auto moe=work;moe.moe=true;moe.experts=128;moe.top_k=8;moe.batch=16;
  moe.shapes[6]={1536,2048};
  Sm90DenseGemm expert;expert.tile_n=96;expert.slot_bytes=2*96*64;
  auto expert_price=Sm90NativeGemmPrice(moe,6,expert,1,cost,false);
  assert(expert_price.tasks==1319);
  assert(expert_price.approximation==kSm90NativeMoeDomainApproximation);
  }
  int failures = 0;
  for (int batch : {1, 16}) {
    work.batch = batch;
    for (int residency : {1, 2}) {
      for (unsigned family = 0; family < 5; ++family) {
        Sm90DenseGemm g;
        g.tile_n = family == 4 ? 128 : 32;
        g.slot_bytes = 2 * g.tile_n * 64;
        double stage = cost.Gemm(work, family, g, residency);
        auto price = Sm90NativeGemmPrice(work, family, g, residency, cost, /*weight_stream=*/false);
        auto problem = Sm90NativeSingleStageProblem(price, cost, residency);
        double pooled = EvaluateFlow(problem).makespan_ns;
        // Section 2.4: the closed form for a single wave, which is what a
        // stage whose task count fits the grid runs.
        if (price.tasks <= cost.Grid(residency)) {
          double closed = ClosedForm(price, constants.dram_gbps, cost.Grid(residency));
          double gap = std::fabs(pooled - closed) / closed * 100;
          std::printf("    closed form %10.1f  gap %5.2f%%\n", closed, gap);
          if (gap > 0.5) ++failures;
        }
        double deviation = std::fabs(pooled - stage) / stage * 100;
        // Appendix B: (W_tot + A_tot) / BW_l2 has no lane in FlowProblem. Where
        // it is the stage model's largest term the two cannot agree, so the
        // cell is listed and left out of the verdict rather than forced.
        double n = work.shapes[family][0], k = work.shapes[family][1];
        double w_tot = 2.0 * n * k, a_tot = n / g.tile_n * batch * k * 2.0;
        bool l2_bound = (w_tot + a_tot) / constants.l2_gbps > w_tot / constants.dram_gbps;
        std::printf("B%-3d res%d family%u tasks%-5d stage %10.1f pooled %10.1f dev %6.2f%%%s\n",
                    batch, residency, family, price.tasks, stage, pooled, deviation,
                    l2_bound ? "  [L2-bound, excluded by appendix B]" : "");
        (void)l2_bound;  // section 2.4 item 3: diagnostic only, never asserted
      }
      Sm90DenseAttention a;
      a.chunks = 16 / batch > 0 ? 16 / batch : 1;
      double stage = cost.Attention(work, a, residency);
      auto price = Sm90NativeAttentionPrice(work, a, residency, cost);
      auto problem = Sm90NativeSingleStageProblem(price, cost, residency);
      double pooled = EvaluateFlow(problem).makespan_ns;
      // The merge is a handoff, so it is not part of these tasks (appendix A).
      double merge = a.chunks == 1 ? 0 : constants.AttentionMerge(work.head_dim, work.queries, 1, a.chunks);
      double comparable = stage - merge;
      double deviation = std::fabs(pooled - comparable) / comparable * 100;
      std::printf("B%-3d res%d attention tasks%-5d stage %10.1f pooled %10.1f dev %6.2f%%\n",
                  batch, residency, price.tasks, comparable, pooled, deviation);
      // Attention's closed form, same rule as the GEMM families above.
      if (price.tasks <= cost.Grid(residency)) {
        double closed = ClosedForm(price, constants.dram_gbps, cost.Grid(residency));
        double gap = std::fabs(pooled - closed) / closed * 100;
        std::printf("    closed form %10.1f  gap %5.2f%%\n", closed, gap);
        if (gap > 0.5) ++failures;
      }
    }
  }
  // Section 2.4 item 1: the MoE families too. Qwen3-30B-A3B decode, shapes
  // 5-7 per expert, E = 128 top-8 (the solver's own workload for those cells).
  Sm90DenseWorkload moe;
  moe.shapes = {{{6144, 2048}, {4096, 2048}, {0, 0}, {0, 0}, {151936, 2048},
                 {128, 2048}, {1536, 2048}, {2048, 768}}};
  moe.layers = 48; moe.head_dim = 128; moe.queries = 4; moe.hkv = 4;
  moe.qk_norm = true; moe.moe = true; moe.experts = 128; moe.top_k = 8;
  for (int batch : {1, 16}) {
    moe.batch = batch;
    for (unsigned family = 5; family < 8; ++family) {
      // Only the measured geometries exist: the constants are exact-match
      // lookups and throw Sm90NativeMissing otherwise (F17).
      Sm90DenseGemm g;
      g.tile_n = family == 5 ? 8 : family == 6 ? 96 : 128;
      g.warps_k = 4;
      g.slot_bytes = 2 * g.tile_n * 64;
      double stage = cost.Gemm(moe, family, g, 1);
      auto price = Sm90NativeGemmPrice(moe, family, g, 1, cost, /*weight_stream=*/true);
      auto problem = Sm90NativeSingleStageProblem(price, cost, 1);
      double pooled = EvaluateFlow(problem).makespan_ns;
      std::printf("MoE B%-3d family%u tasks%-5d stage %10.1f pooled %10.1f dev %6.2f%%\n",
                  batch, family, price.tasks, stage, pooled,
                  std::fabs(pooled - stage) / stage * 100);
      if (price.tasks <= cost.Grid(1)) {
        double closed = ClosedForm(price, constants.dram_gbps, cost.Grid(1));
        double gap = std::fabs(pooled - closed) / closed * 100;
        std::printf("    closed form %10.1f  gap %5.2f%%\n", closed, gap);
        if (gap > 0.5) ++failures;
      }
    }
  }
  std::printf("cells outside the 0.5%% closed form: %d\n", failures);
  return failures == 0 ? 0 : 1;
}
