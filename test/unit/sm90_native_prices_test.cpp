// SPDX-License-Identifier: BSD-3-Clause
// P1's self-check: for each family, the whole stage priced per task and run
// through the shared evaluator with an all-producer in-edge must agree with
// Sm90DenseCostModel's own stage time to within 2%. Disagreement means the
// mapping in Sm90NativePrices.cpp is wrong, never that a constant is.
#include <tilemega/Solver/Sm90NativePrices.h>
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace tilemega;
using namespace tilemega::solver;

int main(int argc, char** argv) {
  if (argc < 2) {std::fprintf(stderr, "usage: %s <target.json>\n", argv[0]); return 2;}
  auto target = TargetSpec::FromJson(argv[1]);
  auto constants = Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]), argc > 2 ? argv[2] : ".");
  Sm90DenseCostModel cost(target, constants);

  // Qwen3-1.7B decode, the shape table the dense cells are solved on.
  Sm90DenseWorkload work;
  work.shapes = {{{4096, 2048}, {2048, 2048}, {6144, 2048}, {2048, 3072}, {151936, 2048}}};
  work.layers = 28; work.head_dim = 128; work.queries = 2; work.hkv = 8;
  work.qk_norm = true;

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
        if (deviation > 2.0 && !l2_bound) ++failures;
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
      if (deviation > 2.0) ++failures;
    }
  }
  std::printf("families outside 2%%: %d\n", failures);
  return failures == 0 ? 0 : 1;
}
