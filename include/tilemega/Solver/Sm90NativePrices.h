// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/StageFlowModel.h>
#include <string>

namespace tilemega::solver {

// P1: the native cost model's own terms, restated as the per-task prices the
// shared pooled evaluator consumes. Nothing here is a new formula: every part
// is one term of Sm90DenseCostModel, and Sm90NativePrices.cpp cites the line
// it comes from. Eq. (7) splits a task's service into o + max{c, m}, which is
// `fixed_ns`, `compute_ns` and the DRAM demand respectively.
//
// A stage's tasks are not all equally priced: the first task a CTA runs pays
// the stage's first-byte latency, so a stage is two pieces rather than one.
struct Sm90NativeStagePrice {
  std::string name, category;
  int tasks = 0;
  // How many tasks carry `first`: one per CTA that gets any work at all.
  int first_tasks = 0;
  TaskPriceParts first, rest;
  // The boundary ahead of this stage, as a constant (V3 barrier by default).
  double hop_ns = 0;
};

Sm90NativeStagePrice Sm90NativeGemmPrice(Sm90DenseWorkload const& work, unsigned family,
                                         Sm90DenseGemm const& gemm, int residency,
                                         Sm90DenseCostModel const& cost, bool weight_stream,
                                         bool pre = false);
Sm90NativeStagePrice Sm90NativeAttentionPrice(Sm90DenseWorkload const& work,
                                              Sm90DenseAttention const& attention, int residency,
                                              Sm90DenseCostModel const& cost);

// One stage alone, its in-edge an all-producer boundary: the configuration
// P1's self-check compares against Gemm()/Attention() for the same stage.
FlowProblem Sm90NativeSingleStageProblem(Sm90NativeStagePrice const& price,
                                         Sm90DenseCostModel const& cost, int residency);
}  // namespace tilemega::solver
