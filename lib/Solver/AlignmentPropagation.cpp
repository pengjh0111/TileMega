// SPDX-License-Identifier: BSD-3-Clause

#include <tilemega/Solver/AlignmentPropagation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace tilemega::solver {
namespace {

int CeilDiv(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

char const* KindName(StageKind kind) {
  switch (kind) {
    case StageKind::kGemm: return "Gemm";
    case StageKind::kRMSNorm: return "RMSNorm";
    case StageKind::kRoPE: return "RoPE";
    case StageKind::kKVAppend: return "KVAppend";
    case StageKind::kElementwise: return "Elementwise";
    case StageKind::kAttention: return "Attention";
  }
  return "?";
}

}  // namespace

int WaitTiles(int m, int consumer_tile, int producer_tile) {
  if (producer_tile <= 0 || consumer_tile <= 0) return 0;
  return CeilDiv((m + 1) * consumer_tile, producer_tile) -
         m * consumer_tile / producer_tile;
}

int WaitInflation(int consumer_tile, int producer_tile, int extent) {
  if (producer_tile <= 0 || consumer_tile <= 0 || extent <= 0) return 0;
  // The pattern repeats with period lcm(Tm,Tr)/Tm, so scanning that many tasks
  // is exhaustive however long the axis is; the model's own extent caps it so
  // a short axis is never charged for a straddle it will not reach.
  int const gcd = std::gcd(consumer_tile, producer_tile);
  int const period = producer_tile / gcd;
  int const tasks = std::min(CeilDiv(extent, consumer_tile), std::max(period, 1));
  int const ideal = CeilDiv(consumer_tile, producer_tile);
  int worst = 0;
  for (int m = 0; m < tasks; ++m) {
    worst = std::max(worst, WaitTiles(m, consumer_tile, producer_tile));
  }
  return worst - ideal;
}

std::vector<AlignmentConstraint> DeriveAlignmentConstraints(
    ModelDescription const& model) {
  std::vector<AlignmentConstraint> out(model.gemms.size());
  std::vector<int> gemm_stage(model.gemms.size(), -1);
  for (std::size_t i = 0; i < model.stages.size(); ++i) {
    ModelStage const& stage = model.stages[i];
    if (stage.kind == StageKind::kGemm && stage.gemm >= 0) {
      gemm_stage[stage.gemm] = static_cast<int>(i);
    }
  }
  for (std::size_t g = 0; g < model.gemms.size(); ++g) {
    AlignmentConstraint& c = out[g];
    c.gemm = static_cast<int>(g);
    int const at = gemm_stage[g];
    if (at < 0) continue;
    GemmOp const& op = model.gemms[g];
    // Backwards from the output end: the first stage after this GEMM that
    // reads its destination is the one whose task width the columns must line
    // up with.  A later reader is already covered by that stage's own
    // constraint, which is what makes one pass over the list enough.
    for (std::size_t i = at + 1; i < model.stages.size(); ++i) {
      ModelStage const& stage = model.stages[i];
      if (stage.kind == StageKind::kGemm) {
        if (stage.gemm >= 0 && model.gemms[stage.gemm].in_buffer == op.out_buffer) {
          // A GEMM reading a GEMM couples two free variables rather than
          // pinning one; neither reference model does it, so refuse instead of
          // pretending the single-variable form covers it.
          throw std::runtime_error(
              "alignment propagation: GEMM feeds GEMM directly, which needs "
              "the coupled two-variable form");
        }
        continue;
      }
      if (std::find(stage.operands.begin(), stage.operands.end(),
                    op.out_buffer) == stage.operands.end()) {
        continue;
      }
      c.consumer_stage = static_cast<int>(i);
      c.consumer_kind = stage.kind;
      c.consumer_tile = stage.ReadGranularity();
      break;
    }
    for (int i = at - 1; i >= 0; --i) {
      ModelStage const& stage = model.stages[i];
      if (stage.kind == StageKind::kGemm) continue;
      if (std::find(stage.operands.begin(), stage.operands.end(),
                    op.in_buffer) == stage.operands.end()) {
        continue;
      }
      c.producer_stage = i;
      c.producer_kind = stage.kind;
      c.producer_tile = stage.ReadGranularity();
      break;
    }
  }
  return out;
}

std::vector<std::vector<int>> PropagateAlignment(ModelDescription const& model,
                                                 std::vector<TileAxes> const& axes,
                                                 AlignmentStats* stats) {
  std::vector<AlignmentConstraint> const constraints =
      DeriveAlignmentConstraints(model);
  std::vector<std::vector<int>> kept(model.gemms.size());
  AlignmentStats local;
  local.operators = static_cast<int>(model.gemms.size());
  local.before = static_cast<int>(axes.size());
  local.after_min = local.before;
  local.after_max = 0;
  double total = 0.0;
  for (std::size_t g = 0; g < model.gemms.size(); ++g) {
    AlignmentConstraint const& c = constraints[g];
    GemmOp const& op = model.gemms[g];
    if (c.consumer_tile <= 0 && c.producer_tile <= 0) ++local.unconstrained;
    for (std::size_t i = 0; i < axes.size(); ++i) {
      if (c.consumer_tile > 0 &&
          WaitInflation(c.consumer_tile, axes[i].tile_n, op.n) != 0) {
        continue;
      }
      if (c.producer_tile > 0 &&
          WaitInflation(c.producer_tile, axes[i].tile_k, op.k) != 0) {
        continue;
      }
      kept[g].push_back(static_cast<int>(i));
    }
    int const n = static_cast<int>(kept[g].size());
    local.after_min = std::min(local.after_min, n);
    local.after_max = std::max(local.after_max, n);
    total += n;
    local.joint_log10_after += n > 0 ? std::log10(static_cast<double>(n)) : 0.0;
    local.joint_log10_before += std::log10(static_cast<double>(local.before));
  }
  local.after_mean = local.operators > 0 ? total / local.operators : 0.0;
  if (stats) *stats = local;
  return kept;
}

std::string ExplainAlignment(ModelDescription const& model,
                             std::vector<TileAxes> const& axes,
                             std::vector<std::vector<int>> const& kept) {
  std::vector<AlignmentConstraint> const constraints =
      DeriveAlignmentConstraints(model);
  std::string out =
      "gemm  n     k     consumer            Tr    producer            Tk    "
      "kept\n";
  char line[256];
  for (std::size_t g = 0; g < model.gemms.size(); ++g) {
    AlignmentConstraint const& c = constraints[g];
    std::snprintf(line, sizeof(line),
                  "%-5zu %-5d %-5d %-11s@%-7d %-5d %-11s@%-7d %-5d %d/%zu\n", g,
                  model.gemms[g].n, model.gemms[g].k,
                  c.consumer_stage < 0 ? "-" : KindName(c.consumer_kind),
                  c.consumer_stage, c.consumer_tile,
                  c.producer_stage < 0 ? "-" : KindName(c.producer_kind),
                  c.producer_stage, c.producer_tile,
                  static_cast<int>(kept[g].size()), axes.size());
    out += line;
  }
  return out;
}

}  // namespace tilemega::solver
