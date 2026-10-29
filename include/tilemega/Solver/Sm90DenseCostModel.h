// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Solver/Sm90DenseWorkload.h>
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <tilemega/Target/TargetSpec.h>
#include <optional>
#include <string>
#include <vector>

namespace tilemega::solver {
struct Sm90DensePrediction {
  struct Stage {
    std::string kind;
    int instances=1;
    double ns=0;
    json::Value terms;
  };
  std::vector<Stage> stages;
  double step_ns=0;
  json::Value ToJson() const;
};

// Fixed per-stage formulas of prompt §4 L0-2 over measured constants. Stages
// are additive; nothing here is fitted to end-to-end measurements.
class Sm90DenseCostModel {
 public:
  Sm90DenseCostModel(TargetSpec target,Sm90NativeConstants constants);
  // family: 0 qkv, 1 o, 2 gate_up, 3 down, 4 lm_head. pre: the stage's first
  // bytes were streamed by a fully resident predecessor (D6), hiding L_dram.
  double Gemm(Sm90DenseWorkload const&,unsigned family,Sm90DenseGemm const&,int residency,
              json::Value* terms=nullptr,bool pre=false) const;
  double Attention(Sm90DenseWorkload const&,Sm90DenseAttention const& attention,
                   int residency,json::Value* terms=nullptr) const;
  Sm90DensePrediction Predict(Sm90DenseWorkload const&,Sm90DenseCoordinates const&) const;
  int Grid(int residency) const {return target_.res.num_sms*residency;}
  Sm90NativeConstants const& Constants() const {return constants_;}
 private:
  double Waves(double tasks,int grid,double task_bytes) const;
  TargetSpec target_;
  Sm90NativeConstants constants_;
};
}  // namespace tilemega::solver
