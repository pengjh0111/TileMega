// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::solver {
struct Sm90DenseStagePlanBinding {
  std::string epilogue;
  bool input_norm=false,output_sums=false;
  json::Value ToJson() const {
    if(epilogue!="store" && epilogue!="residual" && epilogue!="swiglu" && epilogue!="argmax_partial")
      throw std::invalid_argument("unknown native plan epilogue");
    return json::Object{{"epilogue",epilogue},{"input_norm",input_norm},{"output_sums",output_sums}};
  }
};
// Ordered by program execution, rather than the storage order of plan.gemms.
struct Sm90DenseGemmPlanBinding {
  int stage=0,gemm=0;
  std::string family;
  int n=0,k=0;
  Sm90DenseStagePlanBinding binding;
  std::uint32_t norm_ss=~std::uint32_t{0},ss_out=~std::uint32_t{0};
  json::Value ToJson() const {
    if(binding.input_norm!=(norm_ss!=~std::uint32_t{0}) || binding.output_sums!=(ss_out!=~std::uint32_t{0}))
      throw std::invalid_argument("native binding flags differ from operand presence");
    auto result=binding.ToJson();
    result.Set("stage",stage);result.Set("gemm",gemm);result.Set("family",family);
    result.Set("n",n);result.Set("k",k);
    result.Set("norm_ss",binding.input_norm?json::Value(double(norm_ss)):json::Value());
    result.Set("ss_out",binding.output_sums?json::Value(double(ss_out)):json::Value());return result;
  }
  static Sm90DenseGemmPlanBinding FromJson(json::Value const& value) {
    auto integer=[&](char const* key) {
      double x=value.At(key).AsNumber(key);
      if(!std::isfinite(x) || x<0 || x!=std::floor(x) || x>std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string("invalid native binding integer: ")+key);
      return int(x);
    };
    auto operand=[&](char const* key) {
      auto const& v=value.At(key);if(v.IsNull())return ~std::uint32_t{0};
      double x=v.AsNumber(key);
      if(!std::isfinite(x) || x<0 || x!=std::floor(x) || x>=std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(std::string("invalid native binding operand: ")+key);
      return std::uint32_t(x);
    };
    Sm90DenseGemmPlanBinding result{integer("stage"),integer("gemm"),
        value.At("family").AsString("family"),integer("n"),integer("k"),
        {value.At("epilogue").AsString("epilogue"),value.At("input_norm").AsBool("input_norm"),
            value.At("output_sums").AsBool("output_sums")},operand("norm_ss"),operand("ss_out")};
    result.ToJson();return result;
  }
};
struct Sm90DenseWorkload {
  Sm90DenseShapeTable shapes;
  int batch=0,layers=0,head_dim=0,queries=0,hkv=8;
  bool qk_norm=false;
  // MoE decode (prompt §8): 128 experts, top-8; shapes 5-7 are per expert.
  bool moe=false;
  int experts=0,top_k=0;
  // Absence preserves the unknown binding domain of historical workloads.
  std::optional<std::vector<Sm90DenseGemmPlanBinding>> gemm_bindings;
};
}  // namespace tilemega::solver
