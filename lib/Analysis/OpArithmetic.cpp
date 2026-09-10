// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/OpArithmetic.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <stdexcept>

namespace tilemega::analysis {

std::vector<ArithmeticDeclaration> const& ArithmeticDeclarations() {
  // Semantic DAG counts: FMA=2; scalar +/-/*// each=1, comparisons and
  // conversions excluded. TaskBody reduction/replication overhead is separate.
  // All coefficients live here, never in the solver's per-kind dispatch.
  static const std::vector<ArithmeticDeclaration> table = {
    {"gemm", {0,2,0,0,1,false}, {0,0,0,0,1,false}, true,true,true,
     "implemented", "one length-K dot per output: K FMA = 2K; BF16 MMA, FP32 SIMT"},
    {"attention", {0,0,4,0,1,false}, {0,0,1,0,1,true}, false,true,true,
     "dense_semantic_work", "QK and PV each use 2*total FLOP/output; total exp/head divided by width; causal QK and softmax overhead require task-domain accounting"},
    {"rmsnorm", {1,0,0,4,1,true}, {1,0,0,0,1,true}, false,true,true,
     "semantic_reduction", "W squares + W-1 sum + mean and epsilon + 2W scale/weight = 4W+1; one rsqrt/row; parallel replication is not included"},
    {"rope", {7,0,0,0,2,false}, {1,0,0,0,1,false}, false,false,true,
     "implemented", "per pair: angle multiply + four multiplies + two adds = 7; sin and cos = 2, divided by two outputs"},
    {"silu", {3,0,0,0,1,false}, {1,0,0,0,1,false}, false,false,true,
     "activation_component", "negate, add one, divide; one exp"},
    {"mul", {1,0,0,0,1,false}, {0,0,0,0,1,false}, false,false,true,
     "activation_component", "one multiplication"},
    {"add", {1,0,0,0,1,false}, {0,0,0,0,1,false}, false,false,true,
     "residual_component", "one addition (unit residual coefficient)"},
    {"swiglu", {4,0,0,0,1,false}, {1,0,0,0,1,false}, false,false,true,
     "implemented", "SiLU's three scalar ops followed by gate/up multiply; one exp"},
    {"kv_append", {0,0,0,0,1,false}, {0,0,0,0,1,false}, false,false,true,
     "pure_data_movement", "copy only, index arithmetic excluded from floating-point lanes"},
    {"sum", {-1,1,0,0,1,false}, {0,0,0,0,1,false}, false,false,true,
     "combine_component", "R values require R-1 additions; zero-seeded implementation overhead is separate"},
    {"softmax", {-1,0,0,3,1,true}, {1,0,0,0,1,false}, false,true,false,
     "no_standalone_taskbody", "per row: W subtract + W-1 sum + W divide = 3W-1; W exp; max comparisons excluded"},
    {"layernorm", {1,0,0,8,1,true}, {1,0,0,0,1,true}, false,true,false,
     "no_standalone_taskbody", "two-pass: mean W ops, variance 3W ops, epsilon 1, center/normalize/affine 4W ops = 8W+1; one rsqrt"},
    {"gelu_tanh", {8,0,0,0,1,false}, {1,0,0,0,1,false}, false,false,false,
     "no_standalone_taskbody", "0.5*x*(1+tanh(c*(x+0.044715*x^3))): cube 2, scaled cube 1, add 1, scale 1, add one 1, final multiplies 2 = 8"},
    {"moe_router", {-1,0,0,3,1,true}, {1,0,0,0,1,false}, false,true,false,
     "placeholder_taskbody", "softmax over expert scores as above; top-k comparison/selection is not floating arithmetic; existing body only stores an integer, so pricing rejects it"},
  };
  return table;
}

void ValidateArithmeticDeclaration(ArithmeticDeclaration const& d) {
  if (!d.name || !*d.name || !d.reason || !*d.reason || !d.derivation || !*d.derivation)
    throw std::invalid_argument("incomplete arithmetic declaration schema");
  if (d.flops.denominator<=0 || d.transcendental.denominator<=0)
    throw std::invalid_argument("nonpositive arithmetic denominator");
}

double ArithmeticRatio::Eval(ParamBinding const& theta) const {
  if (denominator<=0) throw std::invalid_argument("nonpositive arithmetic denominator");
  return static_cast<double>(numerator.Eval(theta))/static_cast<double>(denominator);
}

OpArithmetic InstantiateArithmetic(std::string const& name, ArithmeticInputs const& inputs) {
  IslReferenceAudit audit(__func__);
#if !TILEMEGA_OP_ARITHMETIC
  throw std::runtime_error("operator arithmetic declarations disabled");
#endif
  auto const& table = ArithmeticDeclarations();
  auto found = std::find_if(table.begin(),table.end(),[&](auto const& d){return name==d.name;});
  if (found==table.end()) throw std::invalid_argument("missing arithmetic signature: "+name);
  ValidateArithmeticDeclaration(*found);
  auto instantiate = [&](ArithmeticForm const& f) {
    if ((f.width || f.divide_by_width) && inputs.width<=0)
      throw std::invalid_argument("arithmetic signature requires a concrete positive width: "+name);
    std::vector<QuasiPolynomial> terms{QuasiPolynomial::Constant(f.constant)};
    if (f.reduction) {
      if (!inputs.reduction) throw std::invalid_argument("missing reduction extent: "+name);
      terms.push_back(inputs.reduction->Scale(f.reduction));
    }
    if (f.total) {
      if (!inputs.total) throw std::invalid_argument("missing total extent: "+name);
      terms.push_back(inputs.total->Scale(f.total));
    }
    if (f.width) terms.push_back(QuasiPolynomial::Constant(f.width*inputs.width));
    return ArithmeticRatio{QuasiPolynomial::Sum(terms),
                           f.denominator*(f.divide_by_width ? inputs.width : 1)};
  };
  return {instantiate(found->flops),instantiate(found->transcendental),
          found->bf16_mma && inputs.dtype==ScalarType::kBF16,
          found->smem_staged,found->reason,found->runtime_implemented};
}

void RequireArithmeticImplementation(OpArithmetic const& a) {
  if (!a.runtime_implemented)
    throw std::invalid_argument("arithmetic task implementation absent: "+a.reason);
}

MixedArithmetic ComposeArithmetic(std::vector<MixedArithmeticPhase> phases) {
  IslReferenceAudit audit(__func__);
  if (phases.size()<2) throw std::invalid_argument("mixed arithmetic requires at least two phases");
  for (auto const& phase:phases) RequireArithmeticImplementation(phase.arithmetic);
  return {std::move(phases)};
}

MixedArithmetic::Work MixedArithmetic::Eval(ParamBinding const& theta) const {
  IslReferenceAudit audit(__func__);
  Work out;
  for (auto const& phase:phases) {
    auto elements=phase.output_elements.Eval(theta);
    if (elements<0) throw std::invalid_argument("negative mixed arithmetic output work");
    auto const& a=phase.arithmetic;
    double flops=a.flops_per_output_element.Eval(theta)*elements;
    if (a.flops_use_mma) out.mma+=flops;
    else out.simt+=flops;
    out.transcendental+=a.transcendental_per_output_element.Eval(theta)*elements;
  }
  return out;
}
}  // namespace tilemega::analysis
