// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>
#include <cmath>

#if defined(__CUDACC__)
#define TILEMEGA_DM_VALUE_HD __host__ __device__
#else
#define TILEMEGA_DM_VALUE_HD
#endif

namespace tilemega::backend {

struct DmEpilogueInputs {
  float bias = 0, scale = 1;
  float residual = 0, residual_scale = 1;
  float mean = 0, rstd = 1, u = 0, v = 0;
  float gamma = 1, beta = 0;
};

template <codegen::DmRounding Rounding>
TILEMEGA_DM_VALUE_HD inline float DmRound(float value) {
  if constexpr(Rounding==codegen::DmRounding::kBF16)
    return float(cutlass::bfloat16_t(value));
  return value;
}

template <codegen::DmActivation Activation>
TILEMEGA_DM_VALUE_HD inline float DmActivate(float value) {
  using A=codegen::DmActivation;
  if constexpr(Activation==A::kRelu) return value<0 ? 0.0f : value;
  if constexpr(Activation==A::kRelu6) return value<0 ? 0.0f : value>6 ? 6.0f : value;
  if constexpr(Activation==A::kGeluErf)
    return 0.5f*value*(1.0f+erff(value*0.7071067811865475244f));
  if constexpr(Activation==A::kGeluTanh)
    return 0.5f*value*(1.0f+tanhf(0.7978845608028653559f*
        (value+0.044715f*value*value*value)));
  if constexpr(Activation==A::kTanh) return tanhf(value);
  if constexpr(Activation==A::kSilu) return value/(1.0f+expf(-value));
}

// Deferred LN uses statistics of the stored BF16 tensor. Folded W', u and v
// remain FP32 through rstd*(acc-mu*u)+v; the step and store contracts decide
// the next materialization point. Residual LN separately rounds LN(x) to
// BF16 before adding it, reproducing the explicit residual input's storage.
template <class Arch, class Step>
struct DmEpilogueValue {
  static_assert(arch::Caps<Arch>::kBf16TensorCore,
                "DM epilogues require a target with BF16 support");
  static_assert(unsigned(Step::kKind)<=7 && unsigned(Step::kActivation)<=5 &&
                unsigned(Step::kGate)<=1 && unsigned(Step::kInputRounding)<=1 &&
                unsigned(Step::kOutputRounding)<=1,
                "invalid compile-time DM epilogue specification");
  TILEMEGA_DM_VALUE_HD static float Apply(float value, DmEpilogueInputs const& p,
                                          float partner=0) {
    using K=codegen::DmEpilogueKind;
    value=DmRound<Step::kInputRounding>(value);
    if constexpr(Step::kKind==K::kBias) value+=p.bias;
    else if constexpr(Step::kKind==K::kScale) value*=p.scale;
    else if constexpr(Step::kKind==K::kActivation)
      value=DmActivate<Step::kActivation>(value);
    else if constexpr(Step::kKind==K::kResidual)
      value=value*p.residual_scale+p.residual;
    else if constexpr(Step::kKind==K::kGatePair) {
      partner=DmRound<Step::kInputRounding>(partner);
      if constexpr(Step::kGate==codegen::DmGatePair::kSwiGLU) {
        // HF stores SiLU(gate) in BF16 before its product with up.
        value=DmRound<codegen::DmRounding::kBF16>(DmActivate<codegen::DmActivation::kSilu>(value));
      }
      value*=partner;
    }
    else if constexpr(Step::kKind==K::kDeferredRMSNorm) value*=p.rstd;
    else if constexpr(Step::kKind==K::kDeferredLayerNorm)
      value=p.rstd*(value-p.mean*p.u)+p.v;
    else if constexpr(Step::kKind==K::kResidualLN) {
      float residual=DmRound<codegen::DmRounding::kBF16>(
          ((p.residual-p.mean)*p.rstd)*p.gamma+p.beta);
      value+=residual;
    }
    return DmRound<Step::kOutputRounding>(value);
  }
};

template<class Program> struct DmEpilogueWalk;
template<class... Steps>
struct DmEpilogueWalk<codegen::DmEpilogueProgram<Steps...>> {
  static constexpr unsigned kGates=(0u+...+
      unsigned(Steps::kKind==codegen::DmEpilogueKind::kGatePair));
  template<unsigned Index,bool Gated,class Visitor>
  TILEMEGA_DM_VALUE_HD static void Visit(Visitor&) {}
  template<unsigned Index,bool Gated,class First,class... Rest,class Visitor>
  TILEMEGA_DM_VALUE_HD static void Visit(Visitor& visitor) {
    visitor.template Apply<Index,First,Gated>();
    Visit<Index+1,Gated || First::kKind==codegen::DmEpilogueKind::kGatePair,Rest...>(visitor);
  }
  template<class Visitor>
  TILEMEGA_DM_VALUE_HD static void Run(Visitor& visitor) {
    Visit<0,false,Steps...>(visitor);
  }
};

}  // namespace tilemega::backend
#undef TILEMEGA_DM_VALUE_HD
