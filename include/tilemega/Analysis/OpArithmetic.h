// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/QuasiPolynomial.h>
#include <tilemega/Analysis/Semantics.h>
#include <string>
#include <optional>
#include <vector>

#ifndef TILEMEGA_OP_ARITHMETIC
#define TILEMEGA_OP_ARITHMETIC 1
#endif

namespace tilemega::analysis {

// Width is an implementation/model axis, fixed before symbolic seq solving.
// Keeping a rational denominator avoids rounding per-output work prematurely.
struct ArithmeticRatio {
  QuasiPolynomial numerator;
  long denominator = 1;
  double Eval(ParamBinding const& theta) const;
};

struct ArithmeticInputs {
  std::optional<QuasiPolynomial> reduction, total;
  long width = 0;
  ScalarType dtype = ScalarType::kF32;
};

struct OpArithmetic {
  ArithmeticRatio flops_per_output_element;
  ArithmeticRatio transcendental_per_output_element;
  bool flops_use_mma = false;
  bool smem_staged = false;
  std::string reason;
  // Semantic arithmetic is not a claim that a placeholder TaskBody implements it.
  bool runtime_implemented = false;
};

struct ArithmeticForm {
  long constant, reduction, total, width;
  long denominator;
  bool divide_by_width;
};

struct ArithmeticDeclaration {
  char const* name;
  ArithmeticForm flops, transcendental;
  bool bf16_mma, smem_staged, runtime_implemented;
  char const* reason;
  char const* derivation;
};

std::vector<ArithmeticDeclaration> const& ArithmeticDeclarations();
void ValidateArithmeticDeclaration(ArithmeticDeclaration const& declaration);
OpArithmetic InstantiateArithmetic(std::string const& name,
                                  ArithmeticInputs const& inputs);
void RequireArithmeticImplementation(OpArithmetic const& arithmetic);

// Sequential phases retain their own output domains and execution pipes.
// A GEMM+add signature must never route add FLOPs to the MMA lane.
struct MixedArithmeticPhase {
  OpArithmetic arithmetic;
  QuasiPolynomial output_elements;
};
struct MixedArithmetic {
  std::vector<MixedArithmeticPhase> phases;
  struct Work { double mma=0, simt=0, transcendental=0; };
  Work Eval(ParamBinding const& theta) const;
};
MixedArithmetic ComposeArithmetic(std::vector<MixedArithmeticPhase> phases);

}  // namespace tilemega::analysis
