// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/OpArithmetic.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <iostream>
#include <set>
#include <stdexcept>

int main(int argc,char** argv) try {
  using namespace tilemega::analysis;
  IslContext context;
  if (argc>1) {
    if (argc!=2) throw std::invalid_argument("usage: tilemega-op-audit [fused.mlir]");
    mlir::MLIRContext mlir_context;
    mlir_context.getOrLoadDialect<tilemega::dialect::CGDialect>();
    auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&mlir_context);
    if (!module) throw std::invalid_argument("cannot parse fusion arithmetic input");
    auto tasks=tilemega::solver::ReadFusedTaskInputs(*module);
    if (tasks.empty()) throw std::invalid_argument("fusion arithmetic input contains no fused tasks");
    for (auto const& task:tasks) {
      std::cout << "FUSED_ARITHMETIC task=" << task.name << " phases=" << task.phases.size()
                << " runtime_lowering=not_implemented\n";
      for (std::size_t i=0;i<task.phases.size();++i) {
        auto const& phase=task.arithmetic.phases[i];
        std::cout << "PHASE name=" << task.semantics[i].op.name
                  << " pipe=" << (phase.arithmetic.flops_use_mma ? "mma" : "simt")
                  << " output_elements=" << phase.output_elements.ToString()
                  << " flops=" << phase.arithmetic.flops_per_output_element.numerator.ToString()
                  << "/" << phase.arithmetic.flops_per_output_element.denominator << '\n';
      }
    }
    return 0;
  }
  ArithmeticInputs inputs;
  inputs.reduction = QuasiPolynomial::Constant(64);
  inputs.total = QuasiPolynomial::FromIslText("[S] -> { S+3 }");
  inputs.width = 128;
  inputs.dtype = ScalarType::kBF16;
  ParamBinding theta; theta.Bind("S",128);
  std::set<std::string> names;
  int unavailable = 0;
  for (auto const& d : ArithmeticDeclarations()) {
    ValidateArithmeticDeclaration(d);
    if (!names.insert(d.name).second) throw std::runtime_error("duplicate signature");
    auto a = InstantiateArithmetic(d.name,inputs);
    if (a.flops_per_output_element.Eval(theta)<0 ||
        a.transcendental_per_output_element.Eval(theta)<0)
      throw std::runtime_error("negative arithmetic work");
    if (!a.runtime_implemented) {
      bool rejected = false;
      try { RequireArithmeticImplementation(a); }
      catch (std::invalid_argument const&) { rejected=true; }
      if (!rejected) throw std::runtime_error("missing implementation silently accepted");
      ++unavailable;
    }
    std::cout << "OP name=" << d.name << " flops="
              << a.flops_per_output_element.numerator.ToString() << "/"
              << a.flops_per_output_element.denominator << " transc="
              << a.transcendental_per_output_element.numerator.ToString() << "/"
              << a.transcendental_per_output_element.denominator
              << " mma=" << a.flops_use_mma << " staged=" << a.smem_staged
              << " implemented=" << a.runtime_implemented << " reason=" << a.reason
              << " derivation=" << d.derivation << '\n';
  }
  for (char const* required : {"gemm","attention","rmsnorm","rope","silu","mul",
        "add","swiglu","kv_append","sum","softmax","layernorm","gelu_tanh","moe_router",
        "attention_scores","attention_normalize","attention_mac","attention_sum"})
    if (!names.count(required)) throw std::runtime_error(std::string("missing required signature: ")+required);
  auto attention = InstantiateArithmetic("attention",inputs);
  auto mixed=ComposeArithmetic({{InstantiateArithmetic("gemm",inputs),QuasiPolynomial::Constant(128)},
                               {InstantiateArithmetic("add",inputs),QuasiPolynomial::Constant(64)}});
  auto mixed_work=mixed.Eval(theta);
  if (mixed_work.mma!=16384 || mixed_work.simt!=64 || mixed_work.transcendental!=0)
    throw std::runtime_error("mixed GEMM/add signature lost phase output domain or pipe");
  std::cout << "MIXED_ARITHMETIC phases=2 mma=" << mixed_work.mma << " simt=" << mixed_work.simt
            << " per_phase_output_domains=1 status=PASS\n";
  if (attention.flops_per_output_element.Eval(theta)!=4.0*131 ||
      attention.transcendental_per_output_element.Eval(theta)!=131.0/128 || attention.flops_use_mma)
    throw std::runtime_error("attention semantic arithmetic gate");
  inputs.dtype = ScalarType::kF32;
  if (InstantiateArithmetic("gemm",inputs).flops_use_mma)
    throw std::runtime_error("FP32 SIMT signature incorrectly uses MMA");
  int before = context.ReferenceCount(), rejected = 0;
  try { (void)InstantiateArithmetic("missing_operator",inputs); }
  catch (std::invalid_argument const&) { ++rejected; }
  inputs.total.reset();
  try { (void)InstantiateArithmetic("attention",inputs); }
  catch (std::invalid_argument const&) { ++rejected; }
  inputs.width = 0;
  try { (void)InstantiateArithmetic("rmsnorm",inputs); }
  catch (std::invalid_argument const&) { ++rejected; }
  auto invalid = ArithmeticDeclarations().front(); invalid.reason = nullptr;
  try { ValidateArithmeticDeclaration(invalid); }
  catch (std::invalid_argument const&) { ++rejected; }
  try { ComposeArithmetic({}); }
  catch (std::invalid_argument const&) { ++rejected; }
  auto absent=mixed.phases; absent.front().arithmetic.runtime_implemented=false;
  try { ComposeArithmetic(absent); }
  catch (std::invalid_argument const&) { ++rejected; }
  auto negative=mixed; negative.phases.front().output_elements=QuasiPolynomial::Constant(-1);
  try { negative.Eval(theta); }
  catch (std::invalid_argument const&) { ++rejected; }
  if (rejected!=7 || before!=context.ReferenceCount())
    throw std::runtime_error("arithmetic error-path gate");
  std::cout << "OP_ERRORS rejected=" << rejected << " before=" << before
            << " after=" << context.ReferenceCount() << '\n';
  std::cout << "OP_AUDIT declarations=" << names.size() << " schema_failures=0 unavailable_implementations="
            << unavailable << " price_integration=not_implemented\n";
} catch (std::exception const& e) {
  std::cerr << "op-audit: " << e.what() << '\n'; return 2;
}
