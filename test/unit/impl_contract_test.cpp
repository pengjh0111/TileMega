// SPDX-License-Identifier: BSD-3-Clause
// Part 5: an implementation contract is only worth having if a wrong one is
// rejected, so every check here is paired with the mutation it must catch.
#include <tilemega/Dialect/CouplingGraph/CGContract.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Solver/ImplementationContract.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tilemega;
using analysis::ClosedForm;

#define REQUIRE(condition)                                                  \
  do {                                                                      \
    if (!(condition)) {                                                     \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace {

analysis::TensorAxis Ax(std::string name, ClosedForm extent) {
  analysis::TensorAxis axis;
  axis.name = std::move(name);
  axis.extent = std::move(extent);
  return axis;
}

/// The attention score matmul of §2.7, which is where a transposed operand
/// actually costs something: Q is (h, q, d) and K is (h, kv, d), and the two
/// tiled axes are both 128 wide, so a swap changes no extent at all.
analysis::OperatorNode ScoreMatmul() {
  analysis::OperatorNode node;
  node.name = "score";
  node.kind = analysis::OperatorKind::kMatmul;
  node.output.name = "score";
  node.output.axes = {Ax("h", ClosedForm::Symbol("n_h")),
                      Ax("q", ClosedForm::Symbol("S")),
                      Ax("kv", ClosedForm::Symbol("L_s"))};
  node.tile = {ClosedForm::Constant(1), ClosedForm::Symbol("Tm"),
               ClosedForm::Symbol("Tn")};

  analysis::Operand query;
  query.producer = "q_rope";
  query.tensor.name = "q";
  query.tensor.axes = {Ax("h", ClosedForm::Symbol("n_h")),
                       Ax("q", ClosedForm::Symbol("S")),
                       Ax("d", ClosedForm::Symbol("d"))};
  query.axes = {analysis::OperandAxisMap::Indexed(0),
                analysis::OperandAxisMap::Indexed(1),
                analysis::OperandAxisMap::FullRange()};

  analysis::Operand key;
  key.producer = "k_cache";
  key.tensor.name = "k";
  key.tensor.axes = {Ax("h", ClosedForm::Symbol("n_kv")),
                     Ax("kv", ClosedForm::Symbol("L_s")),
                     Ax("d", ClosedForm::Symbol("d"))};
  key.axes = {analysis::OperandAxisMap::Indexed(0, ClosedForm::Constant(1),
                                                ClosedForm::Symbol("G")),
              analysis::OperandAxisMap::Indexed(2),
              analysis::OperandAxisMap::FullRange()};

  node.operands = {std::move(query), std::move(key)};
  return node;
}

std::vector<analysis::AccessRelation> ReadMaps(analysis::OperatorNode const& node) {
  std::vector<analysis::AccessRelation> result;
  for (std::size_t i = 0; i < node.operands.size(); ++i)
    result.push_back(analysis::BuildReadMap(node, i));
  return result;
}

void Report(char const* label, bool ok, std::string const& error) {
  std::printf("%-28s %s%s%s\n", label, ok ? "accepted" : "rejected",
              error.empty() ? "" : ": ", error.c_str());
}

}  // namespace

int main() {
  analysis::OperatorNode node = ScoreMatmul();
  std::vector<analysis::AccessRelation> derived = ReadMaps(node);

  analysis::ParamBinding theta;
  theta.Bind("S", 4).Bind("L_s", 512).Bind("d", 128).Bind("n_h", 32)
       .Bind("n_kv", 8).Bind("G", 4);

  TargetSpec target = TargetSpec::Probe();
  solver::CandidateGenerator generator(target);
  solver::TileCandidate g;
  solver::ImplementationContract impl;
  analysis::ParamBinding binding;
  // §4.3: one decision, not two -- the chosen tile is what binds the symbols
  // the access maps are written in.
  REQUIRE(solver::SelectImplementation(generator, {4, 512, 128}, "impl_qkv_17",
                                       "score", derived, theta, {}, &g, &impl,
                                       &binding));
  std::printf("selected g: %dx%dx%d s%d, impl %s threads=%d smem=%d arch=%d\n",
              g.m, g.n, g.k, g.stages, impl.backend.c_str(), impl.threads,
              impl.smem_bytes, impl.arch_required);
  for (auto const& operand : impl.operands) {
    std::printf("  operand %-8s", operand.name.c_str());
    for (std::size_t i = 0; i < operand.coordinate.size(); ++i)
      std::printf(" %s:%d",
                  operand.coordinate[i].empty() ? "*"
                                                : operand.coordinate[i].c_str(),
                  operand.span[i]);
    std::printf("\n");
  }

  std::string error;
  REQUIRE(solver::VerifyContract(impl, target, derived, theta, binding, &error));
  Report("selected contract", true, "");

  // The transposition negative is run at the §2.7 granularity (Tm = d = 128),
  // where the two axes a swap exchanges are the same width. That is the hard
  // case: no extent changes, so only the coordinate check can catch it.
  analysis::ParamBinding table27;
  table27.Bind("Tm", 128).Bind("Tn", 128).Bind("Tk", 16);
  solver::ImplementationContract control = impl;
  control.name = "impl_qkv_17_ctl";
  control.tile_m = control.tile_n = 128;
  control.tile_k = 16;
  control.stages = 3;
  control.smem_bytes = solver::SimtF32SmemBytes(128, 128, 16, 3);
  REQUIRE(solver::LowerAccess(derived, theta, table27, &control.operands, &error));
  REQUIRE(solver::VerifyContract(control, target, derived, theta, table27, &error));
  REQUIRE(control.operands[0].span[1] == control.operands[0].span[2]);

  solver::ImplementationContract transposed = control;
  std::swap(transposed.operands[0].coordinate[1],
            transposed.operands[0].coordinate[2]);
  REQUIRE(transposed.operands[0].span == control.operands[0].span);
  error.clear();
  REQUIRE(!solver::VerifyAccess(transposed, derived, theta, table27, &error));
  Report("transposed operand", false, error);

  solver::ImplementationContract mistiled = control;
  mistiled.name = "impl_qkv_17_mistiled";
  mistiled.operands[1].span[1] /= 2;
  error.clear();
  REQUIRE(!solver::VerifyAccess(mistiled, derived, theta, table27, &error));
  Report("tile shape mismatch", false, error);

  solver::ImplementationContract mispriced = impl;
  mispriced.smem_bytes -= 4;
  error.clear();
  REQUIRE(!solver::VerifyTraits(mispriced, &error));
  Report("restated smem", false, error);

  solver::ImplementationContract wrong_arity = impl;
  wrong_arity.operands.pop_back();
  error.clear();
  REQUIRE(!solver::VerifyAccess(wrong_arity, derived, theta, binding, &error));
  Report("missing operand", false, error);

  // The same three checks through the IR, which is where a hand-written or
  // pass-mutated contract actually reaches the verifier.
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  mlir::ArrayAttr index_map;
  REQUIRE(dialect::emitIndexMap(builder, derived, theta, table27, &index_map,
                                &error));
  mlir::OperationState space(builder.getUnknownLoc(), "tilemega.task_space");
  space.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr("score"));
  space.addAttribute("kind", dialect::TaskKindAttr::get(
                                 &context, builder.getStringAttr("gemm")));
  space.addAttribute("granularity", builder.getDictionaryAttr({}));
  space.addAttribute("write_map",
                     dialect::AccessMapAttr::get(
                         &context, builder.getDictionaryAttr(
                                       {builder.getNamedAttr(
                                           "kind", builder.getStringAttr("gemm"))})));
  space.addAttribute("stage", builder.getI64IntegerAttr(0));
  space.addAttribute("operator_name", builder.getStringAttr("aten.matmul.default"));
  space.addAttribute("index_map", index_map);
  builder.create(space);

  dialect::ImplementationOp good =
      dialect::emitImplementation(builder, builder.getUnknownLoc(), control);
  REQUIRE(mlir::succeeded(mlir::verify(module)));
  Report("IR: selected contract", true, "");

  // Same mutation, now applied to the attribute the op actually carries.
  dialect::ImplementationOp bad =
      dialect::emitImplementation(builder, builder.getUnknownLoc(), transposed);
  bad.setSymName("impl_qkv_17_transposed");
  REQUIRE(mlir::failed(mlir::verify(module)));
  Report("IR: transposed operand", false, "verifier rejected the module");
  bad.erase();
  REQUIRE(mlir::succeeded(mlir::verify(module)));

  dialect::ImplementationOp mistiled_op =
      dialect::emitImplementation(builder, builder.getUnknownLoc(), mistiled);
  mistiled_op.setSymName("impl_qkv_17_mistiled");
  REQUIRE(mlir::failed(mlir::verify(module)));
  Report("IR: tile shape mismatch", false, "verifier rejected the module");
  mistiled_op.erase();
  REQUIRE(mlir::succeeded(mlir::verify(module)));
  (void)good;

  std::printf("impl_contract: all checks behaved as required\n");
  return 0;
}
