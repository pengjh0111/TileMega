// Part 1.2 probe: build an mlir::ModuleOp, an isl_map, and a barvinok
// isl_pw_qpolynomial cardinality, all in one process/binary, to test whether
// MLIR (its own bundled Presburger implementation) and isl/barvinok
// (GMP-backed) conflict when linked together.
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Analysis/Presburger/PresburgerSpace.h>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/val.h>
#include <isl/printer.h>
#include <barvinok/isl.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

// Exercises mlir::MLIRContext + mlir::ModuleOp + the LLVM-bundled Presburger
// IntegerRelation type, so both bignum implementations are actually live in
// the process, not just link-present.
bool ProbeMlir() {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  if (!module) return false;

  // MLIR's own Presburger implementation (LLVM APInt-backed, no GMP): build
  // { [i] : 0 <= i < 10 } and check it is non-empty, to prove this path is
  // genuinely exercised alongside isl's.
  mlir::presburger::PresburgerSpace space =
      mlir::presburger::PresburgerSpace::getSetSpace(/*numDims=*/1);
  mlir::presburger::IntegerRelation relation(space);
  // i >= 0
  {
    llvm::SmallVector<int64_t, 4> coeffs = {1, 0};
    relation.addInequality(coeffs);
  }
  // 10 - i - 1 >= 0  (i <= 9)
  {
    llvm::SmallVector<int64_t, 4> coeffs = {-1, 9};
    relation.addInequality(coeffs);
  }
  bool empty = relation.isIntegerEmpty();
  std::fprintf(stderr, "[mlir-presburger] { [i] : 0<=i<10 } isIntegerEmpty=%d\n",
              empty);
  module.erase();
  return !empty;
}

// Exercises isl (barvinok's dependency) directly, and barvinok's counting
// entry point isl_set_card, to make sure the isl/barvinok half is genuinely
// alive too, not just linked.
bool ProbeIslBarvinok() {
  isl_ctx *ctx = isl_ctx_alloc();
  if (!ctx) return false;

  isl_set *set = isl_set_read_from_str(
      ctx, "[n] -> { [i] : 0 <= i < n }");
  isl_pw_qpolynomial *card = isl_set_card(isl_set_copy(set));

  isl_printer *printer = isl_printer_to_str(ctx);
  printer = isl_printer_print_pw_qpolynomial(printer, card);
  char *text = isl_printer_get_str(printer);
  std::string result = text ? text : "";
  std::fprintf(stderr, "[isl+barvinok] card([n]->{[i]:0<=i<n}) = %s\n",
              result.c_str());
  free(text);
  isl_printer_free(printer);
  isl_pw_qpolynomial_free(card);
  isl_set_free(set);

  // The I2 substitutability primitive we actually need: isl_set_is_subset.
  isl_set *narrow = isl_set_read_from_str(ctx, "{ [i] : 0 <= i < 5 }");
  isl_set *wide = isl_set_read_from_str(ctx, "{ [i] : 0 <= i < 10 }");
  isl_bool contains = isl_set_is_subset(narrow, wide);
  std::fprintf(stderr, "[isl] {[i]:0<=i<5} subset of {[i]:0<=i<10} = %d\n",
              static_cast<int>(contains));
  isl_set_free(narrow);
  isl_set_free(wide);

  isl_ctx_free(ctx);
  // Expect "n" as the cardinality and a true subset check.
  return result.find('n') != std::string::npos && contains == isl_bool_true;
}

}  // namespace

int main() {
  bool mlir_ok = ProbeMlir();
  bool isl_ok = ProbeIslBarvinok();
  std::fprintf(stderr, "RESULT mlir_ok=%d isl_barvinok_ok=%d\n", mlir_ok,
              isl_ok);
  return (mlir_ok && isl_ok) ? 0 : 1;
}
