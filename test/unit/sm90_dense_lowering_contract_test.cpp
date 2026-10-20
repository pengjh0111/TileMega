// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <cassert>
#include <stdexcept>
#include <string>

int main() {
  mlir::MLIRContext context;mlir::OpBuilder b(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module=mlir::ModuleOp::create(b.getUnknownLoc());
  (*module)->setAttr("tilemega.model_plan",b.getDictionaryAttr({
      b.getNamedAttr("dtype",b.getStringAttr("bf16"))}));
  (*module)->setAttr("tilemega.serving",b.getDictionaryAttr({
      b.getNamedAttr("seq",b.getI64IntegerAttr(1))}));
  (*module)->setAttr("tmexec.solved_arch",b.getStringAttr("sm_90"));
  tilemega::codegen::CouplingGraphToCUDA emitter;
  bool single=false,variants=false;
  try {(void)emitter.Lower(*module);}
  catch(std::invalid_argument const& e) {single=std::string(e.what()).find("native CTG/tmexec")!=std::string::npos;}
  try {(void)emitter.LowerVariants({{*module,1,1}});}
  catch(std::invalid_argument const& e) {variants=std::string(e.what()).find("native CTG/tmexec")!=std::string::npos;}
  assert(single && variants);
}
