// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Target/TargetSpec.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <cassert>
#include <stdexcept>

int main() {
  mlir::MLIRContext context;
  mlir::OpBuilder b(&context);
  auto module=mlir::ModuleOp::create(b.getUnknownLoc());
  module->setAttr("tilemega.dimension_roles",b.getDictionaryAttr({}));
  module->setAttr("tilemega.serving",b.getDictionaryAttr({b.getNamedAttr("seq",b.getI64IntegerAttr(1))}));
  module->setAttr("tilemega.model_plan",b.getDictionaryAttr({
      b.getNamedAttr("stages",b.getArrayAttr({b.getDictionaryAttr({
          b.getNamedAttr("kind",b.getStringAttr("kGemm"))})})),
      b.getNamedAttr("buffers",b.getArrayAttr({}))}));
  module->setAttr("tmexec.solved_grid",b.getI64IntegerAttr(264));
  tilemega::TargetSpec target;
  target.res.num_sms=132;target.res.l2_bytes=52428800;
  auto config=[&]{return module->getAttrOfType<mlir::DictionaryAttr>("tmexec.prefetch");};
  tilemega::codegen::ConfigureServingPrefetch(module,target,1,128);
  assert(config().getAs<mlir::IntegerAttr>("worker_bytes").getInt()==52428800/528);
  tilemega::codegen::ConfigureServingPrefetch(module,target,1,64,65536,false);
  assert(config().getAs<mlir::IntegerAttr>("worker_bytes").getInt()==65536);
  assert(!config().getAs<mlir::BoolAttr>("history_enabled").getValue());
  for(int bytes:{-16,17,524304}) {
    bool rejected=false;
    try{tilemega::codegen::ConfigureServingPrefetch(module,target,1,128,bytes);}
    catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);
  }
  tilemega::codegen::ConfigureServingPrefetch(module,target,0,128);
  assert(!config());
}
