// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/PageLayout.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <algorithm>
#include <stdexcept>
namespace tilemega::codegen {
void ConfigureServingPages(mlir::ModuleOp module,TargetSpec const& target,int page_bytes) {
  auto serving=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
  auto model=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if(!serving || !model || mlir::cast<mlir::IntegerAttr>(serving.get("seq")).getInt()!=1)
    throw std::invalid_argument("page execution requires a decode serving graph");
  auto runtime=ReadRuntimePlan(module);int activation=0,scratch=0;
  for(auto const& g:runtime.gemms) {
    if(!solver::PageLayout::StageFits(page_bytes,g.tile_n,g.tile_k))
      throw std::invalid_argument("a GEMM B stage must divide a page or occupy whole pages");
    activation=std::max(activation,4*g.tile_m*g.tile_k);
    scratch=std::max(scratch,4*g.tile_m*g.tile_n);
  }
  for(auto a:mlir::cast<mlir::ArrayAttr>(model.get("stages"))) {
    auto stage=mlir::cast<mlir::DictionaryAttr>(a);
    if(mlir::cast<mlir::StringAttr>(stage.get("kind")).getValue()=="kFusedAttention") {
      int d=mlir::cast<mlir::IntegerAttr>(stage.get("width")).getInt();
      // Query, four warp output fragments and four per-row LSEs.
      scratch=std::max(scratch,16*d*2+4*16*d*4+4*16*4);
    }
  }
  auto layout=solver::PageLayout::Build(target,page_bytes,activation,scratch);
  for(auto const& g:runtime.gemms)
    if(g.tile_n*g.tile_k*2>page_bytes*layout.pages)
      throw std::invalid_argument("a B stage exceeds the available page pool");
  if(auto grid=module->getAttrOfType<mlir::IntegerAttr>("tmexec.solved_grid"))
    if(grid.getInt()>target.res.num_sms)
      throw std::invalid_argument("paged execution requires at most one CTA per SM");
  mlir::OpBuilder b(module.getContext());mlir::NamedAttrList values;
  for(auto [key,value]:std::initializer_list<std::pair<char const*,int>>{
      {"page_bytes",layout.page_bytes},{"pages",layout.pages},
      {"workspace_offset",layout.activation_offset},{"pool_offset",layout.pages_offset},
      {"shared_bytes",layout.shared_bytes},{"activation_bytes",activation},{"scratch_bytes",scratch},
      {"threads",160}})values.set(key,b.getI64IntegerAttr(value));
  module->setAttr("tmexec.pages",values.getDictionary(module.getContext()));
}
}
