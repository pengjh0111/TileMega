// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Codegen/ServingPages.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <algorithm>

namespace tilemega::solver {

unsigned Sm90DenseFamily(frontend::ModelPlan const& plan,frontend::PlanGemm const& g) {
  using Op=frontend::PlanGemm::Epilogue;
  auto const& name=plan.buffers.at(g.b).name;
  if(g.epilogue==Op::kSwiGLU)return 2;
  if(g.epilogue==Op::kArgmaxPartial)return 4;
  if(g.epilogue==Op::kStore && name.find("qkv")!=std::string::npos)return 0;
  if(g.epilogue==Op::kResidual && name.find("o_proj")!=std::string::npos)return 1;
  if(g.epilogue==Op::kResidual && name.find("down_proj")!=std::string::npos)return 3;
  throw std::invalid_argument("unrecognized native dense GEMM: "+name);
}
std::array<std::array<int,2>,5> Sm90DenseShapes(frontend::ModelPlan const& plan) {
  std::array<std::array<int,2>,5> result{};
  if(!plan.serving || plan.dm || plan.forward || plan.serving_seq!=1)
    throw std::invalid_argument("native dense shapes require dense decode");
  for(auto const& g:plan.gemms) {
    auto family=Sm90DenseFamily(plan,g);std::array<int,2> shape={int(g.n),int(g.k)};
    if(result[family][0] && result[family]!=shape)
      throw std::invalid_argument("native GEMM family changes shape between layers");
    result[family]=shape;
  }
  for(auto const& shape:result)if(!shape[0] || !shape[1])
    throw std::invalid_argument("native dense model lacks a required GEMM family");
  return result;
}
frontend::ImportedSemantics ImportSm90Dense(std::string const& path,
    Sm90DenseCoordinates const& c,mlir::MLIRContext& context) {
  auto bridge=frontend::ReadExportBridge(path);
  frontend::ServingOptions options;
  options.interleave_u=c.swiglu_u;options.dn_vector_sums=c.dn_vector_sums;
  options.argmax_tile_n=c.gemms[4].tile_n;
  options.dense_attention_chunks=c.attention.chunks;
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  return frontend::TorchExportImporter{}.ImportSemantics(path,plan,context);
}
mlir::OwningOpRef<mlir::ModuleOp> MaterializeSm90Dense(
    frontend::ImportedSemantics const& imported,Sm90DenseCoordinates const& c,int batch,
    TargetSpec const& target,mlir::MLIRContext& context,std::string const& origin) {
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" ||
     target.res.num_sms<=0 || target.res.max_smem_per_sm<=0 ||
     target.res.max_dynamic_smem_per_cta<=0 || (origin!="fixed" && origin!="solver"))
    throw std::invalid_argument("native dense materialization requires a complete sm_90a target");
  auto const& plan=imported.plan;
  c.Validate(Sm90DenseShapes(plan),batch);
  if(plan.dn_vector_sums!=c.dn_vector_sums)
    throw std::invalid_argument("native structural coordinates require a fresh model plan");
  auto budget=std::min(target.res.max_dynamic_smem_per_cta,
                       target.res.max_smem_per_sm/c.residency);
  frontend::ImportOptions options;
  for(auto const& g:plan.gemms) {
    auto const& cfg=c.gemms[Sm90DenseFamily(plan,g)];
    if(cfg.SharedBytes()>budget)
      throw std::invalid_argument("native GEMM exceeds target shared-memory budget");
    if(cfg.mma!=Sm90DenseMma::kMmaSync || cfg.preissue_smem_bytes || cfg.prefetch_l2_bytes)
      throw std::invalid_argument("requested native mechanism is not implemented yet");
    options.gemms.push_back({16,cfg.tile_n,cfg.TileK(),cfg.slots,1,kSm90DenseGemmImpl});
  }
  for(auto const& stage:plan.stages)if(stage.kind==frontend::PlanTaskKind::kFusedAttention) {
    if(stage.dense_attention_chunks!=c.attention.chunks ||
       c.attention.SharedBytes(batch,stage.width,stage.group,plan.serving_capacity)>budget)
      throw std::invalid_argument("native attention structure or shared-memory budget is invalid");
  }
  if(c.attention.preissue_smem_bytes || c.attention.prefetch_l2_bytes)
    throw std::invalid_argument("native attention preissue is not implemented yet");
  auto module=frontend::TorchExportImporter{}.InstantiateForGranularity(imported,context,options);
  mlir::OpBuilder b(&context);
  auto set_int=[&](char const* name,int value) {(*module)->setAttr(name,b.getI64IntegerAttr(value));};
  (*module)->setAttr("tmexec.sm90_dense",b.getStringAttr(c.ToJson().Dump(0)));
  (*module)->setAttr("tmexec.sm90_dense_origin",b.getStringAttr(origin));
  (*module)->setAttr("tmexec.sm90_dense_target",b.getStringAttr(target.ToJson()));
  (*module)->setAttr("tmexec.solved_arch",b.getStringAttr("sm_90"));
  set_int("tmexec.sm90_dense_batch",batch);
  set_int("tmexec.sm90_dense_grid",target.res.num_sms*c.residency);
  set_int("tmexec.solved_residency",c.residency);
  auto roles=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.dimension_roles");
  auto role=roles?roles.getAs<mlir::StringAttr>("batch"):mlir::StringAttr{};
  if(!role || role.getValue().empty())throw std::invalid_argument("native dense CTG has no batch symbol");
  mlir::NamedAttrList theta((*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.theta"));
  theta.set(role.getValue(),b.getI64IntegerAttr(batch));
  (*module)->setAttr("tilemega.theta",theta.getDictionary(&context));
  (*module)->setAttr("tmexec.nonpaged_weight_layout_tiled",b.getBoolAttr(true));
  codegen::ResolveServingWeightPacking(*module);
  if(mlir::failed(mlir::verify(*module)))throw std::invalid_argument("native dense CTG verification failed");
  return module;
}
}  // namespace tilemega::solver
