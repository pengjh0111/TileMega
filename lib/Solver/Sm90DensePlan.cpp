// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <algorithm>
#include <limits>
#include <set>

namespace tilemega::solver {
namespace {
void NativeBoundaryContracts(mlir::ModuleOp module,bool write) {
  mlir::OpBuilder b(module.getContext());
  auto integer=[](mlir::DictionaryAttr attr,llvm::StringRef key) {
    auto value=attr.getAs<mlir::IntegerAttr>(key);
    if(!value)throw std::invalid_argument("native boundary lacks integer "+key.str());
    return int(value.getInt());
  };
  auto string=[](mlir::DictionaryAttr attr,llvm::StringRef key) {
    auto value=attr.getAs<mlir::StringAttr>(key);
    if(!value)throw std::invalid_argument("native boundary lacks string "+key.str());
    return value.getValue().str();
  };
  auto c=Sm90DenseCoordinates::FromJson(json::Parse(
      module->getAttrOfType<mlir::StringAttr>("tmexec.sm90_dense").getValue().str()));
  int grid=module->getAttrOfType<mlir::IntegerAttr>("tmexec.sm90_dense_grid").getInt();
  int batch=module->getAttrOfType<mlir::IntegerAttr>("tmexec.sm90_dense_batch").getInt();
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto stages=plan.getAs<mlir::ArrayAttr>("stages"),gemms=plan.getAs<mlir::ArrayAttr>("gemms");
  struct Stage {std::string kind,epilogue,implementation;int n=0,groups=0;};
  std::vector<Stage> descriptions;
  int experts=0,gu_tiles=0;
  for(auto value:stages) {
    auto s=mlir::cast<mlir::DictionaryAttr>(value);Stage d;d.kind=string(s,"kind");
    if(d.kind=="kGemm") {
      auto g=mlir::cast<mlir::DictionaryAttr>(gemms[integer(s,"gemm")]);
      d.epilogue=g.get("epilogue")?string(g,"epilogue"):"store";
      d.n=integer(g,"n");d.implementation="mma_sync/"+d.epilogue;
      if(d.epilogue=="moe_router") {
        experts=d.n;
        if(c.GlobalRoute())d.implementation+=std::string("/route-")+c.RouteName();
      }
      if(d.epilogue=="moe_gate_up")gu_tiles=d.n/c.gemms[6].tile_n;
      if(d.epilogue=="moe_down" && c.combine_stage)d.implementation+="/combine-stage";
    }else if(d.kind=="kFusedAttention") {
      d.groups=integer(s,"extent");
      d.implementation="attention/chunks="+std::to_string(c.attention.chunks)+
          (c.attention.dsmem?"/dsmem":c.attention.merge_stage?"/stage":"/global");
    }else if(d.kind=="kArgmaxReduce")d.implementation=c.argmax_stage?"argmax/stage":"argmax/handoff";
    else if(d.kind=="kEmbedding")d.implementation=c.embed_la?"embedding/handoff":"embedding/stage";
    else throw std::invalid_argument("native boundary has an unsupported stage kind");
    descriptions.push_back(std::move(d));
  }
  auto attribute=[&](mlir::Operation* op,llvm::StringRef key,mlir::Attribute expected) {
    if(write)op->setAttr(key,expected);
    else if(op->getAttr(key)!=expected)
      throw std::invalid_argument("native boundary contract mismatch: "+key.str());
  };
  std::map<std::string,int> task_stage;
  module.walk([&](dialect::TileSpaceOp task) {
    int stage=task.getStage();
    auto const& d=descriptions.at(stage);task_stage.emplace(task.getSymName().str(),stage);
    attribute(task,"tmexec.implementation",b.getStringAttr(d.implementation));
    attribute(task,"tmexec.placement",b.getStringAttr("template"));
    if(d.kind=="kFusedAttention" && c.attention.chunks>1) {
      attribute(task,c.attention.merge_stage?"tmexec.merge_event_extent":"tmexec.handoff_event_extent",
          b.getI64IntegerAttr(c.attention.merge_stage?1:batch*d.groups));
      attribute(task,c.attention.merge_stage?"tmexec.merge_required_count":"tmexec.handoff_required_count",
          b.getI64IntegerAttr(c.attention.merge_stage?grid:c.attention.chunks));
    }
    if(d.epilogue=="moe_router" && c.GlobalRoute()) {
      attribute(task,"tmexec.route_implementation",b.getStringAttr(c.RouteName()));
      attribute(task,"tmexec.route_event_extent",b.getI64IntegerAttr(1));
      attribute(task,"tmexec.route_required_count",b.getI64IntegerAttr(
          c.route==Sm90DenseCoordinates::Route::kStage?grid:experts/c.gemms[5].tile_n));
    }
    if(d.epilogue=="moe_down" && c.combine_stage) {
      attribute(task,"tmexec.combine_implementation",b.getStringAttr("stage"));
      attribute(task,"tmexec.combine_event_extent",b.getI64IntegerAttr(1));
      attribute(task,"tmexec.combine_required_count",b.getI64IntegerAttr(grid));
    }
    if(d.epilogue=="residual_ar")
      attribute(task,"tmexec.data_world",b.getI64IntegerAttr(c.tp));
  });
  module.walk([&](dialect::CouplingOp edge) {
    auto const& p=descriptions.at(task_stage.at(edge.getSrc().str()));
    auto const& q=descriptions.at(task_stage.at(edge.getDst().str()));
    std::string kind="aggregate";int extent=1,required=grid;
    if(q.kind=="kFusedAttention" && p.epilogue=="store" && c.group_wait) {
      kind="group";extent=q.groups;required=p.n/c.gemms[0].tile_n/q.groups;
    }else if(q.kind=="kArgmaxReduce" && !c.argmax_stage)kind="handoff";
    else if(p.epilogue=="moe_gate_up" && q.epilogue=="moe_down" && c.expert_fuse) {
      kind="counted";extent=experts;required=gu_tiles;
    }
    // V3 joins CTAs, group/count joins completed producer tiles. TP's data
    // exchange is inside the producer task and leaves this V3 join intact.
    attribute(edge,"tmexec.sync_kind",b.getStringAttr(kind));
    attribute(edge,"tmexec.event_extent",b.getI64IntegerAttr(extent));
    attribute(edge,"tmexec.required_count",b.getI64IntegerAttr(required));
  });
}
}

void ValidateSm90NativeBoundaryContracts(mlir::ModuleOp module) {
  auto version=module->getAttrOfType<mlir::IntegerAttr>("tmexec.native_contract_version");
  if(!version)return; // Artifacts preceding per-edge WriteBack remain readable.
  if(version.getInt()!=1)throw std::invalid_argument("unsupported native boundary contract version");
  NativeBoundaryContracts(module,false);
}

unsigned Sm90DenseFamily(frontend::ModelPlan const& plan,frontend::PlanGemm const& g) {
  using Op=frontend::PlanGemm::Epilogue;
  auto const& name=plan.buffers.at(g.b).name;
  if(g.epilogue==Op::kMoeRouter)return 5;
  if(g.epilogue==Op::kMoeGateUp)return 6;
  if(g.epilogue==Op::kMoeDown)return 7;
  if(g.epilogue==Op::kSwiGLU)return 2;
  if(g.epilogue==Op::kArgmaxPartial)return 4;
  if(g.epilogue==Op::kStore && name.find("qkv")!=std::string::npos)return 0;
  // residual_ar is the tensor parallel form of the same two families.
  bool residual=g.epilogue==Op::kResidual || g.epilogue==Op::kResidualAllReduce;
  if(residual && name.find("o_proj")!=std::string::npos)return 1;
  if(residual && name.find("down_proj")!=std::string::npos)return 3;
  throw std::invalid_argument("unrecognized native dense GEMM: "+name);
}
Sm90DenseShapeTable Sm90DenseShapes(frontend::ModelPlan const& plan) {
  Sm90DenseShapeTable result{};
  if(!plan.serving || plan.dm || plan.forward || plan.serving_seq!=1)
    throw std::invalid_argument("native dense shapes require dense decode");
  bool moe=std::any_of(plan.gemms.begin(),plan.gemms.end(),
      [](auto const& g) {return g.epilogue==frontend::PlanGemm::Epilogue::kMoeRouter;});
  for(auto const& g:plan.gemms) {
    auto family=Sm90DenseFamily(plan,g);std::array<int,2> shape={int(g.n),int(g.k)};
    if(result[family][0] && result[family]!=shape)
      throw std::invalid_argument("native GEMM family changes shape between layers");
    result[family]=shape;
  }
  for(unsigned f=0;f<result.size();++f) {
    auto active=Sm90DenseActiveFamilies(moe);
    bool used=std::find(active.begin(),active.end(),f)!=active.end();
    if(used!=(result[f][0] && result[f][1]))
      throw std::invalid_argument("native model GEMM families differ from its dense/MoE kind");
  }
  return result;
}
Sm90DenseWorkload Sm90DenseWorkloadFromPlan(frontend::ModelPlan const& plan,int batch) {
  using Kind=frontend::PlanTaskKind;
  constexpr auto absent=std::numeric_limits<std::uint32_t>::max();
  Sm90DenseWorkload work;
  work.moe=std::any_of(plan.gemms.begin(),plan.gemms.end(),
      [](auto const& g) {return g.epilogue==frontend::PlanGemm::Epilogue::kMoeRouter;});
  // Dense: qkv, attention, o, gate_up, down; MoE: qkv, attention, o, router,
  // expert gate_up, expert down (prompt §8).
  unsigned per_layer=work.moe?6:5;
  if(!plan.serving || plan.dm || plan.forward || plan.serving_seq!=1 || plan.serving_capacity!=1088 ||
      (batch!=1 && batch!=2 && batch!=4 && batch!=8 && batch!=16) ||
      plan.stages.size()<3+per_layer || (plan.stages.size()-3)%per_layer)
    throw std::invalid_argument("native workload requires the supported decode stage sequence");
  work.batch=batch;work.layers=int((plan.stages.size()-3)/per_layer);
  // Set before the per-layer checks: the attention extent rule depends on it.
  work.tp=plan.serving_tp;
  if(plan.gemms.size()!=std::size_t((per_layer-1)*work.layers+1) ||
      plan.stages.front().kind!=Kind::kEmbedding || plan.stages.back().kind!=Kind::kArgmaxReduce)
    throw std::invalid_argument("native workload GEMM/stage count differs");
  auto operand=[&](std::uint32_t id,bool optional) {
    if(optional && id==absent)return false;
    if(id>=plan.buffers.size())throw std::invalid_argument("native workload operand ID outside buffer table");
    return true;
  };
  std::set<unsigned> seen;
  work.gemm_bindings.emplace();
  auto gemm=[&](unsigned stage,unsigned family) {
    auto const& s=plan.stages.at(stage);
    if(s.kind!=Kind::kGemm || s.gemm>=plan.gemms.size() || !seen.insert(s.gemm).second)
      throw std::invalid_argument("native workload has reordered, repeated or invalid GEMM stages");
    auto const& g=plan.gemms[s.gemm];
    if(family==5)work.softmax_all=g.moe_softmax_all;
    for(auto id:{g.a,g.b,g.c})operand(id,false);
    operand(g.d,g.epilogue!=frontend::PlanGemm::Epilogue::kResidual &&
                g.epilogue!=frontend::PlanGemm::Epilogue::kResidualAllReduce);
    for(unsigned i=0;i<4;++i)if(family>=5)operand(s.operands[i],i>0);
    if(!g.n || !g.k || g.n>std::uint32_t(std::numeric_limits<int>::max()) ||
        g.k>std::uint32_t(std::numeric_limits<int>::max()) || Sm90DenseFamily(plan,g)!=family)
      throw std::invalid_argument("native workload family/shape order differs");
    Sm90DenseStagePlanBinding binding;
    switch(g.epilogue) {
      case frontend::PlanGemm::Epilogue::kStore:binding.epilogue="store";break;
      case frontend::PlanGemm::Epilogue::kResidual:binding.epilogue="residual";break;
      case frontend::PlanGemm::Epilogue::kSwiGLU:binding.epilogue="swiglu";break;
      case frontend::PlanGemm::Epilogue::kArgmaxPartial:binding.epilogue="argmax_partial";break;
      case frontend::PlanGemm::Epilogue::kMoeRouter:binding.epilogue="moe_router";break;
      case frontend::PlanGemm::Epilogue::kMoeGateUp:binding.epilogue="moe_gate_up";break;
      case frontend::PlanGemm::Epilogue::kMoeDown:binding.epilogue="moe_down";break;
      case frontend::PlanGemm::Epilogue::kResidualAllReduce:binding.epilogue="residual_ar";break;
      default:throw std::invalid_argument("unknown native workload epilogue");
    }
    binding.input_norm=operand(g.norm_ss,true);binding.output_sums=operand(g.ss_out,true);
    work.gemm_bindings->push_back({int(stage),int(s.gemm),kSm90DenseFamilies[family],int(g.n),int(g.k),binding,g.norm_ss,g.ss_out});
  };
  for(int layer=0;layer<work.layers;++layer) {
    unsigned start=1+per_layer*layer;gemm(start,0);
    auto const& a=plan.stages.at(start+1);
    // Hkv is the model's, divided by the tensor parallel degree: 8 dense, 4 for
    // Qwen3-30B-A3B, 16 for OLMoE (MHA, Q = 1). A sharded dense model reaches
    // 4, 2 or 1 KV heads per rank, which the attention body already supports.
    bool extent_ok=a.extent==8 || (work.moe && (a.extent==4 || a.extent==16)) ||
        (!work.moe && work.tp>1 && (a.extent==1 || a.extent==2 || a.extent==4));
    if(a.kind!=Kind::kFusedAttention || !extent_ok ||
        (a.width!=64 && a.width!=128) ||
        a.group<1 || a.group>(work.moe?8:4) || a.attention_kv_block!=plan.serving_capacity ||
        a.dense_attention_chunks<1 || a.dense_attention_chunks>64)
      throw std::invalid_argument("native workload attention structure differs");
    for(unsigned i:{0u,1u,2u,3u,4u,7u,8u,9u,10u})operand(a.operands[i],false);
    bool q=operand(a.operands[5],true),k=operand(a.operands[6],true);
    if(q!=k)throw std::invalid_argument("native workload has asymmetric Q/K normalization");
    // The norm mode follows the weights' widths, as codegen reads it: one head
    // (per head) or the whole projected row (OLMoE, before the heads split).
    bool row=false;
    if(q) {
      auto elements=[&](std::uint32_t id) {
        return std::int64_t(plan.buffers[id].constant)+plan.buffers[id].per_seq;
      };
      auto qe=elements(a.operands[5]),ke=elements(a.operands[6]);
      row=qe==std::int64_t(a.extent)*a.group*a.width && ke==std::int64_t(a.extent)*a.width;
      if(!row && (qe!=a.width || ke!=a.width))
        throw std::invalid_argument("native workload Q/K norm is neither per head nor per projected row");
    }
    if(!layer) {work.head_dim=a.width;work.queries=a.group;work.qk_norm=q;work.qk_norm_row=row;work.hkv=a.extent;}
    else if(work.head_dim!=int(a.width) || work.queries!=int(a.group) || work.qk_norm!=q ||
            work.qk_norm_row!=row)
      throw std::invalid_argument("native workload attention tuple changes between layers");
    if(work.moe) {gemm(start+2,1);gemm(start+3,5);gemm(start+4,6);gemm(start+5,7);}
    else {gemm(start+2,1);gemm(start+3,2);gemm(start+4,3);}
  }
  gemm(unsigned(plan.stages.size()-2),4);
  work.shapes=Sm90DenseShapes(plan);
  if(work.moe) {
    // The router's N is the expert count; top-k is fixed by the plan (§8).
    work.experts=work.shapes[5][0];work.top_k=8;
  }
  return work;
}
frontend::ImportedSemantics ImportSm90Dense(std::string const& path,
    Sm90DenseCoordinates const& c,mlir::MLIRContext& context) {
  auto bridge=frontend::ReadExportBridge(path);
  frontend::ServingOptions options;
  options.interleave_u=c.swiglu_u;options.native_rms=true;options.native_shared_scratch=true;
  options.argmax_tile_n=c.gemms[4].tile_n;
  options.dense_attention_chunks=c.attention.chunks;
  options.tp=c.tp;
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
  auto budget=Sm90DenseSharedBudget(target,c.residency);
  frontend::ImportOptions options;
  for(auto const& g:plan.gemms) {
    auto const& cfg=c.gemms[Sm90DenseFamily(plan,g)];
    if(cfg.SharedBytes(batch,Sm90DenseCoordinates::ResidualFamily(Sm90DenseFamily(plan,g)))>budget)
      throw std::invalid_argument("native GEMM exceeds target shared-memory budget");
    if(cfg.mma!=Sm90DenseMma::kMmaSync)
      throw std::invalid_argument("requested native mechanism is not implemented yet");
    options.gemms.push_back({16,cfg.tile_n,cfg.TileK(),cfg.slots,1,kSm90DenseGemmImpl});
  }
  for(auto const& stage:plan.stages)if(stage.kind==frontend::PlanTaskKind::kFusedAttention) {
    if(stage.dense_attention_chunks!=c.attention.chunks ||
       c.attention.SharedBytes(stage.width,stage.group)>budget)
      throw std::invalid_argument("native attention structure or shared-memory budget is invalid");
    c.ValidateSchedule(Sm90DenseShapes(plan),batch,stage.width,stage.group,
        plan.serving_capacity,budget);
  }
  auto module=frontend::TorchExportImporter{}.InstantiateForGranularity(imported,context,options);
  mlir::OpBuilder b(&context);
  auto set_int=[&](char const* name,int value) {(*module)->setAttr(name,b.getI64IntegerAttr(value));};
  (*module)->setAttr("tmexec.sm90_dense",b.getStringAttr(c.ToJson().Dump(0)));
  set_int("tmexec.sm90_dense_budget",budget);
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
  // P4 WriteBack: the decision layer records how each boundary is realized, so
  // codegen reads the synchronization from the IR instead of re-deriving it
  // from the stage structure. The placement is the skeleton's closed-form
  // template this round (prompt §4.0 item 3).
  {
    // Each boundary's kind follows from the coordinate alone (appendix A): the
    // group wait coarsens qkv -> attention, tp realizes o and down as in-
    // epilogue reductions, fusion makes the expert boundary a count, and the
    // head's last arriver is a handoff. The event extents are derived at
    // lowering from the same coordinate, where the shapes are in scope.
    mlir::NamedAttrList kinds;
    auto kind=[&](char const* stage,char const* value) {
      kinds.set(stage,b.getStringAttr(value));
    };
    kind("qkv",c.group_wait?"group":"aggregate");
    kind("attention",c.group_wait?"group":"aggregate");
    kind("o",c.tp>1?"data":"aggregate");
    if(c.moe) {
      kind("router","aggregate");
      kind("expert_gate_up","aggregate");
      kind("expert_down",c.expert_fuse?"counted":"aggregate");
    } else {
      kind("gate_up","aggregate");
      kind("down",c.tp>1?"data":"aggregate");
    }
    kind("lm_head",c.embed_la?"handoff":"aggregate");
    (*module)->setAttr("tmexec.sync_kind",kinds.getDictionary(&context));
    (*module)->setAttr("tmexec.placement",b.getStringAttr("template"));
  }
  codegen::ResolveServingWeightPacking(*module);
  (*module)->setAttr("tmexec.native_contract_version",b.getI64IntegerAttr(1));
  NativeBoundaryContracts(*module,true);
  if(mlir::failed(mlir::verify(*module)))throw std::invalid_argument("native dense CTG verification failed");
  return module;
}
}  // namespace tilemega::solver
