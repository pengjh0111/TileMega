// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90NativeCtg.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>
#include <map>

namespace tilemega::solver {
namespace {
using Role=frontend::OpRole;
std::string TaskKind(Role role) {
  if(role==Role::kProjection || role==Role::kServingQkv || role==Role::kServingSwiGlu ||
      role==Role::kServingArgmaxPartial)return "gemm";
  return frontend::ToString(role);
}
struct Contract {std::string implementation,sync="aggregate",count_source="static";int extent=1,required=0;};
Contract NodeContract(Role role,Sm90DenseCoordinates const& c,int grid,int batch,int groups) {
  Contract result;result.required=grid;
  if(role==Role::kServingMerge) {
    result.implementation=c.attention.merge_stage?"attention_merge/stage":
        c.attention.dsmem?"attention_merge/dsmem":"attention_merge/handoff";
    if(!c.attention.merge_stage) {result.sync="handoff";result.extent=batch*groups;result.required=c.attention.chunks;}
  }else if(role==Role::kMoERouting) {
    result.implementation=std::string("moe_route/")+c.RouteName();
    if(c.route==Sm90DenseCoordinates::Route::kLastArriver)result.sync="handoff";
  }else if(role==Role::kMoECombine) {
    result.implementation=c.combine_stage?"moe_combine/stage":"moe_combine/handoff";
    if(!c.combine_stage) {result.sync="handoff";result.required=0;result.count_source="active_schedule";}
  }else if(role==Role::kServingArgmax) {
    result.implementation=c.argmax_stage?"argmax/stage":"argmax/handoff";
    if(!c.argmax_stage)result.sync="handoff";
  }else if(role==Role::kEmbedding) {
    result.implementation=c.embed_la?"embedding/handoff":"embedding/stage";
    if(c.embed_la) {result.sync="handoff";result.required=1;}
  }else if(role==Role::kServingFusedAttention) {
    result.implementation="attention";
    if(c.group_wait) {result.sync="group";result.extent=groups;}
  }else result.implementation="mma_sync";
  return result;
}
int Integer(mlir::Operation* op,llvm::StringRef key) {
  auto value=op->getAttrOfType<mlir::IntegerAttr>(key);
  if(!value)throw std::invalid_argument("native execution graph lacks "+key.str());
  return value.getInt();
}
std::string String(mlir::DictionaryAttr attr,llvm::StringRef key) {
  auto value=attr.getAs<mlir::StringAttr>(key);
  if(!value)throw std::invalid_argument("native execution stage lacks "+key.str());
  return value.getValue().str();
}
void Contracts(mlir::ModuleOp module,dialect::GraphOp graph,bool write) {
  mlir::OpBuilder b(module.getContext());
  auto c=Sm90DenseCoordinates::FromJson(json::Parse(module->getAttrOfType<mlir::StringAttr>("tmexec.sm90_dense").getValue().str()));
  int grid=Integer(module,"tmexec.sm90_dense_grid"),batch=Integer(module,"tmexec.sm90_dense_batch");
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto stages=plan.getAs<mlir::ArrayAttr>("stages"),gemms=plan.getAs<mlir::ArrayAttr>("gemms");
  auto attr=[&](mlir::Operation* op,llvm::StringRef key,mlir::Attribute expected) {
    if(write)op->setAttr(key,expected);
    else if(op->getAttr(key)!=expected)throw std::invalid_argument("native execution contract mismatch: "+key.str());
  };
  struct Node {Role role;Contract contract;int count,stage,n=0,groups=0;std::string epilogue;};
  std::map<std::string,Node> nodes;
  std::map<std::pair<int,Role>,int> occurrences;
  std::map<std::string,long> ownership_counts;
  graph.walk([&](dialect::TileSpaceOp task) {
    int index=task.getStage();
    auto s=mlir::cast<mlir::DictionaryAttr>(stages[index]);
    Node node;node.stage=index;node.role=static_cast<Role>(Integer(task,"tmexec.native_role"));
    node.count=Integer(task,"tmexec.task_count");
    auto stage_kind=String(s,"kind");
    bool allowed=false;
    if(stage_kind=="kGemm") {
      auto g=mlir::cast<mlir::DictionaryAttr>(gemms[s.getAs<mlir::IntegerAttr>("gemm").getInt()]);
      node.n=g.getAs<mlir::IntegerAttr>("n").getInt();
      node.epilogue=g.get("epilogue")?String(g,"epilogue"):"store";
      allowed=node.role==Role::kServingQkv || node.role==Role::kServingSwiGlu ||
          node.role==Role::kServingArgmaxPartial || node.role==Role::kProjection ||
          (node.role==Role::kMoERouting && node.epilogue=="moe_router") ||
          (node.role==Role::kMoECombine && node.epilogue=="moe_down");
    }else if(stage_kind=="kFusedAttention") {
      node.groups=s.getAs<mlir::IntegerAttr>("extent").getInt();
      allowed=node.role==Role::kServingFusedAttention || (node.role==Role::kServingMerge && c.attention.chunks>1);
    }else if(stage_kind=="kArgmaxReduce")allowed=node.role==Role::kServingArgmax;
    else if(stage_kind=="kEmbedding")allowed=node.role==Role::kEmbedding;
    if(!allowed || node.count<1 || ++occurrences[{index,node.role}]!=1)
      throw std::invalid_argument("native execution task has invalid origin, cardinality or duplicate role");
    auto ownership=task->getAttrOfType<dialect::CouplingMapAttr>("tmexec.ownership");
    if(!ownership)throw std::invalid_argument("native execution task lacks CTG ownership");
    auto key=ownership.getMap().ToString();
    auto found=ownership_counts.find(key);
    if(found==ownership_counts.end())found=ownership_counts.emplace(key,
        ownership.getMap().Reverse().ImageCard().Eval({})).first;
    if(found->second!=node.count)
      throw std::invalid_argument("native execution task count differs from CTG ownership");
    node.contract=NodeContract(node.role,c,grid,batch,node.groups);
    attr(task,"kind",dialect::TaskKindAttr::get(module.getContext(),b.getStringAttr(TaskKind(node.role))));
    attr(task,"tmexec.implementation",b.getStringAttr(node.contract.implementation));
    attr(task,"tmexec.placement",b.getStringAttr("template"));
    nodes.emplace(task.getSymName().str(),node);
  });
  int expected=stages.size();
  for(std::size_t i=0;i<stages.size();++i) {
    auto s=mlir::cast<mlir::DictionaryAttr>(stages[i]);
    if(String(s,"kind")=="kFusedAttention" && c.attention.chunks>1) {
      ++expected;if(occurrences[{int(i),Role::kServingMerge}]!=1)
        throw std::invalid_argument("native execution graph lacks attention reducer");
    }
    if(String(s,"kind")=="kGemm") {
      auto g=mlir::cast<mlir::DictionaryAttr>(gemms[s.getAs<mlir::IntegerAttr>("gemm").getInt()]);
      auto epilogue=g.get("epilogue")?String(g,"epilogue"):"store";
      if(epilogue=="moe_router" || epilogue=="moe_down") {
        ++expected;
        auto role=epilogue=="moe_router"?Role::kMoERouting:Role::kMoECombine;
        if(occurrences[{int(i),role}]!=1)throw std::invalid_argument("native execution graph lacks MoE reducer");
      }
    }
  }
  if(int(nodes.size())!=expected)throw std::invalid_argument("native execution graph has missing or extra task spaces");
  int edges=0;
  graph.walk([&](dialect::CouplingOp edge) {
    ++edges;auto const& p=nodes.at(edge.getSrc().str());auto const& q=nodes.at(edge.getDst().str());
    auto contract=q.contract;
    if(q.role==Role::kMoERouting && c.route==Sm90DenseCoordinates::Route::kLastArriver)
      contract.required=p.n/c.gemms[5].tile_n;
    if(q.role==Role::kMoECombine && !c.combine_stage)contract.extent=q.n/c.gemms[7].tile_n;
    if(q.role==Role::kServingFusedAttention && c.group_wait)
      contract.required=p.n/c.gemms[0].tile_n/q.groups;
    if(p.epilogue=="moe_gate_up" && q.epilogue=="moe_down" && q.role!=Role::kMoECombine && c.expert_fuse) {
      contract.sync="counted";contract.required=p.n/c.gemms[6].tile_n;
      contract.extent=c.moe?Integer(graph,"tmexec.experts"):0;
    }
    if(p.role==Role::kMoERouting && !c.GlobalRoute()) {
      contract.sync="data";contract.extent=grid;contract.required=1;
      contract.count_source="cta_local_schedule";
    }
    attr(edge,"tmexec.sync_kind",b.getStringAttr(contract.sync));
    attr(edge,"tmexec.event_extent",b.getI64IntegerAttr(contract.extent));
    attr(edge,"tmexec.required_count",b.getI64IntegerAttr(contract.required));
    attr(edge,"tmexec.required_count_source",b.getStringAttr(contract.count_source));
    auto event=mlir::SymbolTable::lookupNearestSymbolFrom<dialect::EventTensorOp>(edge,edge.getEventAttr());
    if(!event)throw std::invalid_argument("native execution edge lacks its event tensor");
    attr(event,"event_type",mlir::TypeAttr::get(mlir::RankedTensorType::get({contract.extent},b.getI32Type())));
    attr(event,"extent",dialect::MetricAttr::get(module.getContext(),analysis::QuasiPolynomial::Constant(contract.extent)));
  });
  attr(graph,"tmexec.space_count",b.getI64IntegerAttr(nodes.size()));
  attr(graph,"tmexec.edge_count",b.getI64IntegerAttr(edges));
}
}

void WriteSm90NativeExecutionGraph(mlir::ModuleOp module,frontend::ImportedSemantics const& imported,
    Sm90DenseCoordinates const& c,int batch,int grid,mlir::MLIRContext& context) {
  Sm90NativeCtgCache cache;
  auto ctg=InstantiateSm90NativeCtg(imported,c,batch,context,cache,grid);
  mlir::OpBuilder b(&context);b.setInsertionPointToEnd(module.getBody());
  mlir::OperationState state(b.getUnknownLoc(),dialect::GraphOp::getOperationName());
  state.addAttribute("sym_name",b.getStringAttr("native_execution"));state.addRegion();
  auto graph=mlir::cast<dialect::GraphOp>(b.create(state));
  graph.getBody().push_back(new mlir::Block());b.setInsertionPointToEnd(&graph.getBody().front());
  auto work=Sm90DenseWorkloadFromPlan(imported.plan,batch);
  graph->setAttr("tmexec.experts",b.getI64IntegerAttr(work.experts));
  graph->setAttr("tmexec.domain_approximation",b.getStringAttr(ctg.approximation.Dump(0)));
  auto symbol=[](int i){return "t"+std::to_string(i);};
  for(std::size_t i=0;i<ctg.spaces.size();++i) {
    auto const& task=ctg.spaces[i];
    mlir::OperationState op(b.getUnknownLoc(),dialect::TileSpaceOp::getOperationName());
    op.addAttribute("sym_name",b.getStringAttr(symbol(i)));
    auto kind=TaskKind(task.role);
    op.addAttribute("kind",dialect::TaskKindAttr::get(&context,b.getStringAttr(kind)));
    op.addAttribute("granularity",b.getDictionaryAttr({b.getNamedAttr("ownership",b.getStringAttr("tile_per_block"))}));
    op.addAttribute("write_map",dialect::AccessMapAttr::get(&context,b.getDictionaryAttr({
        b.getNamedAttr("kind",b.getStringAttr("native_ctg_projection"))})));
    op.addAttribute("stage",b.getI64IntegerAttr(task.stage));
    op.addAttribute("operator_name",b.getStringAttr(task.name));
    op.addAttribute("tmexec.native_role",b.getI64IntegerAttr(int(task.role)));
    op.addAttribute("tmexec.task_count",b.getI64IntegerAttr(task.count));
    op.addAttribute("tmexec.ownership",dialect::CouplingMapAttr::get(&context,task.ownership.BindParams(ctg.binding)));
    b.create(op);
  }
  int index=0;
  struct Metrics {analysis::QuasiPolynomial wait,fanout,count;};
  std::map<std::string,Metrics> metrics;
  for(auto const& edge:ctg.edges) {
    auto relation=edge.relation.BindParams(ctg.binding).Reverse();
    auto found=metrics.find(relation.ToString());
    if(found==metrics.end())found=metrics.emplace(relation.ToString(),Metrics{
        relation.Card(),relation.Reverse().Card(),relation.ImageCard()}).first;
    auto const& values=found->second;
    auto metric=[&](auto const& q){return dialect::MetricAttr::get(&context,q);};
    auto event="e"+std::to_string(index);
    mlir::OperationState e(b.getUnknownLoc(),dialect::EventTensorOp::getOperationName());
    e.addAttribute("sym_name",b.getStringAttr(event));
    e.addAttribute("event_type",mlir::TypeAttr::get(mlir::RankedTensorType::get({1},b.getI32Type())));
    e.addAttribute("extent",metric(analysis::QuasiPolynomial::Constant(1)));b.create(e);
    mlir::OperationState op(b.getUnknownLoc(),dialect::CouplingOp::getOperationName());
    op.addAttribute("sym_name",b.getStringAttr("c"+std::to_string(index++)));
    op.addAttribute("src",mlir::FlatSymbolRefAttr::get(&context,symbol(edge.producer)));
    op.addAttribute("dst",mlir::FlatSymbolRefAttr::get(&context,symbol(edge.consumer)));
    op.addAttribute("read_map",dialect::AccessMapAttr::get(&context,b.getDictionaryAttr({
        b.getNamedAttr("kind",b.getStringAttr("native_ctg_projection"))})));
    op.addAttribute("relation",dialect::CouplingMapAttr::get(&context,relation));
    op.addAttribute("wait",metric(values.wait));op.addAttribute("fanout",metric(values.fanout));
    op.addAttribute("volume",metric(values.wait));op.addAttribute("count",metric(values.count));
    op.addAttribute("tmexec.metric_unit",b.getStringAttr("dependency_instances"));
    op.addAttribute("tier",dialect::TierAttr::get(&context,3));
    op.addAttribute("sync_kind",dialect::SyncKindAttr::get(&context,b.getStringAttr("global")));
    op.addAttribute("event",mlir::FlatSymbolRefAttr::get(&context,event));b.create(op);
  }
  Contracts(module,graph,true);
  module->setAttr("tmexec.native_execution_graph",mlir::FlatSymbolRefAttr::get(&context,"native_execution"));
}

void ValidateSm90NativeExecutionGraph(mlir::ModuleOp module) {
  auto name=module->getAttrOfType<mlir::FlatSymbolRefAttr>("tmexec.native_execution_graph");
  auto graph=name?mlir::dyn_cast_or_null<dialect::GraphOp>(mlir::SymbolTable::lookupSymbolIn(module,name.getValue())):dialect::GraphOp{};
  if(!graph)throw std::invalid_argument("native execution graph is missing");
  Contracts(module,graph,false);
}
} // namespace tilemega::solver
