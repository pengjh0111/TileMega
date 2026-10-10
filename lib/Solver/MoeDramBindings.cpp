// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MoeDramBindings.h>
#include <mlir/IR/BuiltinOps.h>
#include <limits>
#include <climits>
#include <sstream>

namespace tilemega::solver {
void BindMoeDramInputs(mlir::ModuleOp module,ModelDescription const& model,
    analysis::DramFloorOptions& options,MoeRoutingProfile const* profile,unsigned first_layer) {
  using namespace analysis;using namespace codegen;
  if(!model.dm || model.gemm_access.size()!=model.gemms.size())
    throw std::invalid_argument("MoE DRAM binding requires DM access descriptors");
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto buffers=plan?plan.getAs<mlir::ArrayAttr>("buffers"):mlir::ArrayAttr{};
  if(!buffers)throw std::invalid_argument("MoE DRAM binding lacks buffer declarations");
  auto name=[&](unsigned id) {
    if(id>=buffers.size())throw std::invalid_argument("MoE DRAM buffer outside plan");
    return llvm::cast<mlir::DictionaryAttr>(buffers[id]).getAs<mlir::StringAttr>("name").getValue().str();
  };
  auto theta=model.MetricBindings();
  auto image=[&](TensorSpace const& tensor) {
    std::ostringstream text;text<<"{ [] -> [";
    for(unsigned i=0;i<tensor.axes.size();++i)text<<(i?",":"")<<"i"<<i;
    text<<"] : true";
    for(unsigned i=0;i<tensor.axes.size();++i) {
      auto const& axis=tensor.axes[i];auto origin=axis.origin.Eval(theta,{});
      auto extent=axis.extent.Eval(theta,{});
      if(extent<0 || origin>LONG_MAX-extent)throw std::invalid_argument("invalid MoE tensor envelope");
      text<<" and "<<origin<<" <= i"<<i<<" < "<<origin+extent;
    }
    return CouplingRelation::FromIslText(text.str()+" }");
  };
  struct Region {unsigned layer,experts,top_k,rows,bm;std::set<unsigned> internal;};
  std::map<unsigned,Region> regions;
  for(unsigned g=0;g<model.gemms.size();++g) {
    auto const& access=model.gemm_access[g];if(access.b!=DmBAccess::kExpertIndirect)continue;
    auto tokens=std::uint64_t(model.dims.seq)*model.dims.batch;
    if(!tokens || tokens>UINT32_MAX || !access.routing_topk || access.routing_topk>access.experts ||
        tokens*access.routing_topk!=access.binding_rows || !access.block_rows)
      throw std::invalid_argument("MoE DRAM assignment conservation differs from descriptor");
    if(!regions.count(access.binding) && regions.size()>UINT32_MAX-first_layer)
      throw std::overflow_error("routing profile layer offset overflows");
    auto [it,inserted]=regions.try_emplace(access.binding,Region{
        first_layer+unsigned(regions.size()),access.experts,access.routing_topk,
        access.binding_rows,access.block_rows,{}});
    auto& region=it->second;
    if(region.experts!=access.experts || region.top_k!=access.routing_topk ||
        region.rows!=access.binding_rows || region.bm!=access.block_rows)
      throw std::invalid_argument("expert stages disagree on one binding table");
    region.internal.insert(access.binding);region.internal.insert(access.rows);
    region.internal.insert(model.gemms[g].out_buffer);
    auto weight=name(llvm::cast<mlir::DictionaryAttr>(plan.getAs<mlir::ArrayAttr>("gemms")[g])
        .getAs<mlir::IntegerAttr>("b").getInt());
    auto per_expert=std::uint64_t(model.gemms[g].n)*model.gemms[g].k;
    if(!per_expert || per_expert>LONG_MAX/region.experts)
      throw std::overflow_error("expert traffic exceeds exact count range");
    auto count=QuasiPolynomial::Constant(region.top_k).Scale(per_expert);
    std::string source="inferred: distinct top-k expert lower bound; binding="+name(access.binding);
    std::string kind="lower_bound";
    if(profile) {
      auto const& routing=profile->At(region.layer,tokens);
      if(routing.experts!=region.experts || routing.top_k!=region.top_k ||
          routing.tokens!=tokens || !routing.windows || routing.tokens_per_expert.size()!=region.experts ||
          routing.windows>std::uint64_t(LONG_MAX)/region.rows)
        throw std::invalid_argument("routing profile differs from expert descriptor");
      std::uint64_t total=0,assignments=0;
      for(auto const& histogram:routing.tokens_per_expert) {
        std::uint64_t windows=0;
        for(auto const& [rows,frequency]:histogram) {
          if(rows>tokens || !frequency || frequency>routing.windows-windows ||
              (rows && frequency>(std::uint64_t(LONG_MAX)-assignments)/rows))
            throw std::invalid_argument("invalid expert occupancy histogram");
          windows+=frequency;assignments+=std::uint64_t(rows)*frequency;
          if(rows) {
            if(frequency>std::uint64_t(LONG_MAX)/per_expert-total)
              throw std::overflow_error("routing expected byte count exceeds exact range");
            total+=frequency;
          }
        }
        if(windows!=routing.windows)throw std::invalid_argument("expert occupancy omits windows");
      }
      if(assignments!=std::uint64_t(region.rows)*routing.windows)
        throw std::invalid_argument("expert occupancy violates assignment conservation");
      count=QuasiPolynomial::Constant(total*per_expert).ScaleRational("1/"+std::to_string(routing.windows));
      source="verified: routing profile "+profile->profile_id+" layer="+std::to_string(region.layer);
      kind="expectation";
    }
    options.expected_indirect_reads[weight]={count,std::move(source),std::move(kind)};
    for(auto const& semantic:model.task_semantics)if(semantic.stage>=0 &&
        model.stages.at(semantic.stage).gemm==int(g) && IsGemmStage(model.stages.at(semantic.stage).kind)) {
      options.active_runtime_rows[semantic.op.name]=QuasiPolynomial::Constant(region.rows);
      if(access.a==DmAAccess::kRowGather)for(auto const& operand:semantic.op.operands) {
        if(operand.tensor.name==weight)continue;
        bool gathered=false;
        for(auto const& index:operand.map.results)
          gathered|=index.kind==IndexResult::Kind::kDataDependent && index.binding_source==name(access.rows);
        if(gathered)options.indirect_read_images[operand.tensor.name]=image(operand.tensor);
      }
    }
  }
  for(auto const& stage:model.stages)if(stage.kind==StageKind::kMoETopK) {
    auto found=regions.find(stage.operands.at(4));
    if(found==regions.end())throw std::invalid_argument("dispatch has no expert binding consumer");
    if(stage.moe.row_capacity!=found->second.rows || stage.moe.experts!=found->second.experts ||
        stage.moe.top_k!=found->second.top_k || stage.moe.block_rows!=found->second.bm)
      throw std::invalid_argument("dispatch and expert binding contracts differ");
    for(unsigned slot:{6u,7u,8u})found->second.internal.insert(stage.operands.at(slot));
  }
  std::map<std::string,std::string> internal;
  for(auto const& [binding,region]:regions)for(auto id:region.internal) {
    auto record=llvm::cast<mlir::DictionaryAttr>(buffers[id]);
    if(record.getAs<mlir::StringAttr>("role").getValue()!="internal" ||
        record.getAs<mlir::StringAttr>("source").getValue()!="zero")
      throw std::invalid_argument("binding-produced buffer must be private internal storage");
    internal.emplace(name(id),"MoE dispatch bijection and active-row binding="+name(binding));
  }
  auto store=[&](SemanticOp const& op,TensorSpace const& tensor) {
    auto found=internal.find(tensor.name);if(found==internal.end())return;
    auto& contract=options.binding_internal[tensor.name];
    contract.envelope=contract.envelope.Union(image(tensor));
    contract.writers.insert(op.name);contract.source=found->second;
  };
  for(auto const& semantic:model.task_semantics) {
    store(semantic.op,semantic.op.result);
    for(auto const& side:semantic.op.additional_writes)store(semantic.op,side.tensor);
  }
}
}
