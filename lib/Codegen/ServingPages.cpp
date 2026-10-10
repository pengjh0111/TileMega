// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Solver/PageLayout.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
namespace tilemega::codegen {
void ConfigureServingPrefetch(mlir::ModuleOp module,TargetSpec const& target,int depth,int stride) {
  if(stride==0) {
    stride=target.CalibrationFor("bf16").l2_prefetch_bytes;
    // Targets without the optional L2 sector microbenchmark record zero.
    // A 128-byte line is the conservative supported codegen fallback; the
    // explicit CLI coordinate still overrides it for controlled experiments.
    if(stride==0)stride=128;
  }
  if(depth<1 || depth>2 || (stride!=32 && stride!=64 && stride!=128))
    throw std::invalid_argument("prefetch depth must be 1/2 and stride 32/64/128");
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto roles=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.dimension_roles");
  auto info=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
  auto past=roles.getAs<mlir::StringAttr>("past");
  bool historical=false;
  if(past && info.getAs<mlir::IntegerAttr>("seq").getInt()==1)
    for(auto attr:plan.getAs<mlir::ArrayAttr>("stages"))
      historical|=mlir::cast<mlir::DictionaryAttr>(attr).getAs<mlir::StringAttr>("kind").getValue()=="kFusedAttention";
  std::map<std::string,analysis::CouplingRelation> written,reads;
  // Only historical KV requests consume this frontier. Spatial DNN maps
  // retain their exact coupling proof without constructing an unused union.
  if(historical) {
    auto model=solver::ModelDescription::FromCouplingGraph(module,{0,0,0},"prefetch-frontier");
    analysis::SemanticGraph sem;std::set<std::string> seen;
    for(auto const& s:model.task_semantics)if(seen.insert(s.op.name).second)sem.ops.push_back(s.op);
    auto graph=analysis::Instantiate(sem,{});
    for(auto const& op:sem.ops) {
      auto const& task=*graph.Find(op.name);
      written[op.result.name]=written[op.result.name].Union(analysis::ElementAccess(task,
          analysis::BuildWriteMap(task),{},analysis::AccessDomain::kPhysicalTensor).Image());
      for(auto const& w:op.additional_writes)
        written[w.tensor.name]=written[w.tensor.name].Union(analysis::ExactElementRead(op,task,
            {w.tensor,w.map,w.nonnegative},{}).Image());
      for(auto const& read:op.element_reads)
        reads[read.tensor.name]=reads[read.tensor.name].Union(analysis::ExactElementRead(op,task,read,{}).Image());
    }
  }
  auto buffers=plan.getAs<mlir::ArrayAttr>("buffers");
  auto batch=roles.getAs<mlir::StringAttr>("batch");
  std::string B=batch?batch.getValue().str():std::string();
  mlir::OpBuilder b(module.getContext());std::vector<mlir::Attribute> stages,proofs;
  for(auto attr:plan.getAs<mlir::ArrayAttr>("stages")) {
    auto stage=mlir::cast<mlir::DictionaryAttr>(attr);mlir::NamedAttrList updated(stage);unsigned mask=0;
    if(past && info.getAs<mlir::IntegerAttr>("seq").getInt()==1 &&
        stage.getAs<mlir::StringAttr>("kind").getValue()=="kFusedAttention") {
      auto operands=stage.getAs<mlir::DenseI64ArrayAttr>("operands");
      if(!operands)throw std::invalid_argument("missing prefetch operand identities");
      long heads=stage.getAs<mlir::IntegerAttr>("extent").getInt();
      long width=stage.getAs<mlir::IntegerAttr>("width").getInt();
      std::string P=past.getValue().str();
      if(B.empty() || P.empty())
        throw std::invalid_argument("historical prefetch requires batch and past roles");
      auto history=analysis::CouplingRelation::FromIslText("["+B+","+P+"] -> { [] -> [b,g,pos,d] : 0<=b<"+B+
          " and 0<=g<"+std::to_string(heads)+" and 0<=pos<"+P+
          " and pos<"+std::to_string(info.getAs<mlir::IntegerAttr>("capacity").getInt())+
          " and 0<=d<"+std::to_string(width)+" }");
      for(int slot=1;slot<=2;++slot) {
        auto name=mlir::cast<mlir::DictionaryAttr>(buffers[operands[slot]]).getAs<mlir::StringAttr>("name").getValue().str();
        auto frontier=reads[name].Subtract(written[name]);
        if(!history.IsSubset(frontier))throw std::invalid_argument("historical prefetch is not in the no-producer read frontier: "+name+" requested="+history.ToString()+" frontier="+frontier.ToString());
        mask|=1u<<(slot-1);
        proofs.push_back(b.getDictionaryAttr({b.getNamedAttr("tensor",b.getStringAttr(name)),
            b.getNamedAttr("frontier",b.getStringAttr(frontier.ToString())),
            b.getNamedAttr("prefetch",b.getStringAttr(history.ToString()))}));
      }
    }
    updated.set("prefetch_history_mask",b.getI64IntegerAttr(mask));stages.push_back(updated.getDictionary(module.getContext()));
  }
  mlir::NamedAttrList updated(plan);updated.set("stages",b.getArrayAttr(stages));module->setAttr("tilemega.model_plan",updated.getDictionary(module.getContext()));
  int grid=target.res.num_sms;
  if(auto g=module->getAttrOfType<mlir::IntegerAttr>("tmexec.solved_grid"))grid=g.getInt();
  auto l2=target.res.l2_bytes;
  if(l2<=0) {
    auto actual=TargetSpec::Probe();
    if(actual.arch_tag!=target.arch_tag)throw std::invalid_argument("cross target must specify L2 size for prefetch budget");
    l2=actual.res.l2_bytes;
  }
  if(l2<=0)throw std::invalid_argument("prefetch requires target L2 capacity");
  long budget=std::min<long>(512*1024,l2/(2*std::max(1,grid)));
  module->setAttr("tmexec.prefetch",b.getDictionaryAttr({b.getNamedAttr("depth",b.getI64IntegerAttr(depth)),
      b.getNamedAttr("stride",b.getI64IntegerAttr(stride)),b.getNamedAttr("worker_bytes",b.getI64IntegerAttr(budget)),
      b.getNamedAttr("history_proofs",b.getArrayAttr(proofs))}));
}
void ConfigureServingPages(mlir::ModuleOp module,TargetSpec const& target,int page_bytes) {
  auto serving=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
  auto model=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if(!serving || !model)
    throw std::invalid_argument("page execution requires a serving graph");
  bool dm=model.getAs<mlir::BoolAttr>("dm") && model.getAs<mlir::BoolAttr>("dm").getValue();
  auto runtime=ReadRuntimePlan(module);
  std::vector<std::array<int,3>> gemm_shapes;
  for(auto const& g:runtime.gemms) {
    if(!dm && g.tile_n*g.tile_k*2>page_bytes)
      throw std::invalid_argument("page plan requires one-page GEMM stages");
    if(!solver::PageLayout::StageFits(page_bytes,g.tile_n,g.tile_k))
      throw std::invalid_argument("a GEMM B stage must divide a page or occupy whole pages");
    gemm_shapes.push_back({g.tile_m,g.tile_n,g.tile_k});
  }
  std::vector<std::array<int,2>> attention_shapes;
  for(auto a:mlir::cast<mlir::ArrayAttr>(model.get("stages"))) {
    auto stage=mlir::cast<mlir::DictionaryAttr>(a);
    if(mlir::cast<mlir::StringAttr>(stage.get("kind")).getValue()=="kFusedAttention")
      attention_shapes.push_back({
          int(mlir::cast<mlir::IntegerAttr>(stage.get("width")).getInt()),
          int(mlir::cast<mlir::IntegerAttr>(stage.get("group")).getInt())});
  }
  int task_workspace=0;
  for(auto a:mlir::cast<mlir::ArrayAttr>(model.get("stages"))) {
    auto stage=mlir::cast<mlir::DictionaryAttr>(a);
    if(stage.getAs<mlir::StringAttr>("kind").getValue()=="kDwPwFused") {
      auto const& g=runtime.gemms.at(stage.getAs<mlir::IntegerAttr>("gemm").getInt());
      int channels=stage.getAs<mlir::IntegerAttr>("width").getInt();
      task_workspace=std::max(task_workspace,2*g.tile_m*channels+
          std::max(solver::DmServingPageActivationBytes(g.tile_m,g.tile_n,g.tile_k),
                   solver::DmServingPageScratchBytes(g.tile_m,g.tile_n)));
    }
    if(auto bytes=stage.getAs<mlir::IntegerAttr>("dm_workspace_bytes")) {
      if(bytes.getInt()<0 || bytes.getInt()>std::numeric_limits<int>::max())
        throw std::invalid_argument("task workspace exceeds the page layout range");
      task_workspace=std::max(task_workspace,int(bytes.getInt()));
    }
  }
  bool prefill=mlir::cast<mlir::IntegerAttr>(serving.get("seq")).getInt()>1;
  int kv_tile=64;
  if(auto tile=module->getAttrOfType<mlir::IntegerAttr>("tmexec.attention_kv_tile"))kv_tile=tile.getInt();
  auto [activation,scratch]=solver::PageLayout::ServingWorkspace(gemm_shapes,attention_shapes,prefill,dm,task_workspace,kv_tile);
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
  int lookahead=0;
  if(auto choice=module->getAttrOfType<mlir::IntegerAttr>("tmexec.lookahead_bytes"))
    lookahead=int(choice.getInt());
  if(lookahead<0 || lookahead>target.res.l2_bytes/(4*target.res.num_sms))
    throw std::invalid_argument("L2 lookahead exceeds the per-SM budget");
  values.set("lookahead_bytes",b.getI64IntegerAttr(lookahead));
  module->setAttr("tmexec.pages",values.getDictionary(module.getContext()));
}

void ResolveServingWeightPacking(mlir::ModuleOp module) {
  if (!module->getAttr("tmexec.pages") &&
      !module->getAttr("tmexec.nonpaged_weight_layout_tiled")) return;
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if (!plan) throw std::invalid_argument("page plan lacks model buffers");
  auto runtime=ReadRuntimePlan(module);
  auto old_buffers=plan.getAs<mlir::ArrayAttr>("buffers");
  auto old_gemms=plan.getAs<mlir::ArrayAttr>("gemms");
  auto old_stages=plan.getAs<mlir::ArrayAttr>("stages");
  if (runtime.gemms.size()!=old_gemms.size())
    throw std::invalid_argument("page geometry does not cover each GEMM");
  mlir::OpBuilder b(module.getContext());
  std::vector<mlir::Attribute> buffers(old_buffers.begin(),old_buffers.end());
  std::vector<mlir::Attribute> gemms(old_gemms.begin(),old_gemms.end());
  std::vector<mlir::Attribute> stages(old_stages.begin(),old_stages.end());
  std::map<std::tuple<int,int,int>,int> packed;
  for (std::size_t i=0;i<gemms.size();++i) {
    auto g=mlir::cast<mlir::DictionaryAttr>(gemms[i]);
    int source=mlir::cast<mlir::IntegerAttr>(g.get("b")).getInt();
    int n=mlir::cast<mlir::IntegerAttr>(g.get("n")).getInt();
    int k=mlir::cast<mlir::IntegerAttr>(g.get("k")).getInt();
    int tn=runtime.gemms[i].tile_n,tk=runtime.gemms[i].tile_k;
    if(source<0 || source>=int(old_buffers.size()) || tn<=0 || tk<=0)
      throw std::invalid_argument("invalid page weight source or tile");
    auto original=mlir::cast<mlir::DictionaryAttr>(old_buffers[source]);
    auto recipe=original.getAs<mlir::StringAttr>("pack_json");
    if(!recipe || recipe.getValue().empty())
      throw std::invalid_argument("page weight lacks a packing recipe");
    if(recipe.getValue().starts_with("{\"kind\":\"tile_pages\"")) continue;
    auto access=g.get("dm_access")?frontend::DecodeDmAccess(g.get("dm_access")):DmGemmAccess{};
    bool expert=access.b==DmBAccess::kExpertIndirect;
    auto stride=std::int64_t((n+tn-1)/tn)*((k+tk-1)/tk)*tn*tk;
    auto elements=stride*(expert?access.experts:1);
    if(expert && (!access.experts || elements>std::numeric_limits<std::uint32_t>::max()))
      throw std::invalid_argument("packed expert stack exceeds the buffer extent range");
    auto key=std::make_tuple(source,tn,tk);
    int destination;
    if(auto found=packed.find(key);found!=packed.end()) destination=found->second;
    else {
      bool replace=expert;
      for(auto const& [prior,id]:packed)if(std::get<0>(prior)==source)replace=false;
      // Expert stacks have no non-GEMM consumer. Replacing their source slot
      // prevents retaining an unused row-major stack beside its tiled copy.
      if(replace)for(auto const& item:old_gemms) {
        auto other=mlir::cast<mlir::DictionaryAttr>(item);
        for(auto operand:{"a","c","d"})
          if(other.getAs<mlir::IntegerAttr>(operand).getInt()==source)
            throw std::invalid_argument("expert weight storage is also an activation");
        if(other.getAs<mlir::IntegerAttr>("b").getInt()==source &&
            (!other.get("dm_access") ||
             frontend::DecodeDmAccess(other.get("dm_access")).b!=DmBAccess::kExpertIndirect))
          throw std::invalid_argument("expert weight storage has a dense consumer");
      }
      destination=replace?source:int(buffers.size());
      auto name=original.getAs<mlir::StringAttr>("name").getValue().str();
      if(!replace)name+="@t"+std::to_string(tn)+"x"+std::to_string(tk);
      auto nested="{\"kind\":\"tile_pages\",\"tile_n\":"+
          std::to_string(tn)+",\"tile_k\":"+std::to_string(tk)+
          ",\"source\":"+recipe.getValue().str()+"}";
      mlir::NamedAttrList updated(original);
      updated.set("name",b.getStringAttr(name));
      updated.set("external_name",b.getStringAttr(name));
      updated.set("constant",b.getI64IntegerAttr(elements));
      updated.set("pack_json",b.getStringAttr(nested));
      if(replace)buffers[source]=updated.getDictionary(module.getContext());
      else buffers.push_back(updated.getDictionary(module.getContext()));
      packed.emplace(key,destination);
    }
    mlir::NamedAttrList rewritten(g);
    rewritten.set("b",b.getI64IntegerAttr(destination));
    if(expert) {
      access.expert_stride=stride;
      rewritten.set("dm_access",frontend::EncodeDm(b,access));
    }
    gemms[i]=rewritten.getDictionary(module.getContext());
    for(auto& attr:stages) {
      auto stage=mlir::cast<mlir::DictionaryAttr>(attr);
      if(stage.getAs<mlir::StringAttr>("kind").getValue()!="kGemm" ||
         mlir::cast<mlir::IntegerAttr>(stage.get("gemm")).getInt()!=int(i)) continue;
      auto ids=mlir::cast<mlir::DenseI64ArrayAttr>(stage.get("operands")).asArrayRef();
      std::vector<std::int64_t> operands(ids.begin(),ids.end());
      for(auto& id:operands) if(id==source) id=destination;
      mlir::NamedAttrList changed(stage);
      changed.set("operands",b.getDenseI64ArrayAttr(operands));
      attr=changed.getDictionary(module.getContext());
    }
  }
  mlir::NamedAttrList updated(plan);
  updated.set("buffers",b.getArrayAttr(buffers));
  updated.set("gemms",b.getArrayAttr(gemms));
  updated.set("stages",b.getArrayAttr(stages));
  module->setAttr("tilemega.model_plan",updated.getDictionary(module.getContext()));
  module->setAttr("tmexec.weight_layout_tiled",b.getBoolAttr(true));
}
}
