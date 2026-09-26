// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/PageLayout.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <algorithm>
#include <set>
#include <stdexcept>
namespace tilemega::codegen {
void ConfigureServingPrefetch(mlir::ModuleOp module,TargetSpec const& target,int depth,int stride) {
  if(stride==0)stride=target.CalibrationFor("bf16").l2_prefetch_bytes;
  if(depth<1 || depth>2 || (stride!=32 && stride!=64 && stride!=128))
    throw std::invalid_argument("prefetch depth must be 1/2 and stride 32/64/128");
  auto model=solver::ModelDescription::FromCouplingGraph(module,{0,0,0},"prefetch-frontier");
  analysis::SemanticGraph sem;std::set<std::string> seen;
  for(auto const& s:model.task_semantics)if(seen.insert(s.op.name).second)sem.ops.push_back(s.op);
  auto graph=analysis::Instantiate(sem,{});
  std::map<std::string,analysis::CouplingRelation> written,reads;
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
  auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto roles=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.dimension_roles");
  auto buffers=plan.getAs<mlir::ArrayAttr>("buffers");
  auto info=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
  std::string B=roles.getAs<mlir::StringAttr>("batch").getValue().str();
  auto past=roles.getAs<mlir::StringAttr>("past");
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
  if(!serving || !model || mlir::cast<mlir::IntegerAttr>(serving.get("seq")).getInt()!=1)
    throw std::invalid_argument("page execution requires a decode serving graph");
  auto runtime=ReadRuntimePlan(module);
  std::vector<std::array<int,3>> gemm_shapes;
  for(auto const& g:runtime.gemms) {
    if(!solver::PageLayout::StageFits(page_bytes,g.tile_n,g.tile_k))
      throw std::invalid_argument("a GEMM B stage must divide a page or occupy whole pages");
    gemm_shapes.push_back({g.tile_m,g.tile_n,g.tile_k});
  }
  std::vector<int> attention_widths;
  for(auto a:mlir::cast<mlir::ArrayAttr>(model.get("stages"))) {
    auto stage=mlir::cast<mlir::DictionaryAttr>(a);
    if(mlir::cast<mlir::StringAttr>(stage.get("kind")).getValue()=="kFusedAttention")
      attention_widths.push_back(mlir::cast<mlir::IntegerAttr>(stage.get("width")).getInt());
  }
  auto [activation,scratch]=solver::PageLayout::ServingWorkspace(gemm_shapes,attention_widths);
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
