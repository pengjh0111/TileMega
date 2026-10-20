// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/AscendCodegen.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/Verifier.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace tilemega::codegen {
namespace {
unsigned U(mlir::DictionaryAttr d,char const* key) {
  auto value=d.getAs<mlir::IntegerAttr>(key);
  if(!value || value.getInt()<0 || value.getInt()>0xffffffffu)
    throw std::invalid_argument(std::string("Invalid native field: ")+key);
  return value.getInt();
}
mlir::DictionaryAttr At(mlir::ArrayAttr a,unsigned i) {
  if(!a || i>=a.size())throw std::invalid_argument("Missing native model table entry");
  return mlir::cast<mlir::DictionaryAttr>(a[i]);
}
}
std::string LowerAscendGraph(mlir::ModuleOp module,solver::SymbolicProblem const& p,solver::AscendPlan const& plan) {
  using solver::StageKind;
  auto reject=[](){throw std::invalid_argument("Native graph is outside the fixed BF16 MLP contract");};
  auto model=module?module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan"):mlir::DictionaryAttr{};
  if(!model || mlir::failed(mlir::verify(module)) || !p.model.dm || p.model.dtype!=solver::ScalarType::kBF16 ||
      p.model.stages.size()!=5 || p.model.gemms.size()!=2 || plan.metadata.stages!=5 ||
      plan.stage_implementations.size()!=5 || (p.model.dims.batch!=1 && p.model.dims.batch!=4))reject();
  auto buffers=model.getAs<mlir::ArrayAttr>("buffers"),gemms=model.getAs<mlir::ArrayAttr>("gemms"),
      stages=model.getAs<mlir::ArrayAttr>("stages"),outputs=model.getAs<mlir::ArrayAttr>("outputs");
  if(!buffers || buffers.size()!=plan.metadata.buffers || !stages || stages.size()!=5 ||
      !outputs || outputs.size()!=1)reject();
  std::vector<unsigned> offsets,elements,allocated,weight_rows(buffers.size(),0);
  std::vector<std::string> files;unsigned arena_bytes=0;
  for(auto raw:buffers) {
    auto b=mlir::cast<mlir::DictionaryAttr>(raw);
    auto dtype=b.getAs<mlir::StringAttr>("dtype");
    if(!dtype || dtype.getValue()!="bf16" || U(b,"per_seq") || U(b,"per_past") || U(b,"per_total"))reject();
    unsigned count=U(b,"constant")+U(b,"per_batch")*p.model.dims.batch;
    if(!count || count>65536 || b.get("dm_arena_offset"))reject();
    offsets.push_back(arena_bytes);elements.push_back(count);
    unsigned bytes=(count*2+512+511)/512*512;allocated.push_back(bytes);arena_bytes+=bytes;
    std::string file;
    if(b.getAs<mlir::BoolAttr>("no_producer").getValue()) {
      auto recipe=b.getAs<mlir::StringAttr>("pack_json").getValue().str();
      if(recipe.empty())file="input.bin";
      else {
        auto pack=json::Parse(recipe);
        if(pack.At("kind").AsString("kind")!="alias")reject();
        file=pack.At("source").AsString("source")+".bin";
        if(file.find('/')!=std::string::npos || file.find("..")!=std::string::npos ||
            file.find('"')!=std::string::npos || file.find('\\')!=std::string::npos)reject();
      }
    }
    files.push_back(std::move(file));
  }
  std::vector<ascend::abi::Stage> descriptions;
  unsigned m=p.model.dims.batch;
  for(unsigned index=0;index<5;++index) {
    auto const& s=p.model.stages[index];auto sd=At(stages,index);
    auto expected=index==0?StageKind::kRMSNorm:index==2?StageKind::kElementwise:
        index==4?StageKind::kAdd:StageKind::kGemm;
    if(s.kind!=expected || p.projection.stages[index].logical_stage!=int(index) ||
       p.projection.stages[index].combine || plan.stage_implementations[index].kind!=static_cast<TaskKind>(s.kind))reject();
    ascend::abi::Stage d{};d.kind=unsigned(s.kind);d.rows=m;d.tasks=p.counts[index];
    unsigned a,b,c;
    if(s.kind==StageKind::kGemm) {
      auto gi=s.gemm;auto const& g=p.model.gemms.at(gi);auto ga=At(gemms,gi);
      auto const& geom=p.runtime.gemms.at(gi);auto const& access=p.model.gemm_access.at(gi);
      auto beta=ga.getAs<mlir::FloatAttr>("beta");
      if(!beta || beta.getValueAsDouble()!=0 || ga.get("epilogue") || ga.get("ss_out") || ga.get("norm_ss") ||
          U(ga,"c")!=U(ga,"d") || p.model.epilogue_chains.at(gi).count ||
          access.a!=DmAAccess::kDense || access.b!=DmBAccess::kDense || access.rows_per_batch!=1 ||
          access.a_row_offset || access.a_row_stride>1 || access.write.kind!=DmWriteKind::kDense ||
          geom.tile_m!=32 || geom.tile_n!=64 || geom.tile_k!=64 || geom.stages!=2 || geom.split_k!=1 ||
          plan.stage_implementations[index].impl_id!="catlass_950_bf16_32x64x64_gm")reject();
      d.columns=g.n;d.k=g.k;
      if((index==1 && (g.n!=256 || g.k!=128)) || (index==3 && (g.n!=128 || g.k!=256)))reject();
      a=U(ga,"a");b=U(ga,"b");c=U(ga,"d");
      if(b>=weight_rows.size() || weight_rows[b])reject();weight_rows[b]=g.n;
      if(d.tasks!=((m+31)/32)*((g.n+63)/64))reject();
    }else {
      if(s.rows_per_batch!=1 || s.operands.size()!=(index==2?2:3))reject();
      a=s.operands[0];b=index==2?a:s.operands[1];c=s.operands.back();d.columns=s.width;
      if(s.width!=(index==2?256:128))reject();
      if(index==0) {
        auto eps=sd.getAs<mlir::FloatAttr>("dm_norm_epsilon");
        if(!eps || eps.getValueAsDouble()!=double(1e-5f) ||
            plan.stage_implementations[index].impl_id!="aiv_950_bf16_rmsnorm_row")reject();
        d.epsilon=eps.getValueAsDouble();
      }else if(index==2) {
        auto chain=frontend::DecodeDmChain(sd.get("dm_chain"));
        if(chain.count!=1 || chain.operations[0].kind!=DmEpilogueKind::kActivation ||
            chain.operations[0].activation!=DmActivation::kSilu ||
            plan.stage_implementations[index].impl_id!="aiv_950_bf16_silu_row")reject();
      }else if(plan.stage_implementations[index].impl_id!="aiv_950_bf16_add_tile")reject();
      if(d.tasks!=(index==4?2:m))reject();
    }
    if(a>=offsets.size() || b>=offsets.size() || c>=offsets.size() ||
        elements[a]!=m*(d.kind==0?d.k:d.columns) || elements[c]!=m*d.columns ||
        elements[b]!=(d.kind==0?d.k*d.columns:d.kind==1?d.columns:m*d.columns))reject();
    d.a=offsets[a];d.b=offsets[b];d.c=offsets[c];descriptions.push_back(d);
  }
  auto const& d=descriptions;
  unsigned output=U(At(outputs,0),"buffer");
  if(output>=offsets.size() || d[0].c!=d[1].a || d[1].c!=d[2].a || d[2].c!=d[3].a ||
      d[3].c!=d[4].a || d[4].b!=d[0].a || d[4].c!=offsets[output])reject();
  std::ostringstream table;table<<std::scientific<<std::setprecision(9);
  table<<"static constexpr unsigned kGroups="<<plan.metadata.groups<<",kTaskCount="<<plan.tasks.size()
      <<",kScheduleCount="<<plan.schedule.size()<<",kArenaBytes="<<arena_bytes<<",kOutputBuffer="<<output<<";\n";
  auto const& meta=plan.metadata;
  table<<"static constexpr abi::Metadata kMetadata={"<<meta.version<<','<<meta.groups<<','<<meta.workers<<','
      <<meta.tasks<<','<<meta.dependencies<<','<<meta.epochs<<','<<meta.stages<<','<<meta.buffers<<"};\n";
  table<<"static constexpr abi::Stage kStages[]={";
  for(auto const& s:descriptions)table<<'{'<<s.kind<<','<<s.a<<','<<s.b<<','<<s.c<<','<<s.rows<<','
      <<s.columns<<','<<s.k<<','<<s.tasks<<','<<s.epsilon<<"f},";
  table<<"};\nstatic constexpr abi::TaskRef kTasks[]={";
  for(auto const& t:plan.tasks)table<<'{'<<t.stage<<','<<t.logical_task<<','<<t.dependency_begin<<','<<t.dependency_count
      <<','<<t.wait_begin<<','<<t.wait_count<<','<<t.flags<<"},";
  table<<"};\nstatic constexpr abi::TaskMeta kOwners[]={";
  for(auto const& t:plan.task_metadata)table<<'{'<<t.owner<<','<<t.event<<"},";
  table<<"};\n";
  auto array=[&](char const* name,std::vector<uint32_t> const& values) {
    table<<"static constexpr uint32_t "<<name<<"[]={";
    if(values.empty())table<<0;else for(auto v:values)table<<v<<',';
    table<<"};\n";
  };
  array("kOffsets",plan.schedule_offsets);array("kSchedule",plan.schedule);array("kDependencies",plan.dependencies);
  array("kBufferOffsets",offsets);array("kBufferElements",elements);array("kBufferAllocatedBytes",allocated);
  array("kWeightRows",weight_rows);table<<"static constexpr char const* kBufferFiles[]={";
  for(auto const& f:files)table<<'"'<<f<<"\",";
  table<<"};\n";
  std::ifstream in(std::string(TILEMEGA_SOURCE_DIR)+"/lib/Codegen/Ascend/PersistentGraph.cpp.in");
  if(!in)throw std::runtime_error("Missing native graph template");
  std::string source((std::istreambuf_iterator<char>(in)),{}),marker="@AS1_PLAN_TABLES@";
  auto at=source.find(marker);
  if(at==std::string::npos || source.find(marker,at+marker.size())!=std::string::npos)reject();
  source.replace(at,marker.size(),table.str());return source;
}
} // namespace tilemega::codegen
