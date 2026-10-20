// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/AscendCodegen.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Support/Json.h>
#include <tilemega/Target/TargetSpec.h>
#include <mlir/IR/Verifier.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace tilemega::codegen {
namespace {
unsigned U(mlir::DictionaryAttr d,char const* key) {
  auto v=d.getAs<mlir::IntegerAttr>(key);
  if(!v || v.getInt()<0 || v.getInt()>0xffffffffu)throw std::invalid_argument(std::string("Invalid encoder field: ")+key);
  return v.getInt();
}
mlir::DictionaryAttr At(mlir::ArrayAttr a,unsigned i) {
  if(!a || i>=a.size())throw std::invalid_argument("Missing encoder table entry");
  return mlir::cast<mlir::DictionaryAttr>(a[i]);
}
std::string File(std::string name) {
  if(name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._")!=std::string::npos ||
      name.find("..")!=std::string::npos)throw std::invalid_argument("Unsafe encoder parameter name");
  return name+".bin";
}
}
std::string LowerAscendEncoder(mlir::ModuleOp module,solver::SymbolicProblem const& p,solver::AscendPlan const& plan) {
  using solver::StageKind;
  auto reject=[](){throw std::invalid_argument("Encoder graph is outside the fixed S16/H256 BF16 contract");};
  auto model=module?module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan"):mlir::DictionaryAttr{};
  if(!model || mlir::failed(mlir::verify(module)) || !p.model.dm || p.model.dtype!=solver::ScalarType::kBF16 ||
      p.model.dims.batch!=1 || p.model.dims.seq!=16 || p.model.stages.size()!=10 || p.model.gemms.size()!=4 ||
      p.model.gemm_access.size()!=4 || plan.metadata.stages!=10 || plan.stage_implementations.size()!=10)reject();
  auto buffers=model.getAs<mlir::ArrayAttr>("buffers"),gemms=model.getAs<mlir::ArrayAttr>("gemms"),
      stages=model.getAs<mlir::ArrayAttr>("stages"),outputs=model.getAs<mlir::ArrayAttr>("outputs");
  if(!buffers || buffers.size()!=plan.metadata.buffers || !stages || stages.size()!=10 || !outputs || outputs.size()!=1)reject();
  std::vector<uint32_t> offsets,bytes,allocated,weight_rows(buffers.size(),0),qkv(buffers.size(),0);
  std::vector<std::array<std::string,3>> files;unsigned arena=0;
  for(auto raw:buffers) {
    auto b=mlir::cast<mlir::DictionaryAttr>(raw);auto dtype=b.getAs<mlir::StringAttr>("dtype").getValue();
    if((dtype!="bf16" && dtype!="i64") || U(b,"per_seq") || U(b,"per_past") || U(b,"per_total") || b.get("dm_arena_offset"))reject();
    unsigned count=U(b,"constant")+U(b,"per_batch");
    if(!count || count>1048576)reject();
    unsigned size=count*(dtype=="bf16"?2:8),allocation=(size+512+511)/512*512;
    offsets.push_back(arena);bytes.push_back(size);allocated.push_back(allocation);arena+=allocation;
    std::array<std::string,3> source{};
    if(b.getAs<mlir::BoolAttr>("no_producer").getValue()) {
      auto recipe=b.getAs<mlir::StringAttr>("pack_json").getValue().str();
      if(recipe.empty())source[0]=dtype=="i64"?"mask.bin":"input.bin";
      else {
        auto pack=json::Parse(recipe);auto kind=pack.At("kind").AsString("kind");
        if(kind=="alias")source[0]=File(pack.At("source").AsString("source"));
        else if(kind=="qkv_concat_bias") {
          auto const& names=pack.At("sources").AsArray("sources");
          if(names.size()!=3 || pack.At("heads").AsNumber("heads")!=4 ||
              pack.At("head_dim").AsNumber("head_dim")!=64 || pack.At("part").AsString("part")!="weight")reject();
          for(unsigned i=0;i<3;++i)source[i]=File(names[i].AsString("source"));
          qkv[files.size()]=1;
        }else reject();
      }
    }
    files.push_back(std::move(source));
  }
  if(arena>8*1024*1024)reject();
  std::vector<ascend::abi::Stage> desc;
  std::array<StageKind,10> kinds{StageKind::kGemm,StageKind::kEncoderAttention,StageKind::kGemm,
    StageKind::kAdd,StageKind::kLayerNorm,StageKind::kGemm,StageKind::kElementwise,
    StageKind::kGemm,StageKind::kAdd,StageKind::kLayerNorm};
  std::array<unsigned,4> expected_n{768,256,1024,256},expected_k{256,256,256,1024};
  for(unsigned index=0;index<10;++index) {
    auto const& s=p.model.stages[index];auto sd=At(stages,index);auto const& impl=plan.stage_implementations[index];
    if(s.kind!=kinds[index] || p.projection.stages[index].logical_stage!=int(index) ||
        p.projection.stages[index].combine || impl.kind!=static_cast<TaskKind>(s.kind))reject();
    ascend::abi::Stage d{};d.kind=unsigned(s.kind);d.rows=16;d.tasks=p.counts[index];
    unsigned a,b,c;
    if(s.kind==StageKind::kGemm) {
      unsigned gi=s.gemm;auto const& g=p.model.gemms.at(gi);auto ga=At(gemms,gi);
      auto const& geo=p.runtime.gemms.at(gi);auto const& access=p.model.gemm_access.at(gi);
      auto beta=ga.getAs<mlir::FloatAttr>("beta");
      if(gi>=4 || g.n!=int(expected_n[gi]) || g.k!=int(expected_k[gi]) || !beta || beta.getValueAsDouble()!=0 ||
          ga.get("epilogue") || ga.get("ss_out") || ga.get("norm_ss") || U(ga,"c")!=U(ga,"d") ||
          p.model.epilogue_chains.at(gi).count || access.a!=DmAAccess::kDense || access.b!=DmBAccess::kDense ||
          access.rows_per_batch!=16 || access.a_row_offset || access.a_row_stride>1 || access.write.kind!=DmWriteKind::kDense ||
          geo.tile_m!=32 || geo.tile_n!=64 || geo.tile_k!=64 || geo.stages!=2 || geo.split_k!=1 ||
          impl.impl_id!="catlass_950_bf16_32x64x64_gm")reject();
      a=U(ga,"a");b=U(ga,"b");c=U(ga,"d");d.columns=g.n;d.k=g.k;
      if(a>=bytes.size() || b>=bytes.size() || c>=bytes.size() || bytes[a]!=16*g.k*2 ||
          bytes[b]!=g.n*g.k*2 || bytes[c]!=16*g.n*2 || weight_rows[b] ||
          bool(qkv[b])!=(gi==0) || d.tasks!=(g.n+63)/64)reject();
      weight_rows[b]=g.n;
    }else {
      if(s.rows_per_batch!=16 || s.operands.size()!=(index==1?3:index==6?2:index==4 || index==9?4:3))reject();
      a=s.operands[0];b=index==6?a:s.operands[1];c=s.operands.back();d.columns=s.width;
      if(index==1) {
        b=s.operands[2];c=s.operands[1];d.columns=256;d.k=4;d.epsilon=.125f;
        if(s.width!=16 || s.extent!=4 || s.group!=1 || d.tasks!=64 ||
            impl.impl_id!="aiv_950_bf16_encoder_attention_q1" ||
            a>=bytes.size() || b>=bytes.size() || c>=bytes.size() ||
            bytes[a]!=16*768*2 || bytes[b]!=16*8 || bytes[c]!=16*256*2)reject();
      }else if(index==4 || index==9) {
        unsigned beta=s.operands[2];auto eps=sd.getAs<mlir::FloatAttr>("dm_norm_epsilon");
        if(s.width!=256 || s.group!=1 || d.tasks!=16 || !eps || eps.getValueAsDouble()!=double(1e-12f) ||
            beta>=bytes.size() || bytes[beta]!=512 || impl.impl_id!="aiv_950_bf16_layernorm_row")reject();
        d.k=offsets[beta];d.epsilon=eps.getValueAsDouble();
        if(a>=bytes.size() || b>=bytes.size() || c>=bytes.size() || bytes[a]!=8192 || bytes[b]!=512 || bytes[c]!=8192)reject();
      }else if(index==6) {
        auto chain=frontend::DecodeDmChain(sd.get("dm_chain"));
        if(s.width!=1024 || d.tasks!=16 || chain.count!=1 || chain.operations[0].kind!=DmEpilogueKind::kActivation ||
            chain.operations[0].activation!=DmActivation::kGeluErf || impl.impl_id!="aiv_950_bf16_erf_gelu_row" ||
            a>=bytes.size() || c>=bytes.size() || bytes[a]!=32768 || bytes[c]!=32768)reject();
      }else {
        if(s.width!=256 || d.tasks!=4 || impl.impl_id!="aiv_950_bf16_encoder_add_tile" ||
            a>=bytes.size() || b>=bytes.size() || c>=bytes.size() || bytes[a]!=8192 || bytes[b]!=8192 || bytes[c]!=8192)reject();
      }
    }
    d.a=offsets.at(a);d.b=offsets.at(b);d.c=offsets.at(c);desc.push_back(d);
  }
  unsigned output=U(At(outputs,0),"buffer");auto const& d=desc;
  if(output>=offsets.size() || d[0].c!=d[1].a || d[1].c!=d[2].a || d[2].c!=d[3].a || d[3].b!=d[0].a ||
      d[3].c!=d[4].a || d[4].c!=d[5].a || d[5].c!=d[6].a || d[6].c!=d[7].a ||
      d[7].c!=d[8].a || d[8].b!=d[4].c || d[8].c!=d[9].a || d[9].c!=offsets[output])reject();
  std::ostringstream table;table<<std::scientific<<std::setprecision(9);
  table<<"static constexpr unsigned kGroups="<<plan.metadata.groups<<",kTaskCount="<<plan.tasks.size()
      <<",kScheduleCount="<<plan.schedule.size()<<",kArenaBytes="<<arena<<",kOutputBuffer="<<output<<";\n";
  table<<"static constexpr bool kStageBarrier="<<(plan.stage_barrier?"true":"false")<<";\n";
  auto target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/ascend_3510.json");
  table<<"static constexpr unsigned kExpectedAic="<<target.ascend_capability->TrackCount(ascend::TrackRole::kAic)
      <<",kExpectedAiv="<<target.ascend_capability->TrackCount(ascend::TrackRole::kAiv)<<";\n";
  auto const& m=plan.metadata;
  table<<"static constexpr abi::Metadata kMetadata={"<<m.version<<','<<m.groups<<','<<m.workers<<','<<m.tasks<<','
      <<m.dependencies<<','<<m.epochs<<','<<m.stages<<','<<m.buffers<<"};\nstatic constexpr abi::Stage kStages[]={";
  for(auto const& s:desc)table<<'{'<<s.kind<<','<<s.a<<','<<s.b<<','<<s.c<<','<<s.rows<<','<<s.columns<<','<<s.k<<','<<s.tasks<<','<<s.epsilon<<"f},";
  table<<"};\nstatic constexpr abi::TaskRef kTasks[]={";
  for(auto const& t:plan.tasks)table<<'{'<<t.stage<<','<<t.logical_task<<','<<t.dependency_begin<<','<<t.dependency_count<<','<<t.wait_begin<<','<<t.wait_count<<','<<t.flags<<"},";
  table<<"};\nstatic constexpr abi::TaskMeta kOwners[]={";
  for(auto const& t:plan.task_metadata)table<<'{'<<t.owner<<','<<t.event<<"},";
  table<<"};\n";
  auto array=[&](char const* name,std::vector<uint32_t> const& values) {
    table<<"static constexpr uint32_t "<<name<<"[]={";
    if(values.empty())table<<0;else for(auto v:values)table<<v<<',';
    table<<"};\n";
  };
  array("kOffsets",plan.schedule_offsets);array("kSchedule",plan.schedule);array("kDependencies",plan.dependencies);
  array("kBufferOffsets",offsets);array("kBufferBytes",bytes);array("kBufferAllocatedBytes",allocated);
  array("kWeightRows",weight_rows);array("kQkvWeight",qkv);
  table<<"static constexpr char const* kBufferFiles[][3]={";
  for(auto const& names:files)table<<"{\""<<names[0]<<"\",\""<<names[1]<<"\",\""<<names[2]<<"\"},";
  table<<"};\n";
  std::ifstream in(std::string(TILEMEGA_SOURCE_DIR)+"/lib/Codegen/Ascend/PersistentEncoder.cpp.in");
  if(!in)throw std::runtime_error("Missing native encoder template");
  std::string source((std::istreambuf_iterator<char>(in)),{}),marker="@AS1_PLAN_TABLES@";
  auto at=source.find(marker);if(at==std::string::npos || source.find(marker,at+marker.size())!=std::string::npos)reject();
  source.replace(at,marker.size(),table.str());return source;
}
} // namespace tilemega::codegen
