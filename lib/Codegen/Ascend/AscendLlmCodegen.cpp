// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/AscendLlmCodegen.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Support/Json.h>
#include <mlir/IR/Verifier.h>
#include <fstream>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace tilemega::codegen {
namespace {
uint64_t Add(uint64_t a,uint64_t b) {
  if(a>UINT64_MAX-b)throw std::overflow_error("LLM byte offset overflow");return a+b;
}
uint64_t Mul(uint64_t a,uint64_t b) {
  if(b && a>UINT64_MAX/b)throw std::overflow_error("LLM allocation product overflow");return a*b;
}
uint32_t U(mlir::DictionaryAttr d,char const* name) {
  auto a=d.getAs<mlir::IntegerAttr>(name);
  if(!a || a.getInt()<0 || uint64_t(a.getInt())>UINT32_MAX)
    throw std::invalid_argument(std::string("Missing/unrepresentable LLM field: ")+name);
  return a.getInt();
}
std::string S(mlir::DictionaryAttr d,char const* name) {
  auto a=d.getAs<mlir::StringAttr>(name);return a?a.getValue().str():"";
}
mlir::DictionaryAttr At(mlir::ArrayAttr a,unsigned i) {
  if(!a || i>=a.size())throw std::invalid_argument("Incomplete verified LLM table");
  auto d=mlir::dyn_cast<mlir::DictionaryAttr>(a[i]);
  if(!d)throw std::invalid_argument("Malformed verified LLM entry");return d;
}
template<class T> void Binary(std::filesystem::path const& p,std::vector<T> const& v) {
  if(std::filesystem::exists(p))throw std::invalid_argument("Never overwrite an LLM artifact");
  std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<char const*>(v.data()),v.size()*sizeof(T));
  if(!f)throw std::runtime_error("Cannot write LLM binary table");
}
}
AscendLlmPlan MaterializeAscendLlm(mlir::ModuleOp module,solver::SymbolicProblem const& p,
    ascend::WorkerMap const& map,std::vector<solver::AscendScheduledTask> const& input,
    solver::AscendSchedule const& schedule,bool diagnostic,unsigned attention_precision) {
  using namespace ascend::llm;
  if(attention_precision>8)throw std::invalid_argument("Unknown native attention precision contract");
  if(!module || mlir::failed(mlir::verify(module)) || !p.model.serving || p.model.dm ||
      p.model.dtype!=solver::ScalarType::kBF16 || (p.model.dims.seq!=1 && p.model.dims.seq!=64) || p.model.dims.batch<1 ||
      p.model.serving_capacity!=1088 || (p.model.dims.seq==1?p.model.dims.past!=1087:p.model.dims.past!=0))
    throw std::invalid_argument("Native LLM materialization requires BF16 prefill64 or conservative full-capacity decode CTG");
  solver::AuditAscendSchedule(input,map,schedule);
  if(!diagnostic) {
    if(schedule.start_ns.size()!=input.size() || schedule.finish_ns.size()!=input.size() ||
        !std::isfinite(schedule.makespan_ns) || schedule.makespan_ns<=0)
      throw std::invalid_argument("An unpriced diagnostic queue cannot be labelled as an optimized native Plan");
    for(auto const& task:input)
      if(!std::isfinite(task.price.service_ns) || task.price.service_ns<=0 || task.price.provenance.empty())
        throw std::invalid_argument("Every selected native task requires a positive sourced price");
  }
  auto model=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if(!model || input.size()!=std::size_t(p.offsets.back()) || input.size()>UINT32_MAX)
    throw std::invalid_argument("LLM plan does not cover its real CTG");
  auto bs=model.getAs<mlir::ArrayAttr>("buffers"),ss=model.getAs<mlir::ArrayAttr>("stages"),gs=model.getAs<mlir::ArrayAttr>("gemms");
  if(!bs || bs.size()!=p.model.physical_buffers.size() || !ss || ss.size()!=p.model.stages.size())
    throw std::invalid_argument("LLM verified table cardinality mismatch");
  AscendLlmPlan out;json::Array buffers;std::vector<uint64_t> offsets;uint64_t arena=0;
  std::set<unsigned> required;
  for(auto raw:ss) {
    auto d=mlir::cast<mlir::DictionaryAttr>(raw);auto op=d.getAs<mlir::DenseI64ArrayAttr>("operands");
    if(S(d,"kind")=="kEmbedding")required.insert(op[0]);
    if(S(d,"kind")=="kFusedAttention"){required.insert(op[3]);required.insert(op[4]);}
  }
  for(unsigned i=0;i<bs.size();++i) {
    auto b=At(bs,i);auto dtype=S(b,"dtype");unsigned width=dtype=="bf16"?2:(dtype=="f32"||dtype=="i32")?4:0;
    if(!width || b.get("dm_arena_offset"))throw std::invalid_argument("Unsupported LLM allocation type/alias");
    uint64_t elements=Add(U(b,"constant"),Add(Mul(U(b,"per_batch"),p.model.dims.batch),
      Add(Mul(U(b,"per_seq"),p.model.dims.seq),Add(Mul(U(b,"per_past"),p.model.dims.past),Mul(U(b,"per_total"),p.model.dims.total)))));
    auto bytes=Mul(elements,width);if(!bytes)throw std::invalid_argument("Empty LLM allocation");
    auto allocated=Mul(Add(Add(bytes,512),511)/512,512);
    bool must=required.count(i) || !S(b,"pack_json").empty();
    out.allocations.push_back({arena,bytes,allocated,unsigned(must),0});
    offsets.push_back(arena);
    buffers.emplace_back(json::Object{{"id",int(i)},{"name",S(b,"name")},{"dtype",dtype},
      {"offset_bytes",double(arena)},{"bytes",double(bytes)},{"allocated_bytes",double(allocated)},
      {"external_name",S(b,"external_name")},{"pack_json",S(b,"pack_json")},{"required_input",must},
      {"no_producer",b.getAs<mlir::BoolAttr>("no_producer").getValue()}});
    arena=Add(arena,allocated);
  }
  auto off=[&](uint32_t id){if(id==UINT32_MAX)return kAbsent;if(id>=offsets.size())throw std::invalid_argument("Invalid LLM operand id");return offsets[id];};
  auto epsilon=model.getAs<mlir::FloatAttr>("norm_epsilon");
  if(!epsilon || epsilon.getValueAsDouble()<=0)throw std::invalid_argument("Missing model normalization epsilon");
  uint64_t attention_workspace=kAbsent;
  std::map<unsigned,uint64_t> partial_workspaces;
  auto scratch=[&](std::string name,std::string dtype,uint64_t bytes) {
    uint64_t offset=arena,allocated=Mul(Add(Add(bytes,512),511)/512,512);
    out.allocations.push_back({offset,bytes,allocated,0,0});
    buffers.emplace_back(json::Object{{"id",int(buffers.size())},{"name",std::move(name)},
      {"dtype",std::move(dtype)},{"offset_bytes",double(offset)},{"bytes",double(bytes)},
      {"allocated_bytes",double(allocated)},{"no_producer",false},
      {"external_name",""},{"pack_json",""},{"required_input",false}});
    arena=Add(arena,allocated);return offset;
  };
  for(unsigned s=0;s<p.counts.size();++s) {
    auto projected=p.projection.stages.at(s);
    auto const& source=p.model.stages.at(projected.logical_stage);auto d=At(ss,projected.logical_stage);
    if(projected.combine && source.kind!=solver::StageKind::kGemm)
      throw std::invalid_argument("Only GEMM combine is registered for a decomposed native plan");
    Stage stage{};for(auto& a:stage.operands)a=kAbsent;stage.workspace=kAbsent;
    stage.kind=unsigned(source.kind);stage.tasks=p.counts[s];stage.extent=source.extent;
    stage.width=source.width;stage.group=source.group;stage.kv_block=source.attention_kv_block;
    stage.query_rows=source.attention_query_rows;stage.epsilon=epsilon.getValueAsDouble();
    stage.row_stride=U(d,"row_stride");stage.row_offset=U(d,"row_offset");
    stage.batch_rows=source.batch_rows;stage.rows=p.model.dims.batch*(source.batch_rows?1:p.model.dims.seq);
    if(source.kind==solver::StageKind::kGemm) {
      auto g=At(gs,source.gemm);auto const& geo=p.runtime.gemms.at(source.gemm);
      bool base=geo.tile_m==32 && geo.tile_n==64 && geo.tile_k==64;
      bool small=geo.tile_m==16 && geo.tile_n==128 && geo.tile_k==128;
      if((!base && !small) || (geo.split_k!=1 && geo.split_k!=2 && geo.split_k!=3 && geo.split_k!=7) || geo.stages!=2 || g.get("norm_ss") || g.get("ss_out"))
        throw std::invalid_argument("LLM GEMM geometry/deferred normalization is not implemented");
      stage.variant=small?1:0;
      stage.columns=U(g,"n");stage.k=U(g,"k");
      if(!stage.k || stage.k%16 || stage.columns%16 || stage.k>16384)
        throw std::invalid_argument("LLM GEMM lies outside BF16 BlockMmad legality");
      for(unsigned i=0;i<4;++i)stage.operands[i]=off(U(g,i==0?"a":i==1?"b":i==2?"c":"d"));
      auto e=S(g,"epilogue");stage.epilogue=e.empty()||e=="store"?0:e=="residual"?1:e=="swiglu"?2:e=="argmax_partial"?3:UINT32_MAX;
      auto beta=g.getAs<mlir::FloatAttr>("beta");
      if(!beta || (stage.epilogue==1?beta.getValueAsDouble()!=1:beta.getValueAsDouble()!=0) || stage.epilogue==UINT32_MAX)
        throw std::invalid_argument("Unregistered LLM epilogue");
      if(stage.epilogue==2 && U(g,"interleave_u")!=16)throw std::invalid_argument("SwiGLU packing requires U16");
      if(stage.epilogue==2 && stage.columns%32)throw std::invalid_argument("SwiGLU tail requires complete gate/up U16 pairs");
      if(stage.epilogue==3 && U(g,"partial_tile_n")!=geo.tile_n)throw std::invalid_argument("LM head partial count disagrees with geometry");
      if(geo.split_k>1) {
        const bool store=geo.split_k==7 && stage.columns==1024 && stage.k==4096 && stage.epilogue==0;
        const bool family=store || ((geo.split_k==3 && stage.columns==2048 && (stage.k==6144 || stage.k==8192)) ||
                          (geo.split_k==2 && stage.columns==3072 && stage.k==8192)) && stage.epilogue==1;
        if(!family || !small || (stage.rows!=1 && stage.rows!=64 && !(stage.rows==16 && geo.split_k==3 && stage.columns==2048 && stage.k==8192)) ||
            !p.projection.options.cg_split_task_order)
          throw std::invalid_argument("Split-K requires an observed shape/epilogue and tile-major CTG");
        stage.extent=geo.split_k;stage.width=(stage.k+geo.split_k-1)/geo.split_k;stage.group=1;
        auto found=partial_workspaces.find(projected.logical_stage);
        if(projected.combine) {
          if(found==partial_workspaces.end())throw std::invalid_argument("Combine has no preceding partial stage");
          stage.kind=unsigned(TaskKind::kGemmCombine);stage.workspace=found->second;
          stage.operands[5]=store?stage.operands[3]:scratch("native.gemm.raw."+std::to_string(projected.logical_stage),
            "bf16",Mul(Mul(stage.rows,stage.columns),2));
        }else {
          if(found!=partial_workspaces.end())throw std::invalid_argument("Duplicate partial stage");
          stage.workspace=scratch("native.gemm.partial."+std::to_string(projected.logical_stage),
            "f32",Mul(Mul(Mul(stage.rows,stage.columns),stage.extent),4));
          partial_workspaces.emplace(projected.logical_stage,stage.workspace);stage.epilogue=0;
        }
      }else if(stage.epilogue) {
        stage.workspace=arena;auto bytes=Mul(Mul(stage.rows,stage.columns),2);
        auto allocated=Mul(Add(Add(bytes,512),511)/512,512);
        out.allocations.push_back({arena,bytes,allocated,0,0});
        buffers.emplace_back(json::Object{{"id",int(buffers.size())},{"name","native.gemm.raw."+std::to_string(projected.logical_stage)},
          {"dtype","bf16"},{"offset_bytes",double(arena)},{"bytes",double(bytes)},
          {"allocated_bytes",double(Mul(Add(Add(bytes,512),511)/512,512))},{"no_producer",false},
          {"external_name",""},{"pack_json",""}});
        arena=Add(arena,Mul(Add(Add(bytes,512),511)/512,512));
      }
    }else {
      auto operands=d.getAs<mlir::DenseI64ArrayAttr>("operands");
      if(!operands || operands.size()!=16)throw std::invalid_argument("Missing LLM primitive operands");
      for(unsigned i=0;i<16;++i)stage.operands[i]=off(operands[i]);
      if(source.kind!=solver::StageKind::kEmbedding && source.kind!=solver::StageKind::kRMSNorm &&
          source.kind!=solver::StageKind::kFusedAttention && source.kind!=solver::StageKind::kAttentionMerge &&
          source.kind!=solver::StageKind::kArgmaxReduce && source.kind!=solver::StageKind::kLayoutConvert)
        throw std::invalid_argument("Unregistered native LLM TaskKind");
      if(stage.kind==25) {
        if(source.layout_group_ratio!=4 || stage.extent!=8 || stage.width!=768 || stage.group!=1)
          throw std::invalid_argument("Native QKV pack requires observed eight-group head128/GQA4");
        stage.query_rows=source.layout_group_ratio;stage.kv_block=128;
      }
      if(source.kind==solver::StageKind::kFusedAttention &&
          (p.model.dims.seq==1?(stage.query_rows!=stage.group || (stage.kv_block!=128 && stage.kv_block!=1088)):
           (stage.kv_block!=1088 || !stage.query_rows || !stage.group || stage.query_rows%stage.group)))
        throw std::invalid_argument("Unregistered native attention query/KV geometry");
      if((stage.kind==1 || stage.kind==11) && (!stage.width || stage.width>4096))
        throw std::invalid_argument("Native full-row norm/embedding exceeds its UB contract");
      if(stage.kind==1 && attention_precision>=6) {
        if(stage.width!=1024 && stage.width!=2048 && stage.width!=3072 && stage.width!=4096)
          throw std::invalid_argument("Newton full-row norm requires a probed H1024/H2048/H3072/H4096 family");
        if(stage.width>=3072 && attention_precision<7)
          throw std::invalid_argument("Observed H3072/H4096 normalization requires the validated fold2048 implementation");
        stage.variant=attention_precision>=7?3:2;
      }
      if(stage.kind==13 && ((stage.width!=64 && stage.width!=128) || !stage.extent || !stage.group || stage.group>8))
        throw std::invalid_argument("Native GQA head geometry is not registered");
      if(stage.kind==13 && attention_precision) {
        if(stage.kv_block!=1088)throw std::invalid_argument("FP32 chunk/merge candidate requires the unified KV ownership domain");
        stage.variant=attention_precision==8?6:attention_precision>=5?5:attention_precision;
        if(attention_precision>=4) {
          if(map.launch.num_groups!=8 || !((stage.width==128 && (stage.group==2 || stage.group==3 || stage.group==4)) || (stage.width==64 && stage.group==4)))
            throw std::invalid_argument("Experimental cooperative FP32 attention requires probed G8/head128/GQA2/3/4 or head64/GQA4");
          if(attention_workspace==kAbsent) {
            attention_workspace=arena;auto bytes=Mul(map.launch.num_groups,attention::kGroupBytes);
            auto allocated=Mul(Add(Add(bytes,512),511)/512,512);
            out.allocations.push_back({arena,bytes,allocated,0,0});
            buffers.emplace_back(json::Object{{"id",int(buffers.size())},{"name","native.attention.group_workspace"},
              {"dtype","f32"},{"offset_bytes",double(arena)},{"bytes",double(bytes)},
              {"allocated_bytes",double(allocated)},{"no_producer",true},{"required_input",false},
              {"external_name",""},{"pack_json",""}});
            arena=Add(arena,allocated);
          }
          stage.workspace=attention_workspace;
        }
      }
    }
    uint64_t expected=stage.tasks;
    if(stage.kind==0 || stage.kind==unsigned(TaskKind::kGemmCombine)) {
      unsigned tm=stage.variant?16:32,tn=stage.variant?128:64;
      expected=Mul((stage.rows+tm-1)/tm,(stage.columns+tn-1)/tn);
      if(stage.kind==0 && stage.extent>1)expected=Mul(expected,stage.extent);
    }
    else if(stage.kind==1 || stage.kind==11 || stage.kind==15 || stage.kind==25)expected=stage.rows;
    else if(stage.kind==14)expected=Mul(p.model.dims.batch,stage.extent);
    else if(stage.kind==13)expected=Mul(Mul(p.model.dims.batch,stage.extent),Mul(
      (p.model.dims.seq*stage.group+stage.query_rows-1)/stage.query_rows,(1088+stage.kv_block-1)/stage.kv_block));
    if(expected!=stage.tasks)throw std::invalid_argument("Native task coordinates disagree with CTG ownership");
    out.stages.push_back(stage);
  }
  for(unsigned s=0;s<out.stages.size();++s)if(out.stages[s].kind==6) {
    auto const& combine=out.stages[s];
    if(!s || p.projection.stages[s-1].logical_stage!=p.projection.stages[s].logical_stage ||
        out.stages[s-1].kind!=0 || out.stages[s-1].tasks!=combine.tasks*combine.extent ||
        out.stages[s-1].workspace!=combine.workspace)
      throw std::invalid_argument("Combine does not own its complete CTG partial domain");
    for(unsigned tile=0;tile<combine.tasks;++tile) {
      auto const& task=input.at(p.offsets[s]+tile);
      for(unsigned part=0;part<combine.extent;++part) {
        unsigned producer=p.offsets[s-1]+tile*combine.extent+part;
        if(std::find(task.predecessors.begin(),task.predecessors.end(),producer)==task.predecessors.end())
          throw std::invalid_argument("CTG combine is missing a required partial dependency");
      }
    }
  }
  for(auto& s:out.stages)if(s.kind==0 && s.epilogue==3) {
    unsigned matches=0;
    for(auto const& reduce:out.stages)if(reduce.kind==15 && reduce.operands[0]==s.operands[3]) {
      s.operands[4]=reduce.operands[1];++matches;
      unsigned tn=s.variant?128:64;
      if(reduce.width!=(s.columns+tn-1)/tn)throw std::invalid_argument("Vocabulary reduction domain disagrees with partial tiles");
    }
    if(matches!=1)throw std::invalid_argument("LM head lacks a unique final argmax consumer");
  }
  for(unsigned i=0;i<input.size();++i) {
    auto const& t=input[i];unsigned lanes=t.role==ascend::ImplementationRole::kMix1_2?t.mix_aiv_lanes:0;
    if(lanes>1)throw std::invalid_argument("LLM epilogue currently registers one AIV participant");
    auto const& stage=out.stages.at(t.stage);
    auto expected=stage.kind==0?(stage.epilogue?ascend::ImplementationRole::kMix1_2:ascend::ImplementationRole::kAicOnly):
      stage.kind==13 && (stage.variant==4 || stage.variant==5 || stage.variant==6)?ascend::ImplementationRole::kMix1_2:ascend::ImplementationRole::kAivOnly;
    if(t.role!=expected || t.logical_task>=stage.tasks || out.waits.size()>UINT32_MAX)
      throw std::invalid_argument("LLM task role/coordinate disagrees with implementation contract");
    Task task{t.stage,t.logical_task,schedule.owners[i],schedule.ordinals[i],uint32_t(out.waits.size()),uint32_t(schedule.waits[i].size()),lanes};
    for(auto w:schedule.waits[i])out.waits.push_back({w.producer_worker,w.required_prefix});
    out.tasks.push_back(task);
  }
  std::set<std::string> allocation_names;
  for(auto const& buffer:buffers)
    if(!allocation_names.insert(buffer.At("name").AsString("allocation name")).second)
      throw std::invalid_argument("Native allocation identity is not unique");
  if(out.waits.size()>UINT32_MAX || schedule.queue.size()>UINT32_MAX)throw std::overflow_error("LLM queue/wait table overflow");
  out.offsets=schedule.offsets;out.queue=schedule.queue;
  if(!diagnostic)for(auto const& task:input)out.task_prices.push_back(task.price);
  out.metadata={kVersion,unsigned(map.launch.num_groups),unsigned(map.workers.size()),unsigned(out.tasks.size()),
    unsigned(out.stages.size()),unsigned(out.waits.size()),unsigned(out.queue.size()),schedule.epoch_stride,
    unsigned(p.model.dims.batch),unsigned(p.model.dims.seq),1088,unsigned(diagnostic),arena,131072,2000000000ULL};
  if(solver::AscendProgressValue(1088,schedule.epoch_stride-1,schedule.epoch_stride)>UINT32_MAX/4)
    throw std::overflow_error("Native MIX phase-token encoding overflow");
  out.buffers_json=json::Value(json::Object{{"schema","tilemega.ascend.llm.buffers.v4"},
    {"arena_bytes",double(arena)},{"buffers",std::move(buffers)},
    {"pricing_status",diagnostic?"diagnostic_unpriced_bootstrap":"sourced_native_empirical_context"},
    {"phase",p.model.dims.seq==1?"decode":"prefill"},{"complete_generation",false},{"device_execution_verified",false}}).Dump();
  return out;
}
void AscendLlmPlan::Write(std::filesystem::path const& root) const {
  std::filesystem::create_directory(root);
  Binary(root/"metadata.bin",std::vector<ascend::llm::Metadata>{metadata});
  Binary(root/"stages.bin",stages);Binary(root/"tasks.bin",tasks);Binary(root/"waits.bin",waits);
  Binary(root/"allocations.bin",allocations);
  Binary(root/"offsets.bin",offsets);Binary(root/"queue.bin",queue);
  std::ofstream out(root/"buffers.json");out<<buffers_json<<'\n';if(!out)throw std::runtime_error("Cannot write LLM allocation manifest");
  json::Array contracts;
  for(unsigned i=0;i<stages.size();++i) {
    auto const& s=stages[i];std::string id;
    if(s.kind==6)id="as2_aiv_fp32_split"+std::to_string(s.extent)+"_ordered_combine_bf16_"+(s.epilogue?"residual":"store");
    if(s.kind==0)id=s.epilogue==0?"as2_catlass_bf16_32x64x64_nk":s.epilogue==1?"as2_mix_gemm_residual_gm":
      s.epilogue==2?"as2_mix_gemm_swiglu_u16_gm":"as2_mix_gemm_argmax64_gm";
    if(s.kind==0 && s.variant) {
      if(s.epilogue==0)id="as2_catlass_bf16_16x128x128_nk";
      else if(s.epilogue==3)id="as2_mix_gemm_argmax128_gm_16x128x128";
      else id+="_16x128x128";
    }
    if(s.kind==0 && s.extent>1)id="as2_catlass_bf16_fp32_split"+std::to_string(s.extent)+"_k"+std::to_string(s.width)+"_16x128x128_l0k64";
    else if(s.kind==1)id=s.variant==3?"as2_aiv_rmsnorm_cann_newton_fold"+std::to_string(s.width==3072?2048:s.width/2)+"_h"+std::to_string(s.width):s.variant==2?"as2_aiv_rmsnorm_cann_newton_h"+std::to_string(s.width):"as2_aiv_rmsnorm_full_row";
    else if(s.kind==25)id="as2_aiv_qkv_group_pack_h128_gqa4";
    else if(s.kind==11)id="as2_aiv_embedding_i32_bf16";
    else if(s.kind==13)id=s.kv_block==128?"as2_aiv_gqa_fp32_log2_partials_kv128":
      metadata.seq==1?"as2_aiv_gqa_bf16_scores_probability_k32":"as2_aiv_causal_prefill_private_kv_k32";
    if(s.kind==13 && s.variant)id=s.variant==6?"as2_mix_fp32_cube_newton_fold_softmax1088_gm":s.variant==5?"as2_mix_fp32_cube_cann_newton_qknorm_gm":s.variant==4?"as2_mix_fp32_qk_vector_div_softmax_pv_gm":s.variant==1?"as2_aiv_gqa_fp32_chunk32_normalized_merge":
      s.variant==2?"as2_aiv_gqa_fp32_scores_bf16_probability_k32":"as2_aiv_gqa_math_sdpa_fp32_split_scale";
    else if(s.kind==14)id="as2_aiv_normalized_partial_log2_merge";
    else if(s.kind==15)id="as2_aiv_ordered_argmax_scalar_reduce";
    if(id.empty())throw std::invalid_argument("Missing native body contract");
    double low=std::numeric_limits<double>::infinity(),high=0;std::set<std::string> provenance;
    if(!task_prices.empty())for(unsigned t=0;t<tasks.size();++t)if(tasks[t].stage==i) {
      low=std::min(low,task_prices.at(t).service_ns);high=std::max(high,task_prices.at(t).service_ns);
      provenance.insert(task_prices.at(t).provenance);
    }
    json::Array sources;for(auto const& source:provenance)sources.emplace_back(source);
    contracts.emplace_back(json::Object{{"stage",int(i)},{"impl_id",id},{"semantic_kind",int(s.kind)},
      {"role",s.kind==0?(s.epilogue?"AIC_AIV0":"AIC_ONLY"):s.kind==13&&(s.variant==4 || s.variant==5 || s.variant==6)?"AIC_AIV0":"AIV_ONLY"},
      {"dtype","bf16"},{"opmath","fp32"},
      {"input_dtype",s.kind==6?"fp32":"bf16"},{"output_dtype",s.kind==0&&s.extent>1?"fp32":"bf16"},
      {"input_weight_layout",s.kind==0?"logical_NK":"ND"},
      {"tile_m",s.kind==0?(s.variant?16:32):0},{"tile_n",s.kind==0?(s.variant?128:64):0},
      {"tile_k",s.kind==0?(s.variant?128:64):0},
      {"split_count",(s.kind==0 || s.kind==6)&&s.extent>1?int(s.extent):1},
      {"partial_k",(s.kind==0 || s.kind==6)&&s.extent>1?int(s.width):0},
      {"partial_layout",(s.kind==0 || s.kind==6)&&s.extent>1?"logical[row,col,part] -> physical[part,row,col]":"not_applicable"},
      {"partial_dtype",(s.kind==0 || s.kind==6)&&s.extent>1?"fp32":"not_applicable"},
      {"partial_workspace_bytes",(s.kind==0 || s.kind==6)&&s.extent>1?double(uint64_t(s.rows)*s.columns*s.extent*4):0},
      {"attention_score_probability",s.kind==13?((s.variant==4 || s.variant==5 || s.variant==6)?"fp32_cube_split_scale_vector_div_softmax":s.variant==1?"fp32_opmath_no_bf16_intermediate":s.variant==2?"fp32_scores_bf16_probability":s.variant==3?"fp32_split_scale_capacity_softmax":"bf16_storage_rounding"):"not_applicable"},
      {"qk_norm_method",s.kind!=13?"not_applicable":s.operands[5]==ascend::llm::kAbsent && s.operands[6]==ascend::llm::kAbsent?"none":(s.variant==5 || s.variant==6)?"CANN9.1.0_register_square_reduce_Newton_refinement":"existing_Rsqrt"},
      {"softmax_sum_method",s.kind==13&&s.variant==6?"CANN9.1.0_register_fold1024_capacity1088":"existing_ReduceSum"},
      {"full_row_norm_method",s.kind==1&&s.variant==3?"CANN9.1.0_power_two_fold_square_reduce_Newton_refinement":s.kind==1&&s.variant==2?"CANN9.1.0_register_square_reduce_Newton_refinement":"existing_Rsqrt"},
      {"aic_l1_bytes",s.kind==0||(s.kind==13&&(s.variant==4 || s.variant==5 || s.variant==6))?524288:0},{"aic_l0a_bytes",s.kind==0||(s.kind==13&&(s.variant==4 || s.variant==5 || s.variant==6))?65536:0},
      {"aic_l0b_bytes",s.kind==0||(s.kind==13&&(s.variant==4 || s.variant==5 || s.variant==6))?65536:0},{"aic_l0c_bytes",s.kind==0||(s.kind==13&&(s.variant==4 || s.variant==5 || s.variant==6))?262144:0},
      {"aiv_ub_bytes",s.kind!=0||s.epilogue?106496:0},
      {"internal_hard_event_slot",7},{"logical_task_completion_publishers",1},
      {"gm_group_workspace_bytes",s.kind==13&&(s.variant==4 || s.variant==5 || s.variant==6)?double(ascend::llm::attention::kGroupBytes):0},
      {"mix_gm_phases",s.kind==13&&(s.variant==4 || s.variant==5 || s.variant==6)?3:s.kind==0&&s.epilogue?1:0},
      {"handoff","GM_single_owner_frontier"},{"service_ns",high && low==high?json::Value(low):json::Value{}},
      {"service_ns_min",high?json::Value(low):json::Value{}},{"service_ns_max",high?json::Value(high):json::Value{}},
      {"price_provenance",std::move(sources)},
      {"calibration_status",high?"sourced_empirical_context_not_global_hardware_calibration":"not_calibrated"}});
  }
  std::ofstream c(root/"implementations.json");c<<json::Value(std::move(contracts)).Dump()<<'\n';
  if(!c)throw std::runtime_error("Cannot write native body contracts");
}
std::string LowerAscendLlm(AscendLlmPlan const& p) {
  if(p.metadata.version!=ascend::llm::kVersion)throw std::invalid_argument("Wrong native LLM ABI");
  std::ifstream file(std::string(TILEMEGA_SOURCE_DIR)+"/lib/Codegen/Ascend/PersistentLlm.cpp.in");
  if(!file)throw std::runtime_error("Missing native LLM template");
  int variant=-1;
  for(auto const& s:p.stages)if(s.kind==0) {
    if(s.variant>1 || (variant>=0 && unsigned(variant)!=s.variant))
      throw std::invalid_argument("One compiled native candidate requires a uniform verified GEMM geometry");
    variant=s.variant;
  }
  if(variant<0)throw std::invalid_argument("Native LLM has no registered GEMM body");
  bool split=false;for(auto const& s:p.stages)if(s.kind==6)split=true;
  return std::string("// Generated by TileMega: verified CTG tables and candidate geometry, LLM ABI v4.\n")+
    "#define TILEMEGA_LLM_SPLIT_K "+std::to_string(unsigned(split))+"\n"+
    "#define TILEMEGA_LLM_VARIANT "+std::to_string(variant)+"\n"+
    "#define TILEMEGA_LLM_TM "+std::to_string(variant?16:32)+"\n"+
    "#define TILEMEGA_LLM_TN "+std::to_string(variant?128:64)+"\n"+
    "#define TILEMEGA_LLM_TK "+std::to_string(variant?128:64)+"\n"+
    std::string((std::istreambuf_iterator<char>(file)),{});
}
} // namespace tilemega::codegen
