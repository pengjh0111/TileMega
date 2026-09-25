// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/ServingPruning.h>
#include <set>

namespace tilemega::solver {
struct OperatorClass {
  std::string signature;
  std::vector<std::size_t> gemms;
  std::vector<std::string> operators;
};
inline std::vector<OperatorClass> BuildOperatorClasses(frontend::ImportedSemantics const& imported) {
  std::vector<OperatorClass> classes;std::map<std::string,std::size_t> by_signature;
  for(std::size_t i=0;i<imported.lifted.ops.size();++i) {
    auto const& info=imported.lifted.ops[i];auto const& op=imported.lifted.sem.ops[i];
    if(op.kind!=analysis::OperatorKind::kMatmul)continue;
    int gemm=imported.plan.stages.at(info.stage).gemm;
    if(gemm<0)throw std::invalid_argument("matmul lacks GEMM invocation");
    auto signature=analysis::SemanticSignature(op);
    auto [it,added]=by_signature.emplace(signature,classes.size());
    if(added)classes.push_back({signature,{},{}});
    classes[it->second].gemms.push_back(std::size_t(gemm));
    classes[it->second].operators.push_back(op.name);
  }
  return classes;
}
inline auto ClassGeometryKey(GemmConfig const& g) {return std::make_tuple(g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k);}
inline frontend::ImportOptions ClassGranularity(frontend::ImportedSemantics const& imported,
    std::vector<OperatorClass> const& classes,std::vector<GemmConfig> const& config) {
  if(classes.size()!=config.size())throw std::invalid_argument("one tile required per semantic class");
  frontend::ImportOptions result;result.gemms.resize(imported.plan.gemms.size());
  std::vector<bool> covered(result.gemms.size());
  std::set<std::tuple<int,int,int,int>> shapes;
  for(std::size_t c=0;c<classes.size();++c) {
    auto const& g=config[c];shapes.emplace(g.tile_m,g.tile_n,g.tile_k,g.stages);
    for(auto index:classes[c].gemms) {
      if(covered.at(index))throw std::invalid_argument("GEMM belongs to two semantic classes");
      covered[index]=true;result.gemms[index]={g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k};
    }
  }
  if(shapes.size()>16)throw std::invalid_argument("per-class plan exceeds 16 compiled GEMM variants");
  if(std::find(covered.begin(),covered.end(),false)!=covered.end())throw std::invalid_argument("unclassified GEMM");
  result.rope_tile_per_block=result.kv_tile_per_block=result.activation_tile_per_block=result.combiner_tile_per_block=true;
  return result;
}
inline std::vector<GemmConfig> ClassCandidates(OperatorClass const& cls,
    frontend::ImportedSemantics const& imported,TargetSpec const& target,ScalarType dtype) {
  CandidateGenerator generator(target,dtype,{256,64,5});std::vector<GemmConfig> result;
  for(auto const& candidate:generator.Enumerate()) {
    auto const& t=candidate.traits();
    bool legal=true;
    for(auto id:cls.gemms) {
      auto const& op=imported.plan.gemms.at(id);
      legal &= op.k%t.alignment.a==0 && op.k%t.alignment.b==0;
    }
    if(!legal)continue;
    for(int split:{1,2,4,8,16,32})result.push_back({t.tile_m,t.tile_n,t.tile_k,t.stages,split});
  }
  return result;
}

struct ServingClassDomain {
  std::vector<GemmConfig> candidates;
  std::size_t raw=0,removed_r1=0,removed_r2=0,removed_r3=0;
};

inline ServingClassDomain ServingClassCandidates(
    OperatorClass const& cls,frontend::ImportedSemantics const& imported,
    TargetSpec const& target,int batch,int seq,
    bool enable_r2=true,bool enable_r3=false) {
  if(!imported.plan.serving || batch<1 || seq<1 || cls.gemms.empty())
    throw std::invalid_argument("serving domain requires a serving model and bound batch");
  auto id=cls.gemms.front();auto const& gemm=imported.plan.gemms.at(id);
  auto stage=std::find_if(imported.plan.stages.begin(),imported.plan.stages.end(),
      [&](auto const& s){return s.kind==frontend::PlanTaskKind::kGemm && s.gemm==id;});
  if(stage==imported.plan.stages.end())throw std::invalid_argument("serving GEMM has no stage");
  int const rows=stage->batch_rows ? batch : batch*seq;
  ServingPruneContext pruning{rows,int(gemm.n),int(gemm.k),
      gemm.epilogue==frontend::PlanGemm::Epilogue::kSwiGLU ? int(gemm.interleave_u):0,
      target.res.num_sms*std::max(1,target.res.max_threads_per_sm/128),&target};
  int group_width=0;
  if(imported.plan.buffers.at(gemm.b).pack_json.find("qkv_group_interleave")!=std::string::npos)
    for(auto const& attention:imported.plan.stages)
      if(attention.kind==frontend::PlanTaskKind::kFusedAttention &&
         attention.operands[0]==gemm.d && attention.extent>0) {
        group_width=int(gemm.n/attention.extent);break;
      }
  ServingClassDomain domain;
  for(int m:{16,32,64,128})for(int n:{32,64,128,256})for(int k:{64,128}) {
    int const per_stage=2*k*(m+n);
    int const stage_limit=target.res.max_dynamic_smem_per_cta/per_stage;
    for(int stages=2;stages<=stage_limit;++stages)
      for(int split:{1,2,4,8,16,32}) {
        if(gemm.epilogue==frontend::PlanGemm::Epilogue::kArgmaxPartial && split!=1)
          continue;
        ++domain.raw;
        GemmConfig candidate{m,n,k,stages,split};
        if(group_width && group_width%n) {++domain.removed_r1;continue;}
        if(PruneServingR1(candidate,pruning)) {++domain.removed_r1;continue;}
        if(enable_r2 && PruneServingR2(candidate,pruning)) {++domain.removed_r2;continue;}
        if(PruneServingR3(candidate,pruning,enable_r3)) {++domain.removed_r3;continue;}
        domain.candidates.push_back(candidate);
      }
  }
  return domain;
}
}
