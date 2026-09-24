// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/OperatorClasses.h>
#include <tilemega/Solver/SolverTiming.h>
#include <functional>
#include <optional>

namespace tilemega::solver {
struct VariantResources { int registers=0,shared_bytes=0,threads=0;bool compiled=false; };
struct ResourceEstimate { int registers=0,shared_bytes=0,threads=0,resident_limit=0; };
class VariantResourceCache {
 public:
  using Probe=std::function<VariantResources(std::string const&,GemmConfig const*,ScalarType)>;
  explicit VariantResourceCache(Probe probe,SolverTiming* timing=nullptr):probe_(std::move(probe)),timing_(timing) {}
  ResourceEstimate Estimate(std::vector<OperatorClass> const& classes,
      std::vector<GemmConfig> const& config,TargetSpec const& target,ScalarType dtype) {
    SolverPhase phase(timing_,"resource_probe");
    if(!nongemm_)nongemm_=ProbeOne("nongemm",nullptr,dtype);
    ResourceEstimate result{nongemm_->registers,nongemm_->shared_bytes,nongemm_->threads,0};
    for(std::size_t i=0;i<classes.size();++i) {
      auto key=std::make_pair(classes[i].signature,ClassGeometryKey(config[i]));
      auto it=variants_.find(key);
      if(it==variants_.end())it=variants_.emplace(key,ProbeOne(classes[i].signature,&config[i],dtype)).first;
      result.registers=std::max(result.registers,it->second.registers);
      result.shared_bytes=std::max(result.shared_bytes,it->second.shared_bytes);
      if(result.threads!=it->second.threads)throw std::invalid_argument("variant thread counts disagree");
    }
    result.resident_limit=ResidentLimit(result,target);
    return result;
  }
  static int ResidentLimit(ResourceEstimate const& r,TargetSpec const& target) {
    if(r.threads<=0 || r.registers<=0 || r.shared_bytes<0 || r.shared_bytes>target.res.max_dynamic_smem_per_cta)return 0;
    int warps=(r.threads+target.res.warp_size-1)/target.res.warp_size;
    // CUDA allocates register file slices per warp (256 registers on supported targets).
    int regs_per_warp=((r.registers*target.res.warp_size+255)/256)*256;
    int shared=((r.shared_bytes+255)/256)*256;
    return std::min({target.res.regs_per_sm/(warps*regs_per_warp),
      shared ? target.res.max_smem_per_sm/shared : target.res.max_threads_per_sm/r.threads,
      target.res.max_threads_per_sm/r.threads});
  }
 private:
  VariantResources ProbeOne(std::string const& signature,GemmConfig const* g,ScalarType dtype) {
    if(!probe_)throw std::invalid_argument("resource probe callback required");
    auto r=probe_(signature,g,dtype);
    if(r.registers<=0 || r.threads<=0 || r.shared_bytes<0)throw std::invalid_argument("invalid compiled variant resources");
    if(timing_ && r.compiled)timing_->Add("variant_compile");
    return r;
  }
  Probe probe_;SolverTiming* timing_;
  std::optional<VariantResources> nongemm_;
  std::map<std::pair<std::string,std::tuple<int,int,int,int,int>>,VariantResources> variants_;
};
}
