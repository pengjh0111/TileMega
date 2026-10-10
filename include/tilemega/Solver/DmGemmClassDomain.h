// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Solver/DmGemmCandidates.h>
#include <limits>
#include <map>
#include <set>

namespace tilemega::solver {

struct DmGemmClassDomain {
  std::vector<GemmConfig> candidates;
  std::size_t raw=0,removed_r1=0,removed_r2=0,removed_r3=0;
};

inline int DmGemmActiveRows(frontend::PlanGemm const& gemm,
    frontend::ModelPlan const& plan,std::size_t id,int batch,int seq) {
  if(batch<1 || seq<1)throw std::invalid_argument("DM class needs bound batch and sequence");
  auto const& access=gemm.access;
  if(access.b==codegen::DmBAccess::kExpertIndirect) {
    if(!access.block_rows || access.block_rows>std::uint32_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("DM expert class has no bounded row block");
    // Virtual blocks repeat this M geometry. Their count affects placement,
    // while padding dominance is determined inside each binding block.
    return int(access.block_rows);
  }
  auto stage=std::find_if(plan.stages.begin(),plan.stages.end(),[&](auto const& s) {
    return frontend::IsGemmStage(s.kind) && s.gemm==id;
  });
  if(stage==plan.stages.end())throw std::invalid_argument("DM class GEMM has no stage");
  auto rows=std::uint64_t(batch)*(access.rows_per_batch?access.rows_per_batch:
      (stage->batch_rows?1:seq));
  if(rows>std::uint64_t(std::numeric_limits<int>::max()))
    throw std::overflow_error("DM class row count exceeds runtime range");
  return int(rows);
}

inline DmGemmClassDomain DmClassCandidates(std::vector<std::size_t> const& ids,
    frontend::ModelPlan const& plan,TargetSpec const& target,int batch,int seq,
    std::map<std::size_t,DmWeightLayoutConstraint> const& shared={}) {
  if(!plan.dm || (!plan.serving && !plan.forward) || plan.dtype!="bf16" || ids.empty() ||
     target.res.max_dynamic_smem_per_cta<=0)
    throw std::invalid_argument("DM class requires a serving BF16 plan and target budget");
  std::set<std::size_t> seen;
  for(auto id:ids) {
    if(!seen.insert(id).second)throw std::invalid_argument("DM class repeats a GEMM");
    (void)DmGemmActiveRows(plan.gemms.at(id),plan,id,batch,seq);
  }
  DmGemmClassDomain result;
  for(int m:{16,32,64,128})for(int n:{16,32,64,128,256})for(int k:{16,32,64,128}) {
    if(m*n>16384)continue;
    int limit=std::min(8,target.res.max_dynamic_smem_per_cta/(2*k*(m+n)));
    for(int stages=2;stages<=limit;++stages)for(int split:{1,2,4,8,16,32}) {
      ++result.raw;
      GemmConfig g{m,n,k,stages,split};
      bool legal=true;
      for(std::size_t member=0;member<ids.size();++member) {
        auto id=ids[member];auto const& gemm=plan.gemms.at(id);
        auto constraint=shared.find(id);
        auto layout=constraint==shared.end()?DmWeightLayoutConstraint{}:constraint->second;
        if(!DmGemmCandidateRejection(gemm,plan,g,target,layout).empty()) {
          legal=false;break;
        }
      }
      if(!legal){++result.removed_r1;continue;}
      // Novel narrow-N layouts change warp and K-partial work. Legacy R-2
      // padding dominance is not proved across these body families.
      result.candidates.push_back(g);
    }
  }
  return result;
}

} // namespace tilemega::solver
