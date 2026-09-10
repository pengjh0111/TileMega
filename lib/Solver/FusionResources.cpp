// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/FusionResources.h>
#include <tilemega/Analysis/ISLContext.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
int FusionCtasPerSm(FusionResources const& r, TargetSpec const& target,
                    int register_allocation_per_warp, int static_shared_bytes) {
  auto const& res=target.res;
  if (r.threads<=0 || r.registers<=0 || r.shared_bytes<0 ||
      static_shared_bytes<0 || register_allocation_per_warp<=0 ||
      res.warp_size<=0 || r.threads%res.warp_size!=0 ||
      res.regs_per_sm<=0 || res.max_threads_per_sm<=0 ||
      res.max_smem_per_sm<=0 || res.max_dynamic_smem_per_cta<=0)
    throw std::invalid_argument("fusion residency requires complete resource budgets");
  if (r.shared_bytes>res.max_dynamic_smem_per_cta) return 0;
  long long warp_regs=static_cast<long long>(r.registers)*res.warp_size;
  long long allocated=((warp_regs+register_allocation_per_warp-1)/
      register_allocation_per_warp)*register_allocation_per_warp;
  long long cta_regs=allocated*(r.threads/res.warp_size);
  long long shared=static_cast<long long>(r.shared_bytes)+static_shared_bytes;
  long long limit=std::min(res.regs_per_sm/cta_regs,
      static_cast<long long>(res.max_threads_per_sm/r.threads));
  if (shared) limit=std::min(limit,res.max_smem_per_sm/shared);
  return static_cast<int>(limit);
}

FusionResources DeriveFusionResources(analysis::FusionAccesses const& accesses,
    analysis::ParamBinding const& theta, std::map<std::string,int> const& element_bytes,
    BackendTraits const& producer, int producer_registers,
    BackendTraits const& consumer, int consumer_registers) {
  analysis::IslReferenceAudit audit(__func__);
  if (producer.threads<=0 || producer.threads!=consumer.threads ||
      producer.smem_bytes<0 || consumer.smem_bytes<0 ||
      producer_registers<=0 || consumer_registers<=0)
    throw std::invalid_argument("fusion requires compatible threads and measured resources");
  std::map<std::vector<long>,long> live;
  for (auto const& [tensor,relation]:accesses.intermediate_tiles) {
    auto bytes=element_bytes.find(tensor);
    if (bytes==element_bytes.end() || bytes->second<=0)
      throw std::invalid_argument("fusion intermediate dtype is missing");
    // Concrete tier-3 resource query; each unique element comes from the
    // exact access set. It does not sample theta or fit a symbolic footprint.
    for (auto const& [task,element]:relation.BindParams(theta).Points()) {
      auto& total=live[task];
      if (total>std::numeric_limits<int>::max()-bytes->second)
        throw std::overflow_error("fusion intermediate exceeds resource representation");
      total+=bytes->second;
    }
  }
  long peak=0;
  for (auto const& [task,bytes]:live) peak=std::max(peak,bytes);
  long scratch=std::max(producer.smem_bytes,consumer.smem_bytes);
  if (peak>std::numeric_limits<int>::max()-scratch)
    throw std::overflow_error("fusion shared resource overflow");
  return {peak,static_cast<int>(scratch+peak),
          std::max(producer_registers,consumer_registers),producer.threads};
}
}  // namespace tilemega::solver
