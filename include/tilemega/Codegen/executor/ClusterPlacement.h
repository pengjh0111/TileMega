// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifndef TILEMEGA_SERVING_CLUSTER_DIM
#define TILEMEGA_SERVING_CLUSTER_DIM 1
#endif

namespace tilemega::codegen::executor {
static_assert(TILEMEGA_SERVING_CLUSTER_DIM==1 ||
              TILEMEGA_SERVING_CLUSTER_DIM==2 ||
              TILEMEGA_SERVING_CLUSTER_DIM==4,
              "serving cluster dimension must be 1, 2 or 4");

struct ClusterTaskOwner { unsigned worker, slot; };

// A cluster launch does not rewrite sigma. Each logical task must retain one
// physical CTA owner before any DSMEM cooperation may be considered.
template<class Task>
inline std::vector<std::vector<ClusterTaskOwner>> ValidateClusterPlacement(
    unsigned cluster_dim,unsigned grid,std::vector<unsigned> const& counts,
    std::vector<std::uint32_t> const& offsets,std::vector<Task> const& tasks) {
  if((cluster_dim!=1 && cluster_dim!=2 && cluster_dim!=4) || !grid || grid%cluster_dim)
    throw std::invalid_argument("cluster placement requires a nonempty divisible grid");
  if(offsets.size()!=std::size_t(grid)+1 || offsets.front()!=0 ||
      offsets.back()!=tasks.size())
    throw std::invalid_argument("cluster placement queue offsets do not cover sigma");
  constexpr auto absent=std::numeric_limits<unsigned>::max();
  std::vector<std::vector<ClusterTaskOwner>> owners(counts.size());
  for(unsigned stage=0;stage<counts.size();++stage)
    owners[stage].assign(counts[stage],{absent,absent});
  for(unsigned worker=0;worker<grid;++worker) {
    if(offsets[worker]>offsets[worker+1] || offsets[worker+1]>tasks.size())
      throw std::invalid_argument("cluster placement queue offsets are not monotonic");
    for(unsigned slot=offsets[worker];slot<offsets[worker+1];++slot) {
      auto const& task=tasks[slot];
      if(task.stage>=counts.size() || task.logical_task>=counts[task.stage])
        throw std::invalid_argument("cluster placement task is out of range");
      auto& owner=owners[task.stage][task.logical_task];
      if(owner.worker!=absent)
        throw std::invalid_argument("cluster placement duplicates a logical task");
      owner={worker,slot-offsets[worker]};
    }
  }
  for(auto const& stage:owners)for(auto const& owner:stage)
    if(owner.worker==absent)
      throw std::invalid_argument("cluster placement omits a logical task");
  return owners;
}

// Callers must supply participants in their fixed numerical reduction order.
// This proves placement only; mailbox empty/full phases are a separate protocol.
inline void ValidateClusterCooperation(unsigned cluster_dim,
    std::vector<ClusterTaskOwner> const& participants) {
  if((cluster_dim!=2 && cluster_dim!=4) || participants.empty() ||
      participants.size()>cluster_dim)
    throw std::invalid_argument("cluster cooperation has an illegal participant count");
  unsigned cluster=participants.front().worker/cluster_dim,mask=0;
  for(auto const& owner:participants) {
    unsigned rank=owner.worker%cluster_dim;
    if(owner.worker/cluster_dim!=cluster || (mask&(1u<<rank)))
      throw std::invalid_argument("cluster cooperation requires one task per rank in one cluster");
    mask|=1u<<rank;
  }
}
} // namespace tilemega::codegen::executor
