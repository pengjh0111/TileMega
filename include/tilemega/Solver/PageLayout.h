// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/TargetSpec.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace tilemega::solver {
struct PageLayout {
  int page_bytes=0, pages=0, activation_offset=0, scratch_offset=0;
  int pages_offset=0, shared_bytes=0;
  static constexpr int Align(int n,int alignment) {
    return (n+alignment-1)/alignment*alignment;
  }
  static constexpr bool StageFits(int page,int n,int k) {
    int b=n*k*2;
    return page>0 && b>0 && (page%b==0 || b%page==0);
  }
  static PageLayout Build(TargetSpec const& target,int page,int activation_bytes,
                          int scratch_bytes,int task_control_bytes=0) {
    if((page!=8192 && page!=16384) || activation_bytes<0 || scratch_bytes<0 || task_control_bytes<0)
      throw std::invalid_argument("invalid page-pool layout input");
    int limit=target.res.max_dynamic_smem_per_cta;
    // Each 32-byte slot owns full/empty barriers and a generation tag. Scratch
    // aliases activation storage only after the compute mainloop has drained.
    for(int count=limit/page;count>0;--count) {
      int operands=Align(count*4*sizeof(std::uint64_t)+task_control_bytes,128);
      int start=Align(operands+std::max(activation_bytes,scratch_bytes),1024);
      if(start+count*page<=limit)
        return {page,count,operands,operands,start,start+count*page};
    }
    throw std::invalid_argument("target shared memory cannot hold one page and task workspace");
  }
};
} // namespace tilemega::solver
