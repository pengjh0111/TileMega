// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/TargetSpec.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

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
  // The solver and codegen use one calculation for the workspace outside the
  // static page ring. Each GEMM shape is (M,N,K); attention widths are D.
  static std::pair<int,int> ServingWorkspace(
      std::vector<std::array<int,3>> const& gemms,
      std::vector<std::array<int,2>> const& attention_shapes) {
    int activation=0,scratch=0;
    for(auto const& g:gemms) {
      activation=std::max(activation,4*2*g[0]*g[2]);
      scratch=std::max(scratch,4*g[0]*g[1]+4*g[0]);
    }
    for(auto const& shape:attention_shapes) {
      int d=shape[0],q=shape[1];
      scratch=std::max(scratch,16*d*2+4*q*d*4+4*q*4);
    }
    return {activation,scratch};
  }
  static PageLayout Build(TargetSpec const& target,int page,int activation_bytes,
                          int scratch_bytes,int task_control_bytes=0) {
    if((page!=8192 && page!=16384) || activation_bytes<0 || scratch_bytes<0 || task_control_bytes<0)
      throw std::invalid_argument("invalid page-pool layout input");
    int limit=target.res.max_dynamic_smem_per_cta;
    // Each 32-byte slot owns full/empty barriers and a generation tag. Scratch
    // aliases activation storage only after the compute mainloop has drained.
    for(int count=limit/page;count>0;--count) {
      // Keep a dedicated 16-byte LA flag after the slot descriptors: the
      // reducer may reuse work[0] before all warps have read the flag.
      int operands=Align(count*4*sizeof(std::uint64_t)+16+task_control_bytes,128);
      int start=Align(operands+std::max(activation_bytes,scratch_bytes),1024);
      if(start+count*page<=limit)
        return {page,count,operands,operands,start,start+count*page};
    }
    throw std::invalid_argument("target shared memory cannot hold one page and task workspace");
  }
};
} // namespace tilemega::solver
