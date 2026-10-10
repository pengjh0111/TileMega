// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/Async.cuh>
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cutlass/bfloat16.h>

namespace tilemega::codegen {
struct LayoutConvertOperands {
  cutlass::bfloat16_t const* input=nullptr;
  cutlass::bfloat16_t* output=nullptr;
  DmBufferLayout layout{};
};
template<class Arch,int RowsPerTask=128>
struct LayoutConvertTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore && RowsPerTask>0);
  static constexpr int kThreads=128,kSharedBytes=0;
  __device__ static void Run(LayoutConvertOperands const& p,std::uint64_t task) {
    using namespace executor;
    auto const& l=p.layout;
    if(!p.input || !p.output || l.kind!=DmLayout::kNHWC || l.rank!=4 ||
       !l.logical[1] || !l.logical[2] || !l.logical[3] || l.strides[3]!=1 ||
       l.physical[3]<l.logical[3] || l.strides[2]<l.physical[3]) {
      asm volatile("trap;");return;
    }
    auto spatial=std::uint64_t(l.logical[1])*l.logical[2];
    auto rows=std::uint64_t(l.logical[0])*spatial;
    auto first=task*RowsPerTask;
    auto count=first<rows?(rows-first<RowsPerTask?rows-first:RowsPerTask):0;
    // Only interior pixels and their channel padding are writable. The halo
    // was initialized when the plan's storage was created and is read-only.
    for(std::uint64_t i=ComputeThread();i<count*l.strides[2];i+=kThreads) {
      auto row=first+i/l.strides[2],column=i%l.strides[2];
      auto image=row/spatial,pixel=row%spatial;
      auto target=image*l.strides[0]+(pixel/l.logical[2]+l.halo_top)*l.strides[1]+
          (pixel%l.logical[2]+l.halo_left)*l.strides[2]+column;
      p.output[target]=column<l.logical[3]?
          p.input[(image*l.logical[3]+column)*spatial+pixel]:cutlass::bfloat16_t(0);
    }
  }
};
} // namespace tilemega::codegen
