// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Backend/DmMoeOperands.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <type_traits>
#include <utility>

namespace tilemega::backend {
struct DmUnscaledActivation {
  template<class Fragment,class Coordinates>
  __device__ void operator()(Fragment,Coordinates const&,int) const {}
};
template<class T,class=void>struct HasDmActivationScale:std::false_type {};
template<class T>struct HasDmActivationScale<T,std::void_t<decltype(std::declval<T>().a_scale)>>:std::true_type {};

// SCA's multiply rounds to BF16 before the next GEMM consumes it. The
// multiplication takes place in the ldmatrix fragment, preserving that point
// without materializing the scaled image or changing its halo ownership.
template<class Operands,int TileM,int TileK>
struct DmScaledActivation {
  Operands const& p;
  int tile_m;
  template<class Fragment,class Coordinates>
  __device__ void operator()(Fragment values,Coordinates const& coordinates,int iteration) const {
    using namespace cute;
    using namespace codegen;
    if(!p.a_scale)return;
    bool conv=p.access.a==DmAAccess::kIm2Col;
    unsigned channels=conv?p.convolutions[p.access.conv].c:p.k_total;
    unsigned image_rows=conv?p.convolutions[p.access.conv].p*p.convolutions[p.access.conv].q:p.access.rows_per_batch;
    if(!image_rows) {asm volatile("trap;");return;}
    unsigned dtype=1;std::uint64_t pitch=channels;
    if(p.access.a_scale!=kDmNoIndex) {
      if(p.access.a_scale>=p.dm_buffers.count || !p.dm_buffers.dtypes || !p.dm_buffers.layouts) {
        asm volatile("trap;");return;
      }
      dtype=p.dm_buffers.dtypes[p.access.a_scale];
      auto const& layout=p.dm_buffers.layouts[p.access.a_scale];
      if(dtype>1 || (layout.rank && (layout.rank!=2 || layout.logical[1]!=channels || layout.strides[1]!=1))) {
        asm volatile("trap;");return;
      }
      if(layout.rank)pitch=layout.strides[0];
    }
    for(int i=0;i<size(values);++i) {
      unsigned row=tile_m*TileM+get<0>(coordinates(i));
      unsigned column=p.k_begin+iteration*TileK+get<1>(coordinates(i));
      bool valid=row<unsigned(p.m) && column<unsigned(p.k_begin+p.k_count);
      if(conv) {
        auto point=p.conv_iteration.At(p.k_begin/TileK+iteration,get<1>(coordinates(i)));
        column=point.c;valid=valid && point.valid;
      }
      valid=valid && column<channels;
      float scale=0;
      if(valid) {
        auto source_row=conv?row:DmSourceRow(p,row);
        auto index=std::uint64_t(source_row/image_rows)*pitch+column;
        scale=dtype==0?float(reinterpret_cast<cutlass::bfloat16_t const*>(p.a_scale)[index]):p.a_scale[index];
      }
      values(i)=cutlass::bfloat16_t(valid?float(values(i))*scale:0);
    }
  }
};
} // namespace tilemega::backend
