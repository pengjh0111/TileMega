// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90Gmma.h>
#include <cstdio>
#include <stdexcept>

using namespace cute;
using namespace tilemega::backend;
using Gmma=Sm90Gmma<tilemega::arch::Sm90>;
using QkMma=decltype(make_tiled_mma(typename Gmma::SS<128>{}));
using PvMma=decltype(make_tiled_mma(typename Gmma::RS<64,gmma::Major::MN>{}));

template<class Mma,int N>
void CheckOutput() {
  for(int thread=0;thread<128;++thread) {
    auto owned=Mma{}.get_slice(thread).partition_C(
        make_identity_tensor(Shape<_64,Int<N>>{}));
    for(int block=0;block<N/8;++block)for(int e=0;e<4;++e) {
      int row=thread/32*16+(thread%32)/4+8*(e/2);
      int col=block*8+2*(thread%4)+e%2;
      auto coord=owned(4*block+e);
      if(row!=int(get<0>(coord)) || col!=int(get<1>(coord)))
        throw std::runtime_error("warp-MMA/GMMA output coordinate mismatch");
    }
  }
}

int main() {
  CheckOutput<QkMma,128>();CheckOutput<PvMma,64>();
  using BLayout=decltype(tile_to_shape(gmma::Layout_MN_SW128_Atom<
      cutlass::bfloat16_t>{},Shape<_64,_128>{}));
  auto physical=as_position_independent_swizzle_layout(BLayout{});
  for(int n=0;n<64;++n)for(int k=0;k<128;++k) {
    int expected=k*128+(((n/8)^(k%8))*16)+(n%8)*2;
    if(2*physical(n,k)!=expected)
      throw std::runtime_error("PV V-panel SW128 coordinate mismatch");
  }
  for(int thread=0;thread<128;++thread) {
    auto a=PvMma{}.get_slice(thread).partition_A(
        make_identity_tensor(Shape<_64,_128>{}));
    for(int step=0;step<8;++step)for(int reg=0;reg<4;++reg)
      for(int half=0;half<2;++half) {
        int row=thread/32*16+(thread%32)/4+8*(reg%2);
        int col=step*16+2*(thread%4)+8*(reg/2)+half;
        auto coord=a(2*reg+half,0,step);
        if(row!=int(get<0>(coord)) || col!=int(get<1>(coord)))
          throw std::runtime_error("PV packed-probability RS coordinate mismatch");
      }
  }
  std::puts("28672 QK/PV fragment and SW128 coordinates PASS");
}
