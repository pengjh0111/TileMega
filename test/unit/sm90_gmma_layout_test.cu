// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90Gmma.h>
#include <cstdio>
#include <cstdlib>

template<int Rows,int K>
void Check() {
  using namespace cute;
  using LegacyAtom=decltype(composition(Swizzle<3,3,3>{},
      Layout<Shape<_8,_64>,Stride<_64,_1>>{}));
  auto legacy=tile_to_shape(LegacyAtom{},Shape<Int<Rows>,Int<K>>{});
  // GMMA carries a byte-address swizzle in the pointer flag. Compare its
  // physical element offsets at the required aligned base.
  auto hopper=as_position_independent_swizzle_layout(
      tilemega::backend::Sm90KLayout<Rows,K>{});
  for(int row=0;row<Rows;++row)for(int k=0;k<K;++k)
    if(legacy(row,k)!=hopper(row,k)) {
      std::fprintf(stderr,"layout %dx%d [%d,%d]: %d != %d\n",Rows,K,row,k,
          int(legacy(row,k)),int(hopper(row,k)));std::exit(1);
    }
}
int main() {
  Check<8,64>();Check<16,64>();Check<32,64>();Check<64,64>();
  Check<128,64>();Check<256,64>();
  Check<8,128>();Check<16,128>();Check<32,128>();Check<64,128>();
  Check<128,128>();Check<256,128>();
  std::puts("SW128 GEMM weights, activations and uniform KV: 12 layouts PASS");
}
