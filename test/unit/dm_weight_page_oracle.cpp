// SPDX-License-Identifier: BSD-3-Clause
#include <cute/layout.hpp>
#include <cute/swizzle_layout.hpp>
#include <cstdint>
#include <cstdio>

template<int N,int K> void Emit() {
  constexpr int atom=K<64?K:64,bits=K==16?1:K==32?2:3;
  auto layout=cute::tile_to_shape(cute::composition(cute::Swizzle<bits,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::Int<atom>>,
                   cute::Stride<cute::Int<atom>,cute::_1>>{}),
      cute::Shape<cute::Int<N>,cute::Int<K>>{});
  std::uint32_t header[]{N,K};
  std::fwrite(header,sizeof(std::uint32_t),2,stdout);
  for(int n=0;n<N;++n)for(int k=0;k<K;++k) {
    auto offset=std::uint32_t(layout(n,k));
    std::fwrite(&offset,sizeof(offset),1,stdout);
  }
}
template<int N> void Row() { Emit<N,16>();Emit<N,32>();Emit<N,64>();Emit<N,128>(); }
int main() { Row<8>();Row<16>();Row<32>();Row<64>();Row<128>();Row<256>(); }
