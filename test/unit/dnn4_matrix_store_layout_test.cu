// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Sm90F3Gemm.cuh>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cstdio>
#include <stdexcept>

template<int TN>
void Check() {
  using namespace cute;
  using namespace tilemega::backend;
  using Side=Sm90F3SideLayout<128,TN,false,0,false>;
  using Gemm=Sm90F3Gemm<128,TN,2,1,Sm90F3Mode::kNormal,
      128,TN,64,TN,Sm90F3Chain<Side>>;
  typename Gemm::Mma mma;
  typename Copy_Traits<SM90_U32x2_STSM_N>::SrcLayout source;
  for(int thread=0;thread<128;++thread) {
    auto owned=mma.get_slice(thread).partition_C(
        make_identity_tensor(Shape<_64,Int<TN>>{}));
    for(int group=0;group<TN/32;++group)for(int g=0;g<4;++g)
      for(int matrix=0;matrix<2;++matrix)for(int half=0;half<2;++half) {
        // Compare independent CuTe atom traits, including both packed rows.
        int bits=source(thread%32,matrix*32+half*16);
        int row=thread/32*16+bits/128;
        int column=group*32+g*8+(bits%128)/16;
        int i=group*16+g*4+2*matrix+half;
        if(row!=int(get<0>(owned(i))) || column!=int(get<1>(owned(i))))
          throw std::runtime_error("GMMA/STSM source coordinate mismatch");
        int address_row=thread/32*16+thread%16;
        int address_column=(group%2)*32+g*8;
        int address=address_row*128+
            (((address_column/8)^(address_row%8))*16);
        if(address<0 || address+16>8192 || address%16)
          throw std::runtime_error("STSM address outside SW128 64x64 box");
      }
  }
}

int main() {
  Check<64>();Check<128>();Check<192>();Check<256>();
  std::puts("40960 GMMA/STSM source coordinates and SW128 bounds PASS");
}
