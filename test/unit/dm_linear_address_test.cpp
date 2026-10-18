// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/DmTensorAddress.h>
#include <cassert>
#include <cstdio>
#include <initializer_list>
using namespace tilemega;
std::uint64_t Reference(codegen::DmBufferLayout const& l,std::uint64_t row,unsigned col) {
  std::uint64_t result=col*l.strides[l.rank-1];
  for(unsigned axis=l.rank-1;axis-->0;) {
    auto coordinate=row%l.logical[axis];row/=l.logical[axis];
    if(l.kind==codegen::DmLayout::kNHWC) {
      if(axis==1)coordinate+=l.halo_top;
      if(axis==2)coordinate+=l.halo_left;
    }
    result+=coordinate*l.strides[axis];
  }
  return result;
}
void Check(codegen::DmBufferLayout const& l) {
  unsigned rows=1;for(unsigned i=0;i+1<l.rank;++i)rows*=l.logical[i];
  for(unsigned row=0;row<2*rows+1;++row)for(unsigned col=0;col<l.logical[l.rank-1];++col) {
    using A=backend::DmTensorAddress<codegen::DmWriteKind::kDense>;
    assert(A::Offset(l,row,col,0)==Reference(l,row,col));
  }
  unsigned scatter[]={rows-1,0,1,rows};
  for(unsigned i=0;i<4;++i)for(unsigned col=0;col<l.logical[l.rank-1];++col)
    assert(backend::DmTensorAddress<codegen::DmWriteKind::kRowScatter>::Offset(l,i,col,0,scatter)==Reference(l,scatter[i],col));
}
int main() {
  for(unsigned rank=1;rank<=4;++rank)for(unsigned gap:{0u,3u})for(unsigned element_stride:{1u,2u}) {
    codegen::DmBufferLayout l{};l.rank=rank;unsigned extents[]={2,3,5,7};
    for(unsigned i=0;i<rank;++i)l.logical[i]=l.physical[i]=extents[i];
    l.physical[rank-1]+=gap;l.strides[rank-1]=element_stride;
    for(unsigned i=rank-1;i-->0;)l.strides[i]=l.strides[i+1]*l.physical[i+1];
    Check(l);
    if(rank>2) {l.strides[0]+=11;Check(l);}
  }
  codegen::DmBufferLayout l{};l.kind=codegen::DmLayout::kNHWC;l.rank=4;
  unsigned logical[]={2,3,5,7},physical[]={2,5,7,8};
  for(unsigned i=0;i<4;++i){l.logical[i]=logical[i];l.physical[i]=physical[i];}
  l.halo_top=l.halo_left=1;l.strides[3]=1;
  for(unsigned i=3;i-->0;)l.strides[i]=l.strides[i+1]*l.physical[i+1];
  Check(l);std::puts("linear DM address: padded channels, strided axes, halo, scatter and wrap PASS");
}
