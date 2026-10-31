// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include <iostream>
using namespace tilemega::backend::mx;
template<int TN,int WN>
int Chunks(int n,int k,int part) {
  if(part)return MxFirstChunksV2<TN,WN,MxPart::kTileRange>(n,k,104);
  if constexpr(WN==1)return MxFirstChunksV2<TN,WN,MxPart::kStreamK>(n,k,104);
  return 0;
}
template<int TN>
int Width(int wn,int n,int k,int part) {
  if(wn==1)return Chunks<TN,1>(n,k,part);
  if(wn==2)return Chunks<TN,2>(n,k,part);
  if(wn==4)return Chunks<TN,4>(n,k,part);
  if(wn==8)return Chunks<TN,8>(n,k,part);
  return 0;
}
int main() {
  int n,k,b,tn,wn,depth,op,part,grouped;
  while(std::cin>>n>>k>>b>>tn>>wn>>depth>>op>>part>>grouped) {
    bool const legal=MxGemmLegalV3(n,k,b,tn,wn,depth,MxOpV2(op),
        part?MxPart::kTileRange:MxPart::kStreamK,bool(grouped));
    int chunks=0;
    if(legal) {
      if(tn==16)chunks=Width<16>(wn,n,k,part);
      if(tn==32)chunks=Width<32>(wn,n,k,part);
      if(tn==64)chunks=Width<64>(wn,n,k,part);
      if(tn==128)chunks=Width<128>(wn,n,k,part);
    }
    std::cout<<legal<<' '<<chunks<<'\n';
  }
}
