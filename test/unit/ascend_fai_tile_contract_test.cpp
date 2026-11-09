#include <tilemega/Target/Ascend/FaiTileContract.h>
#include <iostream>
int main() {
  using namespace tilemega::ascend::fai_tile;
  for(unsigned d:{64u,128u})for(unsigned rows:{1u,2u,3u,4u,16u,32u})
    for(unsigned keys:{65u,127u,128u,129u,255u,256u,257u,1087u}) {
      if(!Shape(rows,d,keys,1088) || UsefulRows(rows,0)+UsefulRows(rows,1)!=rows)return 1;
      auto b=Budget(d);
      if(b.l1>524288 || b.l0a>65536 || b.l0b>65536 || b.l0c>262144 || b.ub>221184)return 1;
    }
  if(Shape(0,128,65,1088) || Shape(33,128,65,1088) || Shape(3,32,65,1088) ||
     Shape(3,128,1089,1088) || Shape(3,128,65,64) || Shape(3,128,1,1089) || kModelAdmitted)return 1;
  if(UsefulRows(3,1)!=0 || UsefulRows(32,1)!=16 || UsefulRows(3,2))return 1;
  if(kMmadK<kKeys || Budget(64).l0a!=Budget(128).l0a ||
     Budget(64).l0b!=Budget(128).l0b || Budget(64).l1!=Budget(128).l1)return 1;
  unsigned flags[]={kFreeOutput[0],kFreeOutput[1],kFreeScore[0],kFreeScore[1],
                    kScoreReady,kOutputReady,kProbabilityReady};
  for(unsigned i=0;i<7;++i)for(unsigned j=i+1;j<7;++j)if(flags[i]==flags[j])return 1;
  std::cout<<"{\"status\":\"pass\",\"scope\":\"host shape/bank/participant contract; no device proof\"}\n";
}
