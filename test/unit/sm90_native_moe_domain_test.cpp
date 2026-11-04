// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90NativeMoeDomain.h>
#include <tilemega/Solver/Sm90NativeCtg.h>
#include <tilemega/Analysis/ISLContext.h>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace tilemega::solver;
int main() {
  tilemega::analysis::IslContext isl;
  // Independent 60-digit decimal evaluations of E*(1-(1-8/E)^B)*tiles.
  int expected[2][5][4]={{{64,128,192,256},{120,240,360,480},{212,424,636,848},
      {336,672,1008,1344},{452,903,1355,1806}},
      {{64,128,192,256},{124,248,372,496},{233,466,699,932},
      {413,826,1239,1652},{659,1319,1978,2638}}};
  int batches[]={1,2,4,8,16},tiles[]={8,16,24,32};
  Sm90NativeCtgCache cache;
  for(int e=0;e<2;++e)for(int b=0;b<5;++b)for(int g=0;g<4;++g) {
    auto gu=Sm90NativeMoeDomain(e?128:64,8,batches[b],tiles[g]);
    assert(gu.tasks==expected[e][b][g]);
    assert(gu.ToJson().At("approximation").AsString("approximation")==kSm90NativeMoeDomainApproximation);
    assert(gu.ownership.Reverse().ImageCard().Eval({})==gu.tasks);
    for(int d=0;d<4;++d) {
      auto dn=Sm90NativeMoeDomain(e?128:64,8,batches[b],tiles[d]);
      auto relation=Sm90NativeMoeExpertCoupling(gu,dn);
      auto releases=Sm90NativeReleaseTable(relation,gu.tasks,dn.tasks,1,{},cache);
      assert(releases->size()==unsigned(dn.tasks));
      for(auto [last,consumer]:*releases) {
        int slot=consumer/dn.tiles;
        assert(last==std::min(gu.tasks,(slot+1)*gu.tiles)-1);
        assert(last/gu.tiles==slot);
      }
      auto combine=Sm90NativeReleaseTable(Sm90NativeMoeCombineCoupling(dn),
          dn.tasks,dn.tiles,1,{},cache);
      assert(combine->size()==unsigned(dn.tiles));
      for(auto [last,column]:*combine)
        assert(last==column+(dn.tasks-1-column)/dn.tiles*dn.tiles);
    }
  }
  // Round the task count, not the number of experts before tiling.
  assert(Sm90NativeExpectedExpertTasks(128,8,16,16)==1319);
  assert(Sm90NativeExpectedExpertTasks(128,8,16,16)!=82*16);
  for(auto args:{std::array<int,4>{0,8,1,16},{64,65,1,16},{64,8,0,16},
      {64,8,1,0},{128,8,16,std::numeric_limits<int>::max()}}) {
    bool rejected=false;
    try {(void)Sm90NativeExpectedExpertTasks(args[0],args[1],args[2],args[3]);}
    catch(std::invalid_argument const&) {rejected=true;}
    assert(rejected);
  }
}
