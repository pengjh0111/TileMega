// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <cassert>
#include <set>

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  using namespace tilemega::solver;
  auto target=tilemega::TargetSpec::FromJson(argv[1]);
  assert(Sm90DenseSharedBudget(target,1)==232448);
  assert(Sm90DenseSharedBudget(target,2)==115712);
  // Qwen3-0.6B shapes (prompt appendix A).
  std::array<std::array<int,2>,5> shapes={{{4096,1024},{1024,2048},{6144,1024},
      {1024,3072},{151936,1024}}};
  for(int batch:{1,2,4,8,16})for(int residency:{1,2}) {
    Sm90DenseSearchSpace space{shapes,batch,128,2,target};
    int grid=space.Grid(residency);
    for(unsigned f=0;f<5;++f)for(bool sums:{false,true}) {
      auto candidates=space.FamilyCandidates(f,residency,sums);
      // vs0 needs 32-column o/down tiles, which give Qwen0.6B < G/4 tasks; at
      // residency 2 its TN8 down cannot cover 48 KB within the shared budget.
      if(residency==1 && sums)assert(!candidates.empty());
      if(!sums && (f==1 || f==3))assert(candidates.empty());
      std::set<std::string> keys;
      for(auto const& g:candidates) {
        int n=shapes[f][0],k=shapes[f][1],tasks=n/g.tile_n;
        assert(n%g.tile_n==0 && g.tile_n%8==0 && g.tile_n/g.warps_n<=256 && k%g.TileK()==0);
        if(f<4)assert(4*tasks>=grid && tasks<=2*grid);
        else assert((g.tile_n==128 || g.tile_n==256) && g.TileK()<=256);
        if(!sums && (f==1 || f==3))assert(g.tile_n%32==0);
        // Ring rule: the fewest slots covering min(W_cta, 48 KB).
        double w_cta=double((tasks+grid-1)/grid)*g.tile_n*k*2;
        double need=std::min(w_cta,48.0*1024);
        assert(g.slots*double(g.slot_bytes)>=need);
        assert(g.slots==2 || (g.slots-1)*double(g.slot_bytes)<need);
        assert(keys.insert(std::to_string(g.tile_n)+":"+std::to_string(g.warps_n)+":"+
            std::to_string(g.TileK())).second);
      }
    }
    // D1: S0 is the largest power of two with one task per CTA in one wave.
    auto attention=space.AttentionCandidates(residency);
    int s0=attention.front().chunks;
    assert(s0*batch*8<=grid && 2*s0*batch*8>grid);
    for(auto const& a:attention)assert(a.chunks==s0 || a.chunks==s0/2 || a.chunks==s0/4);
  }
  // REF-R2 used Qwen0.6B B16 down TN16; it stays in the space.
  {
    Sm90DenseSearchSpace space{shapes,16,128,2,target};bool found=false;
    for(auto const& g:space.FamilyCandidates(3,1,true))found|=g.tile_n==16;
    assert(found);
  }
}
