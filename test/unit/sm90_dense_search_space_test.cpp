// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <tilemega/Support/Json.h>
#include <cassert>
#include <set>

int main(int argc,char** argv) {
  if(argc<2)return 2;
  using namespace tilemega::solver;
  auto target=tilemega::TargetSpec::FromJson(argv[1]);
  assert(Sm90DenseSharedBudget(target,1)==232448);
  assert(Sm90DenseSharedBudget(target,2)==115712);
  // Qwen3-0.6B shapes (prompt appendix A).
  Sm90DenseShapeTable shapes={{{4096,1024},{1024,2048},{6144,1024},
      {1024,3072},{151936,1024}}};
  for(int batch:{1,2,4,8,16})for(int residency:{1,2}) {
    Sm90DenseSearchSpace space{shapes,batch,128,2,target};
    int grid=space.Grid(residency);
    for(unsigned f=0;f<5;++f) {
      auto candidates=space.FamilyCandidates(f,residency);
      // At residency 2 the TN8 down cannot cover 48 KB within one arena.
      if(residency==1)assert(!candidates.empty());
      std::set<std::string> keys;
      for(auto const& g:candidates) {
        int n=shapes[f][0],k=shapes[f][1],tasks=n/g.tile_n;
        assert(n%g.tile_n==0 && g.tile_n%8==0 && g.tile_n/g.warps_n<=256 && k%g.TileK()==0);
        if(f<4)assert(4*tasks>=grid && tasks<=2*grid);
        else assert((g.tile_n==128 || g.tile_n==256) && g.TileK()<=256);
        // D5 ring rule: the whole task resident (K/TK <= 8 slots) or the
        // fewest slots covering min(W_cta, 48 KB); either way inside one arena.
        int arena=Sm90DenseArenaBytes(Sm90DenseSharedBudget(target,residency));
        bool residual=Sm90DenseCoordinates::ResidualFamily(f);
        assert(g.SharedBytes(batch,residual)-1024<=arena);
        bool resident=k/g.TileK()<=8 && g.slots==std::max(2,k/g.TileK());
        double w_cta=double((tasks+grid-1)/grid)*g.tile_n*k*2;
        double need=std::min(w_cta,48.0*1024);
        assert(resident || (g.slots*double(g.slot_bytes)>=need &&
            (g.slots==2 || (g.slots-1)*double(g.slot_bytes)<need)));
        assert(keys.insert(std::to_string(g.tile_n)+":"+std::to_string(g.warps_n)+":"+
            std::to_string(g.TileK())).second);
      }
    }
    // D1: S0 is the largest power of two with one task per CTA in one wave.
    auto attention=space.AttentionCandidates(residency);
    int s0=attention.front().chunks;
    assert(s0*batch*8<=grid && 2*s0*batch*8>grid);
    // MR5 (R4) widened this to every power of two down to one (no merge), so
    // the old {s0, s0/2, s0/4} bound no longer holds.
    for(auto const& a:attention) {
      assert(a.chunks>=1 && a.chunks<=s0);
      assert((a.chunks&(a.chunks-1))==0);
    }
  }
  // REF-R2 used Qwen0.6B B16 down TN16; it stays in the space.
  {
    Sm90DenseSearchSpace space{shapes,16,128,2,target};bool found=false;
    for(auto const& g:space.FamilyCandidates(3,1))found|=g.tile_n==16;
    assert(found);
  }
  // A1 (RC1): a whole-row q/k norm reads every KV group of its token, so it
  // cannot follow a qkv that releases only this CTA's own groups. This is the
  // real regression: argv[2] is R4's selected OLMoE coordinate, which the old
  // Legal() admitted with group_wait and which lost OLMoE its C-2.
  if(argc==3) {
    auto coordinates=Sm90DenseCoordinates::FromJson(tilemega::json::ParseFile(argv[2]));
    // OLMoE decode: Hkv 16, Q 1, E 64 top-8 (prompt appendix A).
    Sm90DenseShapeTable olmoe={{{6144,2048},{2048,2048},{0,0},{0,0},
        {50304,2048},{64,2048},{2048,2048},{2048,1024}}};
    Sm90DenseSearchSpace plain{olmoe,1,128,1,target,16};
    plain.moe=true;plain.experts=64;plain.top_k=8;
    Sm90DenseSearchSpace rows=plain;rows.qk_norm_row=true;
    assert(coordinates.group_wait);
    // R4 could reach it; with the row norm declared it is now unreachable.
    assert(plain.Legal(coordinates));
    assert(!rows.Legal(coordinates));
    // Turning the group wait off restores legality under both spaces.
    auto free_running=coordinates;free_running.group_wait=false;
    assert(rows.Legal(free_running) && plain.Legal(free_running));
  }
}
