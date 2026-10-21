// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <cassert>
#include <set>

int main(int argc,char** argv) {
  if(argc!=2)return 2;
  using namespace tilemega::solver;
  auto target=tilemega::TargetSpec::FromJson(argv[1]);
  std::array<std::array<int,2>,5> shapes={{{4096,1024},{1024,2048},{6144,1024},
      {1024,3072},{151936,1024}}};
  for(int batch:{1,2,4,8,16}) {
    Sm90DenseSearchSpace space{shapes,batch,128,2,target};
    auto seeds=space.Seeds();assert(seeds.size()==6);
    assert(seeds[1].gemms[0].preissue_smem_bytes>0);
    assert(seeds[2].attention.prefetch_l2_bytes>0);
    for(auto const& seed:seeds) {
      auto neighbours=space.Neighbours(seed);assert(neighbours.size()>=8);
      std::set<std::string> keys{seed.ToJson().Dump(0)};
      for(auto const& c:neighbours) {
        assert(space.Legal(c));assert(keys.insert(c.ToJson().Dump(0)).second);
        assert(c.gemms[1].tile_n%32==0 || c.dn_vector_sums);
        assert(c.gemms[2].tile_n%(2*c.swiglu_u)==0);
        assert(c.attention.chunks!=1 || c.attention.new_row_work==0);
      }
    }
    if(batch==1) {
      auto const& base=seeds.front();auto neighbours=space.Neighbours(base);
      auto const& gate=base.gemms[2];
      auto impossible=base;impossible.gemms[2].preissue_smem_bytes=gate.slot_bytes;
      assert(!space.Legal(impossible));
      bool relocated=false;
      for(auto const& c:neighbours) {
        if(c.gemms[2].preissue_smem_bytes!=gate.slot_bytes)continue;
        assert(c.gemms[2].workspace_offset>0);
        auto restored=c;restored.gemms[2]=base.gemms[2];
        if(restored.ToJson().Dump(0)==base.ToJson().Dump(0))relocated=true;
      }
      assert(relocated);
    }
    auto invalid=seeds.front();invalid.gemms[1].tile_n=7;
    assert(!space.Legal(invalid));
    auto empty_late=seeds.front();empty_late.attention.preissue_late=true;
    assert(!space.Legal(empty_late));
    auto late=seeds[1];late.attention.preissue_late=true;
    assert(space.Legal(late));
    assert(Sm90DenseCoordinates::FromJson(late.ToJson()).attention.preissue_late);
    bool found_early=false;
    for(auto const& c:space.Neighbours(late)) {
      auto restored=c;restored.attention.preissue_late=true;
      if(!c.attention.preissue_late && restored.ToJson().Dump(0)==late.ToJson().Dump(0))
        found_early=true;
    }
    assert(found_early);
    auto overlap=seeds.front();overlap.gemms[2].preissue_smem_bytes=
        std::min(overlap.gemms[2].slots,shapes[2][1]/overlap.gemms[2].TileK())*
        overlap.gemms[2].slot_bytes;
    assert(!space.Legal(overlap));
    auto misaligned=seeds.front();misaligned.attention.preissue_smem_bytes=16;
    assert(!space.Legal(misaligned));
    auto excess=seeds.front();excess.gemms[0].workspace_offset=1024*1024;
    assert(!space.Legal(excess));
    auto too_large=seeds.front();too_large.attention.chunks=1;
    if(batch<=4)assert(!space.Legal(too_large));
    auto legacy=target;legacy.arch_tag="sm_89";
    assert(!(Sm90DenseSearchSpace{shapes,batch,128,2,legacy}.Legal(seeds.front())));
  }
}
