// SPDX-License-Identifier: BSD-3-Clause
#include "NativeSweep.h"
#include <cassert>
#include <set>

int main(int argc,char** argv) {
  if(argc!=4)return 2;
  using namespace tilemega;
  using namespace tilemega::commands::native_sweep;
  auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]),argv[3]);
  Sm90DenseCostModel cost(target,constants);
  auto config=json::ParseFile(argv[2]);
  auto w=Sm90DenseCoordinates::FromJson(config.At("coordinates"));
  Sm90DenseWorkload work;work.shapes={{{4096,1024},{1024,2048},{6144,1024},
      {1024,3072},{151936,1024}}};
  work.batch=1;work.head_dim=128;work.queries=2;work.hkv=8;work.layers=28;work.qk_norm=true;
  Sm90DenseSearchSpace space{work.shapes,1,128,2,target,8};
  assert(space.Legal(w));
  for(std::string family:{"qkv","o","gate_up","down","lm_head","attention"}) {
    auto sweep=Sweep(space,work,cost,w,family,4),again=Sweep(space,work,cost,w,family,4);
    assert(sweep.size()==again.size());
    assert(sweep.front().reference && sweep.front().selected);
    assert(sweep.front().coordinates.ToJson().Dump(0)==w.ToJson().Dump(0));
    std::set<std::string> keys;int selected=0;
    for(unsigned i=0;i<sweep.size();++i) {
      auto const& v=sweep[i];assert(keys.insert(v.key).second);
      assert(v.key==again[i].key && v.stage_ns==again[i].stage_ns);
      assert(v.coordinates.ToJson().Dump(0)==again[i].coordinates.ToJson().Dump(0));
      assert(v.selected==again[i].selected && v.reason==again[i].reason);
      if(v.selected) {++selected;assert(v.legal && space.Legal(v.coordinates));}
      assert(v.coordinates.residency==w.residency && v.coordinates.cluster==w.cluster);
      assert(v.coordinates.group_wait==w.group_wait && v.coordinates.embed_la==w.embed_la);
      assert(v.coordinates.weight_stream==w.weight_stream && v.coordinates.layer_loop==w.layer_loop);
      assert(v.bytes>0 && v.stage_ns>0);
      if(v.reference || family=="attention")continue;
      unsigned f=0;while(family!=kSm90DenseFamilies[f])++f;
      auto list=space.FamilyCandidates(f,w.residency);
      std::vector<Sm90DenseGemm> group;
      for(auto const& g:list)if(std::to_string(g.tile_n)==v.key)group.push_back(g);
      assert(!group.empty());
      auto rank=[&](auto const& a,auto const& b) {
        auto x=cost.Gemm(work,f,a,w.residency),y=cost.Gemm(work,f,b,w.residency);
        if(x!=y)return x<y;
        auto sx=a.SharedBytes(1,Sm90DenseCoordinates::ResidualFamily(f));
        auto sy=b.SharedBytes(1,Sm90DenseCoordinates::ResidualFamily(f));
        return sx!=sy?sx<sy:a.TileK()>b.TileK();
      };
      std::stable_sort(group.begin(),group.end(),rank);
      auto expected=w;expected.gemms[f]=group.front();Sm90DenseSearchSpace::Structure(expected);
      assert(expected.ToJson().Dump(0)==v.coordinates.ToJson().Dump(0));
    }
    assert(selected>=1 && selected<=4);
  }
  auto broken=w;broken.gemms[0].tile_n=48;
  assert(Reason(space,broken)=="D8_qkv_group");
  assert(Sweep(space,work,cost,w,"qkv",1).front().reference);
  tilemega::json::Array choices;
  auto n=Control(space,work,cost,w,"N",choices);
  assert(space.Legal(n) && !n.embed_la && n.attention.chunks==1);
  auto moe_path=std::string(argv[3])+"/configs/serving/sm90r5/ref_r4/qwen3_30b_a3b_b1.json";
  auto mw=Sm90DenseCoordinates::FromJson(json::ParseFile(moe_path));
  Sm90DenseWorkload m;m.shapes={{{5120,2048},{2048,4096},{0,0},{0,0},{151936,2048},
      {128,2048},{1536,2048},{2048,768}}};
  m.batch=1;m.head_dim=128;m.queries=8;m.hkv=4;m.layers=48;m.moe=true;m.qk_norm=true;m.experts=128;m.top_k=8;
  Sm90DenseSearchSpace ms{m.shapes,1,128,8,target,4};ms.moe=true;ms.experts=128;ms.top_k=8;
  assert(ms.Legal(mw));
  auto fused=Sweep(ms,m,cost,mw,"experts",4);
  assert(fused.front().reference && fused.front().coordinates.ToJson().Dump(0)==mw.ToJson().Dump(0));
  for(auto const& v:fused)if(v.selected) {
    auto c=v.coordinates;assert(ms.Legal(c));assert(c.gemms[6].slots==c.gemms[7].slots);
    if(!v.reference && c.gemms[6].slots<8) {
      ++c.gemms[6].slots;++c.gemms[7].slots;assert(!ms.Legal(c));
    }
  }

}
