#include <tilemega/Solver/Sm90NativeFlow.h>
#include <tilemega/Solver/Sm90NativeCtg.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <cstdio>
#include <iostream>
using namespace tilemega; using namespace tilemega::solver;
// P2: the pool score must respond to the boundary semantics, not only to the
// tile geometry. Turning the group wait on replaces a whole-grid barrier with
// a per-group count on two boundaries, so the step must come down; and every
// stage must be reachable, which a zero or non-finite score would deny.
int main(int argc,char**argv){
  analysis::IslContext isl;
  auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]),argv[2]);
  Sm90DenseCostModel cost(target,constants);
  Sm90NativeCtgCache cache;
  auto relation=analysis::CouplingRelation::FromIslText(
      "{ [p] -> [c] : (c=0 and 4<=p<8) or (c=1 and 0<=p<2) or (c=2 and 2<=p<4) }");
  auto exact=Sm90NativeReleaseTable(relation,8,3,1,{},cache);
  assert((*exact==std::vector<std::pair<int,int>>({{1,1},{3,2},{7,0}})));
  assert(Sm90NativeReleaseTable(relation,8,3,1,{},cache)==exact);
  assert(cache.release_hits==1 && cache.release_misses==1);
  auto coarse=Sm90NativeReleaseTable(relation,8,3,4,{},cache);
  assert((*coarse==std::vector<std::pair<int,int>>({{3,1},{3,2},{7,0}})));
  Sm90DenseWorkload work;
  work.shapes={{{4096,2048},{2048,2048},{6144,2048},{2048,3072},{151936,2048}}};
  work.layers=28;work.head_dim=128;work.queries=2;work.hkv=8;work.qk_norm=true;
  for(int batch:{1,16}){
    work.batch=batch;
    Sm90DenseCoordinates c;
    for(int i=0;i<5;++i){auto&g=c.gemms[i];g.tile_n=i==4?128:32;g.slot_bytes=2*g.tile_n*64;}
    c.swiglu_u=8;c.attention.chunks=16/batch>0?16/batch:1;c.weight_stream=true;
    c.layer_loop=true;c.embed_la=true;c.residency=1;
    if(argc>3) {
      mlir::MLIRContext context;
      auto imported=ImportSm90Dense(argv[3],c,context);
      auto ctg=InstantiateSm90NativeCtg(imported,c,batch,context,cache);
      assert(ctg.spaces.size()==imported.plan.stages.size()+
          (c.attention.chunks>1?work.layers:0));
      assert(ctg.spaces[1].count==work.shapes[0][0]/c.gemms[0].tile_n);
      assert(ctg.spaces[2].count==batch*work.hkv*c.attention.chunks);
      bool qkv_attention=false;
      for(auto const& edge:ctg.edges)if(edge.producer==1 && edge.consumer==2) {
        qkv_attention=true;
        auto release=Sm90NativeReleaseTable(edge.relation,ctg.spaces[1].count,
            ctg.spaces[2].count,1,ctg.binding,cache);
        assert(!release->empty());
        // Each KV group reads its own qkv columns; this is a CTG query,
        // independent of the ratio of producer and consumer task counts.
        assert(release->front().first==work.shapes[0][0]/work.hkv/c.gemms[0].tile_n-1);
        assert(release->back().first==ctg.spaces[1].count-1);
        for(auto [last,consumer]:*release)
          assert(last==(consumer/(batch*c.attention.chunks)+1)*
              (work.shapes[0][0]/work.hkv/c.gemms[0].tile_n)-1);
      }
      assert(qkv_attention);
      auto misses=cache.coupling.misses;
      auto again=InstantiateSm90NativeCtg(imported,c,batch,context,cache);
      assert(cache.coupling.misses==misses && cache.coupling.hits>0);
      assert(!again.module && cache.space_hits>0 && cache.edge_hits>0);
      assert(again.edges.size()==ctg.edges.size());
      for(std::size_t i=0;i<ctg.edges.size();++i)
        assert(again.edges[i].relation.ToString()==ctg.edges[i].relation.ToString());
      assert(again.spaces.size()==ctg.spaces.size());
      if(batch==1) {
        auto changed=c;changed.gemms[0].tile_n*=2;changed.gemms[0].slot_bytes*=2;
        auto variant=InstantiateSm90NativeCtg(imported,changed,batch,context,cache);
        assert(variant.spaces[1].count*2==ctg.spaces[1].count);
        auto restored=InstantiateSm90NativeCtg(imported,c,batch,context,cache);
        assert(!restored.module && restored.spaces[1].count==ctg.spaces[1].count);
      }
    }
    double waited = 0, barriered = 0;
    for(bool wait:{false,true}){
      c.group_wait=wait;
      auto flow=Sm90NativePoolScore(work,c,cost);
      auto additive=cost.Predict(work,c);
      std::printf("B%-3d group_wait=%d  pool %9.1f ns/layer  step %10.1f  additive %10.1f  ratio %.3f\n",
        batch,int(wait),flow.pool_score_ns,flow.step_ns,additive.step_ns,flow.step_ns/additive.step_ns);
      assert(flow.pool_score_ns > 0 && flow.step_ns > 0);
      assert(flow.stage_ns.size() == flow.nodes.size());
      assert(flow.problem.spaces.back().name=="gemm4");
      assert(std::count_if(flow.problem.spaces.begin(),flow.problem.spaces.end(),
          [](auto const& s){return s.name=="gemm4";})==1);
      if (wait) waited = flow.step_ns; else barriered = flow.step_ns;
    }
    // A cheaper boundary cannot make the program slower.
    assert(waited <= barriered);
    std::printf("B%-3d group wait saves %.1f ns per step (%.2f%%)\n",
                batch, barriered - waited, (barriered - waited) / barriered * 100);
  }
  if(argc>3)cache.timing.Write(std::cout,"native","qwen3_1_7b",1);
  return 0;
}
