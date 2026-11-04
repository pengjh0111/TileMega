#include <tilemega/Solver/Sm90NativeFlow.h>
#include <cassert>
#include <cstdio>
using namespace tilemega; using namespace tilemega::solver;
// P2: the pool score must respond to the boundary semantics, not only to the
// tile geometry. Turning the group wait on replaces a whole-grid barrier with
// a per-group count on two boundaries, so the step must come down; and every
// stage must be reachable, which a zero or non-finite score would deny.
int main(int argc,char**argv){
  auto target=TargetSpec::FromJson(argv[1]);
  auto constants=Sm90NativeConstants::FromTarget(json::ParseFile(argv[1]),argv[2]);
  Sm90DenseCostModel cost(target,constants);
  Sm90DenseWorkload work;
  work.shapes={{{4096,2048},{2048,2048},{6144,2048},{2048,3072},{151936,2048}}};
  work.layers=28;work.head_dim=128;work.queries=2;work.hkv=8;work.qk_norm=true;
  for(int batch:{1,16}){
    work.batch=batch;
    Sm90DenseCoordinates c;
    for(int i=0;i<5;++i){auto&g=c.gemms[i];g.tile_n=i==4?128:32;g.slot_bytes=2*g.tile_n*64;}
    c.swiglu_u=8;c.attention.chunks=16/batch>0?16/batch:1;c.weight_stream=true;
    c.layer_loop=true;c.embed_la=true;c.residency=1;
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
  return 0;
}
