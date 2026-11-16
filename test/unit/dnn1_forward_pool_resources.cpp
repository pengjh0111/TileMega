#include <tilemega/Solver/ForwardGeometry.h>
#include <cassert>
using namespace tilemega;
using namespace tilemega::solver;
int main() {
  Sm90ForwardShape shape;shape.body=Sm90ForwardBody::kGlobalPool;
  Sm90ForwardStage coordinate;coordinate.body=shape.body;
  TargetSpec target;
  target.res.max_smem_per_sm=232448;
  target.res.max_dynamic_smem_per_cta=227328;
  for(int batch:{1,2,4,8,16})for(int channels:{3,32,64,144,512,1000}) {
    shape.channels=channels;
    shape.dnn1_geometry=json::Object{{"needs_stats",false},{"batch",batch},{"channels",channels}};
    for(int residency:{1,2}) {
      auto resource=QuerySm90ForwardResources(shape,coordinate,target,residency);
      assert(resource.shared_bytes==17408);
      assert(resource.task_count==batch*((channels+127)/128));
      assert(resource.feasible && !resource.exclusive);
    }
  }
  // The 17-KiB footprint needs exclusive placement on this synthetic target.
  // The resource gate must reject the smaller 16-KiB budget.
  target.res.max_smem_per_sm=target.res.max_dynamic_smem_per_cta=17408;
  auto exact=QuerySm90ForwardResources(shape,coordinate,target,1);
  assert(exact.feasible && exact.exclusive);
  assert(!QuerySm90ForwardResources(shape,coordinate,target,2).feasible);
  target.res.max_smem_per_sm=target.res.max_dynamic_smem_per_cta=16384;
  auto too_small=QuerySm90ForwardResources(shape,coordinate,target,1);
  assert(!too_small.feasible && !too_small.reason.empty());
}
