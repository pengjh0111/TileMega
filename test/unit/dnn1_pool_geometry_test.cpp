// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <cassert>
#include <iostream>
using namespace tilemega;using namespace tilemega::solver;
int main(){
 TargetSpec target;target.res.max_smem_per_sm=233472;target.res.max_dynamic_smem_per_cta=232448;
 Sm90ForwardShape shape;shape.body=Sm90ForwardBody::kPool;Sm90ForwardStage c;c.body=shape.body;
 int cases=0;
 for(int b:{1,2,4,8,16})for(int h:{1,7,56,57})for(int w:{1,4,56})for(int channels:{3,8,32,64,128,144,512}){
  shape.channels=channels;shape.dnn1_geometry=json::Object{{"needs_stats",false},{"batch",b},{"output_width",w},{"output_height",h}};
  long long elements=1LL*b*h*w*channels,tasks=(elements+127)/128;
  for(int residency:{1,2}){
   auto r=QuerySm90ForwardResources(shape,c,target,residency);
   assert(r.task_count==tasks&&Sm90ForwardCostModel::Tasks(shape,c)==tasks);
   assert(r.shared_bytes==1024&&r.feasible);++cases;
  }
  assert((tasks-1)*128<elements&&tasks*128>=elements);
 }
 std::cout<<cases<<" task/resource checks passed\n";
}
