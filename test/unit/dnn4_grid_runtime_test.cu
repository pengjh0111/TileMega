// SPDX-License-Identifier: BSD-3-Clause
// Compile with DNN4_GRID_SOURCE naming a generated fixed-grid BERT source.
#include DNN4_GRID_SOURCE
#include <cstdio>
static_assert(TILEMEGA_F3_H800_GRID==1);

int main() {
  using namespace tilemega::codegen::forward_f3;
  auto* plan=new Plan;
  plan->epoch=42;
  for(int grid:{0,1,131,133,264})for(unsigned mode:{TM_SERVING_L1,TM_SERVING_L2}) {
    plan->grid=grid;
    if(tm_plan_launch(plan,0,mode,0,nullptr)!=-1 || plan->epoch!=42)return 1;
  }
  // No device allocation occurred; avoid Plan's CUDA destructor in this host test.
  std::puts("wrong_grid_rejected_before_epoch PASS");
  return 0;
}
