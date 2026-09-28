// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/ServingLag.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace tilemega::tests::serving_lag_test {
using namespace tilemega::codegen;

static StageDesc Stage(TaskKind kind) {
  StageDesc stage{};
  stage.kind=kind;
  return stage;
}

int TestServingLag(int, char**) {
  std::vector<StageDesc> stages{
      Stage(TaskKind::kEmbedding), Stage(TaskKind::kGemm),
      Stage(TaskKind::kGemmCombine), Stage(TaskKind::kFusedAttention),
      Stage(TaskKind::kAttentionMerge), Stage(TaskKind::kGemm),
      Stage(TaskKind::kGemmCombine), Stage(TaskKind::kGemm),
      Stage(TaskKind::kGemm), Stage(TaskKind::kGemmCombine),
      Stage(TaskKind::kGemm), Stage(TaskKind::kArgmaxReduce)};
  stages[3].handoff_reduce_stage=4;
  stages[4].handoff_elided=true;
  stages[11].handoff_elided=true;
  auto lag=ServingLagDependencies(stages);
  assert(lag.size()==2);
  assert(lag[0].kind==LagDependency::Kind::kToken &&
         lag[0].consumer==0 && lag[0].producer==11);
  assert(lag[1].kind==LagDependency::Kind::kHistoricalKv &&
         lag[1].consumer==3 && lag[1].producer==4);
  std::vector<std::uint32_t> flags(stages.size());
  flags[4]=flags[11]=kNeedsAggregateEvent;
  std::vector<std::uint32_t> offsets(stages.size()+1);
  for(std::size_t i=0;i<stages.size();++i)
    offsets[i+1]=offsets[i]+((flags[i]&kNeedsAggregateEvent)?1u:0u);
  ValidateServingLagDependencies(stages,flags,offsets,lag);
  flags[4]=0;
  bool rejected=false;
  try {ValidateServingLagDependencies(stages,flags,offsets,lag);}
  catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);

  stages[4].handoff_elided=false;
  lag=ServingLagDependencies(stages);
  assert(lag[1].producer==3); // A separate, non-elided merge does not publish the KV lag row.
  stages[3].handoff_reduce_stage=kNoOperand;
  stages.erase(stages.begin()+4);
  lag=ServingLagDependencies(stages);
  assert(lag[1].producer==3 && lag[0].producer==10);
  std::cout << "serving lag passed\n";
  return 0;
}
} // namespace tilemega::tests::serving_lag_test
