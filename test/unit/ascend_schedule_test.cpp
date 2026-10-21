#include <tilemega/Solver/AscendSchedule.h>
#include <tilemega/Solver/PlanSkeleton.h>
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/AscendImplementation.h>
#include <tilemega/Target/Ascend/LlmPlanABI.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <limits>
#include <set>

using namespace tilemega::solver;
using namespace tilemega::ascend;
template<class F> void Reject(F f) {
  bool rejected=false;try{f();}catch(std::exception const&){rejected=true;}assert(rejected);
}
static WorkerMap Map() {
  TargetCapability t;t.arch_tag="ascend_3510";t.product_sku="950PR";t.cann_version="cpu_fixture";
  t.tracks={{"aic",TrackRole::kAic,2,{}, {ImplementationRole::kAicOnly,ImplementationRole::kMix1_2},"mix_group"},
            {"aiv",TrackRole::kAiv,4,{}, {ImplementationRole::kAivOnly,ImplementationRole::kMix1_2},"mix_group"}};
  t.max_verified_mix_groups=2;t.mix_1_2=true;
  return WorkerMap::Build(t,LaunchMode::kMix1_2,2);
}
static AscendScheduledTask Task(ImplementationRole role,double ns,std::vector<unsigned> deps={},std::vector<int> domain={}) {
  AscendScheduledTask t;t.role=role;t.price={ns,0,"CPU_TEST_ONLY_NOT_A_CALIBRATION"};
  t.predecessors=std::move(deps);t.legal_workers=std::move(domain);return t;
}
int main() {
  auto map=Map();AscendSchedulingCosts c{10,1,"CPU_TEST_ONLY_NOT_A_CALIBRATION"};
  auto aic=ImplementationRole::kAicOnly,aiv=ImplementationRole::kAivOnly,mix=ImplementationRole::kMix1_2;
  // A change in service prices changes actual FIFO order, not just a score.
  std::vector<AscendScheduledTask> branch={Task(aic,100,{}, {0}),Task(aic,10,{}, {0}),
      Task(aiv,1,{0},{2}),Task(aiv,200,{1},{2})};
  auto first=ScheduleAscendTasks(branch,map,c);assert(first.queue[first.offsets[0]]==1);
  branch[0].price.service_ns=1000;
  auto changed=ScheduleAscendTasks(branch,map,c);assert(changed.queue[changed.offsets[0]]==0);
  assert(first.queue!=changed.queue);
  auto evaluated=PriceAscendSchedule(branch,map,c,changed);
  assert(evaluated.queue==changed.queue && evaluated.owners==changed.owners && evaluated.makespan_ns>0);
  auto low_price=branch;low_price[0].price.service_ns=1;
  assert(PriceAscendSchedule(low_price,map,c,changed).makespan_ns<evaluated.makespan_ns);
  // Compare two legal body plans; changing a test-only price reverses choice.
  auto slow=branch;slow[0].price.service_ns=1200;
  assert(changed.makespan_ns<ScheduleAscendTasks(slow,map,c).makespan_ns);
  slow[0].price.service_ns=10;
  assert(changed.makespan_ns>ScheduleAscendTasks(slow,map,c).makespan_ns);

  std::vector<AscendScheduledTask> independent(8,Task(aiv,10));
  auto dual=ScheduleAscendTasks(independent,map,c);
  assert(std::set<unsigned>(dual.owners.begin(),dual.owners.end())==std::set<unsigned>({2,3,4,5}));
  auto illegal=independent;illegal[0].legal_workers={0};
  Reject([&]{ScheduleAscendTasks(illegal,map,c);});
  auto composite=ScheduleAscendTasks({Task(aic,5),Task(mix,10,{0},{1}),Task(aiv,5,{1})},map,c);
  assert(composite.participants[1]==std::vector<unsigned>({1,4,5}));
  auto single=Task(mix,10,{}, {0});single.mix_aiv_lanes=1;
  assert(ScheduleAscendTasks({single},map,c).participants[0]==std::vector<unsigned>({0,2}));

  // Three semantic edges become one prefix wait covering all earlier writes.
  auto chain=std::vector<AscendScheduledTask>{Task(aic,1,{}, {0}),Task(aic,1,{0},{0}),
      Task(aic,1,{1},{0}),Task(aiv,1,{0,1,2},{3})};
  auto compact=ScheduleAscendTasks(chain,map,c);
  assert(compact.semantic_edges==5 && compact.lowered_waits==1);
  assert(compact.waits[3].size()==1 && compact.waits[3][0].required_prefix==3);
  auto corrupt=compact;corrupt.waits[3][0].required_prefix=2;
  Reject([&]{AuditAscendSchedule(chain,map,corrupt);});
  corrupt=compact;std::swap(corrupt.queue[0],corrupt.queue[1]);
  Reject([&]{AuditAscendSchedule(chain,map,corrupt);});
  auto cycle=chain;cycle[0].predecessors={3};
  Reject([&]{ScheduleAscendTasks(cycle,map,c);});

  std::vector<AscendScheduledTask> large(5000,Task(aic,1,{}, {0}));
  auto scaled=ScheduleAscendTasks(large,map,c);assert(scaled.epoch_stride==5001);
  assert(AscendProgressValue(4096,5000,5001)==4096u*5001+5000);
  assert(AscendProgressValue(1,0,5001)>AscendProgressValue(0,5000,5001));
  Reject([&]{AscendProgressValue(std::numeric_limits<std::uint64_t>::max(),1,5001);});
  Reject([&]{AscendProgressValue(0,5001,5001);});
  auto unknown=large;unknown[0].price.service_ns=0;
  Reject([&]{ScheduleAscendTasks(unknown,map,c);});
  for(auto& t:unknown)t.price={};
  auto bootstrap=DiagnosticAscendSchedule(unknown,map);
  assert(bootstrap.queue.size()==5000 && bootstrap.epoch_stride==5001);
  assert(bootstrap.start_ns.empty() && bootstrap.finish_ns.empty());

  tilemega::ascend::llm::Stage abi{};abi.operands[0]=(5ULL<<30)+512;abi.epsilon=1e-6f;
  unsigned char raw[sizeof(abi)];std::memcpy(raw,&abi,sizeof(abi));
  tilemega::ascend::llm::Stage restored{};std::memcpy(&restored,raw,sizeof(restored));
  assert(restored.operands[0]==(5ULL<<30)+512 && restored.epsilon==1e-6f);

  // A 64-column LM-head candidate has two partials for vocabulary65. The
  // previous fixed ceil(vocab/32) map incorrectly demanded a third partial.
  tilemega::frontend::ModelPlan head;head.serving=true;head.serving_seq=1;head.serving_capacity=1088;
  head.buffers.resize(3);head.buffers[0].name="values";head.buffers[1].name="indices";head.buffers[2].name="tokens";
  tilemega::frontend::PlanStage reduce;reduce.kind=tilemega::frontend::PlanTaskKind::kArgmaxReduce;
  reduce.width=2;reduce.extent=65;reduce.operands.fill(UINT32_MAX);
  reduce.operands[0]=0;reduce.operands[1]=1;reduce.operands[2]=2;head.stages={reduce};
  tilemega::frontend::LiftOptions options;options.serving=true;options.batch_symbol="B";options.static_seq=1;
  auto lifted=tilemega::frontend::LiftServingSemantics(head,options);
  assert(lifted.sem.ops.size()==1 && lifted.sem.ops[0].domain.back().extent.IsLiteral(2));
  head.stages[0].width=0;
  Reject([&]{tilemega::frontend::LiftServingSemantics(head,options);});

  auto cube=Task(aic,10),vector=Task(aiv,10);cube.price.gm_bytes=vector.price.gm_bytes=100;
  auto shared=ScheduleAscendTasks({cube,vector},map,c);
  assert(shared.gm_floor_ns==20 && shared.makespan_ns==20);
  PlanSkeleton bound;bound.grid=2;bound.residency=1;bound.spaces.resize(2);
  bound.costs_calibrated=true;
  bound.BindAscendWorkers(map,{aic,aiv},true);
  assert(bound.spaces[1].legal_workers==std::vector<int>({2,3,4,5}));
  struct TestPrices final:AscendGemmPriceProvider {
    double second=20;
    AscendGemmEstimate Estimate(BackendCandidate const& c,GemmProblem const&) const override {
      return {c.traits().tile_n==64?10:second,2,"CPU_TEST_ONLY_NOT_A_CALIBRATION"};
    }
  };
  auto target=tilemega::TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/ascend_3510.json");
  std::vector<BackendCandidate> candidates;
  for(unsigned i=0;i<2;++i){auto t=As2Bf16GemmTraits(i);target.ascend_capability->implementations.emplace(t.ascend_implementation->impl_id,t.ascend_implementation->kind);candidates.emplace_back(t);}
  auto prices=std::make_shared<TestPrices>();CandidateGenerator generator(target,ScalarType::kBF16);
  generator.SetAscendDomain(candidates,prices);auto legal=generator.Enumerate();assert(legal.size()==2);
  assert(generator.TopK(legal,{{1,1024,3072}},1)[0].traits().tile_n==64);
  prices->second=5;
  assert(generator.TopK(legal,{{1,1024,3072}},1)[0].traits().tile_n==128);
  auto altered=candidates[0].traits();altered.tile_m=16;
  Reject([&]{generator.Analyze(BackendCandidate(altered),{1,1024,3072});});
  std::cout<<"{\"status\":\"passed\",\"price_changes_queue_and_plan\":true,"
               "\"both_aiv_domains\":true,\"frontier_edges\":5,\"frontier_waits\":1,"
               "\"large_tasks\":5000,\"epoch_overflow_rejected\":true,"
               "\"shared_gm_not_duplicated\":true,\"native_TopK_changes_selection\":true,"
               "\"native_abi_host_64bit_roundtrip\":true,\"argmax_actual_partial_domain\":true,"
               "\"diagnostic_has_no_prices\":true,\"evidence\":\"CPU fixtures only\"}\n";
}
