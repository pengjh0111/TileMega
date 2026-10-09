// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <iostream>
#include <cassert>
#include <set>
#include <stdexcept>

namespace tilemega::tests::fused_runtime_projection_test {

int TestFusedRuntimeProjection(int argc, char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  using R=analysis::CouplingRelation;
  analysis::IslContext isl;
  ModelDescription model;
  model.dims=ModelDims::Symbolic("S",0);
  ModelStage p,c;
  p.kind=c.kind=StageKind::kElementwise;
  p.extent=4; c.extent=8;
  model.stages={p,c,c};
  codegen::RuntimePlan plan;
  plan.dependencies={{0,1,{true,2,1,0,1}},{1,2,{true,1,1,0,1}}};
  auto relation=R::FromIslText("[S] -> { [c] -> [p] : S>=1 and 0<=c<2*S and p=floor(c/2) }");
  int checks=0,errors=0;
  auto reject=[&](auto action) {
    int before=isl.ReferenceCount(); bool caught=false;
    try { action(); } catch (std::invalid_argument const&) { caught=true; }
    if (!caught || before!=isl.ReferenceCount()) throw std::runtime_error("fusion projection error escaped or leaked");
    ++errors;
  };
  for (int kappa:{0,1,2,4}) {
    RuntimeProjectionOptions options{4,4,kappa};
    auto original=ProjectRuntimeQueues(model,plan,options);
    auto fused=FuseProjectedQueues(original,0,1,relation,options);
    for (int seq:{1,2,3,4,7,16}) {
      analysis::ParamBinding theta; theta.Bind("S",seq);
      auto const& result=fused.projection;
      if (result.stages.size()!=2 || result.runtime_task_refs.Eval(theta)!=4*seq ||
          result.max_worker_task_refs.Eval(theta)!=2*((2*seq+3)/4))
        throw std::runtime_error("fused task count or longest queue differs from explicit ownership");
      std::set<std::vector<long>> events;
      for (int task=0;task<2*seq;++task) {
        int worker=task%4;
        if (kappa==0) events.insert({worker,0,0,0});
        else if (kappa!=1) events.insert({worker,0,1,task/kappa});
      }
      if (result.runtime_wait_entries.Eval(theta)!=long(events.size()))
        throw std::runtime_error("fused event cardinality differs from explicit queue polls");
      auto points=result.dependencies.BindParams(theta).Points();
      if (points.size()!=std::size_t(2*seq)) throw std::runtime_error("fusion lost external edges");
      for (auto const& [consumer,producer]:points)
        if (consumer[0]!=1 || producer[0]!=0 || consumer[1]!=producer[1])
          throw std::runtime_error("fusion rewired wrong runtime coordinates");
      ++checks;
    }
    reject([&] { FuseProjectedQueues(original,0,2,relation,options); });
    reject([&] { auto wrong=options; wrong.grid*=2; FuseProjectedQueues(original,0,1,relation,wrong); });
    reject([&] { FuseProjectedQueues(original,0,1,R::FromIslText(
        "[S] -> { [c] -> [p] : 0<=c<S and p=c }"),options); });
    auto external=plan;
    external.dependencies.push_back({0,2,{true,2,1,0,1}});
    auto shared=ProjectRuntimeQueues(model,external,options);
    reject([&] { FuseProjectedQueues(shared,0,1,relation,options); });
  }
  auto optional_model=model;
  optional_model.stages[0].extent=4;
  optional_model.stages[1].extent=12;
  optional_model.stages[2].extent=12;
  RuntimeProjectionOptions options{4,4,1};
  auto optional_source=ProjectRuntimeQueues(optional_model,plan,options);
  auto optional=FuseProjectedQueues(optional_source,0,1,relation,options,true);
  for (int seq:{1,4,16}) {
    analysis::ParamBinding theta; theta.Bind("S",seq);
    if (optional.projection.runtime_task_refs.Eval(theta)!=6*seq ||
        optional.projection.dependencies.BindParams(theta).Points().size()!=std::size_t(3*seq))
      throw std::runtime_error("optional producer dropped independent consumer work");
  }
  // Counted publication survives fusion beside either endpoint. The
  // independent permutation oracle also changes the consumer's tile order.
  for(int selected:{0,2})for(int grid:{1,3,8})for(int kappa:{0,1,4,16})
    for(bool permuted:{false,true})for(int contracts:{1,2})for(bool mixed:{false,true}) {
    ModelDescription counted_model;counted_model.dims.seq=counted_model.dims.total=1;
    ModelStage stage;stage.kind=StageKind::kElementwise;stage.extent=5*128;
    counted_model.stages.assign(4,stage);
    codegen::RuntimePlan counted_plan;
    counted_plan.dependencies={{0,1,{true,1,1,0,1}},{2,3,{true,1,1,0,1}}};
    codegen::DependencyRecord edge{0,2,{true,1,0,0,1}};
    codegen::CountedWaitRecord contract;contract.producers=5;contract.tensor="partial";contract.unit_axes={0};
    contract.contributions.binding_source="rows";
    contract.contributions.target_units=R::FromIslText("{ [c] -> [row] : 0<=c<5 and 2*c<=row<2*c+2 and row<9 }");
    contract.contributions.expected={2,2,2,2,1};
    contract.conservative_relation=R::FromIslText("{ [c] -> [p] : 0<=c<5 and 0<=p<5 }");
    edge.counted=contract;counted_plan.dependencies.push_back(edge);
    if(contracts==2) {
      edge.counted->tensor="partial.other";
      counted_plan.dependencies.push_back(edge);
    }
    auto sparse=R::FromIslText("{ [c] -> [p] : (c=1 and (p=0 or p=4)) or (c=3 and p=1) or (c=4 and (p=2 or p=3)) }");
    if(mixed) {
      codegen::DependencyRecord ordinary{0,2,{}};
      ordinary.table=analysis::BuildDependencyTableLinear(sparse,5,5);
      counted_plan.dependencies.push_back(std::move(ordinary));
    }
    RuntimeProjectionOptions counted_options{grid,128,kappa};
    auto counted_original=ProjectRuntimeQueues(counted_model,counted_plan,counted_options);
    if(permuted)counted_plan.dependencies[selected==0?0:1].window={true,1,-1,4,1};
    counted_original=ProjectRuntimeQueues(counted_model,counted_plan,counted_options);
    auto identity=R::FromIslText(permuted?"{ [c] -> [p=4-c] : 0<=c<5 }":"{ [c] -> [p=c] : 0<=c<5 }");
    auto replacement=FuseProjectedQueues(counted_original,selected,selected+1,identity,counted_options).projection;
    assert(replacement.runtime_counted.size()==unsigned(contracts));
    auto const& transported=replacement.runtime_counted[0];
    assert(transported.producer==0 && transported.consumer==(selected==0?1:2));
    for(unsigned target=0;target<5;++target)
      assert(transported.contract.contributions.expected[target]==
          contract.contributions.expected[selected==2 && permuted?4-target:target]);
    assert(transported.contract.producers==5);
    assert(transported.contract.conservative_relation.IsSubset(contract.conservative_relation));
    assert(contract.conservative_relation.IsSubset(transported.contract.conservative_relation));
    unsigned arrivals=0;
    for(auto const& [task,event]:replacement.waits.Points())if(event[2]==3) {
      assert(task[0]==transported.consumer && event[0]==task[1]%grid);
      assert(event[1]==transported.producer);
      assert(event[3]==task[1] || (contracts==2 && event[3]==5+task[1]));++arrivals;
    }
    assert(arrivals==unsigned(5*contracts));
    if(contracts==2) {
      auto const& other=replacement.runtime_counted[1];
      assert(other.contract.tensor=="partial.other");
      assert(other.contract.contributions.expected==transported.contract.contributions.expected);
    }
    auto rebuilt=RebuildBoundRuntimeDependencies(replacement);
    unsigned counters=0,tables=0;
    for(auto const& dependency:rebuilt) {
      if(dependency.counted) {
        assert(dependency.counted->contributions.expected==transported.contract.contributions.expected);
        ++counters;
      }
      if(dependency.producer==0 && dependency.consumer==transported.consumer && dependency.table) {
        ++tables;
        std::set<std::pair<long,long>> expected,actual;
        for(auto const& [c,p]:sparse.Points())
          expected.emplace(selected==2 && permuted?4-c[0]:c[0],selected==0 && permuted?4-p[0]:p[0]);
        for(auto const& [c,p]:dependency.table->linear_relation.Points())actual.emplace(c[0],p[0]);
        assert(actual==expected);
      }
    }
    assert(counters==unsigned(contracts) && tables==unsigned(mixed));
    assert(replacement.runtime_tables.size()==unsigned(mixed));
    std::set<std::vector<long>> expected_waits,actual_waits;
    if(mixed)for(auto const& [c,p]:sparse.Points()) {
      long ct=selected==2 && permuted?4-c[0]:c[0];
      long pt=selected==0 && permuted?4-p[0]:p[0];
      if(kappa==1 && ct%grid==pt%grid)continue;
      expected_waits.insert({ct,ct%grid,0,kappa==0?0:kappa==1?2:1,kappa==0?0:pt/kappa});
    }
    for(auto const& [task,event]:replacement.waits.Points())
      if(task[0]==transported.consumer && event[1]==0 && event[2]!=3)
        actual_waits.insert({task[1],event[0],event[1],event[2],event[3]});
    assert(actual_waits==expected_waits);
    ++checks;
  }
  if (isl.ReferenceCount()) throw std::runtime_error("fusion projection retained references");
  std::cout << "FUSED_RUNTIME checks=" << checks << " errors=" << errors << " remaining=0\n";

  return 0;
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }

}  // namespace tilemega::tests::fused_runtime_projection_test
