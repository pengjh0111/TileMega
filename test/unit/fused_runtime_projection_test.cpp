// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <iostream>
#include <set>
#include <stdexcept>

int main() try {
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
  if (isl.ReferenceCount()) throw std::runtime_error("fusion projection retained references");
  std::cout << "FUSED_RUNTIME checks=" << checks << " errors=" << errors << " remaining=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
