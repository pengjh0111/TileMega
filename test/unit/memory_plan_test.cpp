// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/MemoryPlan.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
#include <set>

namespace tilemega::tests::memory_plan_test {
namespace {
using namespace analysis;
ClosedForm F(long value){return ClosedForm::Constant(value);}
SemanticOp Op(char const* name,char const* output,unsigned columns) {
  SemanticOp op;op.name=name;op.kind=OperatorKind::kPointwise;
  op.dtype=ScalarType::kBF16;op.exact_task_access=true;
  op.domain={{"m",F(8)},{"n",F(columns)}};
  op.result={output,{{"row",F(8)},{"channel",F(columns)}}};
  op.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  op.task_space={std::string(name)+".owners",op.result.axes};
  op.task_map=op.result_map;return op;
}
frontend::ModelPlan Plan() {
  frontend::ModelPlan p;p.dm=p.forward=true;p.dtype="bf16";p.serving_seq=1;
  p.buffers.resize(3);
  for(unsigned id=0;id<3;++id) {
    auto& b=p.buffers[id];b.name=std::string(1,'a'+id);b.per_batch=8*(id==1?4:8);
    auto& l=b.layout;l.rank=2;l.logical[0]=l.physical[0]=8;
    l.logical[1]=l.physical[1]=id==1?4:8;l.strides[0]=l.logical[1];l.strides[1]=1;
  }
  return p;
}
}
int TestMemoryPlan(int,char**) {
  IslContext isl;auto a=Op("write_a","a",8),b=Op("read_a","b",4),c=Op("write_c","c",8);
  b.operands={{a.name,a.result,b.result_map}};
  Granularity g;g.Tile(a.name,"row",F(4)).Tile(a.name,"channel",F(8));
  g.Tile(b.name,"row",F(2)).Tile(b.name,"channel",F(4));
  g.Tile(c.name,"row",F(4)).Tile(c.name,"channel",F(8));
  auto graph=Instantiate(SemanticGraph{{a,b,c}},g);auto plan=Plan();
  auto no=solver::PlanBufferReuse(plan,graph,{},"none");
  assert(no.aliases.empty() && no.hazards.empty() && !no.arena_bytes);
  for(auto const* policy:{"greedy","l2"}) {
    auto memory=solver::PlanBufferReuse(plan,graph,{},policy,512);
    assert(memory.arena_bytes==512 && memory.live_peak_bytes==192 && memory.fits_l2_budget);
    assert(!memory.retained_internal_bytes && memory.total_internal_bytes==512);
    assert(memory.aliases.size()==3 && memory.aliases[0].buffer==0 && memory.aliases[2].buffer==2);
    assert(memory.aliases[0].offset==memory.aliases[2].offset &&
           memory.aliases[1].offset!=memory.aliases[0].offset);
    assert(memory.hazards.size()==2);
    using Pair=std::pair<std::vector<long>,std::vector<long>>;
    for(auto const& hazard:memory.hazards) {
      std::set<Pair> expected;
      // Independent byte enumeration, including old elements nobody read.
      for(unsigned row=0;row<8;++row)for(unsigned col=0;col<8;++col) {
        if(hazard.kind==StorageHazardKind::kWAR && col<4)
          expected.insert({{long(row/4)},{long(row/2)}});
        if(hazard.kind==StorageHazardKind::kWAW && col>=4)
          expected.insert({{long(row/4)},{long(row/4)}});
      }
      auto points=hazard.coupling.C.Points();
      assert(std::set<Pair>(points.begin(),points.end())==expected);
      assert(hazard.coupling.exact && hazard.coupling.tier==Tier::kAffine);
    }
  }
  auto external=plan;external.buffers[0].role="external";
  auto protected_storage=solver::PlanBufferReuse(external,graph,{},"greedy");
  for(auto const& alias:protected_storage.aliases)assert(alias.buffer!=0);
  auto output=plan;output.outputs.push_back({0,{}});
  auto observed=solver::PlanBufferReuse(output,graph,{},"greedy");
  for(auto const& alias:observed.aliases)assert(alias.buffer!=0);
  auto with_scratch=plan;frontend::PlanBuffer scratch;scratch.name="unproved_scratch";
  scratch.constant=384;with_scratch.buffers.push_back(scratch);
  auto accounted=solver::PlanBufferReuse(with_scratch,graph,{},"l2",1024);
  assert(accounted.arena_bytes==512 && accounted.retained_internal_bytes==768 &&
      accounted.total_internal_bytes==1280 && accounted.live_peak_bytes==960 &&
      !accounted.fits_l2_budget);
  auto at_limit=solver::PlanBufferReuse(with_scratch,graph,{},"l2",1280);
  assert(at_limit.fits_l2_budget && at_limit.aliases.size()==accounted.aliases.size());
  with_scratch.buffers.back().constant=0;with_scratch.buffers.back().per_batch=384;
  ParamBinding batch;batch.Bind("B",2);
  auto batched=solver::PlanBufferReuse(with_scratch,graph,batch,"l2",2048);
  assert(batched.retained_internal_bytes==1536 && batched.total_internal_bytes==2048 &&
      batched.fits_l2_budget);
  auto fill=plan;fill.buffers[2].layout.fill=codegen::DmFill::kNegativeInfinity;
  assert(solver::PlanBufferReuse(fill,graph,{},"greedy").arena_bytes==768);
  auto nonexact=graph;nonexact.nodes[1].element_access.reset();
  auto excluded=solver::PlanBufferReuse(plan,nonexact,{},"greedy");
  for(auto const& alias:excluded.aliases)assert(alias.buffer!=0 && alias.buffer!=1);
  auto too_small=solver::PlanBufferReuse(plan,graph,{},"l2",256);
  assert(!too_small.fits_l2_budget && too_small.arena_bytes==512);
  // A -> smaller B -> A-sized C in one slot: B cannot order the old tail.
  auto shrink=b;shrink.operands.clear();
  auto shrink_graph=Instantiate(SemanticGraph{{a,shrink,c}},g);
  auto shrink_plan=solver::PlanBufferReuse(plan,shrink_graph,{},"greedy");
  assert(shrink_plan.arena_bytes==256 && shrink_plan.aliases.size()==3);
  for(auto const& alias:shrink_plan.aliases)assert(alias.offset==0);
  bool tail=false,latest=false;
  using Pair=std::pair<std::vector<long>,std::vector<long>>;
  for(auto const& hazard:shrink_plan.hazards)if(hazard.coupling.dst.name=="write_c") {
    std::set<Pair> expected;
    auto source=hazard.coupling.src.name;
    // Enumerate flat BF16 element addresses independently of the planner.
    for(unsigned element=0;element<(source=="write_a"?64u:32u);++element)
      expected.insert({{long(element/32)},{long(element/(source=="write_a"?32:8))}});
    auto points=hazard.coupling.C.Points();
    assert(std::set<Pair>(points.begin(),points.end())==expected);
    tail|=source=="write_a";latest|=source=="read_a";
    assert(hazard.kind==StorageHazardKind::kWAW && hazard.coupling.exact);
  }
  assert(tail && latest);
  bool rejected=false;
  try{(void)solver::PlanBufferReuse(plan,graph,{},"l2");}
  catch(std::invalid_argument const&){rejected=true;}
  assert(rejected);
  std::cout<<"Memory plan: independent WAR/WAW enumeration, lifetimes and exclusions PASS\n";
  return 0;
}
}
