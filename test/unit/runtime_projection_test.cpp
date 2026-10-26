// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <set>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <cassert>
#include <iostream>
#include <isl/polynomial.h>
#include <isl/set.h>
#include <isl/space.h>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::tests::runtime_projection_test {
using tilemega::analysis::QuasiPolynomial;
using tilemega::analysis::ParamBinding;

static void require(bool x,char const* message) {if(!x)throw std::runtime_error(message);}
static void exact(QuasiPolynomial const& a,QuasiPolynomial const& b) {
  // Semantic equality's scalar shortcut alone does not establish support equality.
  isl_ctx* ctx=isl_ctx_alloc();
  auto* lhs=isl_pw_qpolynomial_read_from_str(ctx,a.ToString().c_str());
  auto* rhs=isl_pw_qpolynomial_read_from_str(ctx,b.ToString().c_str());
  require(lhs&&rhs,"parse comparison");
  auto names=[](isl_pw_qpolynomial* p) {std::set<std::string> n;auto* space=isl_pw_qpolynomial_get_domain_space(p);int count=isl_space_dim(space,isl_dim_param);for(int i=0;i<count;++i) {char const* name=isl_space_get_dim_name(space,isl_dim_param,i);require(name,"unnamed parameter");n.insert(name);}isl_space_free(space);return n;};
  require(names(lhs)==names(rhs),"parameter sets differ");
  auto* parameters=isl_space_align_params(isl_space_params(isl_pw_qpolynomial_get_domain_space(lhs)),isl_space_params(isl_pw_qpolynomial_get_domain_space(rhs)));
  auto align=[&](isl_pw_qpolynomial* p) {
    auto* space=isl_space_align_params(isl_pw_qpolynomial_get_space(p),isl_space_copy(parameters));
    auto* u=isl_union_pw_qpolynomial_align_params(isl_union_pw_qpolynomial_from_pw_qpolynomial(p),isl_space_copy(parameters));
    auto* result=isl_union_pw_qpolynomial_extract_pw_qpolynomial(u,space);isl_union_pw_qpolynomial_free(u);return result;
  };
  lhs=align(lhs);rhs=align(rhs);require(lhs&&rhs,"parameter alignment failed");
  auto* ld=isl_pw_qpolynomial_domain(isl_pw_qpolynomial_copy(lhs));auto* rd=isl_pw_qpolynomial_domain(isl_pw_qpolynomial_copy(rhs));
  require(isl_set_is_equal(ld,rd)==isl_bool_true,"defined parameter/task domains differ");
  isl_set_free(ld);isl_set_free(rd);
  auto* difference=isl_pw_qpolynomial_sub(lhs,rhs);require(difference,"difference failed");
  require(isl_pw_qpolynomial_is_zero(difference)==isl_bool_true,"exact polynomial values differ");
  isl_pw_qpolynomial_free(difference);isl_space_free(parameters);isl_ctx_free(ctx);
  require(a.SemanticallyEqual(b,{})&&b.SemanticallyEqual(a,{}),"semantic API equality differs");
}
static QuasiPolynomial old_fold(std::vector<QuasiPolynomial> const& terms) {
  QuasiPolynomial result;for(auto const& x:terms)result=result.Add(x);return result;
}
static void term_tests() {
  std::vector<std::vector<std::string>> cases={
    {"[S] -> { S : 1<=S<=8 }","[S] -> { 2*S : 5<=S<=12 }"},
    {"[B,S] -> { B*floor((S+3)/4) : 1<=B<=16 and 1<=S<=64 }","[S,B] -> { 3*B : 2<=B<=8 and 0<=S<=32 }"},
    {"[S] -> { S : 1<=S<=8 }","[S] -> { -S : 1<=S<=8 }","[S] -> { 7 : 10<=S<=12 }"},
    {"[S] -> { 0 : false }","[S] -> { floor((S+1)/2) : 0<=S<=128 }"}};
  for(std::size_t case_id=0;case_id<cases.size();++case_id) {
    auto const& c=cases[case_id];
    std::vector<QuasiPolynomial> terms,seeded{QuasiPolynomial{}};
    for(std::size_t i=0;i<c.size();++i) {auto q=QuasiPolynomial::FromIslText(c[i]);if(case_id==2 && i<2)q=q.ScaleRational("1/2");terms.push_back(q);seeded.push_back(q);}
    exact(old_fold(terms),QuasiPolynomial::Sum(seeded));
  }
  std::cout<<"PASS four exact-domain/parameter/rational/overlap/empty-support sums\n";
}
static void projection_tests() {
  using namespace tilemega;
  for(bool split:{false,true}) for(int stages:{1,3,8}) {
    solver::ModelDescription model;model.dims=solver::ModelDims::Symbolic("S",3);model.seq_metric_parameter="S";model.stages.resize(stages);
    for(auto& s:model.stages)s.kind=solver::StageKind::kRMSNorm;
    codegen::RuntimePlan plan;plan.parameter_ranges={{"S",{1,128}}};
    solver::RuntimeProjectionOptions options{3,128,1};options.split_count_periods=split;options.count_wait_entries=false;
    auto p=solver::ProjectRuntimeQueues(model,plan,options);
    std::vector<QuasiPolynomial> tasks,workers;
    for(int i=0;i<stages;++i) {
      tasks.push_back(p.stages.at(i).task_count);
      auto owned=p.tasks.IntersectRange("{ [s,t] : s="+std::to_string(i)+" and t%3=0 }").Image().ImageCard();
      if(split)owned=owned.SplitPeriods(3);workers.push_back(owned);
    }
    exact(p.runtime_task_refs,old_fold(tasks));exact(p.max_worker_task_refs,old_fold(workers));
    require(p.stages.size()==std::size_t(stages),"stage order/count changed");
    for(int s:{-1,0,1,2,3,4,17,64,128,129}) {ParamBinding t;t.Bind("S",s);long n=s>=1&&s<=128?s:0;require(p.runtime_task_refs.Eval(t)==stages*n,"task closed form");require(p.max_worker_task_refs.Eval(t)==stages*((n+2)/3),"worker closed form");}
  }
  // Large repeated-stage test uses its independent exact closed form, not a slow
  // old prefix fold whose cost is the defect under review.
  solver::ModelDescription model;model.dims=solver::ModelDims::Symbolic("S",3);model.seq_metric_parameter="S";model.stages.resize(64);for(auto& s:model.stages)s.kind=solver::StageKind::kRMSNorm;
  codegen::RuntimePlan plan;plan.parameter_ranges={{"S",{1,128}}};solver::RuntimeProjectionOptions options{3,128,1};options.count_wait_entries=false;
  auto p=solver::ProjectRuntimeQueues(model,plan,options);
  auto tasks=QuasiPolynomial::FromIslText("[S] -> { 64*S : 1<=S<=128; 0 : S<=0 or S>=129 }");
  auto workers=QuasiPolynomial::FromIslText("[S] -> { 64*floor((S+2)/3) : 1<=S<=128; 0 : S<=0 or S>=129 }");
  exact(p.runtime_task_refs,tasks);exact(p.max_worker_task_refs,workers);
  std::cout<<"PASS six bounded old-fold projection comparisons +64stage exact closed form\n";
}

int TestRuntimeProjection(int argc, char** argv) {
  tilemega::analysis::IslContext context;
  term_tests();
  projection_tests();
  {
    for (auto const& text:{"{ [i] -> [j] : -3<=i<=4 and 2<=j<=7 }",
        "{ [i] -> [j] : 0<=i<=9 and j=i+2 }",
        "{ [i] -> [j] : 0<=i<=9 and 0<=j<=8 and (i+j)%2=0 }",
        "{ [i] -> [j] : 0<=i<=3 and 0<=j<=2; [i] -> [j] : 2<=i<=4 and 1<=j<=3 }",
        "{ [i] -> [j] : false }"}) {
      std::set<std::array<long,2>> actual,expected;
      auto relation=tilemega::analysis::CouplingRelation::FromIslText(text);
      for(auto const& [a,b]:relation.Points())expected.insert({a[0],b[0]});
      tilemega::analysis::VisitFiniteRelation(context,text,2,[&](long const* p){actual.insert({p[0],p[1]});});
      assert(actual==expected);
    }
    for(auto const& text:{"{ [s,i] -> [t,j] : s=1 and t=0 and 0<=i<4 and 512*i<=j<512*i+512 }",
        "{ [s,i] -> [t,j] : s=1 and t=0 and 0<=i<4 and 512*i<=j<512*i+512 and j%2=0 }"}) {
      std::set<std::array<long,4>> actual,expected;
      for(auto const& [a,b]:tilemega::analysis::CouplingRelation::FromIslText(text).Points())
        expected.insert({a[0],a[1],b[0],b[1]});
      tilemega::analysis::VisitFiniteRelation(context,text,4,[&](long const* p){actual.insert({p[0],p[1],p[2],p[3]});});
      assert(actual==expected);
    }
    auto band=isl_map_wrap(isl_map_read_from_str(context.raw(),
        "{ [s,i] -> [t,j] : s=1 and t=0 and 0<=i<4 and 512*i<=j<512*i+512 }"));
    int sliced_points=0;
    assert(tilemega::analysis::VisitFiniteSlices(band,4,[&](long const*){++sliced_points;},
        [](isl_set*){throw std::runtime_error("unexpected non-box slice");return isl_stat_error;}));
    isl_set_free(band);assert(sliced_points==2048);
    std::vector<std::array<long,2>> boundary;
    auto maximum=std::to_string(std::numeric_limits<long>::max());
    tilemega::analysis::VisitFiniteRelation(context,"{ [i] -> [j] : i="+maximum+" and j="+maximum+" }",2,
        [&](long const* p){boundary.push_back({p[0],p[1]});});
    assert(boundary.size()==1 && boundary[0][0]==std::numeric_limits<long>::max());
    bool rejected=false;
    try {tilemega::analysis::VisitFiniteRelation(context,"{ [i] -> [j] : i=0 and j=0 }",2,
        [](long const*){throw std::runtime_error("visitor failed");});}
    catch(std::runtime_error const& e){rejected=std::string(e.what()).find("visitor failed")!=std::string::npos;}
    assert(rejected);
  }

  {
    // A2's real counterexample: row-major (m,n,j) must address the same M
    // row as the emitted norm -> partial window for every split contribution.
    for (int chunks : {1,2,4,8,16}) for (int m=0;m<4;++m)
      for (int n=0;n<4;++n) for (int j=0;j<chunks;++j) {
        int task=(m*4+n)*chunks+j;
        auto coordinate=tilemega::codegen::DecodeSplitTask(task,16,chunks,true);
        assert(coordinate.chunk==j && coordinate.tile/4==m);
        assert(task/(4*chunks)==coordinate.tile/4);
      }
    tilemega::solver::ModelDescription model;
    model.dims = tilemega::solver::ModelDims::Symbolic("S",3);
    model.gemms = {{4,4,0,1}};
    tilemega::solver::ModelStage gemm;
    gemm.kind = tilemega::solver::StageKind::kGemm; gemm.gemm = 0;
    tilemega::solver::ModelStage element;
    element.kind = tilemega::solver::StageKind::kElementwise; element.extent = 4;
    model.stages = {gemm,element};
    tilemega::codegen::RuntimePlan plan;
    plan.gemms = {{0,2,2,2,2,1}};
    plan.ownership_flags = tilemega::codegen::kCombinerTileOwnership |
                           tilemega::codegen::kActivationTileOwnership;
    plan.dependencies = {{0,1,{true,2,2,0,2}}};
    for (bool periods : {false,true}) for (bool partition : {false,true})
      for (bool cg_order : {false,true}) for (int kappa : {0,1,2}) {
      auto projected = tilemega::solver::ProjectRuntimeQueues(model,plan,{2,2,kappa,false,cg_order,partition,periods});
      for (int s : {1,2,3,4,7,8,16}) {
        tilemega::analysis::ParamBinding theta; theta.Bind("S",s);
        int tiles_m = (s+1)/2;
        assert(projected.stages.size() == 3);
        assert(projected.runtime_task_refs.Eval(theta) == 6*tiles_m+s);
        assert(projected.max_worker_task_refs.Eval(theta) == 4*tiles_m);
        // In CG order each combiner needs one remote partial; in historical
        // chunk-major order all its partials were local on this two-worker grid.
        int waits = kappa == 0 ? 2+std::min(s,2) :
                    kappa == 1 ? (cg_order ? 2*tiles_m+s : s) :
                    (cg_order ? 2*tiles_m+s : 4*tiles_m+s);
        assert(projected.runtime_wait_entries.Eval(theta) == waits);
      }
    }
    // Graph-only projection must preserve every task/dependency/event relation
    // while refusing to advertise an uncomputed cardinality as a cost metric.
    tilemega::solver::RuntimeProjectionOptions graph_options{2,2,1};
    auto counted = tilemega::solver::ProjectRuntimeQueues(model,plan,graph_options);
    graph_options.count_wait_entries=false;
    auto graph_only = tilemega::solver::ProjectRuntimeQueues(model,plan,graph_options);
    assert(graph_only.tasks.ToString()==counted.tasks.ToString());
    assert(graph_only.dependencies.ToString()==counted.dependencies.ToString());
    assert(graph_only.requested_events.ToString()==counted.requested_events.ToString());
    assert(graph_only.waits.ToString()==counted.waits.ToString());
    bool missing_count_rejected=false;
    try { tilemega::solver::AttachProjectedEventMetrics(model,plan,graph_only); }
    catch (std::invalid_argument const&) { missing_count_rejected=true; }
    assert(missing_count_rejected);
    auto forced = tilemega::solver::ProjectRuntimeQueues(model,plan,{2,2,1,true});
    tilemega::solver::AttachRuntimeEventMetrics(model,plan,{2,2,1});
    assert(model.coupling_metrics.runtime);
    assert(model.coupling_metrics.runtime->stage_count==3);
    for (int seq:{1,4,128}) {
      tilemega::analysis::ParamBinding bindings; bindings.Bind("S",seq);
      auto bound=model.SubstituteParams(bindings);
      auto const& metrics=*bound.coupling_metrics.runtime;
      assert(metrics.task_refs.Eval({})==6*((seq+1)/2)+seq);
      assert(metrics.wait_entries.Eval({})==2*((seq+1)/2)+seq);
    }
    tilemega::analysis::ParamBinding four; four.Bind("S",4);
    assert(forced.runtime_wait_entries.Eval(four) == 4);
    auto balanced=tilemega::solver::BalanceProjectedQueues(forced,four,2);
    assert(balanced.task_ids.size()==16 && balanced.placement.max_queue<=8);
    auto concrete=tilemega::codegen::MaterializeRuntimeTaskGraph({8,4,4},
        {{0,1,true,1,0,0,1},{1,2,true,1,0,0,1}},2);
    auto exact=forced.dependencies.BindParams(four).Points();
    std::size_t edges=0;
    for (auto const& successors:concrete.successors) edges+=successors.size();
    assert(edges==exact.size());
    for (auto const& [consumer,producer]:exact) {
      int p=concrete.stage_offsets[producer[0]]+producer[1];
      int c=concrete.stage_offsets[consumer[0]]+consumer[1];
      assert(std::find(concrete.successors[p].begin(),concrete.successors[p].end(),c)!=concrete.successors[p].end());
    }
    // Element chunks own ceil(S*N/threads) combine and activation tasks.
    plan.ownership_flags = 0;
    for (int kappa : {0,1,2}) {
      auto projected = tilemega::solver::ProjectRuntimeQueues(model,plan,{2,2,kappa});
      for (int s : {1,3,8,16}) {
        tilemega::analysis::ParamBinding theta; theta.Bind("S",s);
        assert(projected.runtime_task_refs.Eval(theta) == 4*((s+1)/2)+4*s);
        // The outgoing split edge becomes aggregate under element ownership.
        assert(projected.runtime_wait_entries.Eval(theta) == 4);
      }
    }
    tilemega::solver::ModelDescription kv;
    kv.dims = tilemega::solver::ModelDims::Symbolic("S",0);
    kv.dims.past_parameter = "P";
    tilemega::solver::ModelStage append;
    append.kind = tilemega::solver::StageKind::kKVAppend;
    append.extent = 2; append.width = 4;
    kv.stages = {append};
    auto kv_projection = tilemega::solver::ProjectRuntimeQueues(kv,{}, {2,2,1});
    for (int s : {1,3,8}) for (int past : {0,3,9}) {
      tilemega::analysis::ParamBinding theta; theta.Bind("S",s).Bind("P",past);
      assert(kv_projection.runtime_task_refs.Eval(theta) == 4*std::max(s,past));
      assert(kv_projection.runtime_wait_entries.Eval(theta) == 0);
    }
    // §6 B2: a per-stage table that names the uniform kappa at every stage is
    // that uniform kappa -- the same relations, not merely the same counts --
    // so the dimension is inert until the search actually moves a stage.
    for (int kappa : {1,2}) {
      tilemega::solver::RuntimeProjectionOptions per_stage{2,2,kappa};
      per_stage.stage_kappa.assign(3,kappa);
      auto uniform = tilemega::solver::ProjectRuntimeQueues(model,plan,{2,2,kappa});
      auto table = tilemega::solver::ProjectRuntimeQueues(model,plan,per_stage);
      assert(table.requested_events.ToString()==uniform.requested_events.ToString());
      assert(table.waits.ToString()==uniform.waits.ToString());
      assert(table.dependencies.ToString()==uniform.dependencies.ToString());
      tilemega::analysis::ParamBinding theta; theta.Bind("S",8);
      assert(table.runtime_wait_entries.Eval(theta)==uniform.runtime_wait_entries.Eval(theta));
    }
    {
      // Tile ownership and chunk-major order are the setting where the two
      // uniform kappas disagree, so it is the setting where a mixed table can
      // be shown to follow the stage it names rather than the global value.
      auto owned=plan;
      owned.ownership_flags = tilemega::codegen::kCombinerTileOwnership |
                              tilemega::codegen::kActivationTileOwnership;
      tilemega::solver::RuntimeProjectionOptions base{2,2,1,false,false};
      auto one=tilemega::solver::ProjectRuntimeQueues(model,owned,base);
      base.kappa=2;
      auto two=tilemega::solver::ProjectRuntimeQueues(model,owned,base);
      tilemega::analysis::ParamBinding theta; theta.Bind("S",8);
      long const at_one=one.runtime_wait_entries.Eval(theta);
      long const at_two=two.runtime_wait_entries.Eval(theta);
      assert(at_one!=at_two);
      for (std::size_t stage=0;stage<3;++stage) {
        auto mixed=base;mixed.kappa=1;mixed.stage_kappa.assign(3,1);
        mixed.stage_kappa[stage]=2;
        long const at_mixed=tilemega::solver::ProjectRuntimeQueues(model,owned,mixed)
            .runtime_wait_entries.Eval(theta);
        // Every stage is either the one the coarsened edge produces for, in
        // which case the mixed table reaches kappa 2's count, or it is not,
        // in which case nothing moved.
        assert(at_mixed==at_one || at_mixed==at_two);
        std::cout << "PROJECTION_STAGE_KAPPA stage=" << stage << " mixed=" << at_mixed
                  << " kappa1=" << at_one << " kappa2=" << at_two << '\n';
      }
    }
    int before = context.ReferenceCount();
    bool rejected = false;
    try { (void)tilemega::solver::ProjectRuntimeQueues(model,plan,{0,2,1}); }
    catch (std::invalid_argument const&) { rejected = true; }
    assert(rejected && context.ReferenceCount() == before);
    std::cout << "PROJECTION_ERROR rejected=" << rejected << " before=" << before
              << " after=" << context.ReferenceCount() << '\n';
  }
  {
    tilemega::solver::ModelDescription model;
    model.dm=model.serving=true;model.dims.seq=model.dims.total=1;
    model.gemms={{64,64,0,1},{64,64,1,2}};
    model.gemm_access.resize(2);
    model.gemm_access[0].rows_per_batch=128;
    model.gemm_access[1].rows_per_batch=96;
    model.stages.resize(2);
    for(int i=0;i<2;++i) {model.stages[i].kind=tilemega::solver::StageKind::kGemm;model.stages[i].gemm=i;}
    tilemega::codegen::RuntimePlan plan;plan.gemms={{0,1,32,64,64,2},{0,1,32,64,64,2}};
    auto relation=tilemega::analysis::CouplingRelation::FromIslText(
        "{ [c] -> [p] : c=0 and p=1; [c] -> [p] : c=2 and (p=0 or p=3) }");
    auto table=tilemega::analysis::BuildDependencyTableLinear(relation,4,3);
    plan.dependencies={{0,1,{true,1,1,0,1},std::nullopt,0,table}};
    for(int grid:{1,4}) for(int kappa:{0,1,2,8}) {
      auto projected=tilemega::solver::ProjectRuntimeQueues(model,plan,{grid,128,kappa});
      assert(projected.dependencies.Points().size()==3);
      std::set<std::vector<long>> expected;
      for(auto [c,p]:std::vector<std::pair<int,int>>{{0,1},{2,0},{2,3}}) {
        if(kappa==1 && c%grid==p%grid)continue;
        expected.insert({c%grid,0,kappa==0?0:1,kappa==0?0:p/kappa});
      }
      assert(projected.runtime_wait_entries.Eval({})==expected.size());
      for(auto const& [consumer,event]:projected.requested_events.Points())assert(consumer[1]!=1);
    }
    auto forced=tilemega::solver::ProjectRuntimeQueues(model,plan,{4,128,1,true});
    assert(forced.dependencies.Points().size()==12 && forced.runtime_wait_entries.Eval({})==3);
    plan.dependencies[0].table->intervals[0].first=0;
    bool rejected=false;
    try { (void)tilemega::solver::ProjectRuntimeQueues(model,plan,{4,128,1}); }
    catch(std::invalid_argument const&) {rejected=true;}
    assert(rejected);
  }
  assert(context.ReferenceCount() == 0);

  return 0;
}

}  // namespace tilemega::tests::runtime_projection_test
