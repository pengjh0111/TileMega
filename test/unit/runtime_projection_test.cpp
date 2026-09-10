// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Codegen/RuntimeTaskGraph.h>
#include <cassert>
#include <iostream>

int main() {
  tilemega::analysis::IslContext context;
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
    int before = context.ReferenceCount();
    bool rejected = false;
    try { (void)tilemega::solver::ProjectRuntimeQueues(model,plan,{0,2,1}); }
    catch (std::invalid_argument const&) { rejected = true; }
    assert(rejected && context.ReferenceCount() == before);
    std::cout << "PROJECTION_ERROR rejected=" << rejected << " before=" << before
              << " after=" << context.ReferenceCount() << '\n';
  }
  assert(context.ReferenceCount() == 0);
}
