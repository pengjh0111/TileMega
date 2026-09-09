// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <cassert>
#include <iostream>

int main() {
  tilemega::analysis::IslContext context;
  {
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
    for (int kappa : {0,1,2}) {
      auto projected = tilemega::solver::ProjectRuntimeQueues(model,plan,{2,2,kappa});
      for (int s : {1,2,3,4,7,8,16}) {
        tilemega::analysis::ParamBinding theta; theta.Bind("S",s);
        int tiles_m = (s+1)/2;
        assert(projected.stages.size() == 3);
        assert(projected.runtime_task_refs.Eval(theta) == 6*tiles_m+s);
        assert(projected.max_worker_task_refs.Eval(theta) == 4*tiles_m);
        // Tile ownership makes every split contribution local for kappa=1.
        // Each elementwise row needs the other worker's one N tile.
        int waits = kappa == 0 ? 2+std::min(s,2) :
                    kappa == 1 ? s : 4*tiles_m+s;
        assert(projected.runtime_wait_entries.Eval(theta) == waits);
      }
    }
    auto forced = tilemega::solver::ProjectRuntimeQueues(model,plan,{2,2,1,true});
    tilemega::analysis::ParamBinding four; four.Bind("S",4);
    assert(forced.runtime_wait_entries.Eval(four) == 4);
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
