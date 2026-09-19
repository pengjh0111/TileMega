// SPDX-License-Identifier: BSD-3-Clause
/// §6 B2: the per-stage kappa a Plan carries is the kappa the placement
/// problem projects with. Without this the search dimension is unfalsifiable:
/// a table that never reaches the projection would report "no difference from
/// global kappa" for every model.
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <cassert>
#include <vector>
#include <iostream>

using namespace tilemega;

static std::string Shape(dialect::PreparedPlacementProblem const& p) {
  return p.projection.requested_events.ToString()+"\n"+p.projection.waits.ToString()+
         "\n"+p.projection.dependencies.ToString();
}

int main() try {
  analysis::IslContext isl;
  mlir::MLIRContext ctx;
  auto module=frontend::TorchExportImporter{}.Import(
      std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SEQSCAN/raw/export/gqa2.json",ctx);
  dialect::PlacementSolveOptions options;
  options.target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+"/configs/targets/sm_89.json");
  options.dims={4,3,7};options.residency=1;options.verified_resident_limit=1;
  options.requested_grid=8;options.kappa=1;
  auto const stages=dialect::PreparePlacementProblem(*module,options).counts.size();
  assert(stages>1);

  // The table is indexed by projected stage, not by model stage: a gemm stage
  // projects to more than one queue, so a table sized to the model would
  // coarsen the wrong producers.
  for (int kappa:{1,2,4}) {
    options.kappa=kappa;options.stage_kappa.clear();
    auto global=dialect::PreparePlacementProblem(*module,options);
    options.stage_kappa.assign(stages,kappa);
    auto table=dialect::PreparePlacementProblem(*module,options);
    assert(Shape(table)==Shape(global));
  }
  options.kappa=1;options.stage_kappa.clear();
  std::string const at_one=Shape(dialect::PreparePlacementProblem(*module,options));
  int moved=0;
  for (std::size_t stage=0;stage<stages;++stage) {
    options.stage_kappa.assign(stages,1);options.stage_kappa[stage]=4;
    if (Shape(dialect::PreparePlacementProblem(*module,options))!=at_one) ++moved;
  }
  // Which stages move is the model's answer; that some do is what makes the
  // dimension real.
  assert(moved>0);
  // A table that does not cover every projected stage would silently leave the
  // tail at the global value, so it is refused rather than padded.
  int refused=0;
  for (std::vector<int> bad:{std::vector<int>(stages-1,1),std::vector<int>(stages,-1)}) {
    options.stage_kappa=bad;
    try { dialect::PreparePlacementProblem(*module,options); }
    catch (std::exception const&) { ++refused; }
  }
  assert(refused==2);
  std::cout << "STAGE_KAPPA stages=" << stages << " moved_by_table=" << moved
            << " uniform_table_matches_global=1 refused=" << refused << "\n";
  return 0;
} catch (std::exception const& e) {
  std::cerr << "stage kappa test failed: " << e.what() << "\n";
  return 1;
}
