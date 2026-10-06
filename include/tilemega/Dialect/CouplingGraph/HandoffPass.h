// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/HandoffAccess.h>
#include <mlir/IR/BuiltinOps.h>
namespace tilemega::dialect {
class HandoffOp;
class TileSpaceOp;
class FusedTileSpaceOp;
analysis::TaskAccesses HandoffTaskAccesses(TileSpaceOp task);
void VerifyWrittenHandoff(FusedTileSpaceOp task);
// Reconstructs exact accesses from the graph's semantics and granularity.
// No model loader is used here: that loader itself invokes IR verification.
analysis::HandoffAccessProof VerifyHandoffAccess(HandoffOp handoff);
void ApplyHandoffs(mlir::ModuleOp module);
// Lower an access-proved serving handoff into the existing runtime stage
// table. The old stage slots remain as no-op release points so the solved
// queue and its conservative event windows stay valid.
void LowerServingHandoffStages(mlir::ModuleOp module);
struct ServingHandoffSelection {int recompute=0,last_arriver=0;};
// Build an access-proved decision plan from a solved serving CG. Unsupported
// edges remain ordinary events. The result is lowered conservatively into the
// already solved runtime stage slots. Class bits: 1 recompute, 2 paged
// merge/argmax, 4 split-K combine, 8 nonpaged attention merge.
ServingHandoffSelection SelectServingHandoffs(mlir::ModuleOp module,
    unsigned selected_classes=7);
void RegisterHandoffPass();
}
