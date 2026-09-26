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
void RegisterHandoffPass();
}
