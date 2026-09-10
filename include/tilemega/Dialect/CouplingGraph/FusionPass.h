// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <string>
namespace mlir { class ModuleOp; }
namespace tilemega::dialect {
// A selected L-task pair, not a heuristic decision. Rejection leaves the
// original module intact. Runtime lowering has a separate readiness gate.
void FuseTaskPair(mlir::ModuleOp module,std::string const& producer,std::string const& consumer);
void RegisterFusionPass();
}
