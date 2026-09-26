// SPDX-License-Identifier: BSD-3-Clause
#pragma once
namespace mlir { class ModuleOp; }
namespace tilemega {
struct TargetSpec;
namespace codegen {
// The physical page layout is an execution decision, carried in the CG.
void ConfigureServingPages(mlir::ModuleOp,TargetSpec const&,int page_bytes);
}
}
