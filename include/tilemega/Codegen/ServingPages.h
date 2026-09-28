// SPDX-License-Identifier: BSD-3-Clause
#pragma once
namespace mlir { class ModuleOp; }
namespace tilemega {
struct TargetSpec;
namespace codegen {
// The physical page layout is an execution decision, carried in the CG.
void ConfigureServingPrefetch(mlir::ModuleOp,TargetSpec const&,int depth,int stride);
void ConfigureServingPages(mlir::ModuleOp,TargetSpec const&,int page_bytes);
// Resolve each selected decode GEMM's weight into its physical page recipe.
void ResolveServingWeightPacking(mlir::ModuleOp);
}
}
