// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/PlacementSolvePass.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/FusionPass.h>

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char** argv) {
  tilemega::analysis::IslContext isl_context;
  tilemega::dialect::RegisterFusionPass();
  tilemega::dialect::RegisterPlacementSolvePass();
  mlir::DialectRegistry registry;
  registry.insert<tilemega::dialect::CGDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TileMega Coupling Graph optimizer\n", registry));
}
