// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<tilemega::dialect::CGDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TileMega Coupling Graph optimizer\n", registry));
}
