// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>

#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <llvm/Support/raw_ostream.h>

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  tilemega::analysis::IslContext isl_context;
  if (argc != 2 && !(argc==3 && std::string(argv[2])=="--separate-residual-tasks")) {
    std::cerr << "usage: tilemega-import STABLE_EXPORT.json [--separate-residual-tasks]\n";
    return 2;
  }
  try {
    mlir::MLIRContext context;
    context.getOrLoadDialect<tilemega::dialect::CGDialect>();
    tilemega::frontend::ImportSummary summary;
    tilemega::frontend::ImportOptions options;
    options.separate_residual_tasks=argc==3;
    auto module = tilemega::frontend::TorchExportImporter{}.Import(argv[1], context, &summary,options);
    module->print(llvm::outs(), mlir::OpPrintingFlags().enableDebugInfo(false));
    llvm::outs() << "\n";
    llvm::errs() << "IMPORT_SUMMARY tasks=" << summary.task_spaces
                 << " couplings=" << summary.couplings
                 << " stages=" << summary.stages
                 << " guards=" << summary.guards << "\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "tilemega-import: " << error.what() << "\n";
    return 1;
  }
}
