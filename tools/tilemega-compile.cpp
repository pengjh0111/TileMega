// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <exception>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
std::string quote(std::string const& value) {
  std::string result = "'";
  for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
  return result + "'";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: tilemega-compile {STABLE_EXPORT.json|CG.mlir} {OUTPUT.cu|OUTPUT.so}\n";
    return 2;
  }
  try {
    mlir::MLIRContext context;
    context.getOrLoadDialect<tilemega::dialect::CGDialect>();
    tilemega::frontend::ImportSummary summary;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::filesystem::path input(argv[1]);
    if (input.extension() == ".mlir") {
      module = mlir::parseSourceFile<mlir::ModuleOp>(input.string(), &context);
      if (!module) throw std::runtime_error("cannot parse CG MLIR input");
      for (auto task : module->getOps<tilemega::dialect::TaskSpaceOp>()) {
        ++summary.task_spaces;
        summary.stages = std::max(summary.stages,
            static_cast<std::size_t>(task.getStage() + 1));
      }
      for (auto coupling : module->getOps<tilemega::dialect::CouplingOp>())
        ++summary.couplings;
      if (auto guards = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
              "tilemega.guard_count"))
        summary.guards = guards.getInt();
    } else {
      module = tilemega::frontend::TorchExportImporter{}.Import(argv[1], context, &summary);
    }
    std::string source = tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
    std::filesystem::path requested(argv[2]);
    bool shared = requested.extension() == ".so";
    std::filesystem::path cuda = shared
        ? std::filesystem::path(requested.string() + ".cu") : requested;
    std::ofstream output(cuda);
    if (!output) throw std::runtime_error("cannot open generated CUDA output");
    output << source;
    output.close();
    if (shared) {
      std::string root = TILEMEGA_SOURCE_DIR;
      std::string nvcc = std::getenv("CUDACXX") ? std::getenv("CUDACXX") :
                                                 "/usr/local/cuda/bin/nvcc";
      std::string command = quote(nvcc) +
          " -std=c++17 -O2 -arch=native -shared -Xcompiler=-fPIC -x cu" +
          " -I" + quote(root + "/include") +
          " -I" + quote(root + "/third_party/cutlass/include") +
          " -I" + quote(root + "/third_party/cutlass/tools/util/include") +
          " -I" + quote(root + "/third_party/cutlass/test") + " " +
          quote(cuda.string()) + " -x cu " +
          quote(root + "/lib/Target/TargetSpec.cpp") +
          " -L/usr/local/cuda/lib64 -lcudart -o " + quote(requested.string());
      int status = std::system(command.c_str());
      if (status != 0) throw std::runtime_error("nvcc failed while building shared object");
    }
    std::cerr << "CODEGEN_SUMMARY tasks=" << summary.task_spaces
              << " couplings=" << summary.couplings
              << " stages=" << summary.stages
              << " output=" << requested.string() << "\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "tilemega-compile: " << error.what() << "\n";
    return 1;
  }
}
