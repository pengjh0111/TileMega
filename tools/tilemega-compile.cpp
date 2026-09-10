// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Frontend/ExportBridge.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>

#include <exception>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
std::string quote(std::string const& value) {
  std::string result = "'";
  for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
  return result + "'";
}

struct VariantRequest {
  std::uint32_t seq_begin = 0;
  std::uint32_t seq_end = 0;
  tilemega::frontend::ImportOptions options;
};

std::int64_t requiredInteger(llvm::json::Object const& object,
                             llvm::StringRef name) {
  auto value = object.getInteger(name);
  if (!value) throw std::runtime_error("variant plan lacks integer " + name.str());
  return *value;
}

tilemega::frontend::GemmGranularity readGemm(llvm::json::Object const& object) {
  tilemega::frontend::GemmGranularity result;
  result.tile_m = requiredInteger(object, "tile_m");
  result.tile_n = requiredInteger(object, "tile_n");
  result.tile_k = requiredInteger(object, "tile_k");
  result.stages = requiredInteger(object, "stages");
  result.split_k = requiredInteger(object, "split_k");
  return result;
}

std::vector<VariantRequest> readVariants(std::string const& path,
                                         std::size_t gemm_count) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file) throw std::runtime_error("cannot read runtime variant JSON: " + path);
  auto parsed = llvm::json::parse(file.get()->getBuffer());
  if (!parsed) throw std::runtime_error("invalid runtime variant JSON: " + path);
  auto* root = parsed->getAsObject();
  if (!root || root->getString("schema") != "tilemega.runtime_variants.v1")
    throw std::runtime_error("unsupported runtime variant schema");
  auto* array = root->getArray("variants");
  if (!array || array->empty()) throw std::runtime_error("variant plan is empty");
  std::vector<VariantRequest> result;
  for (auto const& value : *array) {
    auto* object = value.getAsObject();
    if (!object) throw std::runtime_error("runtime variant is not an object");
    VariantRequest request;
    request.seq_begin = requiredInteger(*object, "seq_begin");
    request.seq_end = requiredInteger(*object, "seq_end");
    if (auto* attention = object->getArray("attention")) {
      for (auto const& value : *attention) {
        auto* choice = value.getAsObject();
        if (!choice) throw std::runtime_error("attention choice must be an object");
        auto stage = requiredInteger(*choice,"stage");
        auto chunks = requiredInteger(*choice,"chunks");
        auto extent = requiredInteger(*choice,"chunk_extent");
        if (stage < 0 || stage > INT_MAX || chunks <= 0 || chunks > INT_MAX ||
            extent <= 0 || extent > INT_MAX)
          throw std::runtime_error("invalid attention stage/chunks/scratch extent");
        request.options.attention.push_back({static_cast<int>(stage),
            {static_cast<std::uint32_t>(chunks),static_cast<std::uint32_t>(extent)}});
      }
    }
    if (auto own = object->getBoolean("rope_tile_per_block"))
      request.options.rope_tile_per_block = *own;
    if (auto own = object->getBoolean("kv_tile_per_block"))
      request.options.kv_tile_per_block = *own;
    if (auto own = object->getBoolean("activation_tile_per_block"))
      request.options.activation_tile_per_block = *own;
    if (auto own = object->getBoolean("combiner_tile_per_block"))
      request.options.combiner_tile_per_block = *own;
    if (auto balanced = object->getBoolean("balanced_placement"))
      request.options.balanced_placement = *balanced;
    if (auto separate = object->getBoolean("separate_residual_tasks"))
      request.options.separate_residual_tasks = *separate;
    if (auto* uniform = object->getObject("uniform")) {
      request.options.gemms.assign(gemm_count, readGemm(*uniform));
    } else if (auto* gemms = object->getArray("gemms")) {
      for (auto const& item : *gemms) {
        auto* gemm = item.getAsObject();
        if (!gemm) throw std::runtime_error("variant GEMM is not an object");
        request.options.gemms.push_back(readGemm(*gemm));
      }
      if (request.options.gemms.size() != gemm_count)
        throw std::runtime_error("variant GEMM list has wrong length");
    } else {
      throw std::runtime_error("variant needs either uniform or gemms plan");
    }
    result.push_back(std::move(request));
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  tilemega::analysis::IslContext isl_context;
  if (argc != 3 && argc != 5) {
    std::cerr << "usage: tilemega-compile {STABLE_EXPORT.json|CG.mlir} "
                 "{OUTPUT.cu|OUTPUT.so} [--variants PLAN.json]\n";
    return 2;
  }
  try {
    mlir::MLIRContext context;
    context.getOrLoadDialect<tilemega::dialect::CGDialect>();
    tilemega::frontend::ImportSummary summary;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::filesystem::path input(argv[1]);
    bool has_variants = argc == 5;
    if (has_variants && std::string(argv[3]) != "--variants")
      throw std::runtime_error("expected --variants before the plan path");
    std::string source;
    if (input.extension() == ".mlir") {
      if (has_variants)
        throw std::runtime_error("runtime variants require stable export JSON input");
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
      source = tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
    } else if (!has_variants) {
      module = tilemega::frontend::TorchExportImporter{}.Import(argv[1], context, &summary);
      source = tilemega::codegen::CouplingGraphToCUDA{}.Lower(*module);
    } else {
      auto bridge = tilemega::frontend::ReadExportBridge(argv[1]);
      auto plan = tilemega::frontend::BuildModelPlan(
          bridge.nodes, bridge.inputs, bridge.outputs);
      auto requests = readVariants(argv[4], plan.gemms.size());
      std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
      std::vector<tilemega::codegen::RuntimeVariantModule> inputs;
      modules.reserve(requests.size());
      inputs.reserve(requests.size());
      for (std::size_t i = 0; i < requests.size(); ++i) {
        modules.push_back(tilemega::frontend::TorchExportImporter{}.Import(
            argv[1], context, i == 0 ? &summary : nullptr,
            requests[i].options));
        inputs.push_back({*modules.back(), requests[i].seq_begin,
                          requests[i].seq_end});
      }
      source = tilemega::codegen::CouplingGraphToCUDA{}.LowerVariants(inputs);
    }
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
              << " symbolic_windows=" << summary.symbolic_windows
              << " fallback_windows=" << summary.fallback_windows
              << " output=" << requested.string() << "\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "tilemega-compile: " << error.what() << "\n";
    return 1;
  }
}
