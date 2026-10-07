// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ModelDescription.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>
#include <cassert>
#include <stdexcept>

namespace tilemega::tests::forward_frontend_test {
int TestForwardFrontend(int argc, char** argv) {
  using namespace frontend;
  using namespace codegen;
  analysis::IslContext isl;
  mlir::MLIRContext context;
  for (bool token_axis : {false, true}) {
    llvm::json::Array shape = token_axis ? llvm::json::Array{"s0", "128"}
                                        : llvm::json::Array{"s0", "17", "128"};
    auto node = [](int index, char const* name, char const* op, char const* target,
                   llvm::json::Array inputs, llvm::json::Array dimensions) {
      return llvm::json::Object{{"index", index}, {"name", name}, {"op", op},
          {"target", target}, {"inputs", std::move(inputs)},
          {"shape", std::move(dimensions)}, {"dtype", "torch.bfloat16"}};
    };
    llvm::json::Value json = llvm::json::Object{
        {"schema", "tilemega.exported_program.v1"}, {"guards", llvm::json::Array{}},
        {"range_constraints", llvm::json::Object{{"s0", "VR[1, 4096]"}}},
        {"nodes", llvm::json::Array{
            node(0, "x", "placeholder", "x", {}, std::move(shape)),
            node(1, "weight", "placeholder", "weight", {}, llvm::json::Array{"64", "128"}),
            node(2, "linear", "call_function", "aten.linear.default", llvm::json::Array{"x", "weight"},
                 token_axis ? llvm::json::Array{"s0", "64"}
                            : llvm::json::Array{"s0", "17", "64"})}},
        {"signature", llvm::json::Object{
            {"inputs", llvm::json::Array{
                llvm::json::Object{{"name", "x"}, {"kind", "USER_INPUT"}, {"target", ""}},
                llvm::json::Object{{"name", "weight"}, {"kind", "PARAMETER"}, {"target", "weight"}}}},
            {"outputs", llvm::json::Array{
                llvm::json::Object{{"name", "linear"}, {"kind", "USER_OUTPUT"}}}}}}};
    int fd;
    llvm::SmallString<128> filename;
    assert(!llvm::sys::fs::createTemporaryFile("dm-forward", "json", fd, filename));
    { llvm::raw_fd_ostream out(fd, true); out << llvm::formatv("{0:2}", json); }
    ModelPlan plan;
    plan.dm = plan.forward = true; plan.forward_token_axis = token_axis;
    plan.dtype = "bf16"; plan.serving_seq = token_axis ? (argc == 4 ? std::stoi(argv[3]) : 16) : 1;
    plan.buffers.resize(3);
    for (unsigned i = 0; i < 3; ++i) {
      auto& buffer = plan.buffers[i];
      buffer.name = i == 0 ? "x" : i == 1 ? "weight" : "out";
      buffer.external_name = buffer.name; buffer.role = "external";
      if (i == 1) buffer.constant = 64 * 128;
      else if (token_axis) buffer.per_seq = i == 0 ? 128 : 64;
      else buffer.per_batch = 17 * (i == 0 ? 128 : 64);
    }
    PlanGemm gemm; gemm.n = 64; gemm.k = 128; gemm.b = 1; gemm.d = 2;
    gemm.access.rows_per_batch = token_axis ? 0 : 17;
    plan.gemms.push_back(gemm);
    PlanStage stage; stage.representative = "linear"; stage.representative_index = 2;
    plan.stages.push_back(stage); plan.outputs.push_back({2, ""});
    ImportOptions options; options.gemms = {{16, 64, 64, 3, 1}};
    auto module = TorchExportImporter{}.ImportPlan(filename.str().str(), plan, context, nullptr, options);
    auto info = module->getOperation()->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
    assert(info.getAs<mlir::IntegerAttr>("phase").getInt() == 2);
    assert(info.getAs<mlir::IntegerAttr>("capacity").getInt() == 0);
    auto roles = module->getOperation()->getAttrOfType<mlir::DictionaryAttr>("tilemega.dimension_roles");
    assert(roles.getAs<mlir::StringAttr>(token_axis ? "seq" : "batch").getValue() == "s0");
    assert(roles.getAs<mlir::StringAttr>("past").getValue().empty());
    auto prepared = TorchExportImporter{}.ImportSemantics(filename.str().str(), plan, context);
    auto bound = analysis::ParamBinding{}.Bind("s0", 8);
    assert(prepared.lifted.sem.ops[0].result.axes[0].extent.Eval(bound, {}) == (token_axis ? 8 : 136));
    auto source = CouplingGraphToCUDA{}.LowerVariants({{*module, unsigned(plan.serving_seq), unsigned(plan.serving_seq)}});
    assert(source.find("#define TILEMEGA_SERVING_PHASE 2\n") != std::string::npos);
    assert(source.find("#define TILEMEGA_SERVING_SEQ " + std::to_string(plan.serving_seq) + "\n") != std::string::npos);
    auto description = solver::ModelDescription::FromCouplingGraph(*module, {plan.serving_seq, 0, plan.serving_seq}, "forward");
    assert(description.serving && description.serving_capacity == 0);
    assert(description.seq_metric_parameter == (token_axis ? "s0" : ""));
    assert(description.batch_metric_parameter == (token_axis ? "" : "s0"));
    if (argc >= 3 && ((std::string(argv[1]) == "--emit" && !token_axis) ||
                     (std::string(argv[1]) == "--emit-token" && token_axis))) {
      std::error_code error;
      llvm::raw_fd_ostream output(argv[2], error); assert(!error); output << source;
    }
    for (auto invalid : {PlanTaskKind::kFusedAttention, PlanTaskKind::kKVAppend,
                         PlanTaskKind::kAttentionMerge, PlanTaskKind::kEmbedding}) {
      plan.stages[0].kind = invalid;
      bool invalid_task = false;
      try { (void)TorchExportImporter{}.ImportPlan(filename.str().str(), plan, context); }
      catch (std::invalid_argument const&) { invalid_task = true; }
      assert(invalid_task);
    }
    plan.stages[0].kind = PlanTaskKind::kGemm;
    plan.serving_capacity = 1;
    bool rejected = false;
    try { (void)TorchExportImporter{}.ImportPlan(filename.str().str(), plan, context); }
    catch (std::invalid_argument const&) { rejected = true; }
    assert(rejected);
    assert(!llvm::sys::fs::remove(filename));
  }
  return 0;
}
}
