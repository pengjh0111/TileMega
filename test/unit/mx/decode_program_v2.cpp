// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/MxDecodeProgramV2.h>
#include "StageProgram.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace tilemega::codegen;

namespace {
mlir::ModuleOp Fixture(mlir::MLIRContext& context) {
  mlir::Builder b(&context);
  auto module = mlir::ModuleOp::create(b.getUnknownLoc());
  auto integer = [&](char const* name, long value) { return b.getNamedAttr(name, b.getI64IntegerAttr(value)); };
  std::vector<mlir::Attribute> buffers;
  auto buffer = [&](char const* name, long constant, long per_batch = 0, char const* recipe = "") {
    int const index = buffers.size();
    buffers.push_back(b.getDictionaryAttr({b.getNamedAttr("name", b.getStringAttr(name)),
        b.getNamedAttr("dtype", b.getStringAttr(std::string(name) == "tokens" ? "i32" : "bf16")),
        integer("constant", constant), integer("per_batch", per_batch),
        b.getNamedAttr("pack_json", b.getStringAttr(recipe))}));
    return index;
  };
  int const tokens = buffer("tokens", 0, 1088), embed = buffer("embed", 65536 * 512);
  int const qkv = buffer("qkv", 768 * 512, 0,
      R"({"kind":"qkv_group_interleave","hkv":2,"qperkv":4,"head_dim":64})");
  int const o = buffer("o", 512 * 512), gu = buffer("gate_up", 2048 * 512), down = buffer("down", 512 * 1024);
  int const head = buffer("head", 65536 * 512), kc = buffer("k_cache", 0, 2 * 1088 * 64);
  int const vc = buffer("v_cache", 0, 2 * 1088 * 64), cos = buffer("cos", 1088 * 64), sin = buffer("sin", 1088 * 64);
  std::vector<mlir::Attribute> gemms, stages;
  auto stage = [&](char const* kind, std::vector<std::int64_t> operands) {
    stages.push_back(b.getDictionaryAttr({b.getNamedAttr("kind", b.getStringAttr(kind)),
        b.getNamedAttr("operands", b.getDenseI64ArrayAttr(operands))}));
  };
  auto gemm = [&](int weight, int n, int k) {
    int const index = gemms.size();
    gemms.push_back(b.getDictionaryAttr({integer("b", weight), integer("n", n), integer("k", k), integer("norm_ss", 0)}));
    stages.push_back(b.getDictionaryAttr({b.getNamedAttr("kind", b.getStringAttr("kGemm")), integer("gemm", index)}));
  };
  stage("kEmbedding", {tokens, embed}); gemm(qkv, 768, 512);
  stage("kFusedAttention", {0, kc, vc, cos, sin, 0xffffffffll, 0xffffffffll});
  gemm(o, 512, 512); gemm(gu, 2048, 512); gemm(down, 512, 1024); gemm(head, 65536, 512);
  stage("kArgmaxReduce", {});
  module->setAttr("tilemega.model_plan", b.getDictionaryAttr({
      b.getNamedAttr("buffers", b.getArrayAttr(buffers)), b.getNamedAttr("gemms", b.getArrayAttr(gemms)),
      b.getNamedAttr("stages", b.getArrayAttr(stages)), b.getNamedAttr("norm_epsilon", b.getF64FloatAttr(1.e-6))}));
  module->setAttr("tilemega.serving", b.getDictionaryAttr({integer("capacity", 1088)}));
  return module;
}
template <class F> void Reject(F f) {
  bool rejected = false;
  try { f(); } catch (std::invalid_argument const&) { rejected = true; }
  assert(rejected);
}
}  // namespace

int main(int argc, char** argv) {
  StageProgramSpec spec;
  spec.loop.end = "1";
  spec.stages = {{"A", "BodyA", "ShapeA", "opsA", "pfA", "A", "pfA", 0, "BodyLast", "Z", "pfZ"}};
  spec.tail_stages = {{"Z", "BodyZ", "ShapeZ", "opsZ", "pfZ", "A", "pfA", 1}};
  assert(EmitStageProgram(spec).find("BodyLast") != std::string::npos);
  auto invalid = spec; invalid.tail_stages.front().name = "A";
  Reject([&] { EmitStageProgram(invalid); });
  invalid = spec; invalid.stages.front().next = "missing";
  Reject([&] { EmitStageProgram(invalid); });
  invalid = spec; invalid.stages.front().next = "Z";
  Reject([&] { EmitStageProgram(invalid); });
  mlir::MLIRContext context;
  auto module = Fixture(context);
  auto const model = AnalyzeMxDecode(module);
  MxDecodeClassV2 const cls{16, 1, 1, false, 1, -1};
  MxDecodeConfigV2 c{cls, cls, cls, cls, {16, 4, 1, false, 1, -1}, 2, 2, 576};
  c.gate_up.tn = 32;
  CheckMxDecodeConfigV2(model, 16, c, 104);
  auto bad = c; bad.qkv.wn = 2;
  Reject([&] { CheckMxDecodeConfigV2(model, 16, bad, 104); });
  bad = c; bad.gate_up.prefetch = 64;
  Reject([&] { CheckMxDecodeConfigV2(model, 16, bad, 104); });
  bad = c; bad.qkv.prefetch = -1;
  Reject([&] { CheckMxDecodeConfigV2(model, 16, bad, 104); });
  bad = c; bad.threads = 1024;
  Reject([&] { CheckMxDecodeConfigV2(model, 16, bad, 104); });
  Reject([&] { CheckMxDecodeConfigV2(model, 16, c, 208); });
  MxDecodeClassV2 derived{16, 1, 1, false, -1, 156};
  assert(MxDecodePrefetchV2(derived, 2048, 2048, 104, 576) == 8);
  auto const rows = MxDecodeBuffersV2(module, c);
  assert(rows[4].name == "head@t16x32");
  assert(rows[5].name == "qkv@t16x32");
  assert(rows[5].recipe.find("\"tile_n\":16") != std::string::npos);
  assert(rows.back().per_batch == 2 * 1088 * 64);
  auto const source = EmitMxDecodeProgramV2(module, 16, 64, 1086, 104, c);
  assert(source.find("kThreads = 576") != std::string::npos);
  assert(source.find("kStageCount = (kLayers - (0)) * 5 + 1") != std::string::npos);
  assert(source.find("MxOpV2::kStore, true") != std::string::npos);
  assert(source.find("QkvWeights(b,l+1)") != std::string::npos);
  assert(source.find("QkvWeights(b,0)") != std::string::npos);
  assert(source.find("DownLast") != std::string::npos);
  assert(source.find("kQkvRope") == std::string::npos);
  if (argc == 2) { std::ofstream out(argv[1]); out << source; assert(out.good()); }
  module.erase();
  std::cout << "PASS v2 configuration geometry, resource payloads and typed dense emission\n";
}
