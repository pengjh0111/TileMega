// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Codegen/MxDecodeProgram.cpp: the MX-4 static decode program
// generator (MX4 prompt §3.9). See the header.
#include "tilemega/Codegen/MxDecodeProgram.h"

#include "tilemega/Backend/Mx/DecodeShapes.h"
#include "tilemega/Support/Json.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tilemega::codegen {
namespace {

constexpr std::int64_t kNone = 0xffffffffll;

mlir::DictionaryAttr Plan(mlir::ModuleOp module) {
  auto plan = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if (!plan) throw std::invalid_argument("mx decode: module lacks tilemega.model_plan");
  return plan;
}

std::int64_t Int(mlir::DictionaryAttr d, char const* key) {
  auto v = d.getAs<mlir::IntegerAttr>(key);
  if (!v) throw std::invalid_argument(std::string("mx decode: missing integer ") + key);
  return v.getInt();
}

std::string Str(mlir::DictionaryAttr d, char const* key) {
  auto v = d.getAs<mlir::StringAttr>(key);
  return v ? v.getValue().str() : std::string();
}

/// hkv, qperkv, head_dim of a qkv_group_interleave recipe, possibly nested.
bool FindQkvGeometry(json::Value const& recipe, int& hkv, int& q, int& d) {
  if (recipe.kind() != json::Value::Kind::kObject) return false;
  if (auto const* kind = recipe.Find("kind"); kind && kind->kind() == json::Value::Kind::kString &&
                                              kind->AsString("kind") == "qkv_group_interleave") {
    hkv = int(recipe.At("hkv").AsNumber("hkv"));
    q = int(recipe.At("qperkv").AsNumber("qperkv"));
    d = int(recipe.At("head_dim").AsNumber("head_dim"));
    return true;
  }
  if (auto const* source = recipe.Find("source")) return FindQkvGeometry(*source, hkv, q, d);
  return false;
}

std::string Escape(std::string const& s) {
  std::string out;
  for (char c : s) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out;
}

char const* PartName(bool stream_k) { return stream_k ? "kStreamK" : "kTileRange"; }

}  // namespace

MxDecodeModel AnalyzeMxDecode(mlir::ModuleOp module) {
  auto plan = Plan(module);
  auto buffers = plan.getAs<mlir::ArrayAttr>("buffers");
  auto gemms = plan.getAs<mlir::ArrayAttr>("gemms");
  auto stages = plan.getAs<mlir::ArrayAttr>("stages");
  auto serving = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
  if (!buffers || !gemms || !stages || !serving) throw std::invalid_argument("mx decode: incomplete model plan");
  if (auto dm = plan.getAs<mlir::BoolAttr>("dm"); dm && dm.getValue())
    throw std::invalid_argument("mx decode: DM (MoE/DNN) plans keep the interpreted executor");
  MxDecodeModel m;
  m.capacity = int(Int(serving, "capacity"));
  if (auto eps = plan.getAs<mlir::FloatAttr>("norm_epsilon")) m.epsilon = eps.getValueAsDouble();
  else throw std::invalid_argument("mx decode: model plan lacks norm_epsilon");

  auto stage = [&](std::size_t i) { return mlir::cast<mlir::DictionaryAttr>(stages[i]); };
  auto kind = [&](std::size_t i) { return Str(stage(i), "kind"); };
  auto operand = [&](std::size_t i, unsigned slot) {
    auto ops = stage(i).getAs<mlir::DenseI64ArrayAttr>("operands").asArrayRef();
    if (slot >= ops.size()) throw std::invalid_argument("mx decode: stage operand missing");
    return ops[slot];
  };
  auto gemm = [&](std::size_t i) {
    if (kind(i) != "kGemm") throw std::invalid_argument("mx decode: expected a GEMM at stage " + std::to_string(i));
    return mlir::cast<mlir::DictionaryAttr>(gemms[Int(stage(i), "gemm")]);
  };
  auto norm_folded = [&](mlir::DictionaryAttr g, char const* what) {
    auto ss = g.getAs<mlir::IntegerAttr>("norm_ss");
    if (!ss || ss.getInt() == kNone)
      throw std::invalid_argument(std::string("mx decode: the ") + what + " GEMM lacks norm_ss (gamma not folded)");
  };

  std::size_t i = 0;
  if (stages.size() < 3 || kind(0) != "kEmbedding")
    throw std::invalid_argument("mx decode: the plan does not start with an embedding");
  m.tokens = int(operand(0, 0));
  m.embed = int(operand(0, 1));
  ++i;
  int qkv_n = 0;
  while (i < stages.size() && kind(i) == "kGemm" && i + 1 < stages.size() && kind(i + 1) == "kFusedAttention") {
    auto qkv = gemm(i);
    norm_folded(qkv, "qkv");
    m.qkv.push_back(int(Int(qkv, "b")));
    qkv_n = int(Int(qkv, "n"));
    m.hidden = int(Int(qkv, "k"));
    ++i;
    m.k_cache.push_back(int(operand(i, 1)));
    m.v_cache.push_back(int(operand(i, 2)));
    m.rope_cos = int(operand(i, 3));
    m.rope_sin = int(operand(i, 4));
    auto qn = operand(i, 5), kn = operand(i, 6);
    m.qk_norm = qn != kNone && kn != kNone;
    if (m.qk_norm) {
      m.q_norm.push_back(int(qn));
      m.k_norm.push_back(int(kn));
    }
    ++i;
    if (i < stages.size() && kind(i) == "kAttentionMerge") ++i;
    auto o = gemm(i++);
    auto gu = gemm(i++);
    norm_folded(gu, "gate_up");
    auto down = gemm(i++);
    m.o.push_back(int(Int(o, "b")));
    m.gate_up.push_back(int(Int(gu, "b")));
    m.down.push_back(int(Int(down, "b")));
    m.inter = int(Int(down, "k"));
    if (Int(gu, "n") != 2 * m.inter || Int(gu, "k") != m.hidden || Int(down, "n") != m.hidden ||
        Int(o, "n") != m.hidden)
      throw std::invalid_argument("mx decode: layer GEMM shapes disagree");
    m.heads = int(Int(o, "k"));
    ++m.layers;
  }
  if (!m.layers) throw std::invalid_argument("mx decode: no decoder layer found");
  auto head = gemm(i++);
  norm_folded(head, "head");
  m.head = int(Int(head, "b"));
  m.vocab = int(Int(head, "n"));
  if (i >= stages.size() || kind(i) != "kArgmaxReduce" || i + 1 != stages.size())
    throw std::invalid_argument("mx decode: the plan does not end with head GEMM + ArgmaxReduce");

  auto qkv_buffer = mlir::cast<mlir::DictionaryAttr>(buffers[m.qkv[0]]);
  auto recipe = Str(qkv_buffer, "pack_json");
  if (recipe.empty() || !FindQkvGeometry(json::Parse(recipe), m.hkv, m.q_per_kv, m.head_dim))
    throw std::invalid_argument("mx decode: the qkv weight lacks a qkv_group_interleave recipe");
  if (m.heads != m.hkv * m.q_per_kv * m.head_dim || qkv_n != (m.q_per_kv + 2) * m.hkv * m.head_dim)
    throw std::invalid_argument("mx decode: attention geometry disagrees with the GEMM shapes");
  m.heads /= m.head_dim;
  if ((m.head_dim != 64 && m.head_dim != 128) || m.q_per_kv > 8 || m.capacity % 8)
    throw std::invalid_argument("mx decode: attention needs D in {64,128}, Q <= 8, capacity % 8 == 0");
  return m;
}

void CheckMxDecodeConfig(MxDecodeModel const& m, int batch, MxDecodeConfig const& c, int grid) {
  using namespace backend::mx;
  auto fail = [](std::string what) { throw std::invalid_argument("mx decode config: " + what); };
  auto check = [&](char const* name, MxDecodeClass const& k, int n, int kk, bool swiglu, bool head, bool qkv) {
    std::string const at = std::string(name) + " ";
    if (!(k.tn == 16 || k.tn == 32 || k.tn == 64 || k.tn == 128)) fail(at + "TN");
    if (swiglu && k.tn < 32) fail(at + "TN >= 32 (interleave_u = 16)");
    if (qkv && !((k.tn == m.head_dim || k.tn == 2 * m.head_dim) && k.tn <= 128 &&
                 ((m.q_per_kv + 2) * m.head_dim) % k.tn == 0))
      fail(at + "TN in {D, 2D}, <= 128, dividing (Q+2)D");
    if (n % k.tn || kk % 32) fail(at + "TN divides N, 32 divides K");
    if (k.d < 1 || k.d > 4 || !MxDepthFits(k.tn, k.d)) fail(at + "D (rule 6)");
    if (!MxReduceRounds(batch, k.tn)) fail(at + "reduction legality");
    if (head && k.stream_k) fail(at + "head is kTileRange");
    if (k.stream_k && (long long)(n / k.tn) * (kk / 32) < grid) fail(at + "stream-K needs U >= G");
  };
  check("qkv", c.qkv, (m.q_per_kv + 2) * m.hkv * m.head_dim, m.hidden, false, false, true);
  check("o", c.o, m.hidden, m.heads * m.head_dim, false, false, false);
  check("gate_up", c.gate_up, 2 * m.inter, m.hidden, true, false, false);
  check("down", c.down, m.hidden, m.inter, false, false, false);
  check("head", c.head, m.vocab, m.hidden, false, true, false);
  if (c.tmin != 32 && c.tmin != 64 && c.tmin != 128) fail("Tmin in {32, 64, 128}");
}

void WriteMxDecodeConfig(mlir::ModuleOp module, MxDecodeConfig const& c) {
  auto model = AnalyzeMxDecode(module);
  (void)model;
  mlir::Builder b(module.getContext());
  auto encode = [&](MxDecodeClass const& k) {
    return b.getDictionaryAttr({b.getNamedAttr("tn", b.getI64IntegerAttr(k.tn)),
                                b.getNamedAttr("d", b.getI64IntegerAttr(k.d)),
                                b.getNamedAttr("part", b.getStringAttr(PartName(k.stream_k)))});
  };
  module->setAttr("tmexec.executor", b.getStringAttr("static_stage"));
  module->setAttr("tmexec.attention_tmin", b.getI64IntegerAttr(c.tmin));
  // The GEMM class of every plan stage, in program order.
  auto stages = Plan(module).getAs<mlir::ArrayAttr>("stages");
  std::vector<MxDecodeClass const*> by_stage(stages.size(), nullptr);
  int gemm_in_layer = 0;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    auto kind = Str(mlir::cast<mlir::DictionaryAttr>(stages[i]), "kind");
    if (kind != "kGemm") continue;
    bool const head = i + 2 == stages.size();
    MxDecodeClass const* cls[] = {&c.qkv, &c.o, &c.gate_up, &c.down};
    by_stage[i] = head ? &c.head : cls[gemm_in_layer];
    gemm_in_layer = head ? 0 : (gemm_in_layer + 1) % 4;
  }
  module.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() != "tmcg.tile_space") return;
    auto s = op->getAttrOfType<mlir::IntegerAttr>("stage");
    if (s && s.getInt() >= 0 && std::size_t(s.getInt()) < by_stage.size() && by_stage[s.getInt()])
      op->setAttr("tmexec.implementation", encode(*by_stage[s.getInt()]));
  });
  module->setAttr("tmexec.mx_classes",
                  b.getDictionaryAttr({b.getNamedAttr("qkv", encode(c.qkv)), b.getNamedAttr("o", encode(c.o)),
                                       b.getNamedAttr("gate_up", encode(c.gate_up)),
                                       b.getNamedAttr("down", encode(c.down)),
                                       b.getNamedAttr("head", encode(c.head))}));
}

MxDecodeConfig ReadMxDecodeConfig(mlir::ModuleOp module) {
  auto executor = module->getAttrOfType<mlir::StringAttr>("tmexec.executor");
  auto classes = module->getAttrOfType<mlir::DictionaryAttr>("tmexec.mx_classes");
  auto tmin = module->getAttrOfType<mlir::IntegerAttr>("tmexec.attention_tmin");
  if (!executor || executor.getValue() != "static_stage" || !classes || !tmin)
    throw std::invalid_argument("mx decode: the module carries no static_stage choice");
  auto decode = [&](char const* name) {
    auto d = classes.getAs<mlir::DictionaryAttr>(name);
    if (!d) throw std::invalid_argument(std::string("mx decode: no choice for ") + name);
    return MxDecodeClass{int(Int(d, "tn")), int(Int(d, "d")), Str(d, "part") == "kStreamK"};
  };
  MxDecodeConfig c;
  c.qkv = decode("qkv");
  c.o = decode("o");
  c.gate_up = decode("gate_up");
  c.down = decode("down");
  c.head = decode("head");
  c.tmin = int(tmin.getInt());
  return c;
}

std::vector<MxDecodeBuffer> MxDecodeBuffers(mlir::ModuleOp module) {
  auto const m = AnalyzeMxDecode(module);
  auto const c = ReadMxDecodeConfig(module);
  auto buffers = Plan(module).getAs<mlir::ArrayAttr>("buffers");
  std::vector<MxDecodeBuffer> rows;
  // External buffers: the front end's names and recipes; GEMM weights
  // wrapped in tile_pages(xcore_mma, TN, 32), named as ServingPages.cpp does.
  auto row = [&](int index, char const* slot, int layer, int tn) {
    auto buffer = mlir::cast<mlir::DictionaryAttr>(buffers[index]);
    MxDecodeBuffer r;
    r.name = Str(buffer, "external_name");
    if (r.name.empty()) r.name = Str(buffer, "name");
    r.dtype = Str(buffer, "dtype");
    if (r.dtype.empty()) r.dtype = "bf16";
    auto per_batch = buffer.getAs<mlir::IntegerAttr>("per_batch");
    r.per_batch = per_batch ? per_batch.getInt() : 0;
    r.constant = Int(buffer, "constant");
    r.recipe = Str(buffer, "pack_json");
    if (tn) {
      if (r.recipe.empty()) r.recipe = "{\"kind\":\"alias\",\"source\":\"" + Escape(r.name) + "\"}";
      r.recipe = "{\"kind\":\"tile_pages\",\"layout\":\"xcore_mma\",\"tile_n\":" + std::to_string(tn) +
                 ",\"tile_k\":32,\"source\":" + r.recipe + "}";
      r.name += "@t" + std::to_string(tn) + "x32";
    }
    r.slot = slot;
    r.layer = layer;
    rows.push_back(r);
  };
  row(m.tokens, "kSlotTokens", 0, 0);
  row(m.rope_cos, "kSlotRopeCos", 0, 0);
  row(m.rope_sin, "kSlotRopeSin", 0, 0);
  row(m.embed, "kSlotEmbed", 0, 0);
  row(m.head, "kSlotHead", 0, c.head.tn);
  for (int l = 0; l < m.layers; ++l) {
    row(m.qkv[l], "kSlotQkv", l, c.qkv.tn);
    row(m.o[l], "kSlotO", l, c.o.tn);
    row(m.gate_up[l], "kSlotGateUp", l, c.gate_up.tn);
    row(m.down[l], "kSlotDown", l, c.down.tn);
    if (m.qk_norm) {
      row(m.q_norm[l], "kSlotQNorm", l, 0);
      row(m.k_norm[l], "kSlotKNorm", l, 0);
    }
    row(m.k_cache[l], "kSlotKCache", l, 0);
    row(m.v_cache[l], "kSlotVtCache", l, 0);
  }
  return rows;
}

std::string EmitMxDecodeProgram(mlir::ModuleOp module, int batch, int past_lo, int past_hi, int grid) {
  auto const m = AnalyzeMxDecode(module);
  auto const c = ReadMxDecodeConfig(module);
  CheckMxDecodeConfig(m, batch, c, grid);
  auto buffers = Plan(module).getAs<mlir::ArrayAttr>("buffers");
  int const n_qkv = (m.q_per_kv + 2) * m.hkv * m.head_dim;
  int const max_blocks =
      backend::mx::MxAttentionMaxBlocks(batch * m.hkv, m.capacity, c.tmin, grid * backend::mx::kDecodeWaves);
  std::ostringstream o;
  o << "// Generated by tilemega::codegen::EmitMxDecodeProgram (MX-4 static decode program).\n"
    << "#define TILEMEGA_MX_STATIC_DECODE 1\n"
    << "#include <tilemega/Codegen/tasks/mx/DecodeProgram.cuh>\n\n"
    << "namespace tm_mx_plan {\n"
    << "using namespace tilemega::backend::mx;\n"
    << "namespace mx = tilemega::codegen::mx;\n"
    << "struct Plan {\n"
    << "  static constexpr int kLayers = " << m.layers << ", kHidden = " << m.hidden << ", kCapacity = " << m.capacity
    << ", kBatch = " << batch << ";\n"
    << "  static constexpr int kGrid = " << grid << ", kPastLo = " << past_lo << ", kPastHi = " << past_hi
    << ", kQ = " << m.q_per_kv << ", kHeadDim = " << m.head_dim << ";\n"
    << "  static constexpr float kEpsilon = " << std::hexfloat << float(m.epsilon) << std::defaultfloat
    << "f;  // " << m.epsilon << "\n"
    << "  using Binding = mx::MxBinding<kLayers>;\n"
    << "  using QkvShape = MxStaticShape<" << n_qkv << ", " << m.hidden << ", kBatch>;\n"
    << "  using OShape = MxStaticShape<" << m.hidden << ", " << m.heads * m.head_dim << ", kBatch>;\n"
    << "  using GateUpShape = MxStaticShape<" << 2 * m.inter << ", " << m.hidden << ", kBatch>;\n"
    << "  using DownShape = MxStaticShape<" << m.hidden << ", " << m.inter << ", kBatch>;\n"
    << "  using HeadShape = MxStaticShape<" << m.vocab << ", " << m.hidden << ", kBatch>;\n"
    << "  using AttnShape = MxStaticAttentionShape<kBatch, " << m.hkv << ">;\n"
    << "  using Qkv = MxDecodeGemm<" << c.qkv.tn << ", " << c.qkv.d << ", MxDecodeOp::kQkvRope, true, MxPart::"
    << PartName(c.qkv.stream_k) << ", MxRope<" << m.head_dim << ", " << m.q_per_kv << ", " << m.hkv << ", "
    << (m.qk_norm ? "true" : "false") << ">>;\n"
    << "  using O = MxDecodeGemm<" << c.o.tn << ", " << c.o.d << ", MxDecodeOp::kResidual, false, MxPart::"
    << PartName(c.o.stream_k) << ">;\n"
    << "  using GateUp = MxDecodeGemm<" << c.gate_up.tn << ", " << c.gate_up.d
    << ", MxDecodeOp::kSwiGLU, true, MxPart::" << PartName(c.gate_up.stream_k) << ">;\n"
    << "  using Down = MxDecodeGemm<" << c.down.tn << ", " << c.down.d << ", MxDecodeOp::kResidual, false, MxPart::"
    << PartName(c.down.stream_k) << ">;\n"
    << "  using Head = MxDecodeGemm<" << c.head.tn << ", " << c.head.d
    << ", MxDecodeOp::kArgmax, true, MxPart::kTileRange>;\n"
    << "  using Attn = MxDecodeAttention<" << m.head_dim << ", " << m.q_per_kv << ", " << c.tmin << ", " << max_blocks
    << ">;\n"
    << "};\n\n";

  o << "inline constexpr mx::MxBufferRow kBuffers[] = {\n";
  for (auto const& r : MxDecodeBuffers(module))
    o << "  {\"" << Escape(r.name) << "\", "
      << (r.dtype == "i32" ? "TM_DTYPE_I32" : r.dtype == "f32" ? "TM_DTYPE_F32" : "TM_DTYPE_BF16") << ", "
      << r.constant << "ull, " << r.per_batch << "ull, "
      << (r.recipe.empty() ? std::string("nullptr") : "\"" + Escape(r.recipe) + "\"") << ", mx::" << r.slot << ", "
      << r.layer << "},\n";
  o << "};\n}  // namespace tm_mx_plan\n\n"
    << "#include <tilemega/Codegen/tasks/mx/DecodeRuntime.cuh>\n";
  return o.str();
}

}  // namespace tilemega::codegen
