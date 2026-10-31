// SPDX-License-Identifier: BSD-3-Clause
#include "tilemega/Codegen/MxDecodeProgramV2.h"

#include "StageProgram.h"
#include "tilemega/Backend/Mx/DecodeGeometryV2.h"

#include <mlir/IR/BuiltinAttributes.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace tilemega::codegen {
namespace {
using namespace backend::mx;

[[noreturn]] void Fail(std::string const& message) {
  throw std::invalid_argument("mx decode v2: " + message);
}
std::string Str(mlir::DictionaryAttr d, char const* key) {
  auto value = d.getAs<mlir::StringAttr>(key);
  return value ? value.getValue().str() : std::string();
}
std::string Escape(std::string const& value) {
  std::string out;
  for (char c : value) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out;
}
int Align16(int value) { return (value + 15) / 16 * 16; }
int GemmSmem(int batch, MxDecodeClassV2 const& c) {
  int const rounds = MxReduceRounds(batch, c.tn);
  int const squares = c.wn == 8 ? 0 : (rounds == 1 ? 8 : 4) * batch * c.tn;
  int const slots = rounds == 2 && c.wn > 1 ? c.wn / 2 : c.wn;
  int const flags = squares + 512 + (c.wn == 8 ? 0 : slots) * batch * (c.tn + 1);
  return Align16((flags + c.wn) * 4);
}
int BodySmem(MxDecodeModel const& m, int batch, MxDecodeConfigV2 const& c) {
  int const slots = std::min(m.capacity / 32, 104);
  int const wave = 64 + m.head_dim / 2 + m.q_per_kv * (m.head_dim + 2);
  int const scratch = std::max(slots * m.q_per_kv * 3, m.q_per_kv * (m.head_dim + 2));
  int const attn = (8 * wave + scratch + 4) * 4;
  int const head = Align16(std::max(GemmSmem(batch, c.head), batch * 128 * 8) + 16);
  return Align16(std::max({32784, GemmSmem(batch, c.qkv), GemmSmem(batch, c.o),
                           GemmSmem(batch, c.gate_up), GemmSmem(batch, c.down), head, attn}));
}

template <int TN, int WN>
int FirstChunks(MxDecodeClassV2 const& c, int n, int k, int grid) {
  if (c.stream_k) {
    if constexpr (WN == 1) return MxFirstChunksV2<TN, WN, MxPart::kStreamK>(n, k, grid);
    return 0;
  }
  return MxFirstChunksV2<TN, WN, MxPart::kTileRange>(n, k, grid);
}
template <int TN>
int FirstChunks(MxDecodeClassV2 const& c, int n, int k, int grid) {
  switch (c.wn) {
    case 1: return FirstChunks<TN, 1>(c, n, k, grid);
    case 2: return FirstChunks<TN, 2>(c, n, k, grid);
    case 4: return FirstChunks<TN, 4>(c, n, k, grid);
    case 8: return FirstChunks<TN, 8>(c, n, k, grid);
  }
  Fail("WN outside {1,2,4,8}");
}
int FirstChunks(MxDecodeClassV2 const& c, int n, int k, int grid) {
  switch (c.tn) {
    case 16: return FirstChunks<16>(c, n, k, grid);
    case 32: return FirstChunks<32>(c, n, k, grid);
    case 64: return FirstChunks<64>(c, n, k, grid);
    case 128: return FirstChunks<128>(c, n, k, grid);
  }
  Fail("TN outside {16,32,64,128}");
}
}  // namespace

int MxDecodeBodySmemBytesV2(MxDecodeModel const& m, int batch, MxDecodeConfigV2 const& c) {
  return BodySmem(m, batch, c);
}

int MxDecodePrefetchV2(MxDecodeClassV2 const& c, int n, int k, int grid, int threads) {
  if (threads != 512 && threads != 576) Fail("threads outside {512,576}");
  if (n <= 0 || k <= 0 || grid <= 0) Fail("prefetch requires positive geometry");
  if (c.tn != 16 && c.tn != 32 && c.tn != 64 && c.tn != 128) Fail("prefetch TN outside {16,32,64,128}");
  if (c.prefetch >= 0) return c.prefetch;
  if (c.prefetch != -1 || c.r_next < 0) Fail("derived prefetch requires measured r_next");
  int const available = std::min(64, (threads == 576 ? 220 : 248) - c.r_next);
  return std::max(c.d, std::min(FirstChunks(c, n, k, grid), available / (c.tn / 4)));
}

void CheckMxDecodeConfigV2(MxDecodeModel const& m, int batch, MxDecodeConfigV2 const& c, int grid) {
  if (grid != 104) Fail("grid must be 104 (one CTA per AP)");
  if (c.threads != 512 && c.threads != 576) Fail("threads outside {512,576}");
  if (m.capacity <= 0 || m.capacity % 32) Fail("fragment KV capacity must be divisible by 32");
  if (c.attn_version != 2 && c.attn_version != 3) Fail("invalid attention version");
  if (c.split != 1 && c.split != 2 && c.split != 4) Fail("invalid attention split");
  if (c.attn_version == 2 && c.split != 1) Fail("attention V2 requires split1");
  if (c.split > 1 && batch * m.hkv * c.split > grid) Fail("attention split exceeds grid");
  if (c.b_min != 1 && c.b_min != 2 && c.b_min != 4 && c.b_min != 8) Fail("b_min outside {1,2,4,8}");
  if (c.attn_prefetch < 0 || c.attn_prefetch > 2) Fail("attention prefetch outside {0,1,2}");
  int const qkv_n = (m.q_per_kv + 2) * m.hkv * m.head_dim;
  auto check = [&](char const* name, MxDecodeClassV2 const& cls, int n, int k, MxOpV2 op, bool head) {
    if (!MxGemmLegalV2(n, k, batch, cls.tn, cls.wn, cls.d, op,
                       cls.stream_k ? MxPart::kStreamK : MxPart::kTileRange, grid))
      Fail(std::string(name) + " violates production GEMM geometry");
    if (head && (cls.stream_k || (cls.wn != 4 && cls.wn != 8))) Fail("head requires TileRange WN in {4,8}");
    int const pw = MxDecodePrefetchV2(cls, n, k, grid, c.threads);
    if (pw < cls.d) Fail(std::string(name) + " prefetch below D");
    if (cls.r_next >= 0 && pw > MxDecodePrefetchV2(MxDecodeClassV2{cls.tn, cls.wn, cls.d, cls.stream_k, -1, cls.r_next}, n, k, grid, c.threads))
      Fail(std::string(name) + " prefetch exceeds measured resource budget");
  };
  check("qkv", c.qkv, qkv_n, m.hidden, MxOpV2::kStore, false);
  check("o", c.o, m.hidden, m.heads * m.head_dim, MxOpV2::kResidual, false);
  check("gate_up", c.gate_up, 2 * m.inter, m.hidden, MxOpV2::kSwiGLU, false);
  check("down", c.down, m.hidden, m.inter, MxOpV2::kResidual, false);
  check("head", c.head, m.vocab, m.hidden, MxOpV2::kStore, true);
  auto loads = [&](MxDecodeClassV2 const& cls, int n, int k) {
    return MxDecodePrefetchV2(cls, n, k, grid, c.threads) * (cls.tn / 16);
  };
  auto wait = [&](char const* name, MxDecodeClassV2 const& cls, int next_loads) {
    if ((cls.d - 1) * (cls.tn / 16 + 1) + next_loads > 63)
      Fail(std::string(name) + " next-stage wait count exceeds 63");
  };
  int const attn_loads = c.attn_prefetch == 0 ? 0 : (m.head_dim / 16) * c.attn_prefetch;
  wait("qkv->attention", c.qkv, attn_loads);
  wait("o->gate_up", c.o, loads(c.gate_up, 2 * m.inter, m.hidden));
  wait("gate_up->down", c.gate_up, loads(c.down, m.hidden, m.inter));
  wait("down->qkv", c.down, loads(c.qkv, qkv_n, m.hidden));
  wait("down_last->head", c.down, loads(c.head, m.vocab, m.hidden));
  if (BodySmem(m, batch, c) + Align16(batch * 4) + 16 > 65536) Fail("shared memory exceeds 64 KiB");
}

std::vector<MxDecodeBuffer> MxDecodeBuffersV2(mlir::ModuleOp module, MxDecodeConfigV2 const& c) {
  auto const m = AnalyzeMxDecode(module);
  auto plan = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  auto buffers = plan.getAs<mlir::ArrayAttr>("buffers");
  std::vector<MxDecodeBuffer> rows;
  auto row = [&](int index, char const* slot, int layer, int tn) {
    auto buffer = mlir::cast<mlir::DictionaryAttr>(buffers[index]);
    MxDecodeBuffer r;
    r.name = Str(buffer, "external_name");
    if (r.name.empty()) r.name = Str(buffer, "name");
    r.dtype = Str(buffer, "dtype");
    if (r.dtype.empty()) r.dtype = "bf16";
    auto constant = buffer.getAs<mlir::IntegerAttr>("constant");
    auto per_batch = buffer.getAs<mlir::IntegerAttr>("per_batch");
    if (!constant) Fail("external buffer lacks constant element count");
    r.constant = constant.getInt();
    r.per_batch = per_batch ? per_batch.getInt() : 0;
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
  row(m.tokens, "kSlotTokensV2", 0, 0);
  row(m.rope_cos, "kSlotRopeCosV2", 0, 0);
  row(m.rope_sin, "kSlotRopeSinV2", 0, 0);
  row(m.embed, "kSlotEmbedV2", 0, 0);
  row(m.head, "kSlotHeadV2", 0, c.head.tn);
  for (int l = 0; l < m.layers; ++l) {
    row(m.qkv[l], "kSlotQkvV2", l, c.qkv.tn);
    row(m.o[l], "kSlotOV2", l, c.o.tn);
    row(m.gate_up[l], "kSlotGateUpV2", l, c.gate_up.tn);
    row(m.down[l], "kSlotDownV2", l, c.down.tn);
    if (m.qk_norm) {
      row(m.q_norm[l], "kSlotQNormV2", l, 0);
      row(m.k_norm[l], "kSlotKNormV2", l, 0);
    }
    row(m.k_cache[l], "kSlotKCacheV2", l, 0);
    row(m.v_cache[l], "kSlotVtCacheV2", l, 0);
  }
  return rows;
}

std::string EmitMxDecodeProgramV2(mlir::ModuleOp module, int batch, int past_lo, int past_hi, int grid,
                                MxDecodeConfigV2 const& c) {
  auto const m = AnalyzeMxDecode(module);
  CheckMxDecodeConfigV2(m, batch, c, grid);
  if (past_lo < 0 || past_hi < past_lo || past_hi >= m.capacity - 1) Fail("invalid decode past interval");
  int const nq = (m.q_per_kv + 2) * m.hkv * m.head_dim;
  auto pw = [&](MxDecodeClassV2 const& cls, int n, int k) {
    return MxDecodePrefetchV2(cls, n, k, grid, c.threads);
  };
  auto body = [&](MxDecodeClassV2 const& cls, char const* op, bool norm, int n, int k) {
    std::ostringstream out;
    out << "MxGemm<" << cls.tn << ", " << cls.wn << ", " << cls.d << ", MxOpV2::" << op << ", "
        << (norm ? "true" : "false") << ", MxPart::" << (cls.stream_k ? "kStreamK" : "kTileRange")
        << ", " << pw(cls, n, k) << ">";
    return out.str();
  };
  auto shape = [&](int n, int k) {
    return "MxStaticShapeV2<" + std::to_string(n) + ", " + std::to_string(k) + ", kBatch, kThreads>";
  };
  StageProgramSpec spec;
  spec.includes = {"tilemega/Codegen/tasks/mx/DecodeProgramV2.cuh"};
  spec.runtime_header = "tilemega/Codegen/tasks/mx/DecodeRuntimeV2.cuh";
  std::ostringstream constants;
  constants << "  static constexpr int kLayers = " << m.layers << ", kHidden = " << m.hidden
      << ", kCapacity = " << m.capacity << ", kBatch = " << batch << ";\n"
      << "  static constexpr int kGrid = " << grid << ", kThreads = " << c.threads
      << ", kPastLo = " << past_lo << ", kPastHi = " << past_hi << ";\n"
      << "  static constexpr int kQ = " << m.q_per_kv << ", kHeadDim = " << m.head_dim
      << ", kBodySmemBytes = " << BodySmem(m, batch, c) << ";\n"
      << "  static constexpr float kEpsilon = " << std::hexfloat << float(m.epsilon) << std::defaultfloat << "f;\n"
      << "  static constexpr bool kQkvStreamK = " << (c.qkv.stream_k ? "true" : "false")
      << ", kOStreamK = " << (c.o.stream_k ? "true" : "false")
      << ", kGateUpStreamK = " << (c.gate_up.stream_k ? "true" : "false")
      << ", kDownStreamK = " << (c.down.stream_k ? "true" : "false") << ";\n"
      << "  using Binding = mx::MxBindingV2<kLayers>;";
  spec.plan_declarations = constants.str();
  spec.loop.end = "kLayers";
  auto const qkv = body(c.qkv, "kStore", true, nq, m.hidden);
  auto const down = body(c.down, "kResidual", false, m.hidden, m.inter);
  std::string const ops = "mx::DenseOperandsV2<Self>::";
  std::string const attention = c.attn_version == 3
      ? "MxAttentionV3<" + std::to_string(m.head_dim) + "," + std::to_string(m.q_per_kv) + "," +
          (m.qk_norm ? "true" : "false") + "," + std::to_string(c.attn_prefetch) + ",kCapacity,0," + std::to_string(c.split) + ">"
      : "MxAttentionV2<" + std::to_string(m.head_dim) + "," + std::to_string(m.q_per_kv) + "," +
          std::to_string(c.b_min) + "," + (m.qk_norm ? "true" : "false") + "," + std::to_string(c.attn_prefetch) + ",kCapacity>";
  spec.stages = {
    {"Qkv", qkv, shape(nq, m.hidden), ops + "Qkv(b,l,past,tokens)", ops + "QkvWeights(b,l)", "Attn", "", 0},
    {"Attn", attention,
        (c.attn_version == 3 ? "mx::MxAttentionShapeWithSyncV2<kBatch, " : "MxStaticAttentionShapeV2<kBatch, ") + std::to_string(m.hkv) + ", kThreads>",
        ops + "Attn(b,l,past)", ops + "Attn(b,l,past)", "O", "", 1},
    {"O", body(c.o, "kResidual", false, m.hidden, m.heads * m.head_dim), shape(m.hidden, m.heads * m.head_dim),
        ops + "O(b,l,tokens)", ops + "O(b,l,tokens)", "GateUp", "", 2},
    {"GateUp", body(c.gate_up, "kSwiGLU", true, 2 * m.inter, m.hidden), shape(2 * m.inter, m.hidden),
        ops + "GateUp(b,l)", ops + "GateUp(b,l)", "Down", "", 3},
    {"Down", down, shape(m.hidden, m.inter), ops + "Down(b,l)", ops + "Down(b,l)", "Qkv",
        ops + "QkvWeights(b,l+1)", 4, down, "Head", ops + "Head(b,past)"}
  };
  spec.tail_stages = {{"Head", "MxHead<" + std::to_string(c.head.tn) + ", " + std::to_string(c.head.wn) + ", " +
      std::to_string(c.head.d) + ", " + std::to_string(pw(c.head, m.vocab, m.hidden)) + ">", shape(m.vocab, m.hidden),
      ops + "Head(b,past)", ops + "Head(b,past)", "Qkv", ops + "QkvWeights(b,0)", 5}};
  spec.entry_prefetch_operands = ops + "QkvWeights(b,0)";
  for (auto const& r : MxDecodeBuffersV2(module, c)) {
    if (r.dtype != "i32" && r.dtype != "f32" && r.dtype != "bf16") Fail("unsupported external dtype " + r.dtype);
    spec.external_buffers.push_back({r.name,
        r.dtype == "i32" ? "TM_DTYPE_I32" : r.dtype == "f32" ? "TM_DTYPE_F32" : "TM_DTYPE_BF16",
        std::uint64_t(r.constant), std::uint64_t(r.per_batch), r.recipe, "mx::" + r.slot, r.layer});
  }
  return EmitStageProgram(spec);
}

}  // namespace tilemega::codegen
