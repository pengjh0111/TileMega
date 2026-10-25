// SPDX-License-Identifier: BSD-3-Clause
#include "MxDecode.h"
#include "Toolchain.h"

#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include <tilemega/Codegen/MxDecodeProgramV2.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Support/BuildConfig.h>
#include <tilemega/Support/Json.h>

#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tilemega::commands::mx_decode {
namespace {
using json::Value;
using json::Object;
using json::Array;
using codegen::MxDecodeClassV2;
using codegen::MxDecodeConfigV2;

std::string Quote(std::string const& value) {
  std::string out = "'";
  for (char c : value) out += c == '\'' ? "'\\''" : std::string(1, c);
  return out + "'";
}
void Write(std::string const& path, std::string const& text) {
  std::ofstream stream(path);
  stream << text;
  if (!stream.good()) throw std::runtime_error("MX-5 cannot write " + path);
}
int Int(Value const& object, char const* name) {
  double const number = object.At(name).AsNumber(name);
  if (!std::isfinite(number) || std::floor(number) != number ||
      number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string("MX-5 pin requires integer ") + name);
  return int(number);
}
MxDecodeClassV2 ParseClass(Value const& config, char const* name) {
  auto const& value = config.At(name);
  auto const& part = value.At("part").AsString("part");
  if (part != "kStreamK" && part != "kTileRange") throw std::invalid_argument("MX-5 pin has invalid part");
  return {Int(value, "tn"), Int(value, "wn"), Int(value, "d"), part == "kStreamK", Int(value, "prefetch"),
          value.Find("r_next") ? Int(value, "r_next") : -1};
}
Value ClassJson(MxDecodeClassV2 const& c) {
  return Object{{"tn", c.tn}, {"wn", c.wn}, {"d", c.d}, {"part", c.stream_k ? "kStreamK" : "kTileRange"},
                {"prefetch", c.prefetch}, {"r_next", c.r_next}};
}
Value Classes(MxDecodeConfigV2 const& c) {
  return Object{{"qkv", ClassJson(c.qkv)}, {"o", ClassJson(c.o)}, {"gate_up", ClassJson(c.gate_up)},
                {"down", ClassJson(c.down)}, {"head", ClassJson(c.head)}};
}
Value Count(std::uint64_t value) { return Value(double(value)); }
}

int TryCompileV2(Request const& r) {
  auto const pin_doc = json::ParseFile(r.pin);
  auto const& pin = pin_doc.At("mx_decode_v2");
  MxDecodeConfigV2 config{ParseClass(pin, "qkv"), ParseClass(pin, "o"), ParseClass(pin, "gate_up"),
      ParseClass(pin, "down"), ParseClass(pin, "head"), Int(pin, "b_min"), Int(pin, "attn_prefetch"), Int(pin, "threads")};
  std::string const purpose = pin.Find("purpose") ? pin.At("purpose").AsString("purpose") : "manual";
  if (purpose != "manual" && purpose != "resource_probe")
    throw std::invalid_argument("MX-5 pinned builds require purpose manual or resource_probe");
  if (r.watchdog != 0 && r.watchdog != 1) throw std::invalid_argument("MX-5 watchdog must be 0 or 1");
  for (auto const& flag : r.interpreted_flags)
    throw std::invalid_argument(flag + " belongs to the interpreted executor; an explicit MX-5 pin requires static v2");

  auto bridge = frontend::ReadExportBridge(r.input.string());
  frontend::ServingOptions options;
  options.deferred_norm = true;
  options.phase = frontend::ServingOptions::Phase::kDecode;
  options.seq = 1;
  options.capacity = r.capacity;
  auto plan = frontend::BuildModelPlan(bridge.nodes, bridge.inputs, bridge.outputs, options);
  if (plan.dm) throw std::invalid_argument("MX-5 explicit dense pin cannot compile a MoE/DNN plan");
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  context.getOrLoadDialect<dialect::ExecDialect>();
  frontend::ImportOptions import;
  import.gemms.assign(plan.gemms.size(), frontend::GemmGranularity{16, 128, 128, 2, 1});
  auto module = frontend::TorchExportImporter{}.ImportPlan(r.input.string(), plan, context, nullptr, import);
  if (!module) throw std::runtime_error("MX-5 decode import failed");
  auto const model = codegen::AnalyzeMxDecode(*module);
  codegen::CheckMxDecodeConfigV2(model, r.batch, config, r.grid);
  int const nq = (model.q_per_kv + 2) * model.hkv * model.head_dim;
  auto resolve = [&](MxDecodeClassV2& c, int n, int k) {
    c.prefetch = codegen::MxDecodePrefetchV2(c, n, k, r.grid, config.threads);
  };
  resolve(config.qkv, nq, model.hidden);
  resolve(config.o, model.hidden, model.heads * model.head_dim);
  resolve(config.gate_up, 2 * model.inter, model.hidden);
  resolve(config.down, model.hidden, model.inter);
  resolve(config.head, model.vocab, model.hidden);

  std::filesystem::path const parent = r.output.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::string const out = r.output.string(), root = TILEMEGA_SOURCE_DIR;
  std::string const environment = "PYTHONPATH=" + Quote(root + "/python") + " LD_LIBRARY_PATH=" +
      Quote(CudaLibraryDirectory() + ":" + (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : ""));
  auto python = [&](std::string const& arguments) {
    std::string const command = environment + " python3 " + arguments;
    if (std::system(command.c_str())) throw std::runtime_error("MX-5 build helper failed: " + arguments);
  };
  python("-m tilemega.build.identity --root " + Quote(root) + " --snapshot " + Quote(out + ".source.json") +
      " --capture --compiler " + Quote(std::filesystem::read_symlink("/proc/self/exe").string()));

  bool const diag = std::getenv("TILEMEGA_MX_DIAG") && std::string(std::getenv("TILEMEGA_MX_DIAG")) == "1";
  std::string flags = " -DTILEMEGA_SYNC_V3=1 -DTILEMEGA_MX_STATIC_DECODE_V2=1 -DTILEMEGA_WATCHDOG=" +
      std::to_string(r.watchdog) + " -DTILEMEGA_TRACE_STAGE=" + (r.trace_stage ? "1" : "0") +
      " -DTILEMEGA_MX_DIAG=" + (diag ? "1" : "0");
  std::string device_flags = DeviceBuildFlags();
  if (device_flags.find("-fno-strict-aliasing") == std::string::npos) device_flags += " -fno-strict-aliasing";
  std::string const command = Quote(NvccPath()) + " -std=c++17 -O3" + device_flags + flags +
      " -mllvm -print-after=stack-frame-layout -shared -Xcompiler=-fPIC -x cu -I" + Quote(root + "/include") + " " +
      Quote(out + ".cu") + " -L" + Quote(CudaLibraryDirectory()) + " -lcudart -o " + Quote(out);
  Write(out + ".cu", codegen::EmitMxDecodeProgramV2(*module, r.batch, r.past_lo, r.past_hi, r.grid, config));
  Write(out + ".runtime_flags.txt", flags + "\n");
  Write(out + ".build_command.txt", command + "\n");
  Value classes = Classes(config);
  auto actual_class = [&](char const* name, MxDecodeClassV2 const& c, int n, int k, char const* op, bool norm) {
    Value value = ClassJson(c);
    value.Set("n", n); value.Set("k", k); value.Set("b", r.batch);
    value.Set("op", op); value.Set("norm", norm);
    classes.Set(name, value);
  };
  actual_class("qkv", config.qkv, nq, model.hidden, "kStore", true);
  actual_class("o", config.o, model.hidden, model.heads * model.head_dim, "kResidual", false);
  actual_class("gate_up", config.gate_up, 2 * model.inter, model.hidden, "kSwiGLU", true);
  actual_class("down", config.down, model.hidden, model.inter, "kResidual", false);
  actual_class("head", config.head, model.vocab, model.hidden, "argmax", true);
  Value resolved_pin = classes;
  resolved_pin.Set("threads", config.threads);
  resolved_pin.Set("b_min", config.b_min);
  resolved_pin.Set("attn_prefetch", config.attn_prefetch);
  resolved_pin.Set("purpose", purpose);
  Write(out + ".mx5_pin.json", Value(Object{{"mx_decode_v2", resolved_pin}}).Dump() + "\n");
  auto const start = std::chrono::steady_clock::now();
  int const status = std::system((command + " >" + Quote(out + ".ptxas.log") + " 2>&1").c_str());
  double const compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (status) {
    Write(out + ".resource_failure.json", Value(Object{{"status", "compiler_failed"}, {"exit_status", status},
          {"source", out + ".cu"}, {"log", out + ".ptxas.log"}, {"compile_s", compile_s}}).Dump() + "\n");
    throw std::runtime_error("MX-5 cucc failed; generated source/binary/log preserved: " + out + ".ptxas.log");
  }
  std::string const resource_command = environment + " python3 -m tilemega.mx5_resources --so " + Quote(out) +
      " --log " + Quote(out + ".ptxas.log") + " --threads " + std::to_string(config.threads) +
      " --out " + Quote(out + ".resources.json");
  if (std::system((resource_command + " >" + Quote(out + ".resource_report.log") + " 2>&1").c_str())) {
    Write(out + ".resource_failure.json", Value(Object{{"status", "resource_report_failed"},
          {"source", out + ".cu"}, {"compiler_log", out + ".ptxas.log"},
          {"report_log", out + ".resource_report.log"}, {"compile_s", compile_s}}).Dump() + "\n");
    throw std::runtime_error("MX-5 resource report failed at python/tilemega/mx5_resources.py; see " +
                             out + ".resource_report.log (artifact retained)");
  }
  auto const resources = json::ParseFile(out + ".resources.json");
  auto const& violations = resources.At("violations").AsArray("violations");
  if (!violations.empty()) {
    Write(out + ".resource_failure.json", resources.Dump() + "\n");
    std::ostringstream message;
    std::string const header = "include/tilemega/Codegen/tasks/mx/DecodeProgramV2.cuh";
    std::ifstream source(root + "/" + header);
    std::string line;
    int line_number = 0;
    while (std::getline(source, line)) {
      ++line_number;
      if (line.find("mx_decode_kernel_v2(") != std::string::npos) break;
    }
    message << "MX-5 resource check failed at " << header << ':' << line_number << ": ";
    for (auto const& violation : violations) message << violation.AsString("violation") << "; ";
    message << "artifact retained at " << out << " (allow_resources cannot bypass v2 checks)";
    std::cerr << message.str() << '\n';
    throw std::runtime_error(message.str());
  }

  int const stages = 5 * model.layers + 1;
  int const body_smem = codegen::MxDecodeBodySmemBytesV2(model, r.batch, config);
  int const generation = (body_smem + r.batch * 4 + 15) / 16 * 16;
  Value resource_manifest = resources;
  resource_manifest.Set("compile_s", compile_s);
  auto const& kernel = resources.At("kernel");
  resource_manifest.Set("mtreg", kernel.At("registers"));
  resource_manifest.Set("streg", kernel.At("scalar_registers"));
  resource_manifest.Set("static_smem", kernel.At("static_smem_bytes"));
  resource_manifest.Set("stack_bytes", kernel.At("stack_bytes"));
  Array buffers;
  for (auto const& b : codegen::MxDecodeBuffersV2(*module, config)) {
    Value row = Object{{"name", b.name}, {"role", "external"}, {"dtype", b.dtype},
                       {"constant", Count(b.constant)}, {"per_batch", Count(b.per_batch)}};
    if (!b.recipe.empty()) row.Set("recipe", json::Parse(b.recipe));
    buffers.push_back(row);
  }
  auto scratch = [&](char const* name, char const* dtype, std::uint64_t elements, int element_bytes) {
    buffers.emplace_back(Object{{"name", name}, {"role", "internal"}, {"dtype", dtype},
        {"constant", Count(elements)}, {"bytes", Count(std::max<std::uint64_t>(elements * element_bytes, 16))}});
  };
  std::uint64_t const B = r.batch, G = r.grid;
  auto partial = [&](MxDecodeClassV2 const& c, int n) -> std::uint64_t {
    int const tiles = n / c.tn;
    return c.stream_k ? std::uint64_t(tiles) * backend::mx::MxMaxContributors(tiles, r.grid) * B * (c.tn + 1) : 0;
  };
  scratch("mx.x", "bf16", B * model.hidden, 2);
  scratch("mx.qkv", "bf16", B * nq, 2);
  scratch("mx.ctx", "bf16", B * model.heads * model.head_dim, 2);
  scratch("mx.h", "bf16", B * model.inter, 2);
  scratch("mx.gemm_partial", "f32", std::max({partial(config.qkv, nq), partial(config.o, model.hidden),
      partial(config.gate_up, 2 * model.inter), partial(config.down, model.hidden)}), 4);
  scratch("mx.attn_partial", "f32", B * model.hkv * std::min(model.capacity / 32, 104) * model.q_per_kv * (model.head_dim + 2), 4);
  scratch("mx.head_partial", "pair_f32_u32", B * G, 8);
  scratch("mx.t_qkv", "u32", nq / config.qkv.tn, 4);
  scratch("mx.t_o", "u32", model.hidden / config.o.tn, 4);
  scratch("mx.t_gu", "u32", 2 * model.inter / config.gate_up.tn, 4);
  scratch("mx.t_down", "u32", model.hidden / config.down.tn, 4);
  scratch("mx.t_attn", "u32", B * model.hkv, 4);
  scratch("mx.t_head", "u32", 16, 4);
  scratch("mx.barrier", "u64", std::uint64_t(stages) * 8, 8);
  Value runtime_allocations = Object{{"pasts", Object{{"dtype", "i32"}, {"elements", "set_steps.count"}}}};
  if (r.trace_stage) runtime_allocations.Set("trace", Object{{"dtype", "u64"},
      {"elements", "TILEMEGA_TRACE_LAUNCHES(default=16) * stages_per_step * grid * 3"}, {"minimum_bytes", 16}});
  if (diag) runtime_allocations.Set("step_ns", Object{{"dtype", "u64"}, {"elements", "set_steps.count + 1"}});
  if (r.watchdog) runtime_allocations.Set("watchdog", Object{{"bytes", 96}, {"allocation", "host_mapped"},
      {"enabled_when", "TILEMEGA_WATCHDOG_MS != 0"}, {"default_ms", 10000}});
  auto edge = [](char const* from, char const* to, MxDecodeClassV2 const& c) -> Value {
    return Object{{"from", from}, {"to", to}, {"P_w", c.prefetch}, {"loads", c.prefetch * (c.tn / 16)}};
  };
  Array transitions{Object{{"from", "qkv"}, {"to", "attention"}, {"prefetch_kind", config.attn_prefetch},
                           {"loads", (model.head_dim / 16) * config.attn_prefetch}},
      edge("attention", "o", config.o), edge("o", "gate_up", config.gate_up),
      edge("gate_up", "down", config.down), edge("down", "qkv", config.qkv),
      edge("down_last", "head", config.head), edge("head", "qkv_layer0", config.qkv)};
  Value manifest = Object{{"identity_schema", 1}, {"version", 5}, {"model", r.model}, {"input", r.input.string()},
      {"phase", "decode"}, {"executor", "mx_static"}, {"kv_layout", "vt"}, {"kv_storage", "block_fragments"},
      {"batch_lo", r.batch}, {"batch_hi", r.batch}, {"seq", 1}, {"capacity", model.capacity},
      {"past_lo", r.past_lo}, {"past_hi", r.past_hi}, {"grid", r.grid}, {"residency", 1}, {"pg", "off"},
      {"threads", config.threads}, {"kThreads", config.threads}, {"smem", generation + 16},
      {"body_smem", body_smem}, {"token_offset", body_smem}, {"generation_offset", generation},
      {"layers", model.layers}, {"head_dim", model.head_dim}, {"heads_kv", model.hkv}, {"q_per_kv", model.q_per_kv},
      {"stages_per_step", stages}, {"classes", classes}, {"transitions", transitions}, {"b_min", config.b_min}, {"attn_prefetch", config.attn_prefetch},
      {"watchdog", r.watchdog}, {"trace_stage", r.trace_stage}, {"diag", diag}, {"barrier_prefetch", 1},
      {"decisions", Object{{"DT-W", 576}, {"plan_threads", config.threads}, {"DT-R", "fence_atomicAdd"},
          {"DT-P", "P-B"}, {"DT-X", "X-A"}, {"DT-KV", "block_fragments"}}},
      {"selection", Object{{"mode", "manual_pin"}, {"purpose", purpose}, {"pin", r.pin},
          {"solver_selected", false}, {"default_enabled", false}}},
      {"resources", resource_manifest}, {"resources_complete", false}, {"runtime_target", r.runtime_target},
      {"source_snapshot", out + ".source.json"}, {"buffers", buffers}, {"runtime_allocations", runtime_allocations}};
  Write(out + ".plan.json", manifest.Dump() + "\n");
  python("-m tilemega.build.identity --root " + Quote(root) + " --snapshot " + Quote(out + ".source.json") + " --so " + Quote(out));
  python("-c " + Quote("from tilemega.build.identity import verify; verify(" + Value(out).Dump(0) + ")"));
  std::cerr << "MX5_DECODE manual_pin threads=" << config.threads << " layers=" << model.layers << " batch=" << r.batch
            << " stack=0 mtreg=" << kernel.At("registers").AsNumber("registers") << " compile_s=" << compile_s << '\n';
  return 0;
}
}  // namespace tilemega::commands::mx_decode
