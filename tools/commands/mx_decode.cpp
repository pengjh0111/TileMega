// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- tools/commands/mx_decode.cpp
//
// `compile ... --serving decode` for a dense decoder on xcore1000 (MX4
// prompt §3.9): the MX-4 static decode program. The plan is the front end's
// dense decoder; its per-class configuration comes from the solver or from
// an MX-4 pin (--evaluate-configs {"mx_decode": {...}}); the generator emits
// the program, cucc builds it, and the manifest records executor mx_static,
// kv_layout vt, the configuration, resources and compile time. Options of
// the interpreted decode executor are rejected here.
#include "MxDecode.h"
#include "Toolchain.h"

#include <tilemega/Backend/Mx/DecodeShapes.h>
#include <tilemega/Codegen/MxDecodeProgram.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Support/BuildConfig.h>
#include <tilemega/Support/Json.h>

#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace tilemega::commands::mx_decode {
namespace {

std::string Quote(std::string const& value) {
  std::string out = "'";
  for (char c : value) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return out + "'";
}

codegen::MxDecodeClass ParseClass(json::Value const& v, char const* name) {
  auto const& o = v.At(name);
  auto part = o.At("part").AsString(name);
  if (part != "kStreamK" && part != "kTileRange")
    throw std::invalid_argument(std::string("mx_decode pin: part of ") + name + " must be kStreamK or kTileRange");
  return {int(o.At("tn").AsNumber(name)), int(o.At("d").AsNumber(name)), part == "kStreamK"};
}

/// --evaluate-configs for the static program: {"mx_decode": {"qkv": {"tn":
/// 128, "d": 1, "part": "kTileRange"}, "o": ..., "gate_up": ..., "down":
/// ..., "head": ..., "tmin": 64}}.
codegen::MxDecodeConfig ReadPin(std::string const& path) {
  auto doc = json::ParseFile(path);
  auto const* pin = doc.Find("mx_decode");
  if (!pin) throw std::invalid_argument("--evaluate-configs: an MX-4 decode pin needs an \"mx_decode\" object");
  codegen::MxDecodeConfig c;
  c.qkv = ParseClass(*pin, "qkv");
  c.o = ParseClass(*pin, "o");
  c.gate_up = ParseClass(*pin, "gate_up");
  c.down = ParseClass(*pin, "down");
  c.head = ParseClass(*pin, "head");
  c.tmin = int(pin->At("tmin").AsNumber("tmin"));
  return c;
}

std::string ClassJson(codegen::MxDecodeClass const& k) {
  std::ostringstream o;
  o << "{\"tn\": " << k.tn << ", \"d\": " << k.d << ", \"part\": \"" << (k.stream_k ? "kStreamK" : "kTileRange")
    << "\"}";
  return o.str();
}

}  // namespace

int TryCompile(Request const& r) {
  auto bridge = frontend::ReadExportBridge(r.input.string());
  frontend::ServingOptions options;
  options.deferred_norm = true;  // gamma folded into qkv, gate_up and head (§3.9)
  options.phase = frontend::ServingOptions::Phase::kDecode;
  options.seq = 1;
  options.capacity = r.capacity;
  auto plan = frontend::BuildModelPlan(bridge.nodes, bridge.inputs, bridge.outputs, options);
  if (plan.dm) return -1;  // MoE and DNN plans keep the interpreted executor
  for (auto const& flag : r.interpreted_flags)
    throw std::invalid_argument(flag + " belongs to the interpreted decode executor; a dense xcore1000 decode "
                                       "plan is the MX-4 static program");
  if (r.pin.empty()) throw std::invalid_argument("MX-4 decode: pass --evaluate-configs with an mx_decode pin");

  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  context.getOrLoadDialect<dialect::ExecDialect>();
  frontend::ImportOptions import;
  import.gemms.assign(plan.gemms.size(), frontend::GemmGranularity{16, 128, 128, 2, 1});
  frontend::ImportSummary summary;
  auto module = frontend::TorchExportImporter{}.ImportPlan(r.input.string(), plan, context, &summary, import);
  if (!module) throw std::runtime_error("MX-4 decode: import failed");
  auto const model = codegen::AnalyzeMxDecode(*module);
  auto const config = ReadPin(r.pin);
  codegen::CheckMxDecodeConfig(model, r.batch, config, r.grid);
  codegen::WriteMxDecodeConfig(*module, config);
  auto const source = codegen::EmitMxDecodeProgram(*module, r.batch, r.past_lo, r.past_hi, r.grid);

  std::string const out = r.output.string();
  std::string const root = TILEMEGA_SOURCE_DIR;
  auto identity = [&](bool capture) {
    std::string command = "PYTHONPATH=" + Quote(root + "/python") + " LD_LIBRARY_PATH=" +
                          Quote(CudaLibraryDirectory() + ":" +
                                (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "")) +
                          " python3 -m tilemega.build.identity --root " + Quote(root) + " --snapshot " +
                          Quote(out + ".source.json");
    if (capture)
      command += " --capture --compiler " + Quote(std::filesystem::read_symlink("/proc/self/exe").string());
    else
      command += " --so " + Quote(out);
    if (std::system(command.c_str()) != 0) throw std::runtime_error("serving artifact identity failed");
  };
  identity(true);
  std::ofstream(out + ".cu") << source;
  // §6 D3: the compiler hoists every body's loop-invariant address
  // arithmetic out of the program's step and layer loops, where it stays
  // live across all stages and spills; without IR LICM the program has no
  // stack frame (measured, MX4_resources_c550.csv).
  std::string flags = " -DTILEMEGA_SYNC_V3=1 -DTILEMEGA_WATCHDOG=" + std::to_string(r.watchdog) +
                      " -DTILEMEGA_MX_BARRIER_PREFETCH=" + std::to_string(r.barrier_prefetch) +
                      " -mllvm -metaxgpu-disable-licm";
  if (r.trace_stage) flags += " -DTILEMEGA_TRACE_STAGE=1";
  if (char const* diag = std::getenv("TILEMEGA_MX_DIAG"); diag && std::string(diag) == "1")
    flags += " -DTILEMEGA_MX_DIAG=1";  // diagnostic builds only, never measured
  std::ofstream(out + ".runtime_flags.txt") << flags << "\n";
  std::string const command = Quote(NvccPath()) + " -std=c++17 -O3" + DeviceBuildFlags() + flags +
                              " -shared -Xcompiler=-fPIC -x cu -I" + Quote(root + "/include") + " " +
                              Quote(out + ".cu") + " -L" + Quote(CudaLibraryDirectory()) + " -lcudart -o " +
                              Quote(out);
  std::ofstream(out + ".build_command.txt") << command << '\n';
  auto const start = std::chrono::steady_clock::now();
  if (std::system((command + " >" + Quote(out + ".ptxas.log") + " 2>&1").c_str()) != 0)
    throw std::runtime_error("cucc failed while building the MX-4 decode program");
  double const compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  // Resources of the program kernel (-resource-usage).
  std::ifstream log(out + ".ptxas.log");
  std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
  int mtreg = -1, streg = -1, stack = -1, static_smem = -1;
  std::smatch match;
  std::regex const kernel(R"(Function properties for\s+\S*mx_decode_kernel\S*\s*\n\s*(\d+) bytes stack frame[\s\S]*?Used\s+(\d+) MTregisters,\s+(\d+) STregisters,\s+(\d+) bytes shared mem)");
  if (std::regex_search(text, match, kernel)) {
    stack = std::stoi(match[1]);
    mtreg = std::stoi(match[2]);
    streg = std::stoi(match[3]);
    static_smem = std::stoi(match[4]);
  }
  auto const code_bytes = std::filesystem::file_size(out);
  // §4.4 item 4: no stack frame, no static shared memory, <= 255 MTregs.
  if (stack != 0 || static_smem != 0 || mtreg < 0 || mtreg > 255) {
    std::cerr << "MX4_RESOURCES stack=" << stack << " static_smem=" << static_smem << " mtreg=" << mtreg << '\n';
    if (!r.allow_resources) throw std::runtime_error("MX-4 decode program fails the resource check (§4.4.4, D3)");
  }

  int const smem_need = std::max({backend::mx::MxGemmSmem(r.batch, config.qkv.tn).bytes,
                                  backend::mx::MxGemmSmem(r.batch, config.o.tn).bytes,
                                  backend::mx::MxGemmSmem(r.batch, config.gate_up.tn).bytes,
                                  backend::mx::MxGemmSmem(r.batch, config.down.tn).bytes,
                                  backend::mx::MxGemmSmem(r.batch, config.head.tn).bytes,
                                  65536 / (r.grid / 104 + 1) + 16});
  std::ofstream manifest(out + ".plan.json");
  manifest << "{\n  \"identity_schema\": 1,\n  \"model\": " << std::quoted(r.model)
           << ",\n  \"phase\": \"decode\",\n  \"executor\": \"mx_static\",\n  \"kv_layout\": \"vt\""
           << ",\n  \"batch_lo\": " << r.batch << ",\n  \"batch_hi\": " << r.batch << ",\n  \"past_lo\": "
           << r.past_lo << ",\n  \"past_hi\": " << r.past_hi << ",\n  \"seq\": 1,\n  \"capacity\": " << model.capacity
           << ",\n  \"pg\": \"off\",\n  \"watchdog\": " << r.watchdog << ",\n  \"barrier_prefetch\": "
           << r.barrier_prefetch << ",\n  \"trace_stage\": " << int(r.trace_stage) << ",\n  \"grid\": " << r.grid
           << ",\n  \"residency\": " << r.grid / 104 << ",\n  \"threads\": 512,\n  \"smem\": " << smem_need
           << ",\n  \"layers\": " << model.layers << ",\n  \"head_dim\": " << model.head_dim
           << ",\n  \"q_per_kv\": " << model.q_per_kv << ",\n  \"heads_kv\": " << model.hkv
           << ",\n  \"classes\": {\"qkv\": " << ClassJson(config.qkv) << ", \"o\": " << ClassJson(config.o)
           << ", \"gate_up\": " << ClassJson(config.gate_up) << ", \"down\": " << ClassJson(config.down)
           << ", \"head\": " << ClassJson(config.head) << "},\n  \"attention_tmin\": " << config.tmin
           << ",\n  \"attention_max_blocks\": "
           << backend::mx::MxAttentionMaxBlocks(r.batch * model.hkv, model.capacity, config.tmin, r.grid * 8)
           << ",\n  \"stages_per_step\": " << 5 * model.layers + 2 << ",\n  \"resources\": {\"mtreg\": " << mtreg
           << ", \"streg\": " << streg << ", \"stack_bytes\": " << stack << ", \"static_smem\": " << static_smem
           << ", \"code_bytes\": " << code_bytes << ", \"compile_s\": " << std::fixed << std::setprecision(1)
           << compile_s << "},\n  \"selection\": {\"mode\": " << std::quoted(r.selection)
           << ", \"pin\": " << std::quoted(r.pin) << "},\n  \"runtime_target\": " << std::quoted(r.runtime_target)
           << ",\n  \"buffers\": [";
  // External buffers as the ABI reports them (recipes parsed by Python), then
  // the program scratch tm_plan_create allocates (DecodeRuntime.cuh).
  bool comma = false;
  for (auto const& b : codegen::MxDecodeBuffers(*module)) {
    manifest << (comma ? ",\n    " : "\n    ") << "{\"name\": " << std::quoted(b.name)
             << ", \"role\": \"external\", \"dtype\": " << std::quoted(b.dtype) << ", \"constant\": " << b.constant
             << ", \"per_batch\": " << b.per_batch;
    if (!b.recipe.empty()) manifest << ", \"recipe\": " << b.recipe;
    manifest << "}";
    comma = true;
  }
  auto scratch = [&](char const* name, char const* dtype, long long elements) {
    manifest << ",\n    {\"name\": " << std::quoted(name) << ", \"role\": \"internal\", \"dtype\": "
             << std::quoted(dtype) << ", \"constant\": " << elements << "}";
  };
  {
    using backend::mx::MxMaxContributors;
    long long const B = r.batch, G = r.grid;
    auto partial = [&](codegen::MxDecodeClass const& k, long long n) -> long long {
      if (!k.stream_k) return 0;
      long long const tiles = n / k.tn;
      return tiles * MxMaxContributors(int(tiles), r.grid) * B * (k.tn + 1);
    };
    long long const n_qkv = (model.q_per_kv + 2LL) * model.hkv * model.head_dim;
    scratch("mx.x", "bf16", B * model.hidden);
    scratch("mx.qkv", "bf16", B * n_qkv);
    scratch("mx.ctx", "bf16", B * model.heads * model.head_dim);
    scratch("mx.h", "bf16", B * model.inter);
    scratch("mx.gemm_partial", "f32",
            std::max({partial(config.qkv, n_qkv), partial(config.o, model.hidden),
                      partial(config.gate_up, 2LL * model.inter), partial(config.down, model.hidden)}));
    scratch("mx.attn_partial", "f32",
            B * model.hkv * ((model.capacity + 31) / 32) * model.q_per_kv * (model.head_dim + 2LL));
    scratch("mx.head_partial", "f32", 2 * B * G);
    scratch("mx.tickets", "i32", n_qkv / config.qkv.tn + model.hidden / config.o.tn +
                                     2LL * model.inter / config.gate_up.tn + model.hidden / config.down.tn +
                                     B * model.hkv + 16);
    scratch("mx.barrier", "i32", 16LL * (5 * model.layers + 2));
  }
  manifest << "\n  ]\n}\n";
  manifest.close();
  identity(false);
  std::cerr << "MX4_DECODE layers=" << model.layers << " batch=" << r.batch << " stack=" << stack
            << " mtreg=" << mtreg << " compile_s=" << compile_s << '\n';
  return 0;
}

}  // namespace tilemega::commands::mx_decode
