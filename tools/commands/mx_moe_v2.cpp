// SPDX-License-Identifier: BSD-3-Clause
#include "MxDecode.h"
#include "Mx5CompileReceipt.h"
#include "Toolchain.h"

#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include <tilemega/Codegen/MxMoeProgramV2.h>
#include <tilemega/Support/BuildConfig.h>
#include <tilemega/Support/Json.h>
#include <tilemega/Solver/MxMoeSelectionV2.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/ADT/StringExtras.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace tilemega::commands::mx_decode {
namespace {
using json::Array;
using json::Object;
using json::Value;
using codegen::MxDecodeClassV2;
using codegen::MxMoeConfigV2;

std::string Quote(std::string const& text) {
  std::string out = "'";
  for (char c : text) out += c == '\'' ? "'\\''" : std::string(1, c);
  return out + "'";
}
void Write(std::string const& path, std::string const& text) {
  std::ofstream stream(path);
  stream << text;
  if (!stream.good()) throw std::runtime_error("MX-5 MoE cannot write " + path);
}
int Int(Value const& object, char const* name) {
  double const number = object.At(name).AsNumber(name);
  if (!std::isfinite(number) || std::floor(number) != number ||
      number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string("MX-5 MoE pin requires integer ") + name);
  return int(number);
}
MxDecodeClassV2 ParseClass(Value const& pin, char const* name) {
  auto const& c = pin.At(name);
  auto const& part = c.At("part").AsString("part");
  if (part != "kTileRange" && part != "kStreamK")
    throw std::invalid_argument("MX-5 MoE pin has invalid part");
  return {Int(c, "tn"), Int(c, "wn"), Int(c, "d"), part == "kStreamK", Int(c, "prefetch"),
          c.Find("r_next") ? Int(c, "r_next") : -1};
}
Value ClassJson(MxDecodeClassV2 const& c) {
  return Object{{"tn", c.tn}, {"wn", c.wn}, {"d", c.d}, {"part", c.stream_k ? "kStreamK" : "kTileRange"},
                {"prefetch", c.prefetch}, {"r_next", c.r_next}};
}
Value Count(std::uint64_t n) { return Value(double(n)); }
}

int TryPrepareMoeV2(Request const& r, frontend::ModelPlan const& plan) {
  if (!r.v2_results.empty())
    throw std::invalid_argument("MX-5 MoE diagnostic previews do not consume measured model results");
  if (r.grid != 104 || r.past_lo != 64 || r.past_hi != 1086 || r.capacity != 1088)
    throw std::invalid_argument("MX-5 MoE measured costs require grid104, capacity1088 and past64:1086");
  auto table = json::ParseFile(r.v2_solve_table);
  if (Int(table, "schema_version") != 1 ||
      table.At("table_kind").AsString("table_kind") != "mx5_moe_geometry_outcomes")
    throw std::invalid_argument("MX-5 MoE requires an explicit native geometry-outcome table");
  auto hash = [](std::string const& path) {
    auto input = llvm::MemoryBuffer::getFile(path);
    if (!input) throw std::runtime_error("MX-5 MoE cannot hash " + path);
    llvm::SHA256 digest; digest.update(input.get()->getBuffer()); return llvm::toHex(digest.final(), true);
  };
  auto reference = [&](std::string const& path) -> Value {
    return Object{{"path", std::filesystem::absolute(path).string()}, {"sha256", hash(path)}};
  };
  auto table_ref = reference(r.v2_solve_table), input_ref = reference(r.input.string());
  auto doc = solver::MxMoeCandidatesJsonV2(plan, r.v2_threads, table.At("rows").AsArray("rows"),
      table.At("rejections").AsArray("rejections"), r.moe_diagnostic_candidates);
  if (reference(r.v2_solve_table).Dump(0) != table_ref.Dump(0) || reference(r.input.string()).Dump(0) != input_ref.Dump(0))
    throw std::invalid_argument("MX-5 MoE table or bridge changed while preparing candidates");
  std::string out = r.output.string() + ".mx5_candidates.json";
  if (std::filesystem::exists(out))
    throw std::invalid_argument("MX-5 MoE existing preview is preserved: " + out);
  doc.Set("stage_table", table_ref); doc.Set("input", input_ref);
  doc.Set("producer", reference(std::filesystem::read_symlink("/proc/self/exe").string()));
  doc.Set("request", Object{{"batch", r.batch}, {"threads", r.v2_threads}, {"grid", r.grid},
      {"capacity", r.capacity}, {"past_lo", r.past_lo}, {"past_hi", r.past_hi}});
  if (!r.output.parent_path().empty()) std::filesystem::create_directories(r.output.parent_path());
  Write(out, doc.Dump() + "\n");
  std::cerr << "MX5_MOE diagnostic_candidate_preview " << out << " execution_authorized=false final=false\n";
  return 0;
}

int TryCompileMoeV2(Request const& r, frontend::ModelPlan const& plan) {
  // Analyze the DM plan directly; it must never reach the descriptor executor.
  auto const model = codegen::AnalyzeMxMoe(plan);
  if (!r.interpreted_flags.empty())
    throw std::invalid_argument(r.interpreted_flags.front() + " belongs to the interpreted executor; xcore1000 MoE requires native stages");
  if (r.grid != 104 || (r.watchdog != 0 && r.watchdog != 1))
    throw std::invalid_argument("MX-5 MoE requires grid 104 and watchdog 0 or 1");
  if (r.pin.empty())
    throw std::invalid_argument("MX-5 MoE measured solver is not ready; supply an explicit diagnostic mx_moe_v2 pin");
  auto const doc = json::ParseFile(r.pin);
  auto const& pin = doc.At("mx_moe_v2");
  if (pin.Find("version") && Int(pin,"version") != 2)
    throw std::invalid_argument("mx_moe_v2 pin version must be 2");
  std::string const purpose = pin.Find("purpose") ? pin.At("purpose").AsString("purpose") : "manual";
  if (purpose != "manual" && purpose != "resource_probe")
    throw std::invalid_argument("MX-5 MoE pinned builds require purpose manual or resource_probe");
  MxMoeConfigV2 config{ParseClass(pin, "qkv"), ParseClass(pin, "o"), ParseClass(pin, "router"),
      ParseClass(pin, "gate_up"), ParseClass(pin, "down"), ParseClass(pin, "head"),
      Int(pin, "threads"), Int(pin, "b_min"), Int(pin, "attn_prefetch")};
  codegen::CheckMxMoeConfigV2(plan, config);
  int const nq = (model.queries + 2) * model.hkv * model.dim;
  int const active = codegen::MxMoeExpectedActiveV2(model.experts,r.batch);
  auto resolve = [&](MxDecodeClassV2& c, int n, int k) {
    c.prefetch = codegen::MxDecodePrefetchV2(c, n, k, r.grid, config.threads);
  };
  resolve(config.qkv, nq, model.hidden);
  resolve(config.o, model.hidden, model.heads * model.dim);
  resolve(config.router, model.experts, model.hidden);
  resolve(config.down, active * model.hidden, model.intermediate);
  resolve(config.head, model.vocabulary, model.hidden);
  int const past_lo = r.prompt ? 0 : r.past_lo, past_hi = r.prompt ? 63 : r.past_hi;
  auto const source = codegen::EmitMxMoeProgramV2(plan, config, r.prompt, past_lo, past_hi);
  if (!r.output.parent_path().empty()) std::filesystem::create_directories(r.output.parent_path());
  std::string const out = std::filesystem::absolute(r.output).lexically_normal().string(), root = TILEMEGA_SOURCE_DIR;
  compile_receipt_v1::RequireFresh(out);
  std::string const environment = "PYTHONPATH=" + Quote(root + "/python") + " LD_LIBRARY_PATH=" +
      Quote(CudaLibraryDirectory() + ":" + (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : ""));
  auto python = [&](std::string const& arguments) {
    if (std::system((environment + " python3 " + arguments).c_str()))
      throw std::runtime_error("MX-5 MoE build helper failed: " + arguments);
  };
  python("-m tilemega.build.identity --root " + Quote(root) + " --snapshot " + Quote(out + ".source.json") +
      " --capture --compiler " + Quote(std::filesystem::read_symlink("/proc/self/exe").string()));
  bool const diag = std::getenv("TILEMEGA_MX_DIAG") && std::string(std::getenv("TILEMEGA_MX_DIAG")) == "1";
  std::string const flags = " -DTILEMEGA_SYNC_V3=1 -DTILEMEGA_MX_STATIC_DECODE_V2=1 -DTILEMEGA_WATCHDOG=" +
      std::to_string(r.watchdog) + " -DTILEMEGA_TRACE_STAGE=" + (r.trace_stage ? "1" : "0") +
      " -DTILEMEGA_MX_DIAG=" + (diag ? "1" : "0");
  std::string device_flags = DeviceBuildFlags();
  if (device_flags.find("-fno-strict-aliasing") == std::string::npos) device_flags += " -fno-strict-aliasing";
  // Avoid MACA's implicit 168-VGPR cap while preserving the prompt's
  // per-plan limit. Prefetch still reserves headroom with R_cap=220/248.
  device_flags += " -mllvm --max-mtreg-number=" + std::to_string(config.threads == 576 ? 227 : 255);
  std::string const command = Quote(compile_receipt_v1::CompilerPath(NvccPath())) + " -std=c++17 -O3" + device_flags + flags +
      " -mllvm -print-after=stack-frame-layout -shared -Xcompiler=-fPIC -x cu -I" + Quote(root + "/include") + " " +
      Quote(out + ".cu") + " -L" + Quote(CudaLibraryDirectory()) + " -lcudart -o " + Quote(out);
  Value classes = Object{{"qkv", ClassJson(config.qkv)}, {"o", ClassJson(config.o)}, {"router", ClassJson(config.router)},
      {"gate_up", ClassJson(config.gate_up)}, {"down", ClassJson(config.down)}, {"head", ClassJson(config.head)}};
  Value resolved = classes;
  resolved.Set("threads", config.threads); resolved.Set("b_min", config.b_min);
  resolved.Set("attn_prefetch", config.attn_prefetch); resolved.Set("purpose", purpose);
  Write(out + ".cu", source);
  Write(out + ".runtime_flags.txt", flags + "\n");
  Write(out + ".build_command.txt", command + "\n");
  Write(out + ".mx5_pin.json", Value(Object{{"mx_moe_v2", resolved}}).Dump() + "\n");
  auto const start = std::chrono::steady_clock::now();
  auto const argv = compile_receipt_v1::Arguments(command);
  auto const inputs = compile_receipt_v1::Inputs(out, r.pin, argv.at(0), std::filesystem::read_symlink("/proc/self/exe").string());
  int const status = compile_receipt_v1::Run(argv, out, inputs);
  double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (status) {
    Write(out + ".resource_failure.json", Value(Object{{"status", "compiler_failed"}, {"exit_status", status},
        {"compile_s", seconds}}).Dump() + "\n");
    throw std::runtime_error("MX-5 MoE cucc failed; source/log/receipt retained at " + out);
  }
  std::string const resource_command = environment + " python3 -m tilemega.mx5_resources --so " + Quote(out) +
      " --log " + Quote(out + ".ptxas.log") + " --threads " + std::to_string(config.threads) +
      " --kernel mx_moe_kernel_v2 --out " + Quote(out + ".resources.json");
  if (std::system((resource_command + " >" + Quote(out + ".resource_report.log") + " 2>&1").c_str()))
    throw std::runtime_error("MX-5 MoE resource report failed; retain " + out + ".resource_report.log");
  Value resources = json::ParseFile(out + ".resources.json");
  if (!resources.At("violations").AsArray("violations").empty()) {
    Write(out + ".resource_failure.json", resources.Dump() + "\n");
    throw std::runtime_error("MX-5 MoE resource check failed at include/tilemega/Codegen/tasks/mx/MoeProgramV2.cuh; artifact retained: " + out);
  }
  resources.Set("compile_s", seconds);
  auto const kernel = resources.At("kernel");
  resources.Set("mtreg", kernel.At("registers")); resources.Set("streg", kernel.At("scalar_registers"));
  resources.Set("stack_bytes", kernel.At("stack_bytes")); resources.Set("static_smem", kernel.At("static_smem_bytes"));
  Array buffers;
  for (auto const& row : codegen::PrepareMxMoeBuffersV2(plan, {unsigned(config.qkv.tn), unsigned(config.o.tn),
      unsigned(config.router.tn), unsigned(config.gate_up.tn), unsigned(config.down.tn), unsigned(config.head.tn)})) {
    auto const& b = row.buffer;
    Value value = Object{{"name", b.name}, {"role", "external"}, {"dtype", b.dtype},
        {"constant", Count(b.constant)}, {"per_batch", Count(b.per_batch)}};
    if (!b.pack_json.empty()) value.Set("recipe", json::Parse(b.pack_json));
    buffers.push_back(value);
  }
  auto scratch = [&](char const* name, char const* dtype, std::uint64_t count, int bytes) {
    buffers.emplace_back(Object{{"name", name}, {"role", "internal"}, {"dtype", dtype}, {"constant", Count(count)},
        {"bytes", Count(std::max<std::uint64_t>(16, count * bytes))}});
  };
  std::uint64_t const B = r.batch, H = model.hidden, I = model.intermediate, G = r.grid;
  auto partial = [&](MxDecodeClassV2 const& c, int n) -> std::uint64_t {
    int const tiles = n / c.tn;
    return c.stream_k ? std::uint64_t(tiles) * backend::mx::MxMaxContributors(tiles, r.grid) * B * (c.tn + 1) : 0;
  };
  int const stages = 6 * model.layers.size() + 1;
  scratch("mx.x", "bf16", B * H, 2); scratch("mx.qkv", "bf16", B * nq, 2);
  scratch("mx.ctx", "bf16", B * model.heads * model.dim, 2); scratch("mx.h", "bf16", model.experts * B * I, 2);
  scratch("mx.gemm_partial", "f32", std::max({partial(config.qkv, nq), partial(config.o, H), partial(config.router, model.experts)}), 4);
  scratch("mx.attn_partial", "f32", B * model.hkv * std::min(int(model.capacity / 32), 104) * model.queries * (model.dim + 2), 4);
  scratch("mx.head_partial", "pair_f32_u32", B * G, 8);
  scratch("mx.router_logits", "bf16", B * model.experts, 2); scratch("mx.expert_partials", "bf16", B * 8 * H, 2);
  scratch("mx.n_active", "u32", 1, 4); scratch("mx.active", "u32", model.experts, 4);
  scratch("mx.row_count", "u32", model.experts, 4); scratch("mx.rows", "pair_u32", model.experts * B, 8);
  scratch("mx.u_gate_up", "u32", 1, 4); scratch("mx.u_down", "u32", 1, 4);
  scratch("mx.routing_weights", model.routing_weights_fp32 ? "f32" : "bf16", B * 8, model.routing_weights_fp32 ? 4 : 2);
  scratch("mx.t_qkv", "u32", nq / config.qkv.tn, 4); scratch("mx.t_o", "u32", H / config.o.tn, 4);
  scratch("mx.router_tile_tickets", "u32", model.experts / config.router.tn, 4); scratch("mx.router_ticket", "u32", 16, 4);
  scratch("mx.combine_tickets", "u32", H / config.down.tn, 4); scratch("mx.t_attn", "u32", B * model.hkv, 4);
  scratch("mx.t_head", "u32", 16, 4); scratch("mx.barrier", "u64", std::uint64_t(stages) * 8, 8);
  if (diag) {
    scratch("mx.route_debug", "u32", 64 * model.layers.size() * B * 8, 4);
    scratch("mx.route_input", "bf16", 64 * model.layers.size() * B * H, 2);
    scratch("mx.route_logits", "bf16", 64 * model.layers.size() * B * model.experts, 2);
    scratch("mx.route_weights", model.routing_weights_fp32 ? "f32" : "bf16",
        64 * model.layers.size() * B * 8, model.routing_weights_fp32 ? 4 : 2);
  }
  int const body_smem = codegen::MxMoeBodySmemBytesV2(plan, config);
  int const metadata_offset = (body_smem + r.batch * 4 + 15) / 16 * 16, generation_offset = metadata_offset + 16;
  auto edge = [](char const* from, char const* to, MxDecodeClassV2 const& c) -> Value {
    return Object{{"from", from}, {"to", to}, {"P_w", c.prefetch}, {"loads", c.prefetch * (c.tn / 16)}};
  };
  Array transitions{Object{{"from", "qkv"}, {"to", "attention"}, {"prefetch_kind", config.attn_prefetch},
      {"loads", int(model.dim / 16) * config.attn_prefetch}}, edge("attention", "o", config.o), edge("o", "router", config.router),
      edge("router", "gate_up", config.gate_up), edge("gate_up", "down", config.down), edge("down", "qkv", config.qkv),
      edge("down_last", "head", config.head), edge("head", "qkv_layer0", config.qkv)};
  Value runtime_allocations = Object{{"pasts", Object{{"dtype", "i32"}, {"elements", "set_steps.count"}}}};
  if (diag) runtime_allocations.Set("step_ns", Object{{"dtype", "u64"}, {"elements", "set_steps.count + 1"}});
  if (r.trace_stage) runtime_allocations.Set("trace", Object{{"dtype", "u64"},
      {"elements", "TILEMEGA_TRACE_LAUNCHES(default=16) * stages_per_step * grid * 3"}, {"minimum_bytes", 16}});
  if (r.watchdog) runtime_allocations.Set("watchdog", Object{{"bytes", 96}, {"allocation", "host_mapped"},
      {"enabled_when", "TILEMEGA_WATCHDOG_MS != 0"}, {"default_ms", 10000}});
  Value manifest = Object{{"identity_schema", 1}, {"version", 5}, {"program_kind", "moe"}, {"model", r.model}, {"input", r.input.string()},
      {"phase", r.prompt ? "prefill" : "decode"}, {"executor", r.prompt ? "mx_static_prompt" : "mx_static"},
      {"kv_layout", "vt"}, {"kv_storage", "block_fragments"}, {"batch_lo", r.batch}, {"batch_hi", r.batch},
      {"seq", r.prompt ? 64 : 1}, {"capacity", int(model.capacity)}, {"past_lo", past_lo}, {"past_hi", past_hi},
      {"grid", r.grid}, {"residency", 1}, {"pg", "off"}, {"threads", config.threads}, {"kThreads", config.threads},
      {"compiler_mtreg_budget", config.threads == 576 ? 227 : 255},
      {"smem", generation_offset + 16}, {"body_smem", body_smem}, {"token_offset", body_smem},
      {"metadata_offset", metadata_offset}, {"generation_offset", generation_offset}, {"layers", int(model.layers.size())},
      {"head_dim", int(model.dim)}, {"heads_kv", int(model.hkv)}, {"q_per_kv", int(model.queries)},
      {"experts", int(model.experts)}, {"qk_norm_scope", model.row_qk_norm ? "row" : "head"},
      {"routing_softmax_scope", model.softmax_all ? "all" : "selected"}, {"top_k", int(model.top_k)}, {"stages_per_step", stages},
      {"prompt_barriers", r.prompt ? 64 * (stages - 1) + 1 : 0}, {"classes", classes}, {"transitions", transitions},
      {"b_min", config.b_min}, {"attn_prefetch", config.attn_prefetch}, {"watchdog", r.watchdog},
      {"trace_stage", r.trace_stage}, {"diag", diag}, {"barrier_prefetch", 1},
      {"decisions", Object{{"DT-W", 576}, {"plan_threads", config.threads}, {"DT-R", "fence_atomicAdd"},
          {"DT-P", "P-B"}, {"DT-X", "X-A"}, {"DT-KV", "block_fragments"}}},
      {"selection", Object{{"mode", "manual_pin"}, {"purpose", purpose}, {"pin", r.pin}, {"solver_selected", false},
          {"default_enabled", false}, {"diagnostic", true}, {"final", false}}},
      {"resources", resources}, {"resources_complete", false}, {"runtime_target", r.runtime_target},
      {"source_snapshot", out + ".source.json"}, {"buffers", buffers}, {"runtime_allocations", runtime_allocations}};
  Write(out + ".plan.json", manifest.Dump() + "\n");
  python("-m tilemega.build.identity --root " + Quote(root) + " --snapshot " + Quote(out + ".source.json") + " --so " + Quote(out));
  python("-c " + Quote("from tilemega.build.identity import verify; verify(" + Value(out).Dump(0) + ")"));
  std::cerr << "MX5_MOE manual_pin prompt=" << r.prompt << " threads=" << config.threads << " layers=" << model.layers.size()
            << " batch=" << r.batch << " stack=0 mtreg=" << kernel.At("registers").AsNumber("registers") << '\n';
  return 0;
}
}  // namespace tilemega::commands::mx_decode
