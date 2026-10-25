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
#include <tilemega/Solver/MxDecodeSolver.h>
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
#include <map>
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

namespace {
struct Built {
  int mtreg = -1, streg = -1, stack = -1, static_smem = -1;
  std::uintmax_t code_bytes = 0;
  double compile_s = 0;
};

std::string PinJson(codegen::MxDecodeConfig const& c) {
  return "{\"mx_decode\": {\"qkv\": " + ClassJson(c.qkv) + ", \"o\": " + ClassJson(c.o) +
         ", \"gate_up\": " + ClassJson(c.gate_up) + ", \"down\": " + ClassJson(c.down) + ", \"head\": " +
         ClassJson(c.head) + ", \"tmin\": " + std::to_string(c.tmin) + "}}";
}
}  // namespace

int TryCompile(Request const& r) {
  if (!r.pin.empty() && json::ParseFile(r.pin).Find("mx_decode_v2")) return TryCompileV2(r);
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

  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  context.getOrLoadDialect<dialect::ExecDialect>();
  frontend::ImportOptions import;
  import.gemms.assign(plan.gemms.size(), frontend::GemmGranularity{16, 128, 128, 2, 1});
  frontend::ImportSummary summary;
  auto module = frontend::TorchExportImporter{}.ImportPlan(r.input.string(), plan, context, &summary, import);
  if (!module) throw std::runtime_error("MX-4 decode: import failed");
  auto const model = codegen::AnalyzeMxDecode(*module);

  std::string const out = r.output.string();
  std::string const root = TILEMEGA_SOURCE_DIR;
  auto identity = [&](bool capture, std::string const& so) {
    std::string command = "PYTHONPATH=" + Quote(root + "/python") + " LD_LIBRARY_PATH=" +
                          Quote(CudaLibraryDirectory() + ":" +
                                (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "")) +
                          " python3 -m tilemega.build.identity --root " + Quote(root) + " --snapshot " +
                          Quote(out + ".source.json");
    if (capture)
      command += " --capture --compiler " + Quote(std::filesystem::read_symlink("/proc/self/exe").string());
    else
      command += " --so " + Quote(so);
    if (std::system(command.c_str()) != 0) throw std::runtime_error("serving artifact identity failed");
  };
  identity(true, out);

  std::string flags = " -DTILEMEGA_SYNC_V3=1 -DTILEMEGA_WATCHDOG=" + std::to_string(r.watchdog) +
                      " -DTILEMEGA_MX_BARRIER_PREFETCH=" + std::to_string(r.barrier_prefetch);
  if (r.trace_stage) flags += " -DTILEMEGA_TRACE_STAGE=1";
  if (char const* diag = std::getenv("TILEMEGA_MX_DIAG"); diag && std::string(diag) == "1")
    flags += " -DTILEMEGA_MX_DIAG=1";  // diagnostic builds only, never measured
  // Emit and build one configuration at `path` (plan source, flags, command,
  // resource log next to it); §4.4 item 4 checks the program's resources.
  auto build = [&](codegen::MxDecodeConfig const& config, std::string const& path) {
    codegen::CheckMxDecodeConfig(model, r.batch, config, r.grid);
    codegen::WriteMxDecodeConfig(*module, config);
    std::ofstream(path + ".cu") << codegen::EmitMxDecodeProgram(*module, r.batch, r.past_lo, r.past_hi, r.grid);
    std::ofstream(path + ".runtime_flags.txt") << flags << "\n";
    std::string const command = Quote(NvccPath()) + " -std=c++17 -O3" + DeviceBuildFlags() + flags +
                                " -shared -Xcompiler=-fPIC -x cu -I" + Quote(root + "/include") + " " +
                                Quote(path + ".cu") + " -L" + Quote(CudaLibraryDirectory()) + " -lcudart -o " +
                                Quote(path);
    std::ofstream(path + ".build_command.txt") << command << '\n';
    auto const start = std::chrono::steady_clock::now();
    if (std::system((command + " >" + Quote(path + ".ptxas.log") + " 2>&1").c_str()) != 0)
      throw std::runtime_error("cucc failed while building the MX-4 decode program");
    Built b;
    b.compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ifstream log(path + ".ptxas.log");
    std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
    std::smatch match;
    std::regex const kernel(R"(Function properties for\s+\S*mx_decode_kernel\S*\s*\n\s*(\d+) bytes stack frame[\s\S]*?Used\s+(\d+) MTregisters,\s+(\d+) STregisters,\s+(\d+) bytes shared mem)");
    if (std::regex_search(text, match, kernel)) {
      b.stack = std::stoi(match[1]);
      b.mtreg = std::stoi(match[2]);
      b.streg = std::stoi(match[3]);
      b.static_smem = std::stoi(match[4]);
    }
    b.code_bytes = std::filesystem::file_size(path);
    // §4.4 item 4: no stack frame, no static shared memory, <= 255 MTregs.
    if (b.stack != 0 || b.static_smem != 0 || b.mtreg < 0 || b.mtreg > 255) {
      std::cerr << "MX4_RESOURCES " << path << " stack=" << b.stack << " static_smem=" << b.static_smem
                << " mtreg=" << b.mtreg << '\n';
      if (!r.allow_resources) throw std::runtime_error("MX-4 decode program fails the resource check (§4.4.4, D3)");
    }
    return b;
  };

  // The manifest of the plan at `path` (§3.9): configuration, prediction,
  // resources, selection and the buffer table.
  auto write_manifest = [&](codegen::MxDecodeConfig const& config, std::string const& path, Built const& b,
                            double predicted_ns, std::map<std::string, double> const& stage_ns,
                            std::string const& selection_json) {
    int const smem_need = std::max({backend::mx::MxGemmSmem(r.batch, config.qkv.tn).bytes,
                                    backend::mx::MxGemmSmem(r.batch, config.o.tn).bytes,
                                    backend::mx::MxGemmSmem(r.batch, config.gate_up.tn).bytes,
                                    backend::mx::MxGemmSmem(r.batch, config.down.tn).bytes,
                                    backend::mx::MxGemmSmem(r.batch, config.head.tn).bytes,
                                    65536 / (r.grid / 104 + 1) + 16});
    std::ofstream manifest(path + ".plan.json");
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
             << ",\n  \"stages_per_step\": " << 5 * model.layers + 2 << std::setprecision(6)
             << ",\n  \"predicted_tpot_ns\": " << predicted_ns << ",\n  \"predicted_stage_ns\": {";
    {
      bool first = true;
      for (auto const& [cls, ns] : stage_ns) {
        manifest << (first ? "" : ", ") << std::quoted(cls) << ": " << ns;
        first = false;
      }
    }
    manifest << "},\n  \"resources\": {\"mtreg\": " << b.mtreg << ", \"streg\": " << b.streg
             << ", \"stack_bytes\": " << b.stack << ", \"static_smem\": " << b.static_smem
             << ", \"code_bytes\": " << b.code_bytes << ", \"compile_s\": " << std::fixed << std::setprecision(1)
             << b.compile_s << "},\n  \"selection\": " << selection_json << ",\n  \"runtime_target\": "
             << std::quoted(r.runtime_target) << ",\n  \"buffers\": [";
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
  };
  auto predict = [&](codegen::MxDecodeConfig const& config, std::map<std::string, double>* stage_ns) {
    if (r.table.empty() || !std::filesystem::exists(r.table)) return 0.0;
    return solver::MxDecodeSolver(r.table).Predict(model, r.batch, config, r.grid, r.past_lo, r.past_hi, stage_ns);
  };

  // The configuration: a pin, or the solver (§4.4 item 1) and, with
  // --selection measure, both of its plans timed end to end on the final
  // executor (item 2).
  codegen::MxDecodeConfig config;
  std::ostringstream selection;
  if (!r.pin.empty()) {
    config = ReadPin(r.pin);
    selection << "{\"mode\": \"pin\", \"pin\": " << std::quoted(r.pin) << "}";
  } else {
    solver::MxDecodeSolver solver(r.table);
    auto const sel = solver.Solve(model, r.batch, r.tmin, r.grid, r.past_lo, r.past_hi);
    std::ofstream domain(out + ".mx_candidates.tsv");
    domain << "class\ttn\td\tpart\tpredicted_ns\tmeasured\tspill_bytes\n";
    for (auto const& [cls, list] : sel.candidates)
      for (auto const& c : list)
        domain << cls << '\t' << c.k.tn << '\t' << c.k.d << '\t' << (c.k.stream_k ? "kStreamK" : "kTileRange") << '\t'
               << c.ns << '\t' << int(c.measured) << '\t' << c.spill_bytes << '\n';
    config = sel.first;
    std::string const mode = r.selection == "measure" ? "measure" : "predicted";
    double measured[2] = {0, 0};
    int chosen = 1;
    if (mode == "measure") {
      if (r.measure_cmd.empty()) throw std::invalid_argument("MX-4 decode: --selection measure needs --measure-cmd");
      codegen::MxDecodeConfig const plans[2] = {sel.first, sel.second};
      std::string paths[2];
      for (int i = 0; i < 2; ++i) {
        std::filesystem::create_directories(out + ".mx" + std::to_string(i + 1));
        paths[i] = out + ".mx" + std::to_string(i + 1) + "/plan.so";
        std::map<std::string, double> stages;
        double const ns = predict(plans[i], &stages);
        write_manifest(plans[i], paths[i], build(plans[i], paths[i]), ns, stages,
                       "{\"mode\": \"candidate\", \"plan\": " + std::to_string(i + 1) + "}");
        identity(false, paths[i]);
      }
      // 2 fresh processes per plan, alternating; each is the serving measure
      // (warm-up 1 + timed 2) that --measure-cmd names, with the plan as its
      // decode plan.
      std::vector<double> samples[2];
      std::ofstream rounds(out + ".mx_measure_rounds.tsv");
      rounds << "round\tplan\ttpot_ms\tartifact\n";
      for (int round = 0; round < 2; ++round)
        for (int j = 0; j < 2; ++j) {
          int const i = (j + round) % 2;
          std::string const artifact = out + ".mx" + std::to_string(i + 1) + "/measure.r" + std::to_string(round);
          std::string const command = "timeout --signal=TERM --kill-after=5s 2400s " + r.measure_cmd +
                                      " --decode-so " + Quote(paths[i]) + " --out " + Quote(artifact) + " >" +
                                      Quote(artifact + ".stdout") + " 2>" + Quote(artifact + ".stderr");
          int const status = std::system(command.c_str());
          if (status == 75 * 256) return 75;
          if (status) throw std::runtime_error("MX-4 candidate measurement failed: " + artifact + ".stderr");
          double const tpot = json::ParseFile(artifact + "/measurements.json").At("tpot_seconds").AsNumber("tpot") * 1e3;
          samples[i].push_back(tpot);
          rounds << round << '\t' << i + 1 << '\t' << tpot << '\t' << artifact << "/measurements.json\n";
        }
      for (int i = 0; i < 2; ++i) measured[i] = 0.5 * (samples[i][0] + samples[i][1]);
      // Within 1% plan #1 stays; otherwise the faster.
      if (measured[1] < measured[0] && (measured[0] - measured[1]) / measured[0] >= 0.01) {
        config = sel.second;
        chosen = 2;
      }
    }
    selection << std::setprecision(6) << "{\"mode\": \"" << mode << "\", \"table\": " << std::quoted(r.table)
              << ", \"plans\": [{\"pin\": " << PinJson(sel.first) << ", \"predicted_ns\": " << sel.first_ns
              << ", \"measured_tpot_ms\": " << measured[0] << "}, {\"pin\": " << PinJson(sel.second)
              << ", \"predicted_ns\": " << sel.second_ns << ", \"measured_tpot_ms\": " << measured[1]
              << ", \"swapped\": " << std::quoted(sel.swapped) << "}], \"chosen\": " << chosen
              << ", \"fit\": {\"r2\": " << sel.fit_r2 << ", \"max_rel\": " << sel.fit_max_rel
              << ", \"rows\": " << sel.fit_rows << "}}";
  }
  std::map<std::string, double> stage_ns;
  double const predicted_ns = predict(config, &stage_ns);
  std::ofstream(out + ".mx4_pin.json") << PinJson(config) << "\n";  // §4.4 item 3: configs/pins/mx4/
  Built const built = build(config, out);
  write_manifest(config, out, built, predicted_ns, stage_ns, selection.str());
  identity(false, out);
  std::cerr << "MX4_DECODE layers=" << model.layers << " batch=" << r.batch << " stack=" << built.stack
            << " mtreg=" << built.mtreg << " compile_s=" << built.compile_s << " predicted_ms=" << predicted_ns / 1e6
            << '\n';
  return 0;
}

}  // namespace tilemega::commands::mx_decode
