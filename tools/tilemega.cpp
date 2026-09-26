// SPDX-License-Identifier: BSD-3-Clause
#include "commands/Commands.h"
#include "commands/Toolchain.h"
#include <tilemega/Support/BuildFingerprint.h>
#include <tilemega/Support/Json.h>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) try {
  struct Command { char const* name; int (*run)(int, char**); };
  Command const commands[] = {
    {"inspect request-floor", tilemega::commands::request_floor::RunRequestFloor},
    {"audit sass", tilemega::commands::audit_binary::RunSass},
    {"audit arch", tilemega::commands::audit_binary::RunArch},
    {"probe device", tilemega::commands::device::RunDevice},
    {"probe attention-work", tilemega::commands::attention_work::RunAttentionWork},
    {"calibrate", tilemega::commands::calibrate::RunCalibrate},
    {"compile", tilemega::commands::compile::RunCompile},
    {"inspect derive", tilemega::commands::derive::RunDerive},
    {"inspect dram-floor", tilemega::commands::dram_floor::RunDramFloor},
    {"probe event-cost", tilemega::commands::event_cost::RunEventCost},
    {"audit flow", tilemega::commands::flow_audit::RunFlowAudit},
    {"inspect import", tilemega::commands::import::RunImport},
    {"probe interface-probe", tilemega::commands::interface_probe::RunInterfaceProbe},
    {"occupancy", tilemega::commands::occupancy::RunOccupancy},
    {"audit op", tilemega::commands::op_audit::RunOpAudit},
    {"inspect parametric", tilemega::commands::parametric::RunParametric},
    {"inspect runtime-projection", tilemega::commands::runtime_projection::RunRuntimeProjection},
    {"probe scalar-error-probe", tilemega::commands::scalar_error_probe::RunScalarErrorProbe},
    {"audit skeleton", tilemega::commands::skeleton_audit::RunSkeletonAudit},
    {"audit target", tilemega::commands::target_audit::RunTargetAudit},
    {"probe task-work-probe", tilemega::commands::task_work_probe::RunTaskWorkProbe},
    {"calibrate wait-policy", tilemega::commands::wait_policy::RunWaitPolicy},
    {"inspect wiring", tilemega::commands::wiring::RunWiring},
  };
  if (argc < 2 || std::string(argv[1]) == "--help") {
    std::cout << "usage: tilemega <command> [options]\n";
    std::cout << "  version [--json]\n";
    for (auto const& c : commands) std::cout << "  " << c.name << "\n";
    return argc < 2 ? 2 : 0;
  }
  if (std::string(argv[1]) == "version") {
    tilemega::json::Value info(tilemega::json::Object{});
    info.Set("source_sha256", TILEMEGA_SOURCE_SHA256);
    info.Set("supported_arches", tilemega::json::Array{"sm_80", "sm_89", "sm_90", "sm_100", "sm_120"});
    info.Set("nvcc", tilemega::commands::NvccPath());
    info.Set("cuda_library_dir", tilemega::commands::CudaLibraryDirectory());
    std::cout << info.Dump() << "\n";
    return 0;
  }
  std::string route = argv[1];
  int first = 2;
  if (route == "inspect" || route == "audit" || route == "probe" ||
      (route == "calibrate" && argc > 2 && std::string(argv[2]) == "wait-policy")) {
    if (argc < 3) { std::cerr << "missing subcommand\n"; return 2; }
    route += " " + std::string(argv[2]); first = 3;
  }
  for (auto const& c : commands) if (route == c.name) {
    std::vector<std::string> args{ argv[0] };
    for (int i = first; i < argc; ++i) {
      if (route == "compile" && std::string(argv[i]) == "--options") {
        if (++i == argc) throw std::runtime_error("--options requires a JSON file");
        auto options = tilemega::json::ParseFile(argv[i]);
        // An argv array preserves order, repeated flags and exact scalar spelling.
        for (auto const& value : options.AsArray("compile options"))
          args.push_back(value.AsString("compile argument"));
      } else args.emplace_back(argv[i]);
    }
    std::vector<char*> raw;
    for (auto& arg : args) raw.push_back(arg.data());
    raw.push_back(nullptr);
    return c.run(int(args.size()), raw.data());
  }
  std::cerr << "unknown command: " << route << "\n";
  return 2;
} catch (std::exception const& e) {
  std::cerr << "tilemega: " << e.what() << "\n"; return 1;
}
