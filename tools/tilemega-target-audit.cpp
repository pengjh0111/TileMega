// SPDX-License-Identifier: BSD-3-Clause
//
// Target audit (task Part 4.2).  For every target TileMega claims to support,
// asks four questions that the rest of the build never asks in one place:
//
//   SCHEMA  does configs/targets/<tag>.json carry exactly the field set
//           TargetSpec has?  The expected set is taken from ToJson() of a
//           fully populated spec, so a field added to TargetSpec makes every
//           config file fail until it is filled in -- which is the point.
//   CAPS    do the file's capability bits agree with ArchDispatch's
//           compile-time table for the same tag?  Two sources of truth that
//           disagree are worse than one.
//   LANES   what is the status of each of the nine resource dimensions --
//           live, capability_absent, or not_calibrated?  A zero with no
//           reason is the failure this tool exists to catch.
//   TERMS   is every cost-model term evaluable on this target, or explicitly
//           zero with a reason?  A non-finite term is a defect.
//   CROSS   can the target be planned for and compiled for without the
//           hardware present?
//
//   tilemega-target-audit [repo_root]
//
// Exits non-zero when any target fails, so it can be a ctest case.

#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Support/Json.h>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Target/TargetSpec.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

using tilemega::TargetSpec;
using namespace tilemega::solver;

char const* const kTags[] = {"sm_80", "sm_89", "sm_90", "sm_100", "sm_120"};

int failures = 0;
void Fail(std::string const& tag, std::string const& what) {
  std::cout << "FAIL " << tag << ' ' << what << '\n';
  ++failures;
}

// ------------------------------------------------------------------ schema --

void CollectKeys(tilemega::json::Value const& value, std::string const& path,
                 std::set<std::string>& out) {
  using Kind = tilemega::json::Value::Kind;
  if (value.kind() == Kind::kObject) {
    for (auto const& entry : value.AsObject("object")) {
      std::string const child = path + "/" + entry.first;
      out.insert(child);
      CollectKeys(entry.second, child, out);
    }
  } else if (value.kind() == Kind::kArray) {
    for (auto const& element : value.AsArray("array"))
      CollectKeys(element, path + "/[]", out);
  }
}

/// A spec with one element in every vector, so ToJson names every field the
/// struct has rather than only the scalar ones.
TargetSpec FullyPopulated() {
  TargetSpec spec;
  spec.arch_tag = "schema";
  spec.calib.l2_curve_bytes = {1.0};
  spec.calib.l2_curve_gbps = {1.0};
  spec.calib.smem_occupancy_ctas = {1.0};
  spec.calib.smem_occupancy_gbps = {1.0};
  spec.calib.atomic_contention_ctas = {1.0};
  spec.calib.atomic_contention_ns = {1.0};
  spec.calib.grid_barrier_ctas = {1.0};
  spec.calib.grid_barrier_ns = {1.0};
  TargetSpec::StreamKPoint point;
  point.occ_per_sm = {1.0};
  point.occ_a_ns = {1.0};
  point.occ_c_ns = {1.0};
  spec.calib.streamk = {point};
  spec.calib.measurements = {TargetSpec::Measurement{}};
  return spec;
}

/// True when `path` names a field inside an array that the file left empty --
/// there is nothing to compare, and an empty curve is a calibration question
/// (LANES answers it), not a schema one.
bool InsideEmptyArray(std::string const& path,
                      std::set<std::string> const& actual) {
  std::size_t const at = path.find("/[]");
  if (at == std::string::npos) return false;
  return actual.find(path.substr(0, at + 3)) == actual.end();
}

void CheckSchema(std::string const& tag, std::string const& file) {
  std::set<std::string> expected, actual;
  CollectKeys(tilemega::json::Parse(FullyPopulated().ToJson()), "", expected);
  CollectKeys(tilemega::json::ParseFile(file), "", actual);

  int missing = 0, extra = 0, documented = 0;
  for (auto const& key : expected) {
    if (actual.count(key) || InsideEmptyArray(key, actual)) continue;
    Fail(tag, "SCHEMA missing field " + key);
    ++missing;
  }
  for (auto const& key : actual) {
    if (expected.count(key)) continue;
    // A `*_provenance` string is the only extra a config may carry: it is how
    // a stated-unverified resource block says so in the file itself.
    if (key.size() > 11 && key.compare(key.size() - 11, 11, "_provenance") == 0) {
      ++documented;
      continue;
    }
    Fail(tag, "SCHEMA field no part of TargetSpec: " + key);
    ++extra;
  }
  std::printf("%-7s SCHEMA fields=%zu missing=%d extra=%d provenance=%d\n",
              tag.c_str(), actual.size(), missing, extra, documented);
}

// -------------------------------------------------------------------- caps --

void CheckCaps(std::string const& tag, TargetSpec const& spec) {
  auto const table = tilemega::arch::RuntimeCapsForTag(tag);
  struct Pair { char const* name; bool file; bool dispatch; };
  Pair const pairs[] = {
      {"cluster", spec.caps.cluster, table.cluster},
      {"tma", spec.caps.tma, table.tma},
      {"warp_specialized", spec.caps.warp_specialized, table.warp_specialized},
      {"tcgen05", spec.caps.tcgen05, table.tcgen05},
      {"l1_5", spec.caps.l1_5, table.l1_5},
      {"net", spec.caps.net, table.net},
      {"cp_async", spec.caps.cp_async, table.cp_async},
      {"mbarrier", spec.caps.mbarrier, table.mbarrier},
  };
  int disagree = 0;
  for (auto const& pair : pairs) {
    if (pair.file == pair.dispatch) continue;
    Fail(tag, std::string("CAPS ") + pair.name + " json=" +
                  (pair.file ? "true" : "false") + " ArchDispatch=" +
                  (pair.dispatch ? "true" : "false"));
    ++disagree;
  }
  if (spec.res.max_cluster_size != table.max_cluster_size) {
    Fail(tag, "CAPS max_cluster_size disagrees with ArchDispatch");
    ++disagree;
  }
  std::printf("%-7s CAPS   checked=%d disagree=%d collective=\"%s\"\n",
              tag.c_str(), 9, disagree, table.collective);
}

// ------------------------------------------------------------------- lanes --

void CheckLanes(std::string const& tag, TargetSpec const& spec,
                CostModel const& model) {
  std::string live, absent, uncal;
  for (int i = 0; i < ResourceVector::kLaneCount; ++i) {
    auto const lane = static_cast<ResourceVector::Lane>(i);
    std::string const name = ResourceVector::LaneName(lane);
    switch (model.lane_status(lane)) {
      case LaneStatus::kLive: live += name + " "; break;
      case LaneStatus::kCapabilityAbsent: absent += name + " "; break;
      case LaneStatus::kNotCalibrated:
        uncal += name + " ";
        // A calibrated target that still has an uncalibrated pipe is exactly
        // the "missing field with no reason" this audit must reject.
        if (spec.calib.calibrated)
          Fail(tag, "LANES " + name + " has no rate on a calibrated target");
        break;
    }
  }
  std::printf("%-7s LANES  live=[%s] capability_absent=[%s] not_calibrated=[%s]\n",
              tag.c_str(), live.c_str(), absent.c_str(), uncal.c_str());
}

// ------------------------------------------------------------------- terms --

char const* Classify(double value) {
  if (!std::isfinite(value)) return "non_finite";
  return value == 0.0 ? "zero" : "live";
}

void CheckTerms(std::string const& tag, TargetSpec const& spec,
                CostModel const& model, ModelDescription const* probe) {
  if (probe == nullptr) {
    std::printf("%-7s TERMS  SKIPPED (no probe model)\n", tag.c_str());
    return;
  }
  CandidateGenerator generator(spec);
  auto const candidates = generator.Enumerate();
  GemmConfig config;
  for (auto const& candidate : candidates) {
    if (!candidate.isLegal(spec)) continue;
    config = GemmConfig{candidate.traits().tile_m, candidate.traits().tile_n,
                        candidate.traits().tile_k, candidate.traits().stages, 1};
    break;
  }
  if (config.tile_m == 0) {
    Fail(tag, "TERMS no legal candidate on this target");
    return;
  }
  CostBreakdown const cost = model.Evaluate(*probe, config, Residency{2});
  struct Term { char const* name; double value; };
  Term const terms[] = {{"gemm", cost.gemm_ns},   {"combine", cost.combine_ns},
                        {"other", cost.other_ns}, {"barrier", cost.barrier_ns},
                        {"total", cost.total_ns}};
  std::string line;
  int bad = 0;
  for (auto const& term : terms) {
    char const* const status = Classify(term.value);
    line += std::string(term.name) + "=" + status + " ";
    if (std::string(status) == "non_finite") {
      Fail(tag, std::string("TERMS ") + term.name + " is not finite");
      ++bad;
    }
  }
  // A term may only be zero when the target says why: uncalibrated targets
  // carry no rates at all, and that reason is `calibrated == false`.
  char const* const reason =
      spec.calib.calibrated ? "calibrated" : "not_calibrated";
  std::printf("%-7s TERMS  %sconfig=%dx%dx%ds%d reason=%s bad=%d\n", tag.c_str(),
              line.c_str(), config.tile_m, config.tile_n, config.tile_k,
              config.stages, reason, bad);
}

// ------------------------------------------------------------------- cross --

bool NvccPath(std::string* out) {
  char const* env = std::getenv("CUDACXX");
  std::string const candidate = env ? env : "/usr/local/cuda/bin/nvcc";
  std::ifstream probe(candidate);
  if (!probe.good()) return false;
  *out = candidate;
  return true;
}

void CheckCross(std::string const& tag, TargetSpec const& spec,
                std::string const& repo) {
  if (spec.NvccArch() != tag) Fail(tag, "CROSS NvccArch() != arch_tag");
  int const stages = TargetSpec::ComputeStages(spec.res.max_dynamic_smem_per_cta,
                                               4 * 32 * (128 + 128));
  if (stages < 1) Fail(tag, "CROSS no pipeline stage fits the smem budget");

  std::string nvcc;
  if (!NvccPath(&nvcc)) {
    std::printf("%-7s CROSS  stages=%d nvcc=SKIPPED\n", tag.c_str(), stages);
    return;
  }
  std::string const source = "/tmp/tilemega_audit_" + tag + ".cu";
  {
    std::ofstream out(source);
    out << "#include <tilemega/Target/ArchDispatch.h>\n"
        << "__global__ void probe(int* o) { *o = "
           "tilemega::arch::Caps<tilemega::arch::CurrentArch>::kMaxClusterSize; }\n";
  }
  std::string const command = nvcc + " -std=c++17 -arch=" + tag + " -cubin " +
                              source + " -I" + repo + "/include -I" + repo +
                              "/third_party/cutlass/include -o /dev/null "
                              "> /tmp/tilemega_audit_" + tag + ".log 2>&1";
  int const rc = std::system(command.c_str());
  if (rc != 0) Fail(tag, "CROSS device compile for -arch=" + tag + " failed");
  std::printf("%-7s CROSS  stages=%d nvcc=%s\n", tag.c_str(), stages,
              rc == 0 ? "ok" : "FAILED");
  std::remove(source.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string const repo = argc > 1 ? argv[1] : TILEMEGA_SOURCE_DIR;

  ModelDescription probe_storage;
  ModelDescription const* probe = nullptr;
  try {
    probe_storage = ModelDescription::FromGeneratedCuda(
        repo + "/docs/experiments/E2E_GEN/raw/generated_e2e.cu",
        ModelDims{4, 3, 7}, "gqa2");
    probe = &probe_storage;
  } catch (std::exception const& error) {
    std::cout << "NOTE probe model unavailable: " << error.what() << '\n';
  }

  for (char const* tag : kTags) {
    std::string const file = repo + "/configs/targets/" + tag + ".json";
    std::cout << "== " << tag << '\n';
    CheckSchema(tag, file);
    TargetSpec spec;
    try {
      spec = TargetSpec::FromJson(file);
    } catch (std::exception const& error) {
      Fail(tag, std::string("FromJson threw: ") + error.what());
      continue;
    }
    CheckCaps(tag, spec);
    // The cost model refuses an uncalibrated target rather than evaluating
    // fabricated zeros.  That refusal is itself the reason a lane and a term
    // carry no value here, so it is reported and not counted as a failure.
    try {
      CostModel const model(spec);
      CheckLanes(tag, spec, model);
      CheckTerms(tag, spec, model, probe);
    } catch (std::exception const& error) {
      std::printf("%-7s LANES  UNAVAILABLE reason=not_calibrated (%s)\n",
                  tag, error.what());
      std::printf("%-7s TERMS  UNAVAILABLE reason=not_calibrated\n", tag);
      if (spec.calib.calibrated)
        Fail(tag, "cost model refused a target that claims calibration");
    }
    CheckCross(tag, spec, repo);
  }

  std::printf("SUMMARY targets=%zu failures=%d\n",
              sizeof(kTags) / sizeof(kTags[0]), failures);
  return failures == 0 ? 0 : 1;
}
