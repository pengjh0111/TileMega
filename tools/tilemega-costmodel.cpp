// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
//
// P4.4 validation driver: score the cost model against every measured
// configuration in docs/experiments/ORACLE/raw and report §2.4's metrics --
// MAPE, Spearman, the measured optimum's model rank, and how many of the
// model's top k land in the measured top 3% -- against the tier-2 baseline and
// across §2.2's ablation ladder.
//
// The acceptance criterion is the last of those (§2.5, §0.3): the top-34 band
// is within ±10% and two 25-process replicates disagree on the winner, so a
// model is useful when it lands inside the band, not when its error is small.

#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Target/TargetSpec.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifndef TILEMEGA_COSTMODEL_BOTTLENECK_DIAGNOSTICS
#define TILEMEGA_COSTMODEL_BOTTLENECK_DIAGNOSTICS 1
#endif

namespace {

using tilemega::TargetSpec;
using namespace tilemega::solver;

struct Point {
  GemmConfig config;
  double measured_ms = 0.0;
  int ctas_per_sm = 0;
  int smem = 0;
};

std::vector<std::string> Split(std::string const& line, char sep) {
  std::vector<std::string> out;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, sep)) out.push_back(field);
  return out;
}

int SmemBytes(GemmConfig const& c, ScalarType dtype) {
  return dtype == ScalarType::kBF16
             ? TensorBF16SmemBytes(c.tile_m, c.tile_n, c.tile_k, c.stages)
             : SimtF32SmemBytes(c.tile_m, c.tile_n, c.tile_k, c.stages);
}

/// Diagnostic only.  The validation sweep records CUDA's kernel-specific
/// occupancy result, which is authoritative; this closed form identifies
/// feature-construction mismatches without feeding them into the cost score.
int CtasPerSm(TargetSpec const& target, int registers, int smem,
              ScalarType dtype) {
  int const threads = dtype == ScalarType::kBF16 ? kTensorBF16Threads
                                                  : kSimtF32Threads;
  int const granularity = 8;
  int const per_cta_regs =
      granularity * ((registers * 32 + 255) / 256) * threads;
  int by_regs = per_cta_regs > 0 ? target.res.regs_per_sm / per_cta_regs : 1;
  int by_smem = smem > 0 ? target.res.max_dynamic_smem_per_cta / smem : 1;
  int by_threads = target.res.max_threads_per_sm / threads;
  return std::max(1, std::min({by_regs, by_smem, by_threads}));
}

std::string ShapeKey(GemmConfig const& c) {
  std::ostringstream out;
  out << c.tile_m << 'x' << c.tile_n << 'x' << c.tile_k << 's' << c.stages;
  return out.str();
}

/// shape -> the largest register count ptxas reported for any entry point of
/// that megakernel.  Occupancy is a whole-kernel property (§4.3), so the
/// maximum over the TaskBodies is the number that binds.
std::map<std::string, int> ReadRegisters(std::string const& path) {
  std::map<std::string, int> out;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open register table: " + path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto fields = Split(line, '\t');
    if (fields.size() < 2) continue;
    out[fields[0]] = std::stoi(fields[1]);
  }
  return out;
}

std::vector<Point> ReadScreen(std::string const& path,
                              std::map<std::string, int> const& registers,
                              TargetSpec const& target, ScalarType dtype,
                              int* missing, int* occupancy_mismatches) {
  std::vector<Point> out;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open oracle sweep: " + path);
  std::string line;
  std::getline(input, line);  // header
  while (std::getline(input, line)) {
    auto f = Split(line, '\t');
    if (f.size() < 13 || f[8] != "PASS") continue;
    Point point;
    point.config = {std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2]),
                    std::stoi(f[3]), std::stoi(f[4])};
    point.measured_ms = std::stod(f[6]);
    point.smem = SmemBytes(point.config, dtype);
    auto it = registers.find(ShapeKey(point.config));
    if (it == registers.end()) {
      ++*missing;
      continue;
    }
    point.ctas_per_sm = std::stoi(f[11]);
    if (point.ctas_per_sm !=
        CtasPerSm(target, it->second, point.smem, dtype))
      ++*occupancy_mismatches;
    out.push_back(point);
  }
  return out;
}

double Spearman(std::vector<double> const& a, std::vector<double> const& b) {
  std::size_t const n = a.size();
  std::vector<std::size_t> ia(n), ib(n);
  std::iota(ia.begin(), ia.end(), 0);
  std::iota(ib.begin(), ib.end(), 0);
  std::sort(ia.begin(), ia.end(), [&](std::size_t x, std::size_t y) { return a[x] < a[y]; });
  std::sort(ib.begin(), ib.end(), [&](std::size_t x, std::size_t y) { return b[x] < b[y]; });
  std::vector<double> ra(n), rb(n);
  for (std::size_t r = 0; r < n; ++r) { ra[ia[r]] = double(r); rb[ib[r]] = double(r); }
  double const mean = (n - 1) / 2.0;
  double cov = 0.0, va = 0.0, vb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    cov += (ra[i] - mean) * (rb[i] - mean);
    va += (ra[i] - mean) * (ra[i] - mean);
    vb += (rb[i] - mean) * (rb[i] - mean);
  }
  return cov / std::sqrt(va * vb);
}

struct Score {
  double mape = 0.0;
  double spearman = 0.0;
  int top1 = 0, top3 = 0, top10 = 0;
  int optimum_rank = 0;
  int n = 0;
};

Score Rank(std::vector<Point> const& points, std::vector<double> const& predicted) {
  std::size_t const n = points.size();
  std::vector<double> measured(n);
  for (std::size_t i = 0; i < n; ++i) measured[i] = points[i].measured_ms;
  Score s;
  s.n = int(n);
  double sum = 0.0;
  int counted = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (predicted[i] <= 0.0) continue;
    sum += std::abs(predicted[i] - measured[i]) / measured[i];
    ++counted;
  }
  s.mape = counted > 0 ? 100.0 * sum / counted : 0.0;
  s.spearman = Spearman(predicted, measured);
  std::vector<std::size_t> by_model(n), by_measure(n);
  std::iota(by_model.begin(), by_model.end(), 0);
  std::iota(by_measure.begin(), by_measure.end(), 0);
  std::sort(by_model.begin(), by_model.end(),
            [&](std::size_t x, std::size_t y) { return predicted[x] < predicted[y]; });
  std::sort(by_measure.begin(), by_measure.end(),
            [&](std::size_t x, std::size_t y) { return measured[x] < measured[y]; });
  std::size_t const band = std::max<std::size_t>(1, std::size_t(0.03 * n));
  std::vector<char> in_band(n, 0);
  for (std::size_t i = 0; i < band; ++i) in_band[by_measure[i]] = 1;
  for (std::size_t k = 0; k < std::min<std::size_t>(10, n); ++k) {
    if (!in_band[by_model[k]]) continue;
    if (k < 1) ++s.top1;
    if (k < 3) ++s.top3;
    ++s.top10;
  }
  for (std::size_t k = 0; k < n; ++k) {
    if (by_model[k] == by_measure[0]) { s.optimum_rank = int(k) + 1; break; }
  }
  return s;
}

void PrintScore(std::ostream& out, std::string const& model,
                std::string const& layer, Score const& s) {
  out << model << '\t' << layer << '\t' << s.n << '\t' << std::fixed
      << std::setprecision(2) << s.mape << '\t' << std::setprecision(4)
      << s.spearman << '\t' << s.top1 << '\t' << s.top3 << '\t' << s.top10
      << '\t' << s.optimum_rank << '\n';
}

/// The tier-2 ordering key (CandidateGenerator::RankKey), for the baseline row.
double Tier2Key(CandidateGenerator const& generator, GemmConfig const& c,
                std::vector<GemmProblem> const& problems, ScalarType dtype) {
  BackendTraits traits = dtype == ScalarType::kBF16
                             ? TensorBF16Traits(c.tile_m, c.tile_n, c.tile_k,
                                                c.stages)
                             : SimtF32Traits(c.tile_m, c.tile_n, c.tile_k,
                                             c.stages);
  return generator.RankKey(BackendCandidate(traits), problems);
}

#if TILEMEGA_COSTMODEL_BOTTLENECK_DIAGNOSTICS
// One vote per measured configuration, using precisely the steady-state
// vector used by GemmStageNs: its residency and model working-set miss rate.
// Every lane scales linearly with residency, so full and tail waves have the
// same winning lane in the present model. Keep all lane values for audit.
void ReportBottlenecks(std::string const& out_dir, std::string const& name,
                       ModelDescription const& model, CostModel const& cost,
                       std::vector<Point> const& points) {
  if (points.empty()) throw std::runtime_error("empty histogram input: " + name);
  std::ofstream detail(out_dir + "/bottlenecks_" + name + ".tsv");
  std::ofstream histogram(out_dir + "/histogram_" + name + ".tsv");
  if (!detail || !histogram)
    throw std::runtime_error("cannot write bottleneck reports in " + out_dir);
  detail << "tile_m\ttile_n\ttile_k\tstages\tsplit\toccupancy\tbottleneck";
  std::map<std::string, std::size_t> counts;
  for (int i = 0; i < ResourceVector::kLaneCount; ++i) {
    auto const lane = static_cast<ResourceVector::Lane>(i);
    counts[ResourceVector::LaneName(lane)] = 0;
    detail << '\t' << ResourceVector::LaneName(lane) << "_ns";
  }
  detail << '\n' << std::setprecision(17);
  double const miss = 1.0 - cost.CacheHitProbability(model.LiveFootprintBytes());
  for (auto const& point : points) {
    auto const& c = point.config;
    auto const u = cost.Steady(c, point.ctas_per_sm, miss);
    if (!(u.Bottleneck() > 0.0))
      throw std::runtime_error("all-zero resource vector: " + ShapeKey(c));
    ++counts[u.BottleneckName()];
    detail << c.tile_m << '\t' << c.tile_n << '\t' << c.tile_k << '\t'
           << c.stages << '\t' << c.split_k << '\t' << point.ctas_per_sm
           << '\t' << u.BottleneckName();
    for (int i = 0; i < ResourceVector::kLaneCount; ++i)
      detail << '\t' << u[static_cast<ResourceVector::Lane>(i)];
    detail << '\n';
  }
  char const* dtype = model.dtype == ScalarType::kBF16 ? "bf16" : "f32";
  histogram << "dtype\tmodel\tlane\tstatus\tcount\ttotal\tpercent\n";
  for (int i = 0; i < ResourceVector::kLaneCount; ++i) {
    auto const lane = static_cast<ResourceVector::Lane>(i);
    auto const count = counts.at(ResourceVector::LaneName(lane));
    histogram << dtype << '\t' << name << '\t' << ResourceVector::LaneName(lane)
              << '\t' << LaneStatusName(cost.lane_status(lane)) << '\t'
              << count << '\t' << points.size() << '\t' << std::setprecision(9)
              << 100.0 * count / points.size() << '\n';
    std::cout << "BOTTLENECK dtype=" << dtype << " model=" << name
              << " lane=" << ResourceVector::LaneName(lane)
              << " count=" << count << " total=" << points.size() << '\n';
  }
}
#endif

}  // namespace

int main(int argc, char** argv) try {
  tilemega::analysis::IslContext isl_context;
  std::string repo = ".";
  std::string out_dir = "docs/experiments/COST_MODEL/raw";
  std::string screen_dir;
  std::string register_dir;
  bool histogram_only = false;
  bool fp32_partials = true;
  bool task_body_traits = true;
  bool measured_partial_combine = TILEMEGA_MEASURED_PARTIAL_COMBINE;
  std::string target_file;
  std::string gqa_cu;
  std::string mha_cu;
  ScalarType dtype = ScalarType::kF32;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--repo" && i + 1 < argc) repo = argv[++i];
    else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    else if (arg == "--screen-dir" && i + 1 < argc) screen_dir = argv[++i];
    else if (arg == "--register-dir" && i + 1 < argc) register_dir = argv[++i];
    else if (arg == "--histogram-only") histogram_only = true;
    else if (arg == "--fp32-partials") fp32_partials = true;
    else if (arg == "--bf16-partials-baseline") fp32_partials = false;
    else if (arg == "--legacy-task-traits") task_body_traits = false;
    else if (arg == "--measured-partial-combine") measured_partial_combine = true;
    else if (arg == "--analytic-partial-combine") measured_partial_combine = false;
    else if (arg == "--target" && i + 1 < argc) target_file = argv[++i];
    else if (arg == "--gqa-cu" && i + 1 < argc) gqa_cu = argv[++i];
    else if (arg == "--mha-cu" && i + 1 < argc) mha_cu = argv[++i];
    else if (arg == "--dtype" && i + 1 < argc) {
      std::string value = argv[++i];
      if (value == "bf16") dtype = ScalarType::kBF16;
      else if (value == "f32") dtype = ScalarType::kF32;
      else { std::cerr << "--dtype must be f32 or bf16\n"; return 2; }
    } else { std::cerr << "usage: tilemega-costmodel [--repo DIR] [--out DIR]"
                         " [--dtype f32|bf16] [--screen-dir DIR]"
                         " [--register-dir DIR] [--histogram-only]"
                         " [--fp32-partials|--bf16-partials-baseline]"
                         " [--legacy-task-traits]"
                         " [--measured-partial-combine|--analytic-partial-combine] [--target FILE]"
                         " [--gqa-cu FILE] [--mha-cu FILE]\n"; return 2; }
  }
  if (screen_dir.empty()) screen_dir = repo + "/docs/experiments/ORACLE/raw";
  if (register_dir.empty()) register_dir = out_dir;
#if !TILEMEGA_COSTMODEL_BOTTLENECK_DIAGNOSTICS
  if (histogram_only)
    throw std::runtime_error("bottleneck diagnostics disabled at compile time");
#endif
  if (gqa_cu.empty()) gqa_cu = repo + "/docs/experiments/E2E_GEN/raw/generated_e2e.cu";
  if (mha_cu.empty()) mha_cu = repo + "/docs/experiments/P3_GENERALIZATION/raw/generated.cu";
  TargetSpec const target = TargetSpec::FromJson(target_file.empty()
      ? repo + "/configs/targets/sm_89.json" : target_file);

  struct ModelSource { char const* name; char const* cu; };
  ModelSource const sources[] = {
      {"gqa2", gqa_cu.c_str()},
      {"mha4", mha_cu.c_str()},
  };

  CostModelOptions full_options;
  full_options.fp32_partials = fp32_partials;
  full_options.task_body_traits = task_body_traits;
  full_options.measured_partial_combine = measured_partial_combine;
  CostModel const full(target, dtype, full_options);
  std::cout << "fit: lds=" << full.fit().lds_ns << " ns/instr (rel rms "
            << 100 * full.fit().lds_rel_rms << "%), setup=" << full.fit().setup_ns
            << " ns (rms " << full.fit().setup_rms_ns << " ns), over "
            << full.fit().points << " calibrated (shape, CTAs/SM) points\n";

  struct Layer { char const* name; CostModelOptions options; };
  CostModelOptions roofline;
  roofline.fp32_partials = fp32_partials;
  roofline.task_body_traits = task_body_traits;
  roofline.measured_partial_combine = measured_partial_combine;
  roofline.pipeline_envelope = false;
  roofline.wave_tail = false;
  roofline.cache_model = false;
  roofline.split_k = false;
  roofline.non_gemm = false;
  roofline.sync = false;
  CostModelOptions plus_split = roofline;
  plus_split.split_k = true;
  CostModelOptions plus_sync = plus_split;
  plus_sync.sync = true;
  CostModelOptions plus_waves = plus_sync;
  plus_waves.wave_tail = true;
  CostModelOptions plus_cache = plus_waves;
  plus_cache.cache_model = true;
  CostModelOptions plus_nongemm = plus_cache;
  plus_nongemm.non_gemm = true;
  // Two probes off the full model rather than further rungs: the envelope's
  // fill depth and the non-smem lanes are each measured against it.
  CostModelOptions with_envelope = plus_nongemm;
  with_envelope.pipeline_envelope = true;
  CostModelOptions smem_only = plus_nongemm;
  smem_only.resource_lanes = false;

  Layer const ladder[] = {
      {"roofline", roofline},        {"+splitk", plus_split},
      {"+sync", plus_sync},          {"+waves", plus_waves},
      {"+cache", plus_cache},        {"+nongemm(full)", plus_nongemm},
      {"full+envelope", with_envelope}, {"full-lanes(smem only)", smem_only},
  };

  std::ofstream summary;
  if (!histogram_only) {
    summary.open(out_dir + "/summary.tsv");
    if (!summary) throw std::runtime_error("cannot write " + out_dir + "/summary.tsv");
    summary << "model\tlayer\tn\tmape_pct\tspearman\ttop1\ttop3\ttop10\toptimum_rank\n";
  }

  double worst_eval_us = 0.0;
  for (auto const& source : sources) {
    int missing = 0;
    int occupancy_mismatches = 0;
    auto const registers =
        ReadRegisters(register_dir + "/registers_" + source.name + ".tsv");
    auto const points =
        ReadScreen(screen_dir + "/screen_" + source.name + ".tsv",
                   registers, target, dtype, &missing, &occupancy_mismatches);
    ModelDescription const model = ModelDescription::FromGeneratedCuda(
        source.cu, ModelDims{4, 3, 7}, source.name);
    if (model.dtype != dtype)
      throw std::runtime_error("generated model dtype does not match --dtype");
    std::cout << source.name << ": " << points.size() << " measured points, "
              << model.gemms.size() << " GEMMs, " << model.stages.size()
              << " generated stages, footprint "
              << model.LiveFootprintBytes() / (1 << 20) << " MiB, L2 hit "
              << full.CacheHitProbability(model.LiveFootprintBytes())
              << (missing ? " [WARNING: shapes without ptxas registers]" : "")
              << " occupancy_feature_mismatches=" << occupancy_mismatches
              << '\n';
#if TILEMEGA_COSTMODEL_BOTTLENECK_DIAGNOSTICS
    if (missing) throw std::runtime_error("incomplete histogram: missing registers");
    ReportBottlenecks(out_dir, source.name, model, full, points);
#endif
    if (histogram_only) continue;

    std::vector<GemmProblem> problems;
    for (auto const& gemm : model.gemms)
      problems.push_back({model.dims.seq, gemm.n, gemm.k});
    CandidateGenerator const generator(target, dtype);

    for (auto const& layer : ladder) {
      CostModel const cost(target, dtype, layer.options);
      std::vector<double> predicted(points.size());
      auto const start = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < points.size(); ++i) {
        predicted[i] = cost.Evaluate(model, points[i].config,
                                     Residency{points[i].ctas_per_sm})
                           .total_ns / 1e6;
      }
      auto const elapsed = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - start)
                               .count() / double(points.size());
      worst_eval_us = std::max(worst_eval_us, elapsed);
      Score const score = Rank(points, predicted);
      PrintScore(summary, source.name, layer.name, score);
      PrintScore(std::cout, source.name, layer.name, score);

      if (std::string(layer.name) == "+nongemm(full)") {
        std::ofstream detail(out_dir + "/predictions_" + source.name + ".tsv");
        detail << "tile_m\ttile_n\ttile_k\tstages\tsplit_k\tctas_per_sm"
                  "\tmeasured_ms\tmodel_ms\tgemm_ms\tcombine_ms\tother_ms"
                  "\tbarrier_ms\tstages_after_split\n";
        for (std::size_t i = 0; i < points.size(); ++i) {
          CostBreakdown const b = cost.Evaluate(
              model, points[i].config, Residency{points[i].ctas_per_sm});
          auto const& c = points[i].config;
          detail << c.tile_m << '\t' << c.tile_n << '\t' << c.tile_k << '\t'
                 << c.stages << '\t' << c.split_k << '\t'
                 << points[i].ctas_per_sm << '\t' << points[i].measured_ms
                 << '\t' << b.total_ns / 1e6 << '\t' << b.gemm_ns / 1e6 << '\t'
                 << b.combine_ns / 1e6 << '\t' << b.other_ns / 1e6 << '\t'
                 << b.barrier_ns / 1e6 << '\t' << b.stage_count << '\n';
        }
      }
    }

    // One-variable lane ablations from the full model.  Capability-absent
    // lanes are retained deliberately: their identical score is evidence for
    // which dimensions remain structurally zero on this target.
    for (int lane = 0; lane < ResourceVector::kLaneCount; ++lane) {
      CostModelOptions ablated = plus_nongemm;
      ablated.disabled_lanes[lane] = true;
      CostModel const cost(target, dtype, ablated);
      std::vector<double> predicted(points.size());
      for (std::size_t i = 0; i < points.size(); ++i)
        predicted[i] = cost.Evaluate(model, points[i].config,
                                     Residency{points[i].ctas_per_sm})
                           .total_ns / 1e6;
      std::string const name =
          std::string("full-minus-") + ResourceVector::LaneName(
                                            static_cast<ResourceVector::Lane>(lane));
      Score const score = Rank(points, predicted);
      PrintScore(summary, source.name, name, score);
      PrintScore(std::cout, source.name, name, score);
    }

    std::vector<double> tier2(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        tier2[i] = Tier2Key(generator, points[i].config, problems, dtype);
    Score const baseline = Rank(points, tier2);
    // Tier 2 is an ordering in FFMA-issue cycles, not a time, so its MAPE is
    // meaningless and is reported as zero rather than as a number to compare.
    Score reported = baseline;
    reported.mape = 0.0;
    PrintScore(summary, source.name, "tier2-baseline", reported);
    PrintScore(std::cout, source.name, "tier2-baseline", reported);
  }
  if (!histogram_only)
    std::cout << "worst per-configuration evaluation: " << worst_eval_us
              << " us\n";
  return 0;
} catch (std::exception const& error) {
  std::cerr << "tilemega-costmodel: " << error.what() << '\n';
  return 1;
}
