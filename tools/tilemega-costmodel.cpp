// SPDX-License-Identifier: BSD-3-Clause
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

int SmemBytes(GemmConfig const& c) {
  return 4 * c.stages * c.tile_k * ((c.tile_m + 1) + (c.tile_n + 1));
}

/// F-40, exact on 1075 of the 1077 measured configurations of each model.
int CtasPerSm(TargetSpec const& target, int registers, int smem) {
  int const threads = 256;
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
                              TargetSpec const& target, int* missing) {
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
    point.smem = SmemBytes(point.config);
    auto it = registers.find(ShapeKey(point.config));
    if (it == registers.end()) {
      ++*missing;
      continue;
    }
    point.ctas_per_sm = CtasPerSm(target, it->second, point.smem);
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
                std::vector<GemmProblem> const& problems) {
  BackendTraits traits;
  traits.tile_m = c.tile_m;
  traits.tile_n = c.tile_n;
  traits.tile_k = c.tile_k;
  traits.stages = c.stages;
  traits.threads = 256;
  traits.smem_bytes = SmemBytes(c);
  traits.arch_sm = 80;
  traits.shape_legal = true;
  return generator.RankKey(BackendCandidate(traits), problems);
}

}  // namespace

int main(int argc, char** argv) try {
  std::string repo = ".";
  std::string out_dir = "docs/experiments/COST_MODEL/raw";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--repo" && i + 1 < argc) repo = argv[++i];
    else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    else { std::cerr << "usage: tilemega-costmodel [--repo DIR] [--out DIR]\n"; return 2; }
  }
  TargetSpec const target = TargetSpec::FromJson(repo + "/configs/targets/sm_89.json");

  struct ModelSource { char const* name; char const* cu; };
  ModelSource const sources[] = {
      {"gqa2", "/docs/experiments/E2E_GEN/raw/generated_e2e.cu"},
      {"mha4", "/docs/experiments/P3_GENERALIZATION/raw/generated.cu"},
  };

  CostModel const full(target);
  std::cout << "fit: lds=" << full.fit().lds_ns << " ns/instr (rel rms "
            << 100 * full.fit().lds_rel_rms << "%), setup=" << full.fit().setup_ns
            << " ns (rms " << full.fit().setup_rms_ns << " ns), over "
            << full.fit().points << " calibrated (shape, CTAs/SM) points\n";

  struct Layer { char const* name; CostModelOptions options; };
  CostModelOptions roofline;
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

  std::ofstream summary(out_dir + "/summary.tsv");
  if (!summary) throw std::runtime_error("cannot write " + out_dir + "/summary.tsv");
  summary << "model\tlayer\tn\tmape_pct\tspearman\ttop1\ttop3\ttop10\toptimum_rank\n";

  double worst_eval_us = 0.0;
  for (auto const& source : sources) {
    int missing = 0;
    auto const registers =
        ReadRegisters(out_dir + "/registers_" + source.name + ".tsv");
    auto const points =
        ReadScreen(repo + "/docs/experiments/ORACLE/raw/screen_" + source.name +
                       ".tsv",
                   registers, target, &missing);
    ModelDescription const model = ModelDescription::FromGeneratedCuda(
        repo + source.cu, ModelDims{4, 3, 7}, source.name);
    std::cout << source.name << ": " << points.size() << " measured points, "
              << model.gemms.size() << " GEMMs, " << model.stages.size()
              << " generated stages, footprint "
              << model.LiveFootprintBytes() / (1 << 20) << " MiB, L2 hit "
              << full.CacheHitProbability(model.LiveFootprintBytes())
              << (missing ? " [WARNING: shapes without ptxas registers]" : "")
              << '\n';

    std::vector<GemmProblem> problems;
    for (auto const& gemm : model.gemms)
      problems.push_back({model.dims.seq, gemm.n, gemm.k});
    CandidateGenerator const generator(target);

    for (auto const& layer : ladder) {
      CostModel const cost(target, layer.options);
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

    std::vector<double> tier2(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
      tier2[i] = Tier2Key(generator, points[i].config, problems);
    Score const baseline = Rank(points, tier2);
    // Tier 2 is an ordering in FFMA-issue cycles, not a time, so its MAPE is
    // meaningless and is reported as zero rather than as a number to compare.
    Score reported = baseline;
    reported.mape = 0.0;
    PrintScore(summary, source.name, "tier2-baseline", reported);
    PrintScore(std::cout, source.name, "tier2-baseline", reported);
  }
  std::cout << "worst per-configuration evaluation: " << worst_eval_us
            << " us\n";
  return 0;
} catch (std::exception const& error) {
  std::cerr << "tilemega-costmodel: " << error.what() << '\n';
  return 1;
}
