// SPDX-License-Identifier: BSD-3-Clause
//
// Part 4.4 migration check: does a cost model calibrated on one GPU still rank
// configurations on another?  The 4090 and the 5090 have comparable
// shared memory per SM (~100 KB) but 128 vs 170 SMs, so the wave quantization
// term is the one that moves, and rank -- not absolute error -- is what the
// solver consumes.
//
// Scores three variants over the same subset of ORACLE configurations:
//   sm_89/sm_89     the model as shipped, against the 4090 measurements
//   sm_89/sm_89     the same model unchanged, against the 5090 measurements
//   sm_89/sm_120    calibration coefficients from the 4090, resources from the
//                   5090 -- the cross-compilation path TargetSpec supports
// and reports the Spearman and top-3%-band loss between them.

#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Target/TargetSpec.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
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

int CtasPerSm(TargetSpec const& target, int registers, int smem) {
  int const threads = 256;
  int const per_cta_regs = 8 * ((registers * 32 + 255) / 256) * threads;
  int by_regs = per_cta_regs > 0 ? target.res.regs_per_sm / per_cta_regs : 1;
  int by_smem = smem > 0 ? target.res.max_dynamic_smem_per_cta / smem : 1;
  return std::max(1, std::min({by_regs, by_smem,
                               target.res.max_threads_per_sm / threads}));
}

std::string ShapeKey(GemmConfig const& c) {
  std::ostringstream out;
  out << c.tile_m << 'x' << c.tile_n << 'x' << c.tile_k << 's' << c.stages;
  return out.str();
}

std::string ConfigKey(GemmConfig const& c) {
  return ShapeKey(c) + "k" + std::to_string(c.split_k);
}

std::map<std::string, int> ReadRegisters(std::string const& path) {
  std::map<std::string, int> out;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open register table: " + path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto fields = Split(line, '\t');
    if (fields.size() >= 2) out[fields[0]] = std::stoi(fields[1]);
  }
  return out;
}

/// Measured points keyed by configuration, so two machines' sweeps can be
/// intersected rather than assumed to be in the same order.
std::map<std::string, Point> ReadScreen(std::string const& path,
                                        std::map<std::string, int> const& regs,
                                        TargetSpec const& host, int* missing) {
  std::map<std::string, Point> out;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open sweep: " + path);
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    auto f = Split(line, '\t');
    if (f.size() < 13 || f[8] != "PASS") continue;
    Point point;
    point.config = {std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2]),
                    std::stoi(f[3]), std::stoi(f[4])};
    point.measured_ms = std::stod(f[6]);
    auto it = regs.find(ShapeKey(point.config));
    if (it == regs.end()) { ++*missing; continue; }
    point.ctas_per_sm = CtasPerSm(host, it->second, SmemBytes(point.config));
    out[ConfigKey(point.config)] = point;
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
  int n = 0;
  double mape = 0.0, spearman = 0.0;
  int top1 = 0, top3 = 0, top10 = 0, optimum_rank = 0, band = 0;
};

Score Rank(std::vector<double> const& measured,
           std::vector<double> const& predicted) {
  std::size_t const n = measured.size();
  Score s;
  s.n = int(n);
  double sum = 0.0;
  int counted = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (predicted[i] <= 0.0) continue;
    sum += std::abs(predicted[i] - measured[i]) / measured[i];
    ++counted;
  }
  s.mape = counted ? 100.0 * sum / counted : 0.0;
  s.spearman = Spearman(predicted, measured);
  std::vector<std::size_t> by_model(n), by_measure(n);
  std::iota(by_model.begin(), by_model.end(), 0);
  std::iota(by_measure.begin(), by_measure.end(), 0);
  std::sort(by_model.begin(), by_model.end(),
            [&](std::size_t x, std::size_t y) { return predicted[x] < predicted[y]; });
  std::sort(by_measure.begin(), by_measure.end(),
            [&](std::size_t x, std::size_t y) { return measured[x] < measured[y]; });
  std::size_t const band = std::max<std::size_t>(1, std::size_t(0.03 * n));
  s.band = int(band);
  std::vector<char> in_band(n, 0);
  for (std::size_t i = 0; i < band; ++i) in_band[by_measure[i]] = 1;
  for (std::size_t k = 0; k < std::min<std::size_t>(10, n); ++k) {
    if (!in_band[by_model[k]]) continue;
    if (k < 1) ++s.top1;
    if (k < 3) ++s.top3;
    ++s.top10;
  }
  for (std::size_t k = 0; k < n; ++k)
    if (by_model[k] == by_measure[0]) { s.optimum_rank = int(k) + 1; break; }
  return s;
}

void Print(std::ostream& out, std::string const& model, std::string const& arm,
           Score const& s) {
  out << model << '\t' << arm << '\t' << s.n << '\t' << std::fixed
      << std::setprecision(2) << s.mape << '\t' << std::setprecision(4)
      << s.spearman << '\t' << s.top1 << '\t' << s.top3 << '\t' << s.top10
      << '\t' << s.optimum_rank << '\n';
}

/// Top `top` by measured latency plus `sample` drawn from the rest with a
/// fixed seed, so the subset is a property of the committed sweep and not of
/// the run that selected it.
std::vector<std::string> SelectSubset(std::string const& screen, int top,
                                      int sample, unsigned seed) {
  std::vector<std::pair<double, std::string>> rows;
  std::ifstream input(screen);
  if (!input) throw std::runtime_error("cannot open sweep: " + screen);
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    auto f = Split(line, '\t');
    if (f.size() < 13 || f[8] != "PASS") continue;
    GemmConfig c{std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2]),
                 std::stoi(f[3]), std::stoi(f[4])};
    rows.emplace_back(std::stod(f[6]), ConfigKey(c));
  }
  std::sort(rows.begin(), rows.end());
  std::vector<std::string> out;
  int const head = std::min<int>(top, int(rows.size()));
  for (int i = 0; i < head; ++i) out.push_back(rows[i].second);
  std::vector<std::string> rest;
  for (std::size_t i = head; i < rows.size(); ++i) rest.push_back(rows[i].second);
  std::mt19937 rng(seed);
  std::shuffle(rest.begin(), rest.end(), rng);
  for (int i = 0; i < std::min<int>(sample, int(rest.size())); ++i)
    out.push_back(rest[i]);
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

int main(int argc, char** argv) try {
  std::string repo = ".", out_dir, label = "sm120";
  std::string calib_json, host_json;
  int top = 50, sample = 50;
  unsigned seed = 20260904u;
  bool probe_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--repo") repo = next();
    else if (arg == "--out") out_dir = next();
    else if (arg == "--label") label = next();
    else if (arg == "--calib") calib_json = next();
    else if (arg == "--host") host_json = next();
    else if (arg == "--top") top = std::stoi(next());
    else if (arg == "--sample") sample = std::stoi(next());
    else if (arg == "--seed") seed = unsigned(std::stoul(next()));
    else if (arg == "--probe") probe_only = true;
    else {
      std::cerr << "usage: tilemega-migrate [--repo DIR] [--out DIR] "
                   "[--label TAG] [--calib JSON] [--host JSON] "
                   "[--top N] [--sample N] [--seed N] [--probe]\n";
      return 2;
    }
  }
  if (out_dir.empty()) out_dir = repo + "/docs/experiments/MIGRATION/raw";
  if (calib_json.empty()) calib_json = repo + "/configs/targets/sm_89.json";
  if (host_json.empty()) host_json = repo + "/configs/targets/sm_120.json";

  TargetSpec const calib = TargetSpec::FromJson(calib_json);
  TargetSpec const host = TargetSpec::FromJson(host_json);

  // The whole point of the check is that the machine under it is *not* the one
  // the coefficients came from, so refuse rather than silently re-measure the
  // calibration GPU and call the result a migration.
  if (probe_only) {
    TargetSpec const here = TargetSpec::Probe();
    std::cout << "PROBE " << here.Summary() << '\n';
    if (here.arch_tag == calib.arch_tag && here.res.num_sms == calib.res.num_sms) {
      std::cerr << "tilemega-migrate: this is the calibration target ("
                << calib.arch_tag << ", " << calib.res.num_sms
                << " SMs); a migration check needs a different GPU\n";
      return 3;
    }
    if (here.arch_tag != host.arch_tag)
      std::cerr << "tilemega-migrate: warning: probed " << here.arch_tag
                << " but --host declares " << host.arch_tag << '\n';
    return 0;
  }
  // The transfer target: everything the new GPU declares about itself, with
  // the coefficients nobody has re-measured there.  This is what a cross
  // compilation actually has available, and the point of the check is whether
  // it is enough to rank with.
  TargetSpec transfer = host;
  transfer.calib = calib.calib;

  CostModel const model_native(calib);
  CostModel const model_transfer(transfer);

  struct Source { char const* name; char const* cu; };
  Source const sources[] = {
      {"gqa2", "/docs/experiments/E2E_GEN/raw/generated_e2e.cu"},
      {"mha4", "/docs/experiments/P3_GENERALIZATION/raw/generated.cu"},
  };

  std::ofstream summary(out_dir + "/summary.tsv");
  if (!summary) throw std::runtime_error("cannot write " + out_dir + "/summary.tsv");
  summary << "model\tarm\tn\tmape_pct\tspearman\ttop1\ttop3\ttop10\toptimum_rank\n";
  std::cout << "model\tarm\tn\tmape_pct\tspearman\ttop1\ttop3\ttop10\toptimum_rank\n";

  for (auto const& source : sources) {
    std::string const base = repo + "/docs/experiments/ORACLE/raw/screen_" +
                             source.name + ".tsv";
    auto subset = SelectSubset(base, top, sample, seed);
    {
      std::ofstream list(out_dir + "/subset_" + source.name + ".txt");
      for (auto const& key : subset) {
        int m, n, k, s, kc;
        std::sscanf(key.c_str(), "%dx%dx%ds%dk%d", &m, &n, &k, &s, &kc);
        list << m << ' ' << n << ' ' << k << ' ' << s << ' ' << kc << '\n';
      }
    }

    int missing = 0;
    auto const native_regs =
        ReadRegisters(repo + "/docs/experiments/COST_MODEL/raw/registers_" +
                      source.name + ".tsv");
    auto const native = ReadScreen(base, native_regs, calib, &missing);
    ModelDescription const desc = ModelDescription::FromGeneratedCuda(
        repo + source.cu, ModelDims{4, 3, 7}, source.name);

    auto score_arm = [&](std::map<std::string, Point> const& measured,
                         CostModel const& cost, char const* arm) {
      std::vector<double> ms, pred;
      for (auto const& key : subset) {
        auto it = measured.find(key);
        if (it == measured.end()) continue;
        ms.push_back(it->second.measured_ms);
        pred.push_back(cost.Evaluate(desc, it->second.config,
                                     Residency{it->second.ctas_per_sm})
                           .total_ns / 1e6);
      }
      if (ms.empty()) return;
      Score const s = Rank(ms, pred);
      Print(summary, source.name, arm, s);
      Print(std::cout, source.name, arm, s);
    };

    score_arm(native, model_native, "sm_89 measured / sm_89 model");

    std::string const migrated_screen =
        out_dir + "/screen_" + label + "_" + source.name + ".tsv";
    std::ifstream probe(migrated_screen);
    if (!probe) {
      std::cout << source.name << "\tBLOCKED\tno " << migrated_screen
                << " -- run run_on_sm120.sh on the migration target\n";
      continue;
    }
    probe.close();
    int missing_migrated = 0;
    auto const migrated_regs =
        ReadRegisters(out_dir + "/registers_" + label + "_" + source.name + ".tsv");
    auto const migrated =
        ReadScreen(migrated_screen, migrated_regs, host, &missing_migrated);
    score_arm(migrated, model_native, "migrated measured / sm_89 model");
    score_arm(migrated, model_transfer, "migrated measured / transfer model");
  }
  return 0;
} catch (std::exception const& error) {
  std::cerr << "tilemega-migrate: " << error.what() << '\n';
  return 1;
}
