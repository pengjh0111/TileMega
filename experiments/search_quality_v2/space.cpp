// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/Sm90DensePlan.h>
#include <tilemega/Solver/Sm90NativePipeline.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>

using namespace tilemega;
using namespace tilemega::solver;

namespace {
constexpr std::uint64_t kTrials = 100000;
constexpr std::uint64_t kSeed = 20261124;
constexpr std::uint64_t kModes = 2 * 2 * 2 * 2;  // cluster, release, embedding, argmax
constexpr double kZ95 = 1.959963984540054;

void Multiply(std::uint64_t& volume, std::size_t size) {
  if (size && volume > std::numeric_limits<std::uint64_t>::max() / size)
    throw std::runtime_error("domain product overflow");
  volume *= size;
}

struct Domain {
  std::array<std::vector<Sm90DenseGemm>, 5> gemms;
  std::vector<Sm90DenseAttention> attention;
  std::uint64_t volume = kModes;
  json::Array family_sizes;
  json::Array geometry_rows;
};

Domain MakeDomain(Sm90DenseSearchSpace const& space, Sm90DenseCoordinates seed,
                  Sm90NativeConstants const& constants, int residency) {
  Domain d;
  for (unsigned family : seed.Active()) {
    auto catalog = space.FamilyCandidates(family, residency);
    auto& modes = d.gemms.at(family);
    json::Array geometry_rows;
    std::set<std::tuple<int, int, int, bool>> keys;
    for (auto const& geometry : catalog) {
      for (bool multicast : {false, true}) {
        if (multicast && (geometry.slots % 2 ||
            double(space.batch) / geometry.tile_n <= constants.MulticastKappa()))
          continue;
        auto candidate = geometry;
        candidate.multicast = multicast;
        if (!keys.emplace(candidate.tile_n, candidate.TileK(), candidate.warps_n, multicast).second)
          throw std::runtime_error("GEMM catalog columns do not have unique derive keys");
        modes.push_back(candidate);
        seed.gemms[family] = candidate;
        geometry_rows.push_back(seed.ToJson().At("gemms").At(kSm90DenseFamilies[family]));
      }
    }
    Multiply(d.volume, modes.size());
    d.family_sizes.emplace_back(json::Object{
        {"name", kSm90DenseFamilies[family]}, {"family_candidates", double(catalog.size())},
        {"multicast_variants", double(modes.size() - catalog.size())},
        {"proposal_cardinality", double(modes.size())}});
    d.geometry_rows.emplace_back(json::Object{
        {"name", kSm90DenseFamilies[family]}, {"geometries", geometry_rows}});
  }
  d.attention = space.AttentionCandidates(residency);
  std::set<std::pair<int, int>> attention_keys;
  for (auto const& a : d.attention)
    if (!attention_keys.emplace(a.chunks, a.block_keys).second)
      throw std::runtime_error("attention catalog columns do not have unique derive keys");
  Multiply(d.volume, d.attention.size() * 3);
  return d;
}

Sm90DenseCoordinates Decode(Domain const& d, Sm90DenseCoordinates c,
                            int residency, std::uint64_t rank) {
  auto choose = [&rank](std::size_t size) {
    auto i = rank % size;
    rank /= size;
    return i;
  };
  c.residency = residency;
  c.cluster = 1 + choose(2);
  c.group_wait = choose(2);
  c.embed_la = choose(2);
  c.argmax_stage = choose(2);
  for (unsigned family : c.Active()) {
    auto const& domain = d.gemms.at(family);
    c.gemms[family] = domain.at(choose(domain.size()));
  }
  c.attention = d.attention.at(choose(d.attention.size()));
  auto merge = choose(3);
  c.attention.dsmem = merge == 1;
  c.attention.merge_stage = merge == 2;
  if (rank != 0) throw std::logic_error("mixed-radix rank exceeds domain");
  // Slots already come from the matching family and attention catalog entry
  // for this residency, so the native derive lookup would retain those slots.
  Sm90DenseSearchSpace::Structure(c);
  return c;
}

std::string Rejection(Sm90DenseSearchSpace const& space, Sm90DenseWorkload const& work,
                      Sm90NativeConstants const& constants, Sm90DenseCoordinates const& c) {
  if (!space.Legal(c)) return "Legal";
  if (c.cluster > 1 && c.cluster > constants.ClusterStar(c.residency))
    return "native_cluster_star";
  if (c.attention.chunks == 1 && c.attention.merge_stage) return "no_split_to_merge";
  if (c.attention.dsmem) {
    try {
      constants.AttentionMergeDsmem(work.head_dim, work.queries,
                                    work.qk_norm ? (work.qk_norm_row ? 2 : 1) : 0,
                                    c.attention.chunks);
    } catch (Sm90NativeMissing const&) {
      return "native_uncalibrated_dsmem_merge";
    }
  }
  return {};
}

json::Array ConfidenceInterval(std::uint64_t n, std::uint64_t hits, double volume) {
  double p = double(hits) / n;
  double z2 = kZ95 * kZ95;
  double denom = 1 + z2 / n;
  double center = (p + z2 / (2 * n)) / denom;
  double half = kZ95 * std::sqrt(p * (1 - p) / n + z2 / (4.0 * n * n)) / denom;
  return {std::max(0.0, center - half) * volume, std::min(1.0, center + half) * volume};
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument(
          "usage: tilemega-sq2-space target.json source-root export.json seed.json U-count output.json");
    if (std::ifstream(argv[6])) throw std::invalid_argument("choose a new output path");
    auto started = std::chrono::steady_clock::now();
    auto u_count = std::stoul(argv[5]);
    if (!u_count || u_count > 1000) throw std::invalid_argument("U-count must be 1..1000");
    auto seed = Sm90DenseCoordinates::FromJson(json::ParseFile(argv[4]));
    if (seed.moe || seed.tp != 1) throw std::invalid_argument("SQ2 requires dense TP1");
    analysis::IslContext isl;
    mlir::MLIRContext context;
    auto imported = ImportSm90Dense(argv[3], seed, context);
    auto work = Sm90DenseWorkloadFromPlan(imported.plan, 1);
    auto target = TargetSpec::FromJson(argv[1]);
    auto target_json = json::ParseFile(argv[1]);
    auto constants = Sm90NativeConstants::FromTarget(target_json, argv[2]);
    Sm90DenseSearchSpace space{work.shapes, work.batch, work.head_dim, work.queries, target, work.hkv};
    space.qk_norm_row = work.qk_norm_row;
    if (!space.Legal(seed)) throw std::invalid_argument("the supplied seed is not legal");

    std::array<Domain, 2> domains{
        MakeDomain(space, seed, constants, 1), MakeDomain(space, seed, constants, 2)};
    auto upper = domains[0].volume + domains[1].volume;
    if (!upper || upper < domains[0].volume || upper > (1ULL << 53))
      throw std::runtime_error("empty overall domain or too large for exact JSON integers");

    std::mt19937_64 engine(kSeed);
    std::uniform_int_distribution<std::uint64_t> sample_rank(0, upper - 1);
    std::set<std::string> saved;
    std::map<std::string, std::uint64_t> rejected;
    json::Array uniform_sample, random_seeds;
    std::uint64_t hits = 0, legal_only_hits = 0, attempts = 0;
    while (attempts < kTrials || saved.size() < u_count + 4) {
      if (attempts == 10 * kTrials) throw std::runtime_error("insufficient unique legal samples");
      auto rank = sample_rank(engine);
      int r = rank < domains[0].volume ? 0 : 1;
      if (r == 1) rank -= domains[0].volume;
      auto candidate = Decode(domains[r], seed, r + 1, rank);
      auto reason = Rejection(space, work, constants, candidate);
      if (++attempts <= kTrials) {
        if (reason.empty()) ++hits;
        else ++rejected[reason];
        if (reason != "Legal") ++legal_only_hits;
      }
      if (!reason.empty() || saved.size() == u_count + 4) continue;
      auto value = candidate.ToJson();
      if (!saved.insert(value.Dump(0)).second) continue;
      auto& split = uniform_sample.size() < u_count ? uniform_sample : random_seeds;
      split.push_back(std::move(value));
    }

    json::Array domain_rows;
    for (unsigned r = 0; r < domains.size(); ++r) {
      auto const& d = domains[r];
      json::Array attention;
      for (auto const& a : d.attention) {
        auto c = seed;
        c.residency = r + 1;
        c.attention = a;
        attention.push_back(c.ToJson().At("attention"));
      }
      domain_rows.emplace_back(json::Object{
          {"residency", int(r + 1)}, {"families", d.family_sizes}, {"gemms", d.geometry_rows},
          {"attention_candidates", double(d.attention.size())}, {"attention_geometries", attention},
          {"attention_merge_modes", 3}, {"stratum_volume", double(d.volume)}});
    }
    json::Object reason_counts;
    for (auto const& [reason, count] : rejected) reason_counts.emplace_back(reason, double(count));
    double p = double(hits) / kTrials;
    json::Value result = json::Object{
        {"scope", "native dense TP1 B1 coordinate catalogs and search admissibility rules"},
        {"domains", domain_rows}, {"program_dimensions", json::Object{
            {"residency", 2}, {"cluster", 2}, {"group_wait", 2}, {"embed_la", 2}, {"argmax_stage", 2}}},
        {"fixed", json::Object{{"weight_stream", seed.weight_stream}, {"layer_loop", seed.layer_loop},
                              {"tp", seed.tp}, {"moe", seed.moe}}},
        {"product_upper_bound", double(upper)}, {"proposal_trials", double(kTrials)},
        {"attempts_including_sampling", double(attempts)}, {"legal_trials", double(hits)},
        {"rejected_trials", reason_counts}, {"legal_fraction", p},
        {"Legal_predicate_trials", double(legal_only_hits)},
        {"Legal_predicate_count_estimate", double(legal_only_hits) / kTrials * upper},
        {"Legal_predicate_count_wilson_95", ConfidenceInterval(kTrials, legal_only_hits, upper)},
        {"legal_fraction_wilson_95", ConfidenceInterval(kTrials, hits, 1)},
        {"legal_count_estimate", p * upper}, {"legal_count_wilson_95", ConfidenceInterval(kTrials, hits, upper)},
        {"seed", double(kSeed)}, {"uniform_sample", uniform_sample}, {"random_search_seeds", random_seeds},
        {"sampling", "Uniform mixed-radix rank over disjoint residency strata; reject after native derive and legality. First distinct accepted draws form U then four random seeds. Count trials with replacement."},
        {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}};
    std::ofstream output(argv[6]);
    if (!output || !(output << result.Dump(2) << '\n'))
      throw std::runtime_error("failed to write space results");
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
