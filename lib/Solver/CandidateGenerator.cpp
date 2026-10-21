// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/CandidateGenerator.h>
#include <tilemega/Solver/AscendImplementation.h>
#include <tilemega/Support/Json.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <cmath>
#include <set>

namespace tilemega::solver {
namespace {

constexpr int kNativeSimtTileMN = 16;
constexpr int kNativeBF16TileM = 32;
constexpr int kNativeBF16TileN = 16;
constexpr int kNativeStages = 2;

int CeilDiv(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

}  // namespace

CandidateGenerator::CandidateGenerator(TargetSpec target, Envelope envelope)
    : target_(std::move(target)), envelope_(envelope) {}

CandidateGenerator::CandidateGenerator(TargetSpec target, ScalarType dtype,
                                       Envelope envelope)
    : target_(std::move(target)), envelope_(envelope), dtype_(dtype) {}

void CandidateGenerator::SetAscendDomain(std::vector<BackendCandidate> candidates,
    std::shared_ptr<AscendGemmPriceProvider const> prices) {
  if(target_.arch_tag!="ascend_3510" || dtype_!=ScalarType::kBF16 || !prices ||
      candidates.empty() || candidates.size()>8)
    throw std::invalid_argument("Native candidate ranking requires a finite BF16 domain and native prices");
  std::set<std::string> ids;
  for(auto const& candidate:candidates) {
    if(!candidate.isLegal(target_) || !candidate.traits().ascend_implementation ||
        !ids.insert(candidate.traits().ascend_implementation->impl_id).second)
      throw std::invalid_argument("Illegal or duplicate native implementation");
  }
  ascend_candidates_=std::move(candidates);ascend_prices_=std::move(prices);
}

std::vector<BackendCandidate> CandidateGenerator::Enumerate(Stats* stats) const {
  if(target_.arch_tag=="ascend_3510") {
    if(ascend_prices_) {
      if(stats)*stats={int(ascend_candidates_.size()),0,0,int(ascend_candidates_.size()),int(ascend_candidates_.size())};
      return ascend_candidates_;
    }
    if(!target_.ascend_config_json.empty() &&
        json::Parse(target_.ascend_config_json).At("selection").AsString("selection")!="fixed_legal_whitelist")
      throw std::invalid_argument("The native empirical target requires its explicit price provider");
    if(dtype_!=ScalarType::kBF16)throw std::invalid_argument("AS-1 fixed GEMM candidates require BF16");
    BackendCandidate candidate(As1Bf16GemmTraits());
    auto config=json::ParseFile(std::string(TILEMEGA_SOURCE_DIR)+"/configs/ascend/as1_impl_bindings.json");
    if(config.At("arch_tag").AsString("arch_tag")!=target_.arch_tag ||
       config.At("selection").AsString("selection")!="fixed_legal_whitelist" ||
       config.At("automatic_performance_ranking").AsBool("automatic_performance_ranking"))
      throw std::invalid_argument("Invalid AS-1 implementation selection policy");
    bool bound=false;
    for(auto const& entry:config.At("bindings").AsArray("bindings")) {
      if(!entry.At("enabled").AsBool("enabled") || entry.At("task_kind").AsNumber("task_kind")!=0)continue;
      auto const& tile=entry.At("tile_mnk").AsArray("tile_mnk");
      if(bound || entry.At("impl_id").AsString("impl_id")!=candidate.traits().ascend_implementation->impl_id ||
         entry.At("required_role").AsString("required_role")!="AIC_ONLY" || tile.size()!=3 ||
         tile[0].AsNumber("tile_m")!=32 || tile[1].AsNumber("tile_n")!=64 || tile[2].AsNumber("tile_k")!=64 ||
         entry.At("stages").AsNumber("stages")!=2 || entry.At("split_k").AsNumber("split_k")!=1 ||
         entry.At("event_triggers").AsNumber("event_triggers")!=1 ||
         entry.At("dtype").AsString("dtype")!="bf16" || entry.At("accumulator").AsString("accumulator")!="fp32")
        throw std::invalid_argument("GEMM binding disagrees with the verified native implementation");
      bound=true;
    }
    if(!bound)throw std::invalid_argument("Missing enabled native GEMM binding");
    if(!candidate.isLegal(target_))throw std::invalid_argument("The fixed native GEMM lacks verified target binding");
    if(stats)*stats={1,0,0,1,1};
    return {candidate};
  }
  std::vector<BackendCandidate> result;
  Stats local;
  int const native_k = dtype_ == ScalarType::kBF16 ? 16 : 8;
  int const native_m = dtype_ == ScalarType::kBF16 ? kNativeBF16TileM
                                                    : kNativeSimtTileMN;
  int const native_n = dtype_ == ScalarType::kBF16 ? kNativeBF16TileN
                                                    : kNativeSimtTileMN;
  int const budget = target_.res.max_dynamic_smem_per_cta;
  for (int m = native_m; m <= envelope_.max_tile_mn; m *= 2)
    for (int n = native_n; n <= envelope_.max_tile_mn; n *= 2)
      for (int k = native_k; k <= envelope_.max_tile_k; k *= 2)
        for (int s = kNativeStages; s <= envelope_.max_stages; ++s)
          ++local.cartesian;

  // Expansion order is canonical: an axis may only be doubled after the axes
  // before it are done growing, so every shape is reached exactly once and no
  // visited set is needed. Shared storage is monotone in all four axes, which
  // is what makes the wall a subtree cut rather than a per-shape filter.
  struct Node {
    int m, n, k, stages, axis;
  };
  std::vector<Node> work{{native_m, native_n, native_k,
                          kNativeStages, 0}};
  while (!work.empty()) {
    Node node = work.back();
    work.pop_back();
    ++local.touched;
    BackendTraits traits = dtype_ == ScalarType::kBF16
                               ? TensorBF16Traits(node.m, node.n, node.k,
                                                  node.stages)
                               : SimtF32Traits(node.m, node.n, node.k,
                                               node.stages);
    int const smem = dtype_ == ScalarType::kBF16
                         ? TensorBF16SmemBytes(node.m, node.n, node.k,
                                              node.stages)
                         : SimtF32SmemBytes(node.m, node.n, node.k,
                                            node.stages);
    if (smem > budget) {
      ++local.wall_pruned;
      continue;
    }
    BackendCandidate candidate(traits);
    if (!traits.shape_legal)
      ++local.shape_rejected;
    else if (candidate.isLegal(target_)) {
      ++local.legal;
      result.push_back(candidate);
    } else {
      // Legal shape, rejected by a target budget other than smem.
      ++local.shape_rejected;
    }
    // The shape rules are not monotone (16x16 is illegal at k=8 but 32x32 is
    // legal), so a rejected shape still has to be expanded; only the wall cuts.
    for (int axis = node.axis; axis < 4; ++axis) {
      Node next = node;
      next.axis = axis;
      switch (axis) {
        case 0: next.m *= 2; if (next.m > envelope_.max_tile_mn) continue; break;
        case 1: next.n *= 2; if (next.n > envelope_.max_tile_mn) continue; break;
        case 2: next.k *= 2; if (next.k > envelope_.max_tile_k) continue; break;
        default: ++next.stages; if (next.stages > envelope_.max_stages) continue;
      }
      work.push_back(next);
    }
  }
  std::sort(result.begin(), result.end(),
            [](BackendCandidate const& a, BackendCandidate const& b) {
              auto key = [](BackendCandidate const& c) {
                return std::tuple(c.traits().tile_m, c.traits().tile_n,
                                  c.traits().tile_k, c.traits().stages);
              };
              return key(a) < key(b);
            });
  if (stats) *stats = local;
  return result;
}

AnalyticalCost CandidateGenerator::Analyze(BackendCandidate const& candidate,
                                           GemmProblem const& problem) const {
  if(target_.arch_tag=="ascend_3510") {
    if(!ascend_prices_)throw std::invalid_argument("Native performance ranking requires an explicit price provider");
    if(!candidate.isLegal(target_) || problem.m<=0 || problem.n<=0 || problem.k<=0)
      throw std::invalid_argument("Illegal native candidate or GEMM shape");
    const auto id=candidate.traits().ascend_implementation->impl_id;
    if(std::none_of(ascend_candidates_.begin(),ascend_candidates_.end(),[&](auto const& c) {
        auto const& a=c.traits();auto const& b=candidate.traits();
        return a.ascend_implementation->impl_id==id && a.tile_m==b.tile_m &&
            a.tile_n==b.tile_n && a.tile_k==b.tile_k && a.stages==b.stages;}))
      throw std::invalid_argument("Candidate lies outside the selected native implementation domain");
    auto estimate=ascend_prices_->Estimate(candidate,problem);
    if(!std::isfinite(estimate.service_makespan_ns) || estimate.service_makespan_ns<=0 ||
        estimate.provenance.empty() || estimate.groups<1 ||
        estimate.groups>target_.ascend_capability->max_verified_mix_groups)
      throw std::invalid_argument("Missing/invalid native service-time estimate");
    AnalyticalCost result;result.ascend=estimate;
    result.ctas=CeilDiv(problem.m,candidate.traits().tile_m)*CeilDiv(problem.n,candidate.traits().tile_n);
    result.waves=CeilDiv(result.ctas,estimate.groups);return result;
  }
  AnalyticalCost cost;
  BackendTraits const& traits = candidate.traits();
  if (!candidate.isLegal(target_) || problem.m <= 0 || problem.n <= 0 ||
      problem.k <= 0)
    return cost;
  cost.ctas = CeilDiv(problem.m, traits.tile_m) * CeilDiv(problem.n, traits.tile_n);
  cost.co_resident = candidate.CoResidentPerSM(target_);
  int const slots = std::max(1, target_.res.num_sms * cost.co_resident);
  cost.waves = CeilDiv(cost.ctas, slots);
  cost.k_iters = CeilDiv(problem.k, traits.tile_k);
  // One FFMA per thread per cycle: the UniversalFMA atom issues on the CUDA
  // cores, so the tile's own work maps to cycles without a machine constant.
  cost.cta_cycles = static_cast<double>(traits.tile_m) * traits.tile_n *
                    traits.tile_k / traits.threads;
  cost.cycles = static_cast<double>(cost.waves) * cost.k_iters * cost.cta_cycles;
  return cost;
}

double CandidateGenerator::RankKey(BackendCandidate const& candidate,
                                   std::vector<GemmProblem> const& problems) const {
  double total = 0;
  for (auto const& problem : problems) {
    auto cost=Analyze(candidate,problem);
    total+=cost.ascend?cost.ascend->service_makespan_ns:cost.cycles;
  }
  return total;
}

std::vector<BackendCandidate> CandidateGenerator::TopK(
    std::vector<BackendCandidate> const& candidates,
    std::vector<GemmProblem> const& problems, int k) const {
  if(target_.arch_tag=="ascend_3510" && !ascend_prices_)
    throw std::invalid_argument("Native TopK requires explicit empirical prices");
  if(k<1 || problems.empty())throw std::invalid_argument("TopK requires a positive count and actual GEMM problems");
  std::vector<std::pair<double, BackendCandidate>> ranked;
  ranked.reserve(candidates.size());
  for (auto const& candidate : candidates)
    ranked.emplace_back(RankKey(candidate, problems), candidate);
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](auto const& a, auto const& b) {
                     if (a.first != b.first) return a.first < b.first;
                     return a.second.smemBytes() < b.second.smemBytes();
                   });
  std::vector<BackendCandidate> result;
  for (auto const& entry : ranked) {
    if (static_cast<int>(result.size()) >= k) break;
    result.push_back(entry.second);
  }
  return result;
}

std::vector<TileCandidate> CandidateGenerator::Generate() const {
  std::vector<TileCandidate> result;
  for (auto const& candidate : Enumerate())
    result.push_back({candidate.traits().tile_m, candidate.traits().tile_n,
                      candidate.traits().tile_k, candidate.traits().stages});
  return result;
}

}  // namespace tilemega::solver
