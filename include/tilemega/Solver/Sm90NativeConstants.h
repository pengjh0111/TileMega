// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace tilemega::solver {

struct Sm90NativeMissing : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

// `sm90_native_constants` of the H800 target (prompt §4 L0-1). Every lookup is
// an exact measured key; absence is a coverage error, never an extrapolation.
class Sm90NativeConstants {
 public:
  static Sm90NativeConstants FromTarget(json::Value const& target,std::string const& root);
  double Barrier(int grid) const;
  double Epilogue(std::string const& family,int tile_n,int batch,int warps_k) const;
  double Embedding(int grid,int width,int batch) const;
  // t_argmax_la: a stage of per-CTA partials, lm_ctr and the last arriver's
  // B x G reduction, including its V3 barrier (D9).
  double ArgmaxLa(int grid,int batch) const;
  // t_route_la: M4's last arriver (top-k, routes, schedule) plus V3 (§8).
  // Qwen3's form (128 experts, softmax over the selected) keys g<grid>;
  // any other form keys g<grid>:e<E>:all<0|1> as the probe reports it.
  double RouteLa(int grid,int batch,int experts=128,bool softmax_all=false) const;
  // D8: a 16-CTA group counter wait's saving over the V3 barrier (DT-4's
  // round-loop window means).
  double GroupWaitSaving(int grid) const;
  // DT-2: the largest cluster size C with max-active-clusters x C >= 128.
  int ClusterStar(int residency) const;
  // D10b: the leader's DSMEM merge of S chunk partials (beyond the CTA join).
  double AttentionMergeDsmem(int dim,int queries,int norm,int chunks) const;
  // B/TN above kappa = BW_l2 / BW_dram - 1 makes a stage activation-bound (D10c).
  double MulticastKappa() const {return l2_gbps/dram_gbps-1;}
  // N-1: the measured WGMMA shape efficiency and one SM's bf16 throughput
  // (FLOP/ns), which price a forward stage's compute term.
  // eta is measured per tile width: a 32-wide WGMMA sustains about half the
  // rate of a 256-wide one, so a stage is priced with its own N.
  double WgmmaEfficiency(int tile_n) const;
  double SmFlops() const;
  // D2 body: one 64-key block (data resident), q preparation, S-chunk merge.
  double AttentionBlock(int dim,int queries,int norm) const;
  double AttentionPrepare(int dim,int queries,int norm) const;
  double AttentionMerge(int dim,int queries,int norm,int chunks) const;
  // MR6: one CTA streaming cold data with 8 x 16 KB in flight bounds the
  // rate of each CTA in a stage's last, partly occupied wave.
  double CtaDramGbps() const;
  // MR6: the fixed cost of one expert task beyond its bytes and epilogue
  // (family "expert_gate_up" or "expert_down") at a batch.
  double TaskExpert(std::string const& family,int batch) const;
  // Constants whose recorded body header no longer matches the source tree.
  std::vector<std::string> StaleHeaders(std::string const& root) const;
  double l2_latency=0,dram_latency=0,dram_gbps=0,l2_gbps=0,sm_gbps=0,cp_async_gbps=0;
  json::Value section;
 private:
  std::map<int,double> barrier_;
  json::Value epilogue_,embedding_,argmax_la_,route_la_,attention_,round_,clusters_,dsmem_,forward_,task_expert_;
  double cta_dram_gbps_=0;
  json::Value const& AttentionShape(int dim,int queries,int norm) const;
};

}  // namespace tilemega::solver
