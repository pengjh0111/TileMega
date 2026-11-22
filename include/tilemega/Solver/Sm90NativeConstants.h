// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <array>
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

struct Sm80DenseServiceShape {
  std::string epilogue;
  int grid,n,k,batch,tn,tk,slots,warps_n,warps_k,register_stages,u,residual_tiles;
  bool input_norm,live_rows;
  int copy_issue_group;
  std::string Key() const;
};

struct Sm80DenseService {
  double producer_issue_ns=0,consumer_mma_ns=0;
  std::optional<double> epilogue_ns;
};

struct Sm80FusedExpertService {
  // The same producer owns Gu, Dn preissue, activation and the Dn mainloop.
  double gu_issue,gu_mma,gu_epilogue,dn_issue,dn_mma,dn_epilogue,dn_preissue,dn_activation;
  double route_startup_ns=-1,combine_tail_ns=-1;
  // Optional isolated stage measurements for the observed batch-two route sizes.
  std::array<double,17> production_body_ns{};
};

struct Sm80FusedExpertShape {
  int batch,gate_up_n,down_n,hidden,gate_up_tn,gate_up_tk,down_tn,down_tk,slots,
      gate_up_warps_n,down_warps_n,experts,float_routes,page,register_stages,
      softmax_all,segments,live_rows,copy_group,gu_route_bases,reuse_payload=0,residency=1;
  std::array<int,22> Key() const;
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
  // SM80 resident MMA and ldmatrix time per K16 step and split-K warp group.
  static std::string AmpereMmaKey(int grid,int tn,int tk,int wn,int stages);
  std::optional<double> AmpereMmaK16(int grid,int tn,int tk,int wn,int stages) const;
  std::optional<Sm80DenseService> AmpereDenseService(Sm80DenseServiceShape const&) const;
  std::optional<double> AmpereResidualEpilogue(Sm80DenseServiceShape const&) const;
  std::optional<double> AmpereProjectionIssueRate(int batch,int n,int k,int tn,int tk,
      int slots,int grid,bool input_norm) const;
  std::vector<double> const* AmpereExpertRouteWeights(int experts,int top_k,int batch,
      int hidden,int intermediate,int layers) const;
  std::optional<Sm80FusedExpertService> AmpereFusedService(Sm80FusedExpertShape const&) const;
  std::optional<double> AmpereFusedBody(Sm80FusedExpertShape const&) const;
  std::vector<Sm80FusedExpertShape> const& AmpereFusedShapes() const {return ampere_fused_shapes_;}
  std::optional<Sm80DenseService> AmpereGateService(
      int grid,int n,int k,int batch,int tn,int tk,int slots,int wn,int stages,int u) const;
  bool HasDenseGateService() const;
  std::vector<Sm80DenseServiceShape> const& AmpereDenseShapes() const {return ampere_dense_shapes_;}
  // D2 body: one 64-key block (data resident), q preparation, S-chunk merge.
  double AttentionBlock(int dim,int queries,int norm) const;
  std::optional<double> AttentionFresh(int dim,int queries,int norm) const;
  std::optional<double> CompactAttentionBarrier(int grid,int dim,int queries,int norm) const;
  double AttentionPrepare(int dim,int queries,int norm) const;
  bool HasAttentionMerge(int dim,int queries,int norm,int chunks) const;
  double AttentionMerge(int dim,int queries,int norm,int chunks) const;
  // MR6: one CTA streaming cold data with 8 x 16 KB in flight bounds the
  // rate of each CTA in a stage's last, partly occupied wave.
  double CtaDramGbps() const;
  // MR6: the fixed cost of one expert task beyond its bytes and epilogue
  // (family "expert_gate_up" or "expert_down") at a batch.
  double TaskExpert(std::string const& family,int batch) const;
  // Tensor parallel (SM90R5 B9.1, section `tp`): t_ar, what an o/down stage's
  // all-reduce epilogue adds over kResidual on T cards (key b=B:tn=TN:t=T),
  // and t_head_x, what the head's cross-rank argmax exchange adds (b=B:t=T).
  double AllReduce(int batch,int tile_n,int tp) const;
  double HeadExchange(int batch,int tp) const;
  // Constants whose recorded body header no longer matches the source tree.
  std::vector<std::string> StaleHeaders(std::string const& root) const;
  double l2_latency=0,dram_latency=0,dram_gbps=0,l2_gbps=0,sm_gbps=0,cp_async_gbps=0;
  json::Value section;
 private:
  std::map<int,double> barrier_;
  json::Value epilogue_,embedding_,argmax_la_,route_la_,attention_,round_,clusters_,dsmem_,forward_,task_expert_,tp_,ampere_mma_;
  json::Value ampere_gate_service_,ampere_dense_service_,ampere_residual_epilogue_;
  json::Value ampere_attention_fresh_;
  std::map<std::array<int,4>,double> ampere_compact_attention_;
  std::map<std::array<int,22>,Sm80FusedExpertService> ampere_fused_service_;
  std::map<std::array<int,22>,double> ampere_fused_body_;
  std::map<std::array<int,6>,std::vector<double>> ampere_routes_;
  std::vector<Sm80FusedExpertShape> ampere_fused_shapes_;
  std::vector<Sm80DenseServiceShape> ampere_dense_shapes_;
  int ampere_projection_batch_=0,ampere_projection_sms_=0;
  std::array<double,9> ampere_projection_coefficients_{};
  double cta_dram_gbps_=0;
  json::Value const& AttentionShape(int dim,int queries,int norm) const;
};

}  // namespace tilemega::solver
