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
  // D2 body: one 64-key block (data resident), q preparation, S-chunk merge.
  double AttentionBlock(int dim,int queries,bool norm) const;
  double AttentionPrepare(int dim,int queries,bool norm) const;
  double AttentionMerge(int dim,int queries,bool norm,int chunks) const;
  // Constants whose recorded body header no longer matches the source tree.
  std::vector<std::string> StaleHeaders(std::string const& root) const;
  double l2_latency=0,dram_latency=0,dram_gbps=0,l2_gbps=0,sm_gbps=0,cp_async_gbps=0;
  json::Value section;
 private:
  std::map<int,double> barrier_;
  json::Value epilogue_,embedding_,argmax_la_,attention_;
  json::Value const& AttentionShape(int dim,int queries,bool norm) const;
};

}  // namespace tilemega::solver
