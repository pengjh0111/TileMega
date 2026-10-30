// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/MxMoeModel.h>
#include <tilemega/Codegen/MxDecodeProgramV2.h>
#include <tilemega/Codegen/MxMoeProgramV2.h>
#include <tilemega/Support/Json.h>
#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace tilemega::solver {
using MxMoeRanksV2 = std::array<std::size_t, 7>;  // qkv, o, router, gate/up, down, head, attention

struct MxMoeStageGeometryV2 {
  int n, k, partition_n, op, norm, n_active;
};
// Host cost-domain geometry. Grouped N stays per expert; partition_n counts
// active expert tiles. No measurement, prefetch or execution is fabricated.
MxMoeStageGeometryV2 MxMoeGeometryV2(codegen::MxMoeModel const& model,
    std::string const& stage, int batch);
std::vector<codegen::MxDecodeClassV2> MxMoeGemmDomainV2(codegen::MxMoeModel const& model,
    std::string const& stage, int batch, int grid = 104);
// Fixed MX-5 grid {65,256,512,768,1088}, averaged over decode L65..1087.
double MxMoeAttentionMeanV2(std::array<double, 5> const& grid_ns);

struct MxMoeStageCostV2 {
  double ns = 0;
  codegen::MxDecodeClassV2 gemm{};
  int b_min = 0, attn_prefetch = 0;
  int dispatch_gate_tn = 0, dispatch_down_tn = 0;
};
struct MxMoeProgramChoiceV2 {
  codegen::MxMoeConfigV2 config;
  MxMoeRanksV2 source_rows{};
  double predicted_ns = 0;
};
struct MxMoeImportedCostsV2 {
  std::array<std::vector<MxMoeStageCostV2>, 7> stages;
  std::array<std::vector<std::vector<std::size_t>>, 7> source_rows;
};
// Import one batch/thread request from externally verified aggregate rows.
// Checks shape/numeric/resource contracts and averages complete attention grids;
// it does not verify files, qualify a partial domain or grant execution.
MxMoeImportedCostsV2 ImportMxMoeCostRowsV2(codegen::MxMoeModel const& model,
    int threads, std::vector<json::Value> const& rows);
struct MxMoeCostCoverageV2 {
  std::size_t expected = 0, measured = 0, rejected = 0;
  // stage,TN,WN,D,Part,dispatch_GU_TN,dispatch_DN_TN,length;
  // attention uses TN for b_min and length for the fixed five-point grid.
  std::vector<std::array<int, 8>> missing;
  bool Complete() const { return expected && missing.empty() && measured + rejected == expected; }
};
// Check geometric coverage only. File/resource-branch qualification remains
// the catalog producer's responsibility; completeness grants no plan policy.
MxMoeCostCoverageV2 CheckMxMoeCostCoverageV2(codegen::MxMoeModel const& model,
    int threads, MxMoeImportedCostsV2 const& imported,
    std::vector<json::Value> const& resource_rejections);
// The caller supplies one batch/thread shape of source-qualified rows. Router
// dispatch TNs must match the selected grouped bodies, even for tied costs.
std::vector<MxMoeProgramChoiceV2> RankMxMoeProgramsV2(frontend::ModelPlan const& plan,
    int threads, std::array<std::vector<MxMoeStageCostV2>, 7> const& rows,
    std::size_t count = 4);
// Complete geometric coverage is required even for a diagnostic preview. File
// qualification and the per-plan thread policy belong to the caller.
json::Value MxMoeCandidatesJsonV2(frontend::ModelPlan const& plan, int threads,
    std::vector<json::Value> const& rows, std::vector<json::Value> const& rejections,
    std::size_t diagnostic_count = 4);

// Lists already contain source-qualified costs, scaled by stage occurrences.
// This merge does not qualify measurements or authorize a generated program.
std::vector<MxMoeRanksV2> MxMoeKBestV2(std::array<std::vector<double>, 7> const& costs,
    std::size_t count = 4, std::function<bool(MxMoeRanksV2 const&)> const& legal = {});

struct MxMoeMeasuredSelectionV2 {
  std::size_t selected = 0;
  std::array<double, 4> measured_ns{}, pred_over_meas{};
};
MxMoeMeasuredSelectionV2 SelectMxMoeMeasuredV2(std::array<double, 4> const& predicted_ns,
    std::array<std::array<double, 2>, 4> const& fresh_process_ns);
}  // namespace tilemega::solver
