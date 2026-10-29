// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/MxMoeModel.h>
#include <tilemega/Codegen/MxDecodeProgramV2.h>
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
