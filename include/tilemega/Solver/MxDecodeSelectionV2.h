// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/MxDecodeSolverV2.h>
namespace tilemega::solver {
/// Candidate pins remain diagnostic manual inputs until all production gates pass.
json::Value MxDecodeCandidatesJsonV2(MxDecodeSelectionV2 const& selection);
json::Value MxDecodeMeasuredJsonV2(MxDecodeSelectionV2 const& selection,
                                  std::vector<std::array<double, 2>> const& measurements);
}  // namespace tilemega::solver
