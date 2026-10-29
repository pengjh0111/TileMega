// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace tilemega::solver {
using MxMoeRanksV2 = std::array<std::size_t, 7>;  // qkv, o, router, gate/up, down, head, attention

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
