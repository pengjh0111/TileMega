// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §4 principle 2 and §5.3 backend trait query.
#pragma once
#include <cstddef>
namespace tilemega::solver {
struct BackendTraits { bool legal = false; std::size_t shared_bytes = 0; int mma_atoms = 0; };
template <class Collective>
constexpr BackendTraits QueryBackendTraits() {
  return {true, sizeof(typename Collective::SharedStorage), 0};
}
}  // namespace tilemega::solver
