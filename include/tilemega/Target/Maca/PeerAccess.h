// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace tilemega::maca {

// C0: fine-grained LL8 passes 5/5 pair and all-to-all processes at both
// packet counts; docs/results/MX7_tp_probe_c550.csv:313. Never widen stores.
__device__ __forceinline__ void PeerStore64(std::uint64_t* p, std::uint64_t value) {
  using Global = __attribute__((address_space(1))) volatile std::uint64_t;
  *((Global*)p) = value;
}
__device__ __forceinline__ std::uint64_t PeerLoad64(std::uint64_t const* p) {
  return __ldcg(reinterpret_cast<unsigned long long const*>(p));
}

}  // namespace tilemega::maca
