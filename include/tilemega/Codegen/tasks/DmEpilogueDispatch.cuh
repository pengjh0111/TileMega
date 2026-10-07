// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>

namespace tilemega::codegen {
#ifndef TILEMEGA_DM_EPILOGUE_DISPATCH
// Handwritten descriptor probes have no generated plan or finite chain table.
template <class Runner>
__device__ inline void DispatchDmEpilogue(std::uint32_t, Runner const&) {
  asm volatile("trap;");
}
#endif
}  // namespace tilemega::codegen
