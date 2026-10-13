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

template <int Variant, bool Combine = false, class Runner>
__device__ inline void DispatchDmVariantEpilogue(std::uint32_t gemm,
                                                Runner const& runner) {
#if defined(TILEMEGA_DM_TYPED_DISPATCH)
  DispatchDmEpilogue<Variant, Combine>(gemm, runner);
#else
  // Old generated sources and handwritten numerical probes have no bindings.
  DispatchDmEpilogue(gemm, runner);
#endif
}
}  // namespace tilemega::codegen
