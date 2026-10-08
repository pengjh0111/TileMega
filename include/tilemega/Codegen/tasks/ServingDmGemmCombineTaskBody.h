// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingDmEpilogue.h>
#include <tilemega/Codegen/tasks/DmEpilogueDispatch.cuh>

namespace tilemega::codegen {
template <class Arch, int TileM, int TileN>
struct DmCombineRunner {
  float const* partials;
  int chunks, tile_m, tile_n;
  char* shared;
  backend::DmEpilogueArguments operands;
  int partial_rows = 0;

  template <class Spec>
  __device__ void Run() const {
    using Gate = backend::DmGateShape<typename Spec::Chain>;
    if constexpr (!Gate::template kFits<TileN>) {
      asm volatile("trap;");
    } else {
      using Epilogue = backend::ServingDmEpilogue<Arch, Spec, TileM, TileN>;
      float* tile = reinterpret_cast<float*>(shared);
      for (int i = executor::ComputeThread(); i < TileM * TileN;
           i += executor::kComputeThreads) {
        int row = tile_m * TileM + i / TileN;
        int column = tile_n * TileN + i % TileN;
        float sum = 0;
        if (row < operands.m && column < operands.n)
          for (int chunk = 0; chunk < chunks; ++chunk)
            sum += partials[(std::size_t(chunk) * (partial_rows ? partial_rows : operands.m) + row) * operands.n + column];
        tile[Epilogue::Index(i / TileN, i % TileN)] = sum;
      }
      executor::ComputeSync();
      // No chain operation or BF16 rounding is applied to a split partial.
      Epilogue::RunFromTile(tile, operands, tile_m, tile_n);
    }
  }
};
}  // namespace tilemega::codegen
