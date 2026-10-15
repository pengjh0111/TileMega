// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Backend/Mx/ServingGemmConfig.h
//
// The xcore1000 serving GEMM configuration: TileMega-owned POD operand types
// in place of the CUTLASS collectives, so GemmInvocation carries pointers and
// row strides only. One config type serves the MMA and the GEMV bodies.
#pragma once

#include <tilemega/Solver/BackendCostQuery.h>
#include <tilemega/Target/ArchDispatch.h>
#include <tilemega/Target/Maca/Bf16.h>

#include <cstdint>
#include <type_traits>

namespace tilemega::backend {

/// Element stride between consecutive rows; the K dimension is contiguous.
struct RowStride {
  std::int64_t row = 0;
};

template <int M, int N, int K>
struct TileShape3 {
  static constexpr int kM = M, kN = N, kK = K;
};

/// The epilogue operands do not depend on the tile: one type for every
/// variant, as the harness's GemmInvocation ABI requires (GemmStageTaskBody's
/// MakeGemmVariantInfo asserts identity across variants).
struct ServingGemmEpilogue {
  struct ThreadParams {
    float alpha = 1.0f, beta = 0.0f;
  };
  struct Params {
    tilemega::Bf16 const* ptr_C = nullptr;
    RowStride dC;
    tilemega::Bf16* ptr_D = nullptr;
    RowStride dD;
    ThreadParams thread;
  };
  using StrideC = RowStride;
  using StrideD = RowStride;
};

template <class Arch, int TileM, int TileN, int TileK, int Stages>
struct ServingGemmConfig {
  static_assert(std::is_same_v<Arch, arch::Xcore1000>, "xcore1000 serving GEMM only");
  static constexpr int kThreads = solver::kServingBF16Threads;
  static constexpr int kSharedBytes = solver::ServingBF16SmemBytes(TileM, TileN, TileK, Stages);
  static constexpr bool kShapeLegal = solver::ServingBF16ShapeLegal(TileM, TileN, TileK, Stages);

  /// Element and stride types are shared by every variant; only the tile
  /// shape and storage size differ.
  struct Mainloop {
    using ElementA = tilemega::Bf16;
    using ElementB = tilemega::Bf16;
    using StrideA = RowStride;
    using StrideB = RowStride;
    using TileShape = TileShape3<TileM, TileN, TileK>;
    struct alignas(16) SharedStorage {
      unsigned char bytes[kSharedBytes];
    };
  };
  using Epilogue = ServingGemmEpilogue;
};

}  // namespace tilemega::backend
