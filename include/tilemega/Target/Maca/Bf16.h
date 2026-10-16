// SPDX-License-Identifier: BSD-3-Clause
// TileMega -- Target/Maca/Bf16.h
//
// 16-bit bfloat16 storage type shared by host and device code. Replaces
// cutlass::bfloat16_t so the MetaX build includes no CUTLASS header.
// fp32 -> bf16 rounds to nearest even like torch; NaN stays NaN (quiet).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__MACACC__) || defined(__CUDACC__)
#define TILEMEGA_BF16_HD __host__ __device__ __forceinline__
#else
#define TILEMEGA_BF16_HD inline
#endif

namespace tilemega {

struct alignas(2) Bf16 {
  uint16_t storage;

  Bf16() = default;
  TILEMEGA_BF16_HD explicit Bf16(float value) : storage(Round(value)) {}
  TILEMEGA_BF16_HD explicit Bf16(double value) : storage(Round(static_cast<float>(value))) {}
  TILEMEGA_BF16_HD explicit Bf16(int value) : storage(Round(static_cast<float>(value))) {}

  TILEMEGA_BF16_HD static Bf16 bitcast(uint16_t bits) {
    Bf16 result;
    result.storage = bits;
    return result;
  }

  TILEMEGA_BF16_HD explicit operator float() const { return ToFloat(storage); }
  TILEMEGA_BF16_HD uint16_t& raw() { return storage; }
  TILEMEGA_BF16_HD uint16_t raw() const { return storage; }

  TILEMEGA_BF16_HD static uint16_t Round(float value) {
#if defined(__MACA_ARCH__)
    // Hardware RNE conversion: bit-identical to the software path below for
    // every non-NaN input (16.7M random patterns plus specials, 0 mismatches;
    // NaN stays a quiet NaN) and 2.8x faster on a dependent chain (measured).
    // The `_fast` variant is not exact and is not used.
    auto converted = __builtin_mxc_cvt_f32tobf16(value);
    uint16_t result;
    __builtin_memcpy(&result, &converted, sizeof(result));
    return result;
#endif
    uint32_t bits = FloatBits(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
      return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
  }

  TILEMEGA_BF16_HD static float ToFloat(uint16_t bits) { return BitsFloat(static_cast<uint32_t>(bits) << 16); }

 private:
  // Device code uses the bit-cast intrinsics: the memcpy form costs ~40 %
  // more on a dependent float->bf16->float chain on xcore1000 (measured).
  TILEMEGA_BF16_HD static uint32_t FloatBits(float value) {
#if defined(__CUDA_ARCH__) || defined(__MACA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
  }
  TILEMEGA_BF16_HD static float BitsFloat(uint32_t bits) {
#if defined(__CUDA_ARCH__) || defined(__MACA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
  }
};

static_assert(sizeof(Bf16) == 2, "Bf16 must be 16-bit storage");

}  // namespace tilemega
