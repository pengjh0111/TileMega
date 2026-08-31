// SPDX-License-Identifier: BSD-3-Clause
//
// TileMega -- test/harness/alternating_fill.h
//
// Alternating-fill verifier (skeleton §7 P0.3, bullet "交替填充校验器").
//
// Why this exists
// ---------------
// A cross-block synchronisation test that fills producer buffers with a
// CONSTANT pattern is nearly useless: the buffer allocated by run N+1 very
// often lands on the physical pages that run N just wrote, and cudaMalloc does
// not zero memory.  A consumer that reads its input *before* the producer has
// written it therefore reads last run's bytes -- which are bit-identical to
// this run's correct answer.  The test passes while the synchronisation is
// entirely broken.
//
// The fix is to make the expected value a function of a per-run nonce, so no
// stale buffer can ever accidentally equal the current run's correct answer.
//
// Values are integers in [0,255] represented exactly in float, so that a sum of
// up to 2^16 of them is still exact in fp32 (255 * 65536 = 16.7M < 2^24).  This
// removes floating-point associativity from the correctness question: a
// mismatch means a synchronisation bug, not a rounding difference.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __CUDACC__
#define TM_HD __host__ __device__ __forceinline__
#else
#define TM_HD inline
#endif

namespace tilemega {
namespace testing {

/// Fill mode.  `kAlternating` is the real verifier; `kConstant` exists only so
/// experiments can demonstrate what the naive choice hides (V-A question 2).
enum class FillMode : int { kAlternating = 0, kConstant = 1 };

TM_HD uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

/// The value producer `tile` writes at element `idx` during round `iter`.
/// Integer-valued in [0,255].
TM_HD float FillValue(FillMode mode, uint32_t nonce, int iter, int tile, int idx) {
  if (mode == FillMode::kConstant) {
    // Run-invariant: identical bytes every run -> stale memory masks bugs.
    return static_cast<float>((tile * 7 + idx * 3 + iter * 11) & 0xFF);
  }
  uint32_t h = hash32(nonce ^ hash32(static_cast<uint32_t>(iter) * 0x9E3779B9u ^
                                     hash32(static_cast<uint32_t>(tile) * 2654435761u ^
                                            static_cast<uint32_t>(idx))));
  return static_cast<float>(h & 0xFFu);
}

/// Pick the per-run nonce.  Honour TILEMEGA_FILL_NONCE so a failing run can be
/// replayed bit-exactly; otherwise derive one from the clock and the pid.
inline uint32_t PickNonce() {
  if (const char* e = std::getenv("TILEMEGA_FILL_NONCE")) {
    return static_cast<uint32_t>(std::strtoul(e, nullptr, 0));
  }
  uint32_t t = static_cast<uint32_t>(std::time(nullptr));
  uint32_t p = static_cast<uint32_t>(
#ifdef _WIN32
      0
#else
      getpid()
#endif
  );
  uint32_t n = hash32(t ^ hash32(p));
  return n ? n : 1u;  // never 0: a zero nonce is indistinguishable from unset
}

}  // namespace testing
}  // namespace tilemega
