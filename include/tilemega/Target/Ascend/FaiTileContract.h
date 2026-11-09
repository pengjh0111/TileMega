// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace tilemega::ascend::fai_tile {
inline constexpr unsigned kRows=32,kKeys=128,kMmadK=128,kStages=2,kUbBase=131584;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 0
#define TILEMEGA_FAI_INLINE __forceinline__[host, aicore]
#else
#define TILEMEGA_FAI_INLINE inline
#endif
TILEMEGA_FAI_INLINE constexpr unsigned Align(unsigned bytes) {return (bytes+511)/512*512;}
struct Resources {unsigned l1,l0a,l0b,l0c,ub;};
TILEMEGA_FAI_INLINE constexpr Resources Budget(unsigned width) {
  // QK and PV share L0 event rings: both physical banks use the same stride,
  // even when the logical head dimension is64 and PV reduces128 keys.
  const unsigned qk_a=kRows*kMmadK*2,qk_b=kKeys*kMmadK*2;
  const unsigned score=kRows/2*kKeys*4,partial=kRows/2*width*4;
  const unsigned statistics=3*512;
  const unsigned softmax=2*Align(kRows/2*kKeys*2)+2*512;
  return {kRows*kKeys*2+2*(qk_a+qk_b)+2*qk_b,
          2*qk_a,2*qk_b,2*kRows*kKeys*4*2,
          kUbBase+2*(score+partial)+statistics+softmax+Align(partial)};
}
TILEMEGA_FAI_INLINE constexpr bool Shape(unsigned rows,unsigned width,unsigned keys,unsigned capacity) {
  return rows && rows<=kRows && (width==64 || width==128) && keys && keys<=capacity && capacity<=1088;
}
// The padded queries and output belong to the Task's private physical tile.
// The two AIVs own disjoint halves; small decode groups do not imply useful
// work on the padded lane. This recipe is not a model-admitted implementation.
TILEMEGA_FAI_INLINE constexpr unsigned UsefulRows(unsigned rows,unsigned lane) {
  return lane>1 || rows<=lane*16?0:rows-lane*16<16?rows-lane*16:16;
}
inline constexpr uint32_t kFreeOutput[2]={7,8},kFreeScore[2]={9,10};
inline constexpr uint32_t kScoreReady=11,kOutputReady=12,kProbabilityReady=13;
inline constexpr bool kModelAdmitted=false;
static_assert(Budget(128).l1<=524288 && Budget(128).l0a<=65536 &&
              Budget(128).l0b<=65536 && Budget(128).l0c<=262144 && Budget(128).ub<=221184);
#undef TILEMEGA_FAI_INLINE
} // namespace tilemega::ascend::fai_tile
