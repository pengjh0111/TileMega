// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace tilemega::ascend::local_residual {
inline constexpr unsigned kSlot=131584,kTileBytes=4096,kGuardBytes=512;
inline constexpr unsigned kUbBytes=kSlot+kTileBytes+kGuardBytes;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 0
__aicore__
#endif
inline constexpr bool Shape(unsigned rows,unsigned columns,unsigned k,
    unsigned epilogue,unsigned parts,unsigned tasks) {
  return rows==1 && columns==1024 && (k==2048 || k==3072) &&
      epilogue==1 && parts<=1 && tasks==8;
}
static_assert(kSlot%512==0 && kUbBytes<=221184);
} // namespace tilemega::ascend::local_residual
