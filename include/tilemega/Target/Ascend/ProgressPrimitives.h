// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/DevicePrimitives.h>

namespace tilemega::ascend {
// ticks is supplied by a target with a measured system-counter frequency.
// Only a single owner writes each progress cell; payload completion precedes
// PublishSingleOwnerEvent and consumers still acquire their payload explicitly.
__aicore__ inline bool WaitProgress(__gm__ uint32_t* address, uint32_t needed,
                                    __gm__ uint32_t* abort, uint64_t ticks) {
  const uint64_t start = AscendC::GetSystemCycle();
  uint32_t polls = 0;
  while (ticks) {
    if (ReadControl(abort)) return false;
    if (ReadControl(address) >= needed) {
      AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
      return true;
    }
    if ((++polls & 63u) == 0 && uint64_t(AscendC::GetSystemCycle()) - start >= ticks)
      break;
    AscendC::Nop<8>();
  }
  WriteControl(abort, 1);
  return false;
}
} // namespace tilemega::ascend
