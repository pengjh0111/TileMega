// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <kernel_operator.h>
#include <cstdint>

namespace tilemega::ascend {
struct alignas(512) EventCell {
  uint32_t value;
  uint32_t padding[127];
};
static_assert(sizeof(EventCell) == 512);

__aicore__ inline unsigned GetWorkerId(unsigned groups) {
  if ASCEND_IS_AIC { return AscendC::GetBlockIdx(); }
  else { return groups + AscendC::GetBlockIdx(); }
}
__aicore__ inline uint32_t ReadControl(__gm__ uint32_t* address) {
  return AscendC::ReadGmByPassDCache(address);
}
__aicore__ inline void WriteControl(__gm__ uint32_t* address, uint32_t value) {
  AscendC::WriteGmByPassDCache(address, value);
}
// This route has exactly one publisher per event. It is not an atomic
// reduction and cannot implement a multi-producer aggregate event.
__aicore__ inline void PublishSingleOwnerEvent(__gm__ uint32_t* address, uint32_t epoch) {
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
  WriteControl(address, epoch);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}
__aicore__ inline bool WaitEvent(__gm__ uint32_t* address, uint32_t needed,
                                 __gm__ uint32_t* abort, uint32_t limit = 100000) {
  for (uint32_t step = 0; step < limit; ++step) {
    if (ReadControl(abort)) return false;
    if (ReadControl(address) >= needed) {
      AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
      return true;
    }
    AscendC::Nop<8>();
  }
  WriteControl(abort, 1);
  return false;
}
template<class T> __aicore__ inline void AcquireData(AscendC::GlobalTensor<T> const& tensor) {
  AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE>(tensor);
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}
// Slot 7 is separate from CATLASS's MTE2/MTE1 and MMAD/FIX pipeline slots.
template<AscendC::HardEvent event> __aicore__ inline void CompleteDma() {
  AscendC::SetFlag<event>(7);
  AscendC::WaitFlag<event>(7);
}
} // namespace tilemega::ascend
