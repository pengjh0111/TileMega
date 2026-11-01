// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/mx/StageProgram.cuh>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/PeerAccess.h>
#include <cstdint>

#ifndef TILEMEGA_TP_LITMUS
#define TILEMEGA_TP_LITMUS 0
#endif

namespace tilemega::backend::mx {

inline constexpr int kMxTpMaxRanks = 8;
struct MxTpLinkV2 {
  std::uint64_t* recv[kMxTpMaxRanks];
  std::uint32_t region, epoch;
  int rank, world;
  codegen::mx::StageWatchV2 watch;
};
struct alignas(64) MxTpExchangeV2 {
  std::uint64_t data[2][kMxTpMaxRanks][16][2];
};
static_assert(sizeof(MxTpExchangeV2) == 4096);
struct MxTpHeadLinkV2 {
  MxTpExchangeV2* x[kMxTpMaxRanks];
  std::uint32_t epoch, parity;
  int rank, world, vocab_local, prompt_len;
  codegen::mx::StageWatchV2 watch;
};
template <class Base> struct MxTpOperandsV2 : Base { MxTpLinkV2 tp; };
template <class Base> struct MxTpHeadOperandsV2 : Base { MxTpHeadLinkV2 tp; };

__device__ __forceinline__ std::uint64_t MxLl8Pack(Bf16 even, Bf16 odd, std::uint32_t epoch) {
  return (std::uint64_t(epoch) << 32) | (std::uint32_t(odd.raw()) << 16) | even.raw();
}

__device__ __forceinline__ std::uint32_t MxTpAwait(std::uint64_t const* p, std::uint32_t epoch,
    codegen::mx::StageWatchV2 const& watch, int site) {
#if TILEMEGA_WATCHDOG
  unsigned long long start = 0, failures = 0;
#endif
  for (;;) {
    std::uint64_t const observed = maca::PeerLoad64(p);
    if (std::uint32_t(observed >> 32) == epoch) return std::uint32_t(observed);
#if TILEMEGA_WATCHDOG
    if (watch.record && watch.limit_ns && !(++failures & 4095ull)) {
      unsigned long long const now = maca::GlobalTimerNs();
      if (!start) start = now;
      if (now - start > watch.limit_ns && atomicCAS(&watch.record->fired, 0ull, 1ull) == 0ull) {
        watch.record->site = site;
        watch.record->block = blockIdx.x;
        watch.record->thread = threadIdx.x;
        watch.record->waiter_stage = watch.stage;
        watch.record->need = epoch;
        watch.record->value = observed;
        watch.record->iteration = watch.iteration;
        __threadfence_system();
        maca::Trap();
      }
    }
#else
    (void)watch;
    (void)site;
#endif
  }
}

__device__ __forceinline__ void MxTpLitmusDelay(MxTpLinkV2 const& link, int tile) {
#if TILEMEGA_TP_LITMUS
  unsigned hash = unsigned(link.rank) * 0x9e3779b9u ^ link.epoch * 0x85ebca6bu ^
                  unsigned(tile) * 0xc2b2ae35u ^ blockIdx.x;
  hash ^= hash >> 16;
  unsigned long long const until = maca::GlobalTimerNs() + hash % 5000;
  while (maca::GlobalTimerNs() < until) {}
#else
  (void)link;
  (void)tile;
#endif
}

}  // namespace tilemega::backend::mx
