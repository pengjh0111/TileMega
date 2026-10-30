// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Mx/DecodeGemmV2.h>
#include <tilemega/Codegen/tasks/mx/StageSyncV3.cuh>
#include <cmath>
#include <type_traits>

namespace tilemega::backend::mx {

template <bool Fp32>
struct MxRouteTableStorageV3 {
  unsigned n_active, units_gate_up, units_down, pad;
  unsigned char expert[128];
  unsigned char count[128];
  unsigned char row[128][16];
  std::conditional_t<Fp32, float, unsigned short> weight[16][8];
};
using MxRouteTableV3 = MxRouteTableStorageV3<false>;
static_assert(sizeof(MxRouteTableV3) == 2576);
static_assert(sizeof(MxRouteTableStorageV3<true>) == 2832);
template <bool Fp32>
using MxSharedRouteTableV3 = __attribute__((address_space(3))) MxRouteTableStorageV3<Fp32>;

template <int B, int E>
struct MxRouteWorkspaceV3 {
  float logits[B][E], selected[B][8];
  unsigned indices[B][8], counts[E];
  unsigned char ranks[B][E];
};

template <int B, int Hidden, int Intermediate, int TnGate, int TnDown,
          int Experts, bool NormalizeTopK, bool Fp32 = false>
struct MxBuildRouteTableV3 {
  static_assert(B > 0 && B <= 16 && (Experts == 64 || Experts == 128));
  static_assert(Hidden % TnDown == 0 && (2 * Intermediate) % TnGate == 0);
  static constexpr int E = Experts, K = 8;
  static constexpr int kSharedBytes = (sizeof(MxRouteWorkspaceV3<B,E>) + 15) / 16 * 16;
  using Sync = codegen::mx::StageSyncV3<512>;
  using Workspace = __attribute__((address_space(3))) MxRouteWorkspaceV3<B,E>;

  __device__ __forceinline__ static void Run(Bf16 const* input, MxSharedRouteTableV3<Fp32>* table,
                                            char* storage,
                                            unsigned* route_debug = nullptr,
                                            void* route_weights = nullptr,
                                            unsigned step = 64, unsigned layer = 0, unsigned layers = 0) {
    unsigned const t = StageThread();
    __builtin_assume(t < 512);
    auto* work = (Workspace*)storage;
    if (t < B * E / 8) {
      uint4 const packet = LoadGlobal16Gvm(input + t * 8);
      __builtin_mxc_arrive_gvmcnt(0);
      alignas(16) Bf16 values[8];
      *reinterpret_cast<uint4*>(values) = packet;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) work->logits[(t * 8) / E][(t * 8 + j) % E] = float(values[j]);
    }
    Sync::LdsSync();
    for (unsigned item = t; item < B * E; item += 512) {
      unsigned const token = item / E, expert = item % E;
      float const value = work->logits[token][expert];
      if (!isfinite(value)) maca::Trap();
      unsigned rank = 0;
#pragma unroll 4
      for (unsigned base = 0; base < E; base += 4) {
#pragma unroll
        for (unsigned offset = 0; offset < 4; ++offset) {
          unsigned const j = base + offset;
          float const other = work->logits[token][j];
          rank += unsigned(other > value) | (unsigned(other == value) & unsigned(j < expert));
        }
      }
      if (rank < K) {
        work->selected[token][rank] = value;
        work->indices[token][rank] = expert;
      }
      work->ranks[token][expert] = rank < K ? rank : 0xff;
    }
    Sync::LdsSync();
    if (t < B) {
      float probabilities[K], sum = 0;
#pragma unroll
      for (unsigned rank = 0; rank < K; ++rank) {
        probabilities[rank] = expf(work->selected[t][rank] - work->selected[t][0]);
        if constexpr (NormalizeTopK) sum += probabilities[rank];
      }
      if constexpr (!NormalizeTopK) {
#pragma unroll 4
        for (unsigned e = 0; e < E; ++e) sum += expf(work->logits[t][e] - work->selected[t][0]);
      }
#pragma unroll
      for (unsigned rank = 0; rank < K; ++rank) {
        float const weight = probabilities[rank] / sum;
        if constexpr (Fp32) table->weight[t][rank] = weight;
        else table->weight[t][rank] = Bf16(weight).raw();
      }
    }
    if (t < E) {
      unsigned count = 0;
#pragma unroll 4
      for (unsigned token = 0; token < B; ++token) count += work->ranks[token][t] != 0xff;
      work->counts[t] = count;
    }
    Sync::LdsSync();
    if (t < E && work->counts[t]) {
      unsigned ordinal = 0;
#pragma unroll 4
      for (unsigned e = 0; e < t; ++e) ordinal += work->counts[e] != 0;
      table->expert[ordinal] = t;
      table->count[ordinal] = work->counts[t];
      unsigned row = 0;
#pragma unroll 4
      for (unsigned token = 0; token < B; ++token)
        if (work->ranks[token][t] != 0xff)
          table->row[ordinal][row++] = (token << 3) | work->ranks[token][t];
    }
    if (t == 0) {
      unsigned total = 0;
#pragma unroll 4
      for (unsigned e = 0; e < E; ++e) total += work->counts[e] != 0;
      table->n_active = total;
      table->units_gate_up = total * (2 * Intermediate / TnGate);
      table->units_down = total * (Hidden / TnDown);
      table->pad = 0;
    }
    Sync::LdsSync();
#if TILEMEGA_MX_DIAG
    if (blockIdx.x == 0 && step < 64) {
      if (route_debug)
        for (unsigned item = t; item < B * K; item += 512) {
          using Global = __attribute__((address_space(1))) unsigned;
          *((Global*)route_debug + (step * layers + layer) * B * K + item) = work->indices[item/K][item%K];
        }
      if (route_weights)
        for (unsigned item = t; item < B * K; item += 512) {
          using Weight = std::conditional_t<Fp32, float, unsigned short>;
          using Global = __attribute__((address_space(1))) Weight;
          *((Global*)route_weights + (step * layers + layer) * B * K + item) = table->weight[item/K][item%K];
        }
    }
#else
    (void)route_debug; (void)route_weights; (void)step; (void)layer; (void)layers;
#endif
  }
};

}  // namespace tilemega::backend::mx
