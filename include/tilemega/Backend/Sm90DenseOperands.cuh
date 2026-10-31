// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <cutlass/bfloat16.h>
#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#if TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY
#error "native shared-operand ABI does not support per-body phase or boundary diagnostics"
#endif
namespace tilemega::backend {
struct Sm90DenseOperands {
  // A is read only through its tensor map (prompt §5 D3).
  CUtensorMap const* activation;
  cutlass::bfloat16_t const *weight,*residual;
  cutlass::bfloat16_t* output;
  int* argmax_index;
  float epsilon;
};
struct Sm90DenseAttentionOperands {
  // History is read by TMA through the stacked [L] maps at coordinate `layer`
  // (D7); the pointers are this layer's views and write row `past`.
  CUtensorMap const *key_map,*value_map;
  cutlass::bfloat16_t const* qkv;
  cutlass::bfloat16_t *key,*value,*output;
  cutlass::bfloat16_t const *cosine,*sine,*q_norm,*k_norm;
  float *partial,*lse;
  std::uint64_t* arrivals;
  int past;
  float epsilon;
  int layer;
};
// lm_head with the folded argmax (D9): the next tokens, lm_ctr, and for the
// LA embedding the table and the residual stream x.
struct Sm90DenseHeadOperands {
  int* tokens;
  std::uint64_t* counter;
  cutlass::bfloat16_t const* table;
  cutlass::bfloat16_t* x;
  int past;
  // Teacher-forced prefill (§8 A3.5): the argmax is written to tokens only at
  // past + 1 >= prompt_len, so prompt positions keep the frozen prompt. 0
  // leaves pure decode, where the condition always holds. It fills the struct's
  // former tail padding, so the operand bank's size and offsets do not move.
  int prompt_len;
};
// Native MoE stages (prompt §8): M4 router + top-k, M5 grouped gate_up,
// M6 down + combine; see Sm90DenseMoe.cuh for each field's use.
struct Sm90DenseMoeOperands {
  CUtensorMap const* activation;
  cutlass::bfloat16_t const* weight;
  cutlass::bfloat16_t const* input;
  void* output;
  int* schedule;
  void* route_weights;
  int* route_index;
  cutlass::bfloat16_t* x;
  std::uint64_t* counter;
  float epsilon;
  // MR7: set only on the router stage of a program whose experts are fused.
  // Its last arriver clears gu_done[e] beside the schedule record; a null
  // pointer (the aggregate default) means M5 and M6 stay apart.
  unsigned* gate_up_done;
};
// MR7's fused expert stage (M56): one stage runs both task types, so it holds
// both weight bases and both outputs (h for the gate_up tasks, the per-route
// partials P for the down tasks) plus `gate_up_done`, the per-active-slot
// counter a down task waits on. `activation` maps h, as M6's did.
struct Sm90DenseMoeFusedOperands {
  CUtensorMap const* activation;
  cutlass::bfloat16_t const *weight_gate_up,*weight_down;
  cutlass::bfloat16_t const* input;
  void *hidden,*output;
  int* schedule;
  void* route_weights;
  int* route_index;
  cutlass::bfloat16_t* x;
  std::uint64_t* counter;
  unsigned* gate_up_done;
  float epsilon;
};
// Thread zero fully initializes the active member for the next stage. That
// member is immutable from the V3 CTA join until the stage's complete return.
// The other member is not read by the stage or its preissue tail.
struct alignas(16) Sm90DenseOperandBank {
  alignas(16) Sm90DenseOperands gemm;
  alignas(16) Sm90DenseHeadOperands head;
  // A stage reads one of these, never two.
  union alignas(16) {
    Sm90DenseAttentionOperands attention;
    Sm90DenseMoeOperands moe;
    Sm90DenseMoeFusedOperands moe_fused;
  };
};
struct Sm90DenseOperandHeader {
  static constexpr int kOffset=2*sizeof(Sm90DenseRingState);
  static constexpr int kEnd=kOffset+2*sizeof(Sm90DenseOperandBank);
  // D0 (prompt §4 A0): the current stage trace's mark array, published by the
  // boundary at level 2 and read by Sm90DenseMark. It sits after the cluster
  // state, inside the 1024-byte header every program already reserves.
  static constexpr int kTraceSlot=1016;
  static_assert(sizeof(void*)==8 && sizeof(Sm90DenseOperands)==48);
  static_assert(sizeof(float)==4 && offsetof(Sm90DenseOperands,epsilon)==40);
  static_assert(sizeof(Sm90DenseAttentionOperands)==120);
  static_assert(sizeof(Sm90DenseHeadOperands)==40 && sizeof(Sm90DenseMoeOperands)==88);
  // M56 must not widen the bank: the union still measures the attention record.
  static_assert(sizeof(Sm90DenseMoeFusedOperands)==104 &&
      sizeof(Sm90DenseMoeFusedOperands)<=sizeof(Sm90DenseAttentionOperands));
  static_assert(sizeof(Sm90DenseOperandBank)==224 && alignof(Sm90DenseOperandBank)==16);
  static_assert(offsetof(Sm90DenseOperandBank,gemm)==0 && offsetof(Sm90DenseOperandBank,head)==48 &&
      offsetof(Sm90DenseOperandBank,attention)==96 && offsetof(Sm90DenseOperandBank,moe)==96 &&
      offsetof(Sm90DenseOperandBank,moe_fused)==96);
  static_assert(sizeof(Sm90DenseRingState)==256 && alignof(Sm90DenseRingState)==16);
  static_assert(kOffset==512 && kOffset%alignof(Sm90DenseOperandBank)==0);
  static_assert(kEnd==960 && kEnd<=1024);
  static_assert(std::is_standard_layout_v<Sm90DenseOperandBank> &&
      std::is_trivially_copyable_v<Sm90DenseOperandBank>);
  __device__ static __forceinline__ Sm90DenseOperandBank& Bank(char* arena,unsigned stage) {
    return reinterpret_cast<Sm90DenseOperandBank*>(arena+kOffset)[stage%2];
  }
};
// Cluster programs (prompt §7 D10b/D10c) keep their cluster barrier state
// after the operand banks: the DSMEM merge inbox of the CTA leading a (b, g)
// chunk group and the D10c issuer phases of both rings' peer barriers.
struct alignas(8) Sm90DenseClusterState {
  std::uint64_t inbox;
  unsigned inbox_generation;
  unsigned peer_generations[2];
  __device__ static __forceinline__ Sm90DenseClusterState& Of(char* shared) {
    return *reinterpret_cast<Sm90DenseClusterState*>(shared+Sm90DenseOperandHeader::kEnd);
  }
  // rings[i] sits at byte 256 i of the 1024-aligned dynamic shared memory.
  __device__ static __forceinline__ unsigned RingIndex(Sm90DenseRingState const& ring) {
    unsigned address=codegen::executor::Async<arch::Sm90>::Shared(&ring);
    return (address/sizeof(Sm90DenseRingState))&1;
  }
  __device__ static __forceinline__ Sm90DenseClusterState& OfRing(Sm90DenseRingState& ring) {
    return Of(reinterpret_cast<char*>(&ring)-RingIndex(ring)*sizeof(Sm90DenseRingState));
  }
};
static_assert(Sm90DenseOperandHeader::kEnd%8==0 &&
    Sm90DenseOperandHeader::kEnd+sizeof(Sm90DenseClusterState)<=1024);
static_assert(Sm90DenseOperandHeader::kEnd+sizeof(Sm90DenseClusterState)<=
    Sm90DenseOperandHeader::kTraceSlot &&
    Sm90DenseOperandHeader::kTraceSlot+8<=1024 &&
    Sm90DenseOperandHeader::kTraceSlot%8==0,
    "the D0 mark slot must clear the cluster state and stay in the header");
}  // namespace tilemega::backend
