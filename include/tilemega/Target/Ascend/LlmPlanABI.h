// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tilemega::ascend::llm {
inline constexpr uint32_t kVersion=4;
inline constexpr uint64_t kAbsent=~uint64_t(0);
namespace attention {
inline constexpr uint64_t kGroupBytes=2ULL<<20;
inline constexpr uint64_t kQ=0,kK=65536,kV=622592,kScores=1179648;
inline constexpr uint64_t kProbabilities=1474560,kOutput=1769472;
inline constexpr uint32_t kMaxRows=64,kMaxWidth=128,kCapacity=1088;
static_assert(kQ+uint64_t(kMaxRows)*kMaxWidth*4<=kK);
static_assert(kK+uint64_t(kCapacity)*kMaxWidth*4<=kV);
static_assert(kV+uint64_t(kCapacity)*kMaxWidth*4<=kScores);
static_assert(kScores+uint64_t(kMaxRows)*kCapacity*4<=kProbabilities);
static_assert(kProbabilities+uint64_t(kMaxRows)*kCapacity*4<=kOutput);
static_assert(kOutput+uint64_t(kMaxRows)*kMaxWidth*4<=kGroupBytes);
// Probe-only observations fit the existing workspace; production never writes them.
inline constexpr uint64_t kDiagnosticExp=kOutput+uint64_t(kMaxRows)*kMaxWidth*4;
inline constexpr uint64_t kDiagnosticStats=kDiagnosticExp+uint64_t(kMaxRows)*kCapacity*4;
static_assert(kDiagnosticStats+uint64_t(kMaxRows)*64<=kGroupBytes);
}
struct Metadata {
  uint32_t version,groups,workers,tasks,stages,waits,queue_refs,epoch_stride;
  uint32_t batch,seq,capacity,diagnostic;
  uint64_t arena_bytes,worker_scratch_bytes,wait_ticks;
};
// Operand values are byte offsets, including weights beyond 4 GiB.
struct Stage {
  uint64_t operands[16];
  uint64_t workspace;
  uint32_t kind,epilogue,tasks,rows,columns,k,extent,width,group;
  uint32_t kv_block,query_rows,row_stride,row_offset,batch_rows,variant;
  // RMSNorm variants: 0=existing Rsqrt, 2=pinned Newton H1024/H2048,
  // 3=power-two-fold Newton H1024/H2048/H3072. Buffer/event ABI is identical.
  // GEMM variants: 0=32x64x64, 1=16x128x128, 2=same geometry with local tail7.
  // 3=same geometry, M1024 N3584:4096 uses ordered ceil(K/3) slices:
  // Q store epilogue4, O/down residual epilogue1, K4096 or14336.
  // Variant2/3 operand5 is per-group FP32 scratch. The MIX done ack ends
  // its lifetime before the next tile; only the AIC owner publishes completion.
  // 4=same geometry, bounded dense M16 strided U16 SwiGLU; identical casts,
  // GM workspace and MIX lifecycle, existing per-worker UB arrays are borrowed.
  // 5=variant4 with internal L1 K256/L0 K32 and per-task MMAD lifetime.
  // CTG output geometry, BF16 intermediates and completion protocol are unchanged.
  // Split GEMM: extent=2/3, width=ceil(k/extent), group=1 (tile-major/part-minor CTG).
  // Kind6 is its AIV combine; workspace is FP32 [part,row,col], operand5
  // is the BF16 raw sum before the separately preserved residual rounding.
  // Attention variants: 0=BF16 scores/probabilities, 1=FP32 chunks, 2=FP32 scores/BF16 probabilities,
  // 3=math SDPA split scale/capacity softmax; 4=cooperative full-FP32 Cube;
  // 5=the same cooperative path with pinned CANN Newton-refined QK RMSNorm.
  // 6=variant5 with the pinned capacity1088 register-fold softmax sum.
  // 7=variant6 with QK internal tiles32x256x128; unchanged task ownership/PV.
  // 8=variant6 omitting entirely masked QK64 tiles; same physical stride/PV.
  // 9=variant8 with request-lived derived FP32 KV in operands10/11. Prefill
  // writes disjoint owned rows, decode appends one, reset clears both planes.
  float epsilon;
};
struct Task {
  uint32_t stage,tile,owner,ordinal,wait_begin,wait_count,mix_aiv_lanes;
};
struct Wait {uint32_t worker,prefix;};
struct Allocation {uint64_t offset,bytes,allocated;uint32_t required,reserved;};
struct alignas(512) Control {uint32_t value;uint32_t reserved[127];};
// A forward is stream-ordered after the previous forward. The host checks the
// product before launch; request resets clear controls only after completion.
struct Forward {uint32_t epoch,past,diagnostic,reserved;};
namespace body_profile {
inline constexpr uint64_t kAttentionMagic=0x4154325048415345ULL;
enum AttentionPhase : unsigned {kKvAppend,kKvCast,kKvZero,kQuery,kQk,kSoftmax,kPv,kOutput,kPhases};
inline constexpr unsigned kHeaderWords=8,kPhaseWords=2*kPhases;
inline constexpr uint64_t kCubeMask=(1ULL<<kQk)|(1ULL<<kPv);
inline constexpr uint64_t kVectorMask=((1ULL<<kPhases)-1)^kCubeMask;
static_assert((kHeaderWords+2+kPhaseWords)*sizeof(uint64_t)<=512);
}
static_assert(sizeof(Metadata)==72 && offsetof(Metadata,arena_bytes)==48);
static_assert(sizeof(Stage)==200 && offsetof(Stage,workspace)==128 && offsetof(Stage,epsilon)==196);
static_assert(sizeof(Task)==28 && sizeof(Wait)==8 && sizeof(Forward)==16);
static_assert(sizeof(Allocation)==32);
static_assert(sizeof(Control)==512 && alignof(Control)==512);
static_assert(std::is_standard_layout_v<Stage> && std::is_trivially_copyable_v<Stage>);
static_assert(std::is_standard_layout_v<Metadata> && std::is_trivially_copyable_v<Metadata>);
} // namespace tilemega::ascend::llm
