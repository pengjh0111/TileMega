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
  // GEMM variants: 0=32x64x64, 1=16x128x128. Attention variants:
  // 0=BF16 scores/probabilities, 1=FP32 chunks, 2=FP32 scores/BF16 probabilities,
  // 3=math SDPA split scale/capacity softmax; 4=cooperative full-FP32 Cube.
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
static_assert(sizeof(Metadata)==72 && offsetof(Metadata,arena_bytes)==48);
static_assert(sizeof(Stage)==200 && offsetof(Stage,workspace)==128 && offsetof(Stage,epsilon)==196);
static_assert(sizeof(Task)==28 && sizeof(Wait)==8 && sizeof(Forward)==16);
static_assert(sizeof(Allocation)==32);
static_assert(sizeof(Control)==512 && alignof(Control)==512);
static_assert(std::is_standard_layout_v<Stage> && std::is_trivially_copyable_v<Stage>);
static_assert(std::is_standard_layout_v<Metadata> && std::is_trivially_copyable_v<Metadata>);
} // namespace tilemega::ascend::llm
