// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define TILEMEGA_MOE_BINDING_HD __host__ __device__
#else
#define TILEMEGA_MOE_BINDING_HD
#endif

namespace tilemega::codegen {
struct alignas(16) MoeBindingRecord {
  std::uint32_t expert = 0, row_begin = 0, row_count = 0, valid = 0;
};
struct alignas(16) MoeBindingRow {
  std::uint32_t token = 0, rank = 0;
  std::uint16_t weight_bf16 = 0, reserved16 = 0;
  std::uint32_t reserved32 = 0;
};
enum class MoeBindingStatus : std::uint32_t { kActive, kEmpty, kInvalid };
struct MoeBindingView {
  MoeBindingRecord const* blocks = nullptr;
  MoeBindingRow const* rows = nullptr;
  std::uint32_t block_capacity = 0, row_capacity = 0;
  std::uint32_t experts = 0, block_rows = 0;

  TILEMEGA_MOE_BINDING_HD MoeBindingStatus Lookup(std::uint32_t id,
                                                MoeBindingRecord* out) const {
    if (!blocks || !out || id >= block_capacity) return MoeBindingStatus::kInvalid;
    *out = blocks[id];
    // Empty tail records may retain fields from the preceding launch. Only
    // valid is tested before using them; no stale address becomes a page.
    if (!out->valid) return MoeBindingStatus::kEmpty;
    if (out->valid != 1 || !rows || !out->row_count || out->expert >= experts ||
        out->row_count > block_rows ||
        std::uint64_t(out->row_begin) + out->row_count > row_capacity)
      return MoeBindingStatus::kInvalid;
    return MoeBindingStatus::kActive;
  }
};

TILEMEGA_MOE_BINDING_HD inline bool MoeVirtualCapacity(
    std::uint32_t tokens, std::uint32_t topk, std::uint32_t experts,
    std::uint32_t block_rows, bool grouped, std::uint32_t* capacity) {
  if (!tokens || !topk || topk > experts || !block_rows || !capacity) return false;
  auto rows = std::uint64_t(tokens) * topk;
  auto active_experts = rows < experts ? rows : experts;
  auto blocks = grouped ? (rows + block_rows - 1) / block_rows + active_experts : rows;
  if (blocks > std::numeric_limits<std::uint32_t>::max()) return false;
  *capacity = static_cast<std::uint32_t>(blocks);
  return true;
}

// Packed expert strides count BF16 elements, independently of TN/TK/page
// layout. The caller validates the table allocation before publishing binding.
TILEMEGA_MOE_BINDING_HD inline bool MoeExpertOffset(
    MoeBindingRecord const& block, std::uint64_t stride, std::uint64_t* offset) {
  if (block.valid != 1 || !stride || !offset ||
      stride > std::numeric_limits<std::uint64_t>::max() / (std::uint64_t(block.expert) + 1))
    return false;
  *offset = std::uint64_t(block.expert) * stride;
  return true;
}
static_assert(sizeof(MoeBindingRecord) == 16 && sizeof(MoeBindingRow) == 16);
static_assert(std::is_trivially_copyable_v<MoeBindingRecord>);
static_assert(std::is_trivially_copyable_v<MoeBindingRow>);
}  // namespace tilemega::codegen
#undef TILEMEGA_MOE_BINDING_HD
