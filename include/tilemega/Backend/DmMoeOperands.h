// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/MoeBinding.h>
#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(__CUDACC__)
#define TILEMEGA_DM_MOE_HD __host__ __device__
#else
#define TILEMEGA_DM_MOE_HD
#endif

namespace tilemega::backend {
// Virtual task rows include TM padding; intermediate storage uses BM rows
// per binding, independently of the gate/up and down stages' chosen TM.
struct DmMoeRows {
  codegen::MoeBindingRow const* rows = nullptr;
  codegen::MoeBindingRecord block{};
  std::uint32_t virtual_begin = 0, storage_begin = 0;
  TILEMEGA_DM_MOE_HD std::uint32_t Entry(std::uint32_t row) const {
    return block.row_begin + row - virtual_begin;
  }
  TILEMEGA_DM_MOE_HD std::uint32_t Token(std::uint32_t row) const {
    return rows ? rows[Entry(row)].token : row;
  }
  TILEMEGA_DM_MOE_HD std::uint32_t Storage(std::uint32_t row) const {
    return rows ? storage_begin + row - virtual_begin : row;
  }
  TILEMEGA_DM_MOE_HD std::uint32_t Scatter(std::uint32_t row, unsigned topk) const {
    auto const& item = rows[Entry(row)];
    return item.token * topk + item.rank;
  }
};

template<class T,class=void>struct HasDmOperandRows:std::false_type {};
template<class T>struct HasDmOperandRows<T,std::void_t<decltype(std::declval<T>().moe)>>:std::true_type {};
template<class Operands>
TILEMEGA_DM_MOE_HD std::uint32_t DmSourceRow(Operands const& p, std::uint32_t row) {
  if constexpr(HasDmOperandRows<Operands>::value) {
    if(p.access.a==codegen::DmAAccess::kRowGather)
      return p.moe.rows ? p.moe.Token(row) : static_cast<std::uint32_t const*>(p.rows)[row];
    return p.moe.Storage(row);
  }else return row;
}

#if defined(__CUDACC__)
template<class Operands>
__device__ bool ResolveDmMoeTile(Operands& p, unsigned tile_m, unsigned tm) {
  using namespace codegen;
  if(p.access.b!=DmBAccess::kExpertIndirect || p.moe.rows)return true;
  auto const& access=p.access;
  if(!tm || !access.block_rows) {asm volatile("trap;");return false;}
  unsigned tiles=(access.block_rows+tm-1)/tm,id=tile_m/tiles;
  MoeBindingView view{static_cast<MoeBindingRecord const*>(p.binding),
      static_cast<MoeBindingRow const*>(p.rows),access.binding_blocks,
      access.binding_rows,access.experts,access.block_rows};
  MoeBindingRecord block;
  auto status=view.Lookup(id,&block);
  if(status==MoeBindingStatus::kInvalid) {asm volatile("trap;");return false;}
  if(status==MoeBindingStatus::kEmpty || (tile_m%tiles)*tm>=block.row_count)return false;
  std::uint64_t offset;
  if(!MoeExpertOffset(block,access.expert_stride,&offset)) {
    asm volatile("trap;");return false;
  }
  if(p.b)p.b+=offset;
  if(p.weight_base)p.weight_base+=offset;
  p.tensor_map=nullptr;
  p.moe={view.rows,block,id*tiles*tm,id*access.block_rows};
  p.m=p.moe.virtual_begin+block.row_count;
  return true;
}
#endif
} // namespace tilemega::backend
#undef TILEMEGA_DM_MOE_HD
