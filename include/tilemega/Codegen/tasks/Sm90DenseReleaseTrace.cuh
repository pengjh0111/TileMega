// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>

namespace tilemega::codegen::native_dense::release_trace {

struct alignas(16) Header {
  std::uint64_t iteration,begin,end;
  std::int32_t past;
  std::uint32_t valid;
};
static_assert(sizeof(Header)==32 && offsetof(Header,valid)==28);

template<unsigned Stages>
struct alignas(16) Record {
  static_assert(Stages>0);
  Header header;
  std::uint32_t releases[(Stages+3)/4*4];
};

struct Binding {
  std::uint32_t* words=nullptr;
  unsigned slots=0;
};

constexpr unsigned Align16(unsigned bytes) {return (bytes+15)/16*16;}

}  // namespace tilemega::codegen::native_dense::release_trace
