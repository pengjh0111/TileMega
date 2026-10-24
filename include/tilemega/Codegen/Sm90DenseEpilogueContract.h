// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tilemega::codegen::native_dense::epilogue_contract {
inline constexpr std::uint32_t kAbsent=~std::uint32_t{0};
inline constexpr std::uint32_t kInternal=0,kExternal=1,kBF16=0,kF32=1,kI32=2;
template<std::uint32_t Norm,std::uint32_t Sums>
struct Presence {
  static constexpr bool kInputNorm=Norm!=kAbsent,kOutputSums=Sums!=kAbsent;
};
struct Gemm {
  std::uint32_t norm=kAbsent,sums=kAbsent;
  std::uint64_t batch=0,k=0,n=0,tile_n=0;
  bool vector_sums=false,residual=false,active=false;
};
template<class Stage,class=void>
struct SpecFor {static constexpr Gemm value{};};
template<class Stage>
struct SpecFor<Stage,std::void_t<decltype(Stage::kEpilogueBinding)>> {
  static constexpr Gemm value=Stage::kEpilogueBinding;
};
struct Buffer {
  std::uint32_t role,dtype;
  std::uint64_t constant,per_batch,per_past,per_total;
  bool live;
};
inline std::uint64_t Add(std::uint64_t a,std::uint64_t b) {
  if(b>std::numeric_limits<std::uint64_t>::max()-a)
    throw std::invalid_argument("native epilogue extent addition overflows");
  return a+b;
}
inline std::uint64_t Multiply(std::uint64_t a,std::uint64_t b) {
  if(a && b>std::numeric_limits<std::uint64_t>::max()/a)
    throw std::invalid_argument("native epilogue extent multiplication overflows");
  return a*b;
}
inline std::uint64_t Elements(Buffer const& b,std::uint64_t batch) {
  return Add(b.constant,Multiply(b.per_batch,batch));
}
[[noreturn]] inline void Reject(std::size_t stage,std::uint32_t id,
                               char const* feature,char const* reason) {
  throw std::invalid_argument("native epilogue stage "+std::to_string(stage)+
      " "+feature+" buffer "+std::to_string(id)+": "+reason);
}
inline std::uint64_t RequiredBytes(std::uint64_t batch,std::uint64_t width,
                                    bool vector_sums) {
  if(!batch || !width || (!vector_sums && width%32))
    throw std::invalid_argument("native epilogue has invalid batch or reduction width");
  // Divide before rounding up so even malformed near-limit widths cannot wrap.
  auto columns=vector_sums?Add(width/8,width%8!=0):width/32;
  return Multiply(Multiply(batch,columns),4);
}
inline void ValidateBuffer(Buffer const& b,std::uint64_t batch,
                           std::uint64_t minimum) {
  if(!b.live || (b.role!=kInternal && b.role!=kExternal) || b.dtype!=kF32 ||
     b.per_past || b.per_total)
    throw std::invalid_argument("enabled native epilogue buffer needs live fixed-capacity f32 metadata");
  auto bytes=Multiply(Elements(b,batch),4);
  if(!bytes || bytes<minimum)
    throw std::invalid_argument("enabled native epilogue buffer has zero or insufficient extent");
}
inline std::vector<std::uint64_t> ValidateMetadata(Buffer const* buffers,
    std::size_t count,Gemm const* gemms,std::size_t stages,std::uint64_t batch) {
  if(!batch || batch>16 || (!buffers && count) || (!gemms && stages))
    throw std::invalid_argument("invalid native epilogue contract table");
  std::vector<std::uint64_t> required(count,0);
  for(std::size_t stage=0;stage<stages;++stage) {
    auto const& g=gemms[stage];if(!g.active)continue;
    if(g.batch!=batch || !g.k || g.k%32 || !g.n || !g.tile_n ||
       g.n%g.tile_n || g.tile_n%8)
      throw std::invalid_argument("native epilogue GEMM shape or batch mismatch");
    auto require=[&](std::uint32_t id,std::uint64_t width,char const* feature) {
      if(id==kAbsent)return;
      if(id>=count)Reject(stage,id,feature,"ID outside live buffer table");
      auto bytes=RequiredBytes(batch,width,g.vector_sums);
      try {ValidateBuffer(buffers[id],batch,bytes);}
      catch(std::invalid_argument const& e) {Reject(stage,id,feature,e.what());}
      if(required[id]<bytes)required[id]=bytes;
    };
    if(g.sums!=kAbsent && g.residual && !g.vector_sums && g.tile_n%32)
      Reject(stage,g.sums,"sums","legacy writer needs complete 32-column tiles");
    require(g.norm,g.k,"norm");require(g.sums,g.n,"sums");
  }
  return required;
}
template<class Allocate,class Initialize>
inline void* BindBuffer(Buffer const& b,std::uint64_t batch,
    std::uint64_t minimum,void* external,Allocate allocate,Initialize initialize) {
  if(minimum)ValidateBuffer(b,batch,minimum);
  auto elements=Elements(b,batch);void* pointer=nullptr;
  if(b.role==kExternal) {
    pointer=external;
    if(elements && !pointer)throw std::invalid_argument("unbound native external buffer");
  }else if(b.role==kInternal) {
    if(elements) {
      if(b.dtype!=kBF16 && b.dtype!=kF32 && b.dtype!=kI32)
        throw std::invalid_argument("invalid native buffer dtype");
      auto bytes=Multiply(elements,b.dtype==kBF16?2:4);
      if(bytes>std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("native buffer byte count exceeds address space");
      pointer=allocate(std::size_t(bytes));
      if(!pointer)throw std::invalid_argument("native internal allocation returned null");
      initialize(pointer,std::size_t(bytes));
    }
  }else throw std::invalid_argument("invalid native buffer role");
  if(minimum && !pointer)throw std::invalid_argument("enabled native epilogue buffer is null after binding");
  return pointer;
}
}  // namespace tilemega::codegen::native_dense::epilogue_contract
