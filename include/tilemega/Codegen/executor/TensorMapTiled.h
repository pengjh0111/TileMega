// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tilemega::codegen::executor {
// Rank-N BF16 tiled map. With SWIZZLE_128B (the default) the inner box is 64
// elements and a box lands in shared memory as [..][rows][64] panels of 8x64
// swizzle atoms, which is what a WGMMA operand needs. A tile a CUDA-core body
// reads (the depthwise halo) is encoded unswizzled, so it lands row-major.
// Elements outside the global extent are written as zeros either way.
template<int Rank>
struct TiledMapShape {
  std::uint64_t dims[Rank];
  std::uint64_t strides[Rank-1];  // bytes, for dimensions 1..Rank-1
  std::uint32_t box[Rank];
};
template<int Rank>
inline void EncodeTiledMap(CUtensorMap& out,void const* address,TiledMapShape<Rank> const& s,
                           bool swizzled=true) {
  static_assert(Rank>=2 && Rank<=5);
  if(!address || reinterpret_cast<std::uintptr_t>(address)%16 || (swizzled && s.box[0]!=64))
    throw std::invalid_argument("swizzled BF16 tensor map needs a 16-byte base and a 64-element inner box");
  if(!swizzled && s.box[0]*2%16)
    throw std::invalid_argument("an unswizzled BF16 box needs a 16-byte inner row");
  for(int i=0;i<Rank;++i)
    if(!s.dims[i] || !s.box[i] || s.box[i]>256)throw std::invalid_argument("invalid tensor map extent");
  for(int i=0;i<Rank-1;++i)
    if(s.strides[i]%16 || s.strides[i]>=(std::uint64_t(1)<<40))
      throw std::invalid_argument("tensor map strides must be 16-byte multiples");
  void* entry=nullptr;cudaDriverEntryPointQueryResult query;
  if(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&entry,cudaEnableDefault,&query)!=cudaSuccess ||
     query!=cudaDriverEntryPointSuccess || !entry)
    throw std::runtime_error("cuTensorMapEncodeTiled is unavailable");
  auto encode=reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(entry);
  cuuint64_t dims[Rank],strides[Rank-1];cuuint32_t box[Rank],elements[Rank];
  for(int i=0;i<Rank;++i) {dims[i]=s.dims[i];box[i]=s.box[i];elements[i]=1;}
  for(int i=0;i<Rank-1;++i)strides[i]=s.strides[i];
  auto status=encode(&out,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,Rank,const_cast<void*>(address),dims,strides,
      box,elements,CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzled?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(status!=CUDA_SUCCESS)
    throw std::invalid_argument("cuTensorMapEncodeTiled rejected a rank-"+std::to_string(Rank)+
        " map (error "+std::to_string(int(status))+")");
}
}  // namespace tilemega::codegen::executor
