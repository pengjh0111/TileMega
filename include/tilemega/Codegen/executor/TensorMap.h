// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tilemega::codegen::executor {
struct TensorMapShape {
  std::uint64_t columns=0,rows=0,stride_bytes=0;
  std::uint32_t box_columns=64,box_rows=0;
  void Validate(void const* address) const {
    if(!address || reinterpret_cast<std::uintptr_t>(address)%16 || columns<box_columns || !rows ||
       stride_bytes<columns*2 || stride_bytes%16 || box_columns!=64 ||
       !box_rows || box_rows>256)
      throw std::invalid_argument("invalid BF16 tensor map address, dimensions, stride or 128-byte swizzle box");
  }
};
struct alignas(64) TensorMap {
  CUtensorMap descriptor;
};
static_assert(alignof(TensorMap)>=64);
inline CUresult EncodeTensorMap(TensorMap& out,void* address,TensorMapShape const& shape) {
  shape.Validate(address);
  void* entry=nullptr;cudaDriverEntryPointQueryResult query;
  auto status=cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&entry,cudaEnableDefault,&query);
  if(status!=cudaSuccess || query!=cudaDriverEntryPointSuccess || !entry)
    return CUDA_ERROR_NOT_SUPPORTED;
  auto encode=reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(entry);
  cuuint64_t dimensions[]={shape.columns,shape.rows};
  cuuint64_t strides[]={shape.stride_bytes};
  cuuint32_t box[]={shape.box_columns,shape.box_rows},elements[]={1,1};
  return encode(&out.descriptor,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,address,
      dimensions,strides,box,elements,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace tilemega::codegen::executor
