// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/TensorMap.h>
#include <tilemega/Codegen/DmDescriptors.h>
#include <algorithm>
namespace tilemega::codegen::executor {
// A descriptor addresses one physical NHWC image, including materialized
// halos. Per-image descriptors prevent OOB rows from reaching another batch.
struct TensorMap3DShape {
  std::uint64_t channels=0,width=0,height=0,pixel_stride_bytes=0,row_stride_bytes=0;
  std::uint32_t box_channels=0,box_width=0,box_height=0;
  static TensorMap3DShape Depthwise(ConvDesc const& c,DmBufferLayout const& l,
      unsigned rows,unsigned channels) {
    unsigned chunks=(l.physical[2]+255)/256;
    TensorMap3DShape s{l.physical[3],l.physical[2],l.physical[1],l.strides[2]*2,l.strides[1]*2,
        channels,chunks?(l.physical[2]+chunks-1)/chunks:0,
        (rows-1)*c.stride_h+(c.r-1)*c.dilation_h+1};
    if(l.rank!=4 || l.kind!=DmLayout::kNHWC || l.strides[3]!=1 || !rows ||
       c.c!=c.k || l.physical[3]<c.c || !c.stride_h || !c.stride_w ||
       !c.dilation_h || !c.dilation_w || !c.r || !c.s)
      throw std::invalid_argument("TMA depthwise requires NHWC unit-channel-stride depthwise convolution");
    s.Validate(reinterpret_cast<void*>(16));return s;
  }
  void Validate(void const* address) const {
    if(!address || reinterpret_cast<std::uintptr_t>(address)%16 || !channels || !width || !height ||
       channels%8 || pixel_stride_bytes<channels*2 || pixel_stride_bytes%16 ||
       row_stride_bytes<pixel_stride_bytes*width || row_stride_bytes%16 ||
       (box_channels!=32 && box_channels!=64 && box_channels!=128 && box_channels!=256) ||
       !box_width || box_width>256 || !box_height || box_height>256 ||
       std::uint64_t(box_channels)*box_width*box_height*2>0xffffffffu)
      throw std::invalid_argument("invalid nonswizzled BF16 3D halo tensor map");
  }
};
inline CUresult EncodeTensorMap3D(TensorMap& out,void* address,TensorMap3DShape const& s) {
  s.Validate(address);void* entry=nullptr;cudaDriverEntryPointQueryResult query;
  if(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&entry,cudaEnableDefault,&query)!=cudaSuccess ||
     query!=cudaDriverEntryPointSuccess || !entry)return CUDA_ERROR_NOT_SUPPORTED;
  auto encode=reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(entry);
  cuuint64_t dimensions[]={s.channels,s.width,s.height},strides[]={s.pixel_stride_bytes,s.row_stride_bytes};
  cuuint32_t box[]={s.box_channels,s.box_width,s.box_height},elements[]={1,1,1};
  return encode(&out.descriptor,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,address,dimensions,strides,box,elements,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace tilemega::codegen::executor
