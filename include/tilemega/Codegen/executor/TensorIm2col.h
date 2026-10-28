// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/executor/TensorMap.h>
#include <tilemega/Codegen/DmConvRuntime.h>
#include <array>

namespace tilemega::codegen::executor {
struct TensorIm2colShape {
  std::array<cuuint64_t,4> dimensions{}; // C,W,H,N
  std::array<cuuint64_t,3> strides{};
  std::array<int,2> lower{},upper{};
  std::array<cuuint32_t,4> traversal{1,1,1,1};
  cuuint32_t channels=64,pixels=64;
  void Validate(void const* address) const {
    if(!address || reinterpret_cast<std::uintptr_t>(address)%16 || channels!=64 || pixels!=64)
      throw std::invalid_argument("im2col requires an aligned BF16 C64/pixels64 tile");
    for(auto d:dimensions)if(!d || d>(1ull<<32))
      throw std::invalid_argument("im2col dimensions exceed the tensor map contract");
    if(dimensions[0]<64 || dimensions[0]%64 || strides[0]<dimensions[0]*2)
      throw std::invalid_argument("im2col requires complete 64-channel blocks");
    for(int i=0;i<3;++i)if(strides[i]%16 || strides[i]>=(1ull<<40) ||
        (i && strides[i]<strides[i-1]*dimensions[i]))
      throw std::invalid_argument("im2col physical strides overlap or are misaligned");
    for(int i=0;i<2;++i)if(lower[i]<-128 || lower[i]>127 || upper[i]<-128 || upper[i]>127 ||
        std::int64_t(dimensions[i+1])+upper[i]-lower[i]<=0 ||
        !traversal[i+1] || traversal[i+1]>8)
      throw std::invalid_argument("im2col pixel box or traversal is unsupported");
  }
  static TensorIm2colShape Build(ConvDesc const& conv,DmBufferLayout const& layout) {
    auto runtime=DmConvRuntime::Build(conv,layout,conv.n*conv.p*conv.q,conv.k,
        conv.c*conv.r*conv.s,64,1);
    // The descriptor fetches whole 64-channel blocks of the *padded* input, so
    // a 3-channel image padded to 64 (the native forward layout) qualifies
    // while an unpadded channel tail still does not.
    if(layout.physical[3]<64 || layout.physical[3]%64 ||
       runtime.geometry.channels!=layout.physical[3] ||
       conv.dilation_h>128 || conv.dilation_w>128)
      throw std::invalid_argument("im2col candidate excludes channel tails and large dilation");
    TensorIm2colShape result;
    result.dimensions={layout.physical[3],layout.physical[2],layout.physical[1],layout.physical[0]};
    result.strides={layout.strides[2]*2,layout.strides[1]*2,layout.strides[0]*2};
    result.lower={int(layout.halo_left)-int(conv.pad_w),int(layout.halo_top)-int(conv.pad_h)};
    result.upper={int(conv.w+layout.halo_left+conv.pad_w)-int(layout.physical[2])-int((conv.s-1)*conv.dilation_w),
        int(conv.h+layout.halo_top+conv.pad_h)-int(layout.physical[1])-int((conv.r-1)*conv.dilation_h)};
    result.traversal={1,conv.stride_w,conv.stride_h,1};
    result.Validate(reinterpret_cast<void*>(16));
    for(int i=0;i<2;++i) {
      auto count=(std::int64_t(result.dimensions[i+1])+result.upper[i]-result.lower[i]-1)/result.traversal[i+1]+1;
      if(count!=(i?conv.p:conv.q))throw std::invalid_argument("im2col output box differs from ConvDesc");
    }
    return result;
  }
};
inline CUresult EncodeTensorIm2col(TensorMap& out,void* address,TensorIm2colShape const& shape) {
  shape.Validate(address);
  void* entry=nullptr;cudaDriverEntryPointQueryResult query;
  auto status=cudaGetDriverEntryPoint("cuTensorMapEncodeIm2col",&entry,cudaEnableDefault,&query);
  if(status!=cudaSuccess || query!=cudaDriverEntryPointSuccess || !entry)return CUDA_ERROR_NOT_SUPPORTED;
  auto encode=reinterpret_cast<decltype(&cuTensorMapEncodeIm2col)>(entry);
  return encode(&out.descriptor,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,address,shape.dimensions.data(),
      shape.strides.data(),shape.lower.data(),shape.upper.data(),shape.channels,shape.pixels,
      shape.traversal.data(),CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace tilemega::codegen::executor
