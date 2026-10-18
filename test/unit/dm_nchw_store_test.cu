// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_EPILOGUE_FUSED 3
#include <tilemega/Backend/ServingDmEpilogue.h>
#include <cassert>
#include <cstdio>
#include <vector>

using namespace tilemega;
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
template<class T>T* Allocate(std::size_t count) {
  T* result=nullptr;assert(cudaMallocManaged(&result,count*sizeof(T))==cudaSuccess);return result;
}
__host__ __device__ float Value(unsigned row,unsigned column) {
  return float(int((row*7+column*13)%97)-48)/128;
}
template<DmRounding R>
__global__ void Run(backend::DmEpilogueArguments p) {
  using Spec=DmEpilogueSpec<DmEpilogueProgram<>,DmWriteKind::kNCHW,1,R>;
  using Body=backend::ServingDmEpilogue<arch::Sm90,Spec,16,16>;
  __shared__ float tile[16*16];
  for(unsigned i=threadIdx.x;i<256;i+=128)
    tile[Body::Index(i/16,i%16)]=Value(blockIdx.x*16+i/16,blockIdx.y*16+i%16);
  __syncthreads();Body::RunFromTile(tile,p,blockIdx.x,blockIdx.y);
}
template<DmRounding R>void Check(unsigned batch,unsigned channels,unsigned width,bool padded) {
  using T=std::conditional_t<R==DmRounding::kBF16,E,float>;
  auto* layout=Allocate<DmBufferLayout>(1);layout->rank=4;
  unsigned shape[]={batch,channels,1,width};
  for(unsigned i=0;i<4;++i)layout->logical[i]=layout->physical[i]=shape[i];
  layout->strides[3]=1;layout->strides[2]=width;
  layout->strides[1]=width+(padded?4:0);layout->strides[0]=channels*layout->strides[1]+(padded?8:0);
  unsigned size=batch*layout->strides[0];auto* output=Allocate<T>(size+16);
  for(unsigned i=0;i<size+16;++i)output[i]=T(-101.25f);
  backend::DmEpilogueArguments p;p.buffers={nullptr,layout,nullptr,1};
  p.chain.store_rounding=R;p.write.kind=DmWriteKind::kNCHW;p.write.layout=0;
  p.output=output+8;p.m=batch*width;p.n=channels;p.output_stride=channels;
  Run<R><<<dim3((p.m+15)/16,(channels+15)/16),128>>>(p);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  std::vector<bool> written(size);
  for(unsigned n=0;n<batch;++n)for(unsigned c=0;c<channels;++c)for(unsigned pixel=0;pixel<width;++pixel) {
    auto index=n*layout->strides[0]+c*layout->strides[1]+pixel;
    assert(output[8+index]==T(Value(n*width+pixel,c)));written[index]=true;
  }
  for(unsigned i=0;i<size;++i)if(!written[i])assert(output[8+i]==T(-101.25f));
  for(unsigned i=0;i<8;++i)assert(output[i]==T(-101.25f) && output[8+size+i]==T(-101.25f));
  assert(cudaFree(output)==cudaSuccess && cudaFree(layout)==cudaSuccess);
}
int main() {
  unsigned cases=0;
  for(unsigned batch:{1,2})for(unsigned channels:{3,17})for(unsigned width:{39,40})for(bool pad:{false,true}) {
    Check<DmRounding::kBF16>(batch,channels,width,pad);
    Check<DmRounding::kFP32>(batch,channels,width,pad);cases+=2;
  }
  std::printf("NCHW vector stores: %u BF16/FP32 batch/tail/padded/fallback cases PASS\n",cases);
}
