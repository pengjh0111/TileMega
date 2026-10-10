// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/MoECombineTaskBody.h>
#include <tilemega/Backend/ServingDmEpilogue.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace tilemega::codegen;
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<tilemega::arch::CurrentArch>,
    tilemega::arch::Sm89,tilemega::arch::CurrentArch>;
constexpr int Tokens=7,Channels=96,Columns=32,Rows=17;
template<class T>T* Managed(unsigned n) {T* p;assert(cudaMallocManaged(&p,n*sizeof(T))==cudaSuccess);return p;}
void Sync() {assert(cudaGetLastError()==cudaSuccess);assert(cudaDeviceSynchronize()==cudaSuccess);}
__global__ void Combine(MoeCombineOperands p) {
  using Body=MoECombineTaskBody<Arch,8,3,64>;
  __shared__ typename Body::SharedStorage s;Body::Run(p,blockIdx.x,s);
}
using Norm=DmEpilogueStep<DmEpilogueKind::kDeferredRMSNorm,DmActivation::kRelu,
    DmGatePair::kSwiGLU,16,DmRounding::kFP32,DmRounding::kBF16,DmWriteKind::kDense,1>;
using Spec=DmEpilogueSpec<DmEpilogueProgram<Norm>>;
__global__ void Normalize(DmEpilogueArguments p) {
  using Body=ServingDmEpilogue<Arch,Spec,16,Columns,false>;
  __shared__ float tile[16*Columns+32];
  for(unsigned i=threadIdx.x;i<16*Columns;i+=128) {
    unsigned row=blockIdx.x*16+i/Columns,column=i%Columns;
    tile[i]=float(int((row*13+column*7)%101)-50)/8;
  }
  tilemega::codegen::executor::ComputeSync();
  Body::RunFromTile(tile,p,blockIdx.x,0);
}
int main() {
  auto* partial=Managed<E>(Tokens*8*Channels),*weights=Managed<E>(Tokens*8);
  auto* residual=Managed<E>(Tokens*Channels),*output=Managed<E>(Tokens*Channels);
  auto* stats=Managed<float>(Tokens*4),*squares=Managed<float>(Tokens*3+16);
  auto* normalized=Managed<E>(Rows*Columns);
  auto* row_map=Managed<unsigned>(Rows);
  auto** buffers=Managed<void*>(1);auto* layouts=Managed<DmBufferLayout>(1);auto* types=Managed<unsigned>(1);
  for(unsigned i=0;i<Tokens*8*Channels;++i)partial[i]=E(float(int((i*17)%103)-51)/16);
  for(unsigned i=0;i<Tokens*Channels;++i)residual[i]=E(float(int((i*11)%71)-35)/8);
  for(unsigned i=0;i<Tokens*8;++i)weights[i]=E(float(i%8+1)/36);
  for(unsigned i=0;i<Rows;++i)row_map[i]=(i*5+3)%Tokens;
  std::vector<float> inverse(Tokens);
  for(unsigned token=0;token<Tokens;++token) {
    double square=0;
    for(unsigned channel=0;channel<Channels;++channel) {
      float sum=0;
      for(unsigned rank=0;rank<8;++rank)sum+=float(E(float(weights[token*8+rank])*float(partial[(token*8+rank)*Channels+channel])));
      auto value=E(float(E(sum))+float(residual[token*Channels+channel]));square+=double(float(value))*float(value);
    }
    inverse[token]=1/std::sqrt(float(square/Channels)+1e-6f);
  }
  MoeCombineOperands p{partial,weights,residual,output,stats,Tokens,Channels,8*Channels,Channels,Channels,4,squares+8};
  DmEpilogueArguments q;q.buffers={buffers,layouts,types,1};q.output=normalized;
  q.m=Rows;q.n=q.output_stride=Columns;q.norm_width=Channels;q.norm_eps=1e-6f;
  q.statistic_rows=row_map;q.chain.count=1;q.chain.operations[0].kind=DmEpilogueKind::kDeferredRMSNorm;
  q.chain.operations[0].parameter[0]=0;q.chain.operations[0].output_rounding=DmRounding::kBF16;
  buffers[0]=squares+8;types[0]=1;layouts[0]={};layouts[0].rank=2;
  layouts[0].logical[0]=layouts[0].physical[0]=Tokens;
  layouts[0].logical[1]=layouts[0].physical[1]=3;layouts[0].strides[0]=3;layouts[0].strides[1]=1;
  for(unsigned epoch=0;epoch<3;++epoch) {
    for(unsigned i=0;i<Tokens*3+16;++i)squares[i]=-12345;
    Combine<<<6,128>>>(p);Sync();
    for(unsigned i=0;i<8;++i)assert(squares[i]==-12345 && squares[8+Tokens*3+i]==-12345);
    Normalize<<<2,128>>>(q);Sync();
    for(unsigned row=0;row<Rows;++row)for(unsigned column=0;column<Columns;++column) {
      float accumulator=float(int((row*13+column*7)%101)-50)/8;
      float expected=accumulator*inverse[row_map[row]];
      assert(std::abs(float(normalized[row*Columns+column])-expected)<=.016f+.016f*std::abs(expected));
    }
  }
  for(void* value:std::vector<void*>{partial,weights,residual,output,stats,squares,normalized,row_map,buffers,layouts,types})assert(cudaFree(value)==cudaSuccess);
  std::puts("MoE residual statistics to gathered deferred RMSNorm: three epochs PASS");
}
