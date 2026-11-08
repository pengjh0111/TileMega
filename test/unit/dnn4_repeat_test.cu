// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90F3Program.cuh>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tilemega::codegen::forward_f3;
using namespace tilemega::backend;
using E=cutlass::bfloat16_t;
#ifdef DNN4_PP_TEST
constexpr bool kPp=true;
#else
constexpr bool kPp=false;
#endif
#ifndef DNN4_PP_TILE_M
#define DNN4_PP_TILE_M 128
#endif
constexpr int kTileM=kPp?DNN4_PP_TILE_M:128;
constexpr int kM=kTileM*67-3,kN=256,kCount=kPp?29:28,kMaps=kPp?14:12;
#ifdef DNN4_ODD_REPEAT_TEST
constexpr bool kOdd=true;
#else
constexpr bool kOdd=false;
#endif
void Check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
template<class T>struct Device {
 T* data;std::size_t count;
 explicit Device(std::size_t n):count(n){Check(cudaMalloc(&data,n*sizeof(T)));}
 ~Device(){cudaFree(data);}
 void Put(std::vector<T> const& x){Check(cudaMemcpy(data,x.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> Get(){std::vector<T>x(count);Check(cudaMemcpy(x.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
struct Shape {
 static constexpr int kOutputStride=kN,kResidualStride=0,kNormWidth=0;
 static constexpr float kEpsilon=0;
};
template<unsigned Map,unsigned Input,unsigned Weight,unsigned Output>struct Bind {
 static constexpr unsigned kMap=Map,kInput=Input,kWeight=Weight,kOutput=Output;
 static constexpr unsigned kResidualMap=kAbsent,kOutputMap=kAbsent,kVector0=kPp?28:kAbsent,
   kVector1=kAbsent,kVector2=kAbsent,kVector3=kAbsent,kResidual=kAbsent,kStats=kAbsent,
   kRowStats=kAbsent,kRowPartials=kAbsent,kRowCounters=kAbsent,kChannelPartial=kAbsent,
   kPartial=kAbsent,kCounters=kAbsent;
};
template<int TN>using Plain=Sm90F3Chain<Sm90F3SideLayout<128,TN,false,0,false>>;
template<int TN>using Bias=Sm90F3Chain<Sm90F3SideLayout<kTileM/2,TN,false,1,false>,
    Sm90F3Step<Sm90ForwardOp::kBias,0>>;
using ChainA=std::conditional_t<kPp,Bias<128>,Plain<128>>;
using ChainB=std::conditional_t<kPp,Bias<64>,Plain<64>>;
template<int Slots>using A=GemmStage<Sm90F3Gemm<kTileM,128,Slots,1,Sm90F3Mode::kNormal,
    kM,kN,kN,kN,ChainA,Sm90F3NoConv,Sm90F3NoPool,64,false,false,kPp>,Bind<0,1,16,2>,Shape>;
template<int Slots>using B=GemmStage<Sm90F3Gemm<kTileM,64,Slots,1,Sm90F3Mode::kNormal,
    kM,kN,kN,kN,ChainB,Sm90F3NoConv,Sm90F3NoPool,64,false,false,kPp>,Bind<1,2,17,3>,Shape>;
using LoopA=RepeatStage<A<3>,A<6>>;
using LoopB=RepeatStage<B<4>,B<4>,B<8>>;
template<int Slots>using C=GemmStage<Sm90F3Gemm<128,64,Slots,1,Sm90F3Mode::kNormal,
    kM,kN,kN,kN,Plain<64>>,Bind<2,3,18,4>,Shape>;
using LoopC=RepeatStage<C<4>,C<4>,C<8>>;
template<unsigned Input,unsigned Output>struct Copy {
 struct Kind {static constexpr int kSideOffset=0;};
 static constexpr int kSharedBytes=0;
 static constexpr bool kCanPreissue=false;
 template<class Binding>__device__ static void Produce(Binding const&,Sm90F3Ring&,
     char*,char*,Sm90F3Phase&,int){}
 template<class Binding>__device__ static void Consume(Binding const& b,Sm90F3Ring&,
     char*,char*,Sm90F3Phase&,Sm90F3Header*,int) {
   auto* src=b.template Get<Input>();auto* dst=b.template Get<Output>();
   for(int i=int(blockIdx.x)*256+int(threadIdx.x);i<kM*kN;i+=int(gridDim.x)*256)dst[i]=src[i];
 }
};
struct TestTables {
 static constexpr unsigned kRepetitions=kOdd?4:6,kStages=kOdd?3:2;
 inline static const std::array<unsigned,kRepetitions*(kCount+kMaps)> indices=[] {
   std::array<unsigned,kRepetitions*(kCount+kMaps)> x{};
   for(unsigned r=0;r<kRepetitions;++r) {
     auto* row=x.data()+r*(kCount+kMaps);
     for(unsigned i=0;i<kCount;++i)row[i]=i;
     for(unsigned i=0;i<kMaps;++i)row[kCount+i]=i;
     for(unsigned j=0;j<=kStages;++j)row[1+j]=1+kStages*r+j;
     for(unsigned j=0;j<kStages;++j) {
       row[16+j]=16+kStages*r+j;row[kCount+j]=kStages*r+j;
     }
   }
   return x;
 }();
 inline static const std::array<RepeatWeight,15> weights=[] {
   std::array<RepeatWeight,15> x{};
   for(unsigned i=0;i<12;++i)x[i+1]={16+i,2*kN*kN};
   if(kPp) {x[0]={16,2*kN*kN};x[13]={17,2*kN*kN};}
   return x;
 }();
};
namespace tilemega::codegen::forward_f3 {
template<>struct RepeatTables<6,LoopA,LoopB>:TestTables {};
template<>struct RepeatTables<4,LoopA,LoopB,LoopC>:TestTables {};
}
using Before=std::conditional_t<kPp,GemmStage<Sm90F3Gemm<128,128,3,1,Sm90F3Mode::kNormal,
    kM,kN,kN,kN,ChainA>,Bind<12,0,16,1>,Shape>,Copy<0,1>>;
using After=std::conditional_t<kPp,GemmStage<Sm90F3Gemm<128,64,4,1,Sm90F3Mode::kNormal,
    kM,kN,kN,kN,ChainB>,Bind<13,13,17,14>,Shape>,Copy<13,14>>;
using Repeated=std::conditional_t<kOdd,Repeat<4,LoopA,LoopB,LoopC>,Repeat<6,LoopA,LoopB>>;
using P=Program<1,kCount,kMaps,Before,Repeated,After,Copy<14,15>>;
int main()try {
 std::array<E*,kCount> buffers{};
 for(unsigned i=0;i<kCount;++i)Check(cudaMalloc(&buffers[i],sizeof(E)*(i<16?kM*kN:kN*kN)));
 std::vector<E> input(kM*kN);
 for(unsigned i=0;i<input.size();++i)input[i]=E(float((i*3+i/kN)%13+1)/128);
 Check(cudaMemcpy(buffers[0],input.data(),input.size()*2,cudaMemcpyHostToDevice));
 P::Binding binding{};
 for(unsigned i=0;i<12;++i) {
   unsigned tn=(kOdd?i%3!=0:i%2)?64:128;std::vector<E> packed(kN*kN,E(0));
   float scale=i%3==0?.5f:1.f;
   for(unsigned n=0;n<kN;++n) {
     unsigned row=n%tn,col=n%64;
     unsigned offset=((n/tn)*(kN/64)+n/64)*tn*64+row*64+8*((col/8)^(row%8))+col%8;
     packed[offset]=E(scale);
   }
   Check(cudaMemcpy(buffers[16+i],packed.data(),packed.size()*2,cudaMemcpyHostToDevice));
   tilemega::codegen::executor::EncodeTiledMap<2>(binding.maps[i],buffers[1+i],
       {{kN,kM},{kN*2},{64,kPp?kTileM/2:128}});
 }
 if(kPp) {
   std::vector<float> bias(kN,.125f);
   Check(cudaMemcpy(buffers[28],bias.data(),bias.size()*sizeof(float),cudaMemcpyHostToDevice));
   tilemega::codegen::executor::EncodeTiledMap<2>(binding.maps[12],buffers[0],{{kN,kM},{kN*2},{64,128}});
   tilemega::codegen::executor::EncodeTiledMap<2>(binding.maps[13],buffers[13],{{kN,kM},{kN*2},{64,128}});
 }
 Device<E*> pointers(kCount);pointers.Put(std::vector<E*>(buffers.begin(),buffers.end()));binding.buffers=pointers.data;
 auto const& table=P::RepeatIndices();Device<unsigned> indices(table.size());
 indices.Put(std::vector<unsigned>(table.begin(),table.end()));binding.repeat_indices=indices.data;
 auto const& info=P::RepeatWeights();Device<RepeatWeight> weights(info.size());
 weights.Put(std::vector<RepeatWeight>(info.begin(),info.end()));binding.repeat_weights=weights.data;
 Device<ForwardCounter> events(P::kStages);events.Put(std::vector<ForwardCounter>(P::kStages));
 auto kernel=sm90_forward_f3_kernel<P>;
 Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,P::kSharedBytes));
 for(unsigned launch=1;launch<=5;++launch) {
   kernel<<<132,384,P::kSharedBytes>>>(binding,events.data,launch);Check(cudaDeviceSynchronize());
 }
 unsigned long long hash=14695981039346656037ull;
 auto reference=input;
 for(unsigned i=1;i<16;++i) {
   bool gemm=(i>=2 && i<=13) || (kPp && (i==1 || i==14));
   float scale=i==1 || (i>=2 && i<=13 && (i-2)%3==0)?.5f:1.f;
   if(gemm)for(auto& x:reference)x=E(float(x)*scale+(kPp?.125f:0.f));
   std::vector<E> actual(input.size());Check(cudaMemcpy(actual.data(),buffers[i],actual.size()*2,cudaMemcpyDeviceToHost));
   for(unsigned j=0;j<input.size();++j) {
     E expected=reference[j];
     if(actual[j].raw()!=expected.raw())throw std::runtime_error("Repeat remap/order mismatch stage="+std::to_string(i)+" element="+std::to_string(j));
     hash^=actual[j].raw();hash*=1099511628211ull;
   }
 }
 for(auto* p:buffers)Check(cudaFree(p));
 std::cout<<"Repeat ring wrap, first/last spanning variants, cross-repetition preissue, generic readers PASS\noutput_hash="<<std::hex<<hash<<'\n';
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
