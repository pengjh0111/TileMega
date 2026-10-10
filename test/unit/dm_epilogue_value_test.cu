// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/DmEpilogueValue.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace tilemega::codegen;
using namespace tilemega::backend;
using Arch=tilemega::arch::Sm80;
template<DmEpilogueKind Kind,DmActivation Act=DmActivation::kRelu,
         DmRounding Input=DmRounding::kFP32,DmRounding Output=DmRounding::kBF16,
         DmGatePair Gate=DmGatePair::kSwiGLU>
using Step=DmEpilogueStep<Kind,Act,Gate,16,Input,Output,DmWriteKind::kDense,1>;
template<class S> using Value=DmEpilogueValue<Arch,S>;
struct Record {
  std::uint32_t mode;
  float value,partner,reference;
  DmEpilogueInputs inputs;
};
static_assert(sizeof(Record)==56);

template<class Program>
struct ScalarVisitor {
  float value,partner;
  DmEpilogueInputs const& inputs;
  template<unsigned Index,class S,bool Gated>
  __device__ void Apply() {
    value=Value<S>::Apply(value,inputs,partner);
    if constexpr(DmEpilogueWalk<Program>::kGates && !Gated && S::kKind!=DmEpilogueKind::kGatePair)
      partner=Value<S>::Apply(partner,inputs);
  }
};

template<int Mode> __global__ void Evaluate(Record const* records,float* output,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
  auto const& r=records[i]; float value=r.value;
  constexpr auto BF=DmRounding::kBF16;
  if constexpr(Mode==0) {
    value=Value<Step<DmEpilogueKind::kBias,DmActivation::kRelu,DmRounding::kFP32,DmRounding::kFP32>>::Apply(value,r.inputs);
    value=Value<Step<DmEpilogueKind::kActivation>>::Apply(value,r.inputs);
  } else if constexpr(Mode==1) {
    value=Value<Step<DmEpilogueKind::kScale,DmActivation::kRelu,DmRounding::kFP32,DmRounding::kFP32>>::Apply(value,r.inputs);
    value=Value<Step<DmEpilogueKind::kActivation,DmActivation::kRelu6>>::Apply(value,r.inputs);
  } else if constexpr(Mode>=2 && Mode<=5) {
    constexpr DmActivation act=Mode==2 ? DmActivation::kGeluErf :
        Mode==3 ? DmActivation::kGeluTanh : Mode==4 ? DmActivation::kTanh : DmActivation::kSilu;
    value=Value<Step<DmEpilogueKind::kActivation,act,BF>>::Apply(value,r.inputs);
  } else if constexpr(Mode==6 || Mode==7) {
    constexpr auto gate=Mode==6 ? DmGatePair::kSimpleGate : DmGatePair::kSwiGLU;
    value=Value<Step<DmEpilogueKind::kGatePair,DmActivation::kRelu,BF,BF,gate>>::Apply(value,r.inputs,r.partner);
  } else if constexpr(Mode==8 || Mode==9) {
    constexpr auto rounding=Mode==8 ? BF : DmRounding::kFP32;
    value=Value<Step<DmEpilogueKind::kResidual,DmActivation::kRelu,rounding>>::Apply(value,r.inputs);
  } else if constexpr(Mode==10) {
    value=Value<Step<DmEpilogueKind::kDeferredRMSNorm>>::Apply(value,r.inputs);
  } else if constexpr(Mode==11) {
    value=Value<Step<DmEpilogueKind::kDeferredLayerNorm>>::Apply(value,r.inputs);
  } else if constexpr(Mode==12) {
    value=Value<Step<DmEpilogueKind::kResidualLN,DmActivation::kRelu,BF>>::Apply(value,r.inputs);
  } else if constexpr(Mode==13) {
    using Program=DmEpilogueProgram<
        Step<DmEpilogueKind::kBias,DmActivation::kRelu,DmRounding::kFP32,DmRounding::kFP32>,
        Step<DmEpilogueKind::kResidual,DmActivation::kRelu,BF>,
        Step<DmEpilogueKind::kActivation,DmActivation::kRelu,BF>>;
    ScalarVisitor<Program> visitor{value,r.partner,r.inputs};
    DmEpilogueWalk<Program>::Run(visitor); value=visitor.value;
  } else if constexpr(Mode==14) {
    using Program=DmEpilogueProgram<Step<DmEpilogueKind::kBias>,
        Step<DmEpilogueKind::kActivation,DmActivation::kGeluErf,BF>>;
    ScalarVisitor<Program> visitor{value,r.partner,r.inputs};
    DmEpilogueWalk<Program>::Run(visitor); value=visitor.value;
  } else if constexpr(Mode==15) {
    using Program=DmEpilogueProgram<Step<DmEpilogueKind::kBias>,
        Step<DmEpilogueKind::kGatePair,DmActivation::kRelu,BF,BF,DmGatePair::kSimpleGate>>;
    ScalarVisitor<Program> visitor{value,r.partner,r.inputs};
    DmEpilogueWalk<Program>::Run(visitor); value=visitor.value;
  }
  output[i]=value;
}
void Check(cudaError_t error) {
  if(error!=cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  std::ifstream input(argv[1],std::ios::binary); if(!input)return 2;
  std::vector<Record> records; Record record;
  while(input.read(reinterpret_cast<char*>(&record),sizeof(record)))records.push_back(record);
  if(!input.eof() || input.gcount()!=0 || records.empty())return 2;
  for(auto const& r:records)if(r.mode>=16)return 2;
  unsigned tested=0;
  for(unsigned mode=0;mode<16;++mode) {
    std::vector<Record> group; for(auto const& r:records)if(r.mode==mode)group.push_back(r);
    if(group.empty())return 2;
    Record* device; float* output; std::vector<float> actual(group.size());
    Check(cudaMalloc(&device,group.size()*sizeof(Record))); Check(cudaMalloc(&output,group.size()*sizeof(float)));
    Check(cudaMemcpy(device,group.data(),group.size()*sizeof(Record),cudaMemcpyHostToDevice));
    switch(mode) {
#define DM_CASE(N) case N: Evaluate<N><<<(group.size()+127)/128,128>>>(device,output,group.size());break;
      DM_CASE(0) DM_CASE(1) DM_CASE(2) DM_CASE(3) DM_CASE(4) DM_CASE(5) DM_CASE(6) DM_CASE(7)
      DM_CASE(8) DM_CASE(9) DM_CASE(10) DM_CASE(11) DM_CASE(12) DM_CASE(13) DM_CASE(14) DM_CASE(15)
#undef DM_CASE
    }
    Check(cudaGetLastError()); Check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    float max_error=0;
    for(unsigned i=0;i<group.size();++i) {
      float ref=group[i].reference,error=std::fabs(actual[i]-ref);
      max_error=std::fmax(max_error,error);
      if(!std::isfinite(actual[i]) || error>1.6e-2f+1.6e-2f*std::fabs(ref)) {
        std::fprintf(stderr,"mode=%u case=%u actual=%g reference=%g error=%g\n",mode,i,actual[i],ref,error);return 3;
      }
    }
    std::printf("mode=%u cases=%zu max_abs_error=%g\n",mode,group.size(),max_error);
    tested+=group.size(); Check(cudaFree(device)); Check(cudaFree(output));
  }
  std::printf("DM epilogue values: %u PyTorch reference cases passed across 16 variants\n",tested);
}
