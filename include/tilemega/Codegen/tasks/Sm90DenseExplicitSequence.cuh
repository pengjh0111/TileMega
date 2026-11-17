// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/Sm90DenseSequence.cuh>
namespace tilemega::codegen::native_dense {
template<class P,class Layer=typename P::Layer>
struct ExplicitSequence;
template<class P,class... Stages>
struct ExplicitSequence<P,std::tuple<Stages...>> {
  using Start=typename P::Start;
  using Norm=std::tuple_element_t<0,typename P::End>;
  using Head=std::tuple_element_t<1,typename P::End>;
  using Final=std::tuple_element_t<2,typename P::End>;
  using Kernels=KernelSequence<P,Start,Head,Final,Stages...>;
  static void Configure() {Kernels::Configure();Kernels::template ConfigureStage<Norm>();}
  static void Capture(typename P::Binding const* binding,SequenceState* state,int const* pasts,
      float epsilon,unsigned long long* stamps,int grid,cudaStream_t stream,bool=true) {
    Kernels::template LaunchStage<Start,true>(binding,state,pasts,0,epsilon,stamps,grid,stream);
    for(unsigned layer=0;layer<P::kLayers;++layer)
      (Kernels::template LaunchStage<Stages>(binding,state,pasts,layer,epsilon,stamps,grid,stream),...);
    Kernels::template LaunchStage<Norm>(binding,state,pasts,0,epsilon,stamps,grid,stream);
    Kernels::template LaunchStage<Head>(binding,state,pasts,0,epsilon,stamps,grid,stream);
    Kernels::template LaunchStage<Final>(binding,state,pasts,0,epsilon,stamps,grid,stream);
    sm90_dense_sequence_advance<<<1,1,0,stream>>>(state,stamps);
    Kernels::Check(cudaGetLastError());
  }
};
} // namespace tilemega::codegen::native_dense
