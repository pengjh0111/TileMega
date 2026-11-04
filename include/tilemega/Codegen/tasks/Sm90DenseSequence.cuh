// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
#include <stdexcept>

#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY || TILEMEGA_NATIVE_RELEASE_ENABLED
#error "kernel sequence does not implement megakernel trace ledgers"
#endif

namespace tilemega::codegen::native_dense {
struct SequenceState {unsigned step;unsigned long long iteration;};

__global__ void sm90_dense_sequence_begin(SequenceState* state,unsigned first,
    unsigned long long iteration) {
  *state={first,iteration};
}
__global__ void sm90_dense_sequence_advance(SequenceState* state,unsigned long long* stamps) {
  stamps[state->step+1]=Now();++state->step;++state->iteration;
}

template<class P,class Stage,bool First=false>
__global__ __launch_bounds__(P::kThreads,P::kResidency)
void sm90_dense_sequence_stage(typename P::Binding const* binding,SequenceState const* state,
    int const* pasts,int layer,float epsilon,unsigned long long* stamps) {
  static_assert(Stage::kGroupTiles==0,"kernel sequence requires aggregate boundaries");
  extern __shared__ __align__(1024) char shared[];
  auto* rings=reinterpret_cast<backend::Sm90DenseRingState*>(shared);
  rings[0].Initialize();rings[1].Initialize();
  if constexpr(P::kCluster>1)P::ClusterInitialize(shared);
  auto const& b=*binding;
  int past=pasts[state->step];
  auto& record=backend::Sm90DenseOperandHeader::Bank(shared,0);
  if(threadIdx.x==0) {
    if constexpr(First)if(blockIdx.x==0)stamps[state->step]=Now();
    if constexpr(Stage::kUsesOperandRecord)
      Stage::PublishOperands(b,layer,past,epsilon,record,state->iteration);
  }
  __syncthreads();
  // Every kernel starts with empty rings; no predecessor issued its prefix.
  Stage::Run(b,past,shared,epsilon,rings[0],backend::Sm90DenseNoTail{},record,false);
  if constexpr(P::kCluster>1)codegen::executor::Async<arch::Sm90>::ClusterSync();
  rings[0].Invalidate();rings[1].Invalidate();
}

template<class P,class Prologue,class Head,class Final,class... LayerStages>
struct KernelSequence {
  static_assert(P::kTp==1 && !P::kEmbed && P::kPostHead);
  static_assert(sizeof...(LayerStages)==P::kPerLayer);
  static void Check(cudaError_t result) {
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
  }
  template<class Stage,bool First=false>
  static void ConfigureStage() {
    auto kernel=sm90_dense_sequence_stage<P,Stage,First>;
    Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,P::kSharedBytes));
    int resident=0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,P::kThreads,P::kSharedBytes));
    if(resident<P::kResidency)
      throw std::invalid_argument("sequence stage exceeds measured residency");
  }
  static void Configure() {
    ConfigureStage<Prologue,true>();(ConfigureStage<LayerStages>(),...);
    ConfigureStage<Head>();ConfigureStage<Final>();
  }
  template<class Stage,bool First=false>
  static void LaunchStage(typename P::Binding const* binding,SequenceState const* state,
      int const* pasts,int layer,float epsilon,unsigned long long* stamps,int grid,cudaStream_t stream) {
    if constexpr(P::kCluster>1) {
      cudaLaunchConfig_t config{};config.gridDim=dim3(grid);config.blockDim=dim3(P::kThreads);
      config.dynamicSmemBytes=P::kSharedBytes;config.stream=stream;
      cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim.x=P::kCluster;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
      config.attrs=&attribute;config.numAttrs=1;
      Check(cudaLaunchKernelEx(&config,sm90_dense_sequence_stage<P,Stage,First>,
          binding,state,pasts,layer,epsilon,stamps));
    }else {
      sm90_dense_sequence_stage<P,Stage,First><<<grid,P::kThreads,P::kSharedBytes,stream>>>(
          binding,state,pasts,layer,epsilon,stamps);
      Check(cudaGetLastError());
    }
  }
  static void Capture(typename P::Binding const* binding,SequenceState* state,int const* pasts,
      float epsilon,unsigned long long* stamps,int grid,cudaStream_t stream) {
    LaunchStage<Prologue,true>(binding,state,pasts,0,epsilon,stamps,grid,stream);
    for(unsigned layer=0;layer<P::kLayers;++layer)
      (LaunchStage<LayerStages>(binding,state,pasts,layer,epsilon,stamps,grid,stream),...);
    LaunchStage<Head>(binding,state,pasts,0,epsilon,stamps,grid,stream);
    LaunchStage<Final>(binding,state,pasts,0,epsilon,stamps,grid,stream);
    sm90_dense_sequence_advance<<<1,1,0,stream>>>(state,stamps);Check(cudaGetLastError());
  }
};
} // namespace tilemega::codegen::native_dense
