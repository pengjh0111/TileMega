// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>

#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY || TILEMEGA_NATIVE_RELEASE_ENABLED
#error "explicit-stage experiment has no diagnostic ledger implementation"
#endif

namespace tilemega::codegen::native_dense {
struct ExplicitSmallStage {
  static constexpr bool kUsesOperandRecord=true;
  static constexpr int kRingSlots=0,kMaxPreissue=0,kSharedBytes=2048;
  static constexpr backend::Sm90DenseLayout kLayout{};
  template<int Bytes> static constexpr backend::Sm90DensePayload Payload() {return {};}
};

// Follow the original m16n8k16 A-fragment accumulation and WK reduction order.
// Only statistics move; gamma remains folded into W and output scaling stays
// before the original BF16 materialization, avoiding an extra rounded A tensor.
template<int K,int WK,unsigned Input,unsigned Inverse>
struct ExplicitRms:ExplicitSmallStage {
  static constexpr int kTasks=1;
  template<class B>
  __device__ static void PublishOperands(B const& b,int layer,int,float epsilon,
      backend::Sm90DenseOperandBank& r) {
    r.gemm.weight=b.template Get<Input>(layer);r.gemm.output=b.template Get<Inverse>();
    r.gemm.epsilon=epsilon;
  }
  template<int Prefix,class B,class Tail>
  __device__ static void Run(B const&,int,char* shared,float,
      backend::Sm90DenseRingState&,Tail,backend::Sm90DenseOperandBank const& r,bool) {
    if(blockIdx.x==0 && threadIdx.x<128) {
      auto* partial=reinterpret_cast<float*>(shared+1024);
      if(threadIdx.x<32) {
        int lane=threadIdx.x,split=lane/4,col=2*(lane%4);float sum=0;
        if(split<WK)for(int k=16*split;k<K;k+=16*WK)
          for(int j:{0,1,8,9}) {float x=float(r.gemm.weight[k+col+j]);sum+=x*x;}
        sum+=__shfl_xor_sync(0xffffffff,sum,1);
        sum+=__shfl_xor_sync(0xffffffff,sum,2);
        if(split<WK && lane%4==0)partial[split]=sum;
      }
      codegen::executor::ComputeSync();
      if(threadIdx.x==0) {
        float sum=0;for(int s=0;s<WK;++s)sum+=partial[s];
        reinterpret_cast<float*>(r.gemm.output)[0]=rsqrtf(sum/float(K)+r.gemm.epsilon);
      }
    }
    __syncthreads();
  }
};

template<bool Swiglu,int N,int U,unsigned Input,unsigned Residual,unsigned Output>
struct ExplicitOutput:ExplicitSmallStage {
  static constexpr int kTasks=(N+255)/256;
  template<class B>
  __device__ static void PublishOperands(B const& b,int layer,int,float,
      backend::Sm90DenseOperandBank& r) {
    r.gemm.weight=b.template Get<Input>();r.gemm.residual=b.template Get<Residual>(layer);
    r.gemm.output=b.template Get<Output>(layer);
  }
  template<int Prefix,class B,class Tail>
  __device__ static void Run(B const&,int,char*,float,
      backend::Sm90DenseRingState&,Tail,backend::Sm90DenseOperandBank const& r,bool) {
    if(threadIdx.x<128)for(int task=blockIdx.x;task<kTasks;task+=gridDim.x)
      for(int i=task*256+threadIdx.x;i<(task+1)*256 && i<N;i+=128) {
        if constexpr(Swiglu) {
          int gate=2*U*(i/U)+i%U;
          r.gemm.output[i]=backend::ServingEpilogueValue<backend::ServingEpilogueOp::kSwiGLU>::Apply(
              float(r.gemm.weight[gate]),float(r.gemm.weight[gate+U]));
        }else r.gemm.output[i]=Element(float(r.gemm.weight[i])+float(r.gemm.residual[i]));
      }
    __syncthreads();
  }
};

template<class Original,int Q,int D>
struct ExplicitAttentionPrepare:Original {
  static constexpr int kTasks=Original::kGroups;
  template<int Prefix,class B,class Tail>
  __device__ static void Run(B const&,int,char* shared,float,
      backend::Sm90DenseRingState&,Tail,backend::Sm90DenseOperandBank const& r,bool) {
    using Body=typename Original::Body;
    auto& s=*reinterpret_cast<typename Body::Storage*>(shared+1024);
    if(threadIdx.x<128)for(int bg=blockIdx.x;bg<kTasks;bg+=gridDim.x) {
      Body::Prepare(r.attention,s,bg,{r.attention.past,r.attention.past+1});
      codegen::executor::ComputeSync();
      for(int i=threadIdx.x*8;i<Q*D;i+=128*8) {
        int row=i/D,col=i%D;
        *reinterpret_cast<uint4*>(const_cast<Element*>(r.attention.qkv)+std::size_t(bg)*(Q+2)*D+i)=
            *reinterpret_cast<uint4 const*>(reinterpret_cast<char*>(s.query)+Body::template Swizzled<16>(row,col));
      }
      codegen::executor::ComputeSync();
    }
    __syncthreads();
  }
};

template<class Original>
struct ExplicitAttentionMerge:Original {
  static constexpr int kTasks=Original::kGroups;
  template<int Prefix,class B,class Tail>
  __device__ static void Run(B const&,int,char* shared,float,
      backend::Sm90DenseRingState&,Tail,backend::Sm90DenseOperandBank const& r,bool) {
    using Body=typename Original::Body;
    auto& s=*reinterpret_cast<typename Body::Storage*>(shared+1024);
    if(threadIdx.x<128)for(int bg=blockIdx.x;bg<kTasks;bg+=gridDim.x) {
      Body::Merge(r.attention,s,bg);
      codegen::executor::ComputeSync();
    }
    __syncthreads();
  }
};

struct ExplicitHeadPartialFinish {
  __device__ void operator()(backend::Sm90DenseOperands const& p,
      backend::Sm90DenseRingState&,float const* best,int const* index,char*,int) const {
    if(threadIdx.x==0) {
      reinterpret_cast<float*>(p.output)[blockIdx.x]=best[0];p.argmax_index[blockIdx.x]=index[0];
    }
  }
};
template<class G>
struct ExplicitHeadPartial:G {
  static constexpr int kMaxGrid=G::kTasks;
  template<int Prefix,class B,class Tail>
  __device__ static void Run(B const&,int,char* shared,float,
      backend::Sm90DenseRingState& ring,Tail tail,backend::Sm90DenseOperandBank const& r,bool streamed) {
    StageEntry<typename G::Body>::template Head<Prefix>(r.gemm,shared,&ring,tail,streamed,
        ExplicitHeadPartialFinish{});
  }
};

template<unsigned Values,unsigned Indices,unsigned Tokens>
struct ExplicitArgmax:ExplicitSmallStage {
  static constexpr int kTasks=1;
  template<class B>
  __device__ static void PublishOperands(B const& b,int,int past,float,
      backend::Sm90DenseOperandBank& r) {
    r.gemm.weight=b.template Get<Values>();
    r.gemm.argmax_index=reinterpret_cast<int*>(b.template Get<Indices>());
    r.head.tokens=reinterpret_cast<int*>(b.template Get<Tokens>());r.head.past=past;
  }
  template<int Prefix,class B,class Tail>
  __device__ static void Run(B const&,int,char*,float,
      backend::Sm90DenseRingState&,Tail,backend::Sm90DenseOperandBank const& r,bool) {
    if(blockIdx.x==0 && threadIdx.x<32) {
      float best=-INFINITY;int arg=INT_MAX;
      for(int i=threadIdx.x;i<int(gridDim.x);i+=32) {
        float v=reinterpret_cast<float const*>(r.gemm.weight)[i];int id=r.gemm.argmax_index[i];
        if(v>best || (v==best && id<arg)) {best=v;arg=id;}
      }
      for(int d=16;d;d>>=1) {
        float v=__shfl_xor_sync(0xffffffff,best,d);int id=__shfl_xor_sync(0xffffffff,arg,d);
        if(v>best || (v==best && id<arg)) {best=v;arg=id;}
      }
      if(threadIdx.x==0)r.head.tokens[r.head.past+1]=arg;
    }
    __syncthreads();
  }
};

template<unsigned Buffers,unsigned Maps,unsigned Layers,unsigned Slots,int SharedBytes,
    class Prologue,class LayerStages,class EndStages,class Head>
struct ExplicitStageProgram {
  using Binding=Bindings<Buffers,Maps,Layers,Slots>;
  using Layer=LayerStages;using End=EndStages;using Start=Prologue;
  using RoleLaunch=backend::Sm90DenseRoleLaunch<160>;
  static constexpr int kBatch=1,kCapacity=1088,kResidency=1,kThreads=160,kCluster=1,kTp=1;
  static constexpr bool kEmbed=false,kPostHead=true;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps,kLayers=Layers;
  static constexpr unsigned kPerLayer=std::tuple_size_v<Layer>,kEnd=std::tuple_size_v<End>;
  static constexpr unsigned kStages=1+Layers*kPerLayer+kEnd,kCounters=kStages;
  static constexpr int kSharedBytes=SharedBytes,kHeadMaxGrid=Head::kMaxGrid;
  template<class Stage,class Next>
  __device__ static void Boundary(Binding const& b,int past,char* shared,float epsilon,
      GridCounter* events,unsigned long long iteration,unsigned g,int layer,int next_layer) {
    auto& ring=*reinterpret_cast<backend::Sm90DenseRingState*>(shared);
    auto& record=backend::Sm90DenseOperandHeader::Bank(shared,0);
    Stage::Run(b,past,shared,epsilon,ring,backend::Sm90DenseNoTail{},record,false);
    if(threadIdx.x==0) {
      if constexpr(Next::kUsesOperandRecord)Next::PublishOperands(b,next_layer,past,epsilon,record);
      auto* counter=&events[g].arrivals;
      asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(counter),"l"(1ull):"memory");
      unsigned long long seen;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(counter):"memory");}
      while(seen<(iteration+1)*gridDim.x);
    }
    __syncthreads();
  }
  template<bool Last,std::size_t... I>
  __device__ static void RunLayer(Binding const& b,int past,char* shared,float epsilon,
      GridCounter* events,unsigned long long iteration,int layer,std::index_sequence<I...>) {
    (Boundary<std::tuple_element_t<I,Layer>,std::conditional_t<I+1==kPerLayer,
        std::conditional_t<Last,std::tuple_element_t<0,End>,std::tuple_element_t<0,Layer>>,
        std::tuple_element_t<(I+1)%kPerLayer,Layer>>>(b,past,shared,epsilon,events,iteration,
        1+layer*kPerLayer+I,layer,layer+(I+1==kPerLayer)),...);
  }
  template<std::size_t... I>
  __device__ static void RunEnd(Binding const& b,int past,char* shared,float epsilon,
      GridCounter* events,unsigned long long iteration,std::index_sequence<I...>) {
    (Boundary<std::tuple_element_t<I,End>,std::conditional_t<I+1==kEnd,Prologue,
        std::tuple_element_t<(I+1)%kEnd,End>>>(b,past,shared,epsilon,events,iteration,
        1+Layers*kPerLayer+I,0,0),...);
  }
  __device__ static void Step(Binding const& b,int past,char* shared,float epsilon,
      GridCounter* events,unsigned long long iteration,TraceState,unsigned) {
    Boundary<Prologue,std::tuple_element_t<0,Layer>>(b,past,shared,epsilon,events,iteration,0,0,0);
    #pragma unroll 1
    for(int layer=0;layer<int(Layers)-1;++layer)
      RunLayer<false>(b,past,shared,epsilon,events,iteration,layer,std::make_index_sequence<kPerLayer>{});
    RunLayer<true>(b,past,shared,epsilon,events,iteration,Layers-1,std::make_index_sequence<kPerLayer>{});
    RunEnd(b,past,shared,epsilon,events,iteration,std::make_index_sequence<kEnd>{});
  }
};
} // namespace tilemega::codegen::native_dense
