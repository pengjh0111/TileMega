// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90DenseEpilogue.h>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <cute/tensor.hpp>

namespace tilemega::backend {

struct Sm90DenseOperands {
  cutlass::bfloat16_t const* activation;
  cutlass::bfloat16_t const* weight;
  cutlass::bfloat16_t const* residual;
  cutlass::bfloat16_t* output;
  int* argmax_index;
  float const* norm_sums;
  float* output_sums;
};


template<int Batch,int N,int K,int TN,int TK,int Slots,int WarpsN,int WarpsK,
         ServingEpilogueOp Op,int U,int RegisterStages=1>
struct Sm90DenseGemm {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(Batch>0 && Batch<=16 && N%TN==0 && TN%8==0 && TK%64==0 && K%TK==0);
  static_assert(WarpsN*WarpsK==4 && TN%(8*WarpsN)==0 && Slots>=2 && Slots<=8);
  static_assert((RegisterStages==1 || RegisterStages==2 || RegisterStages==4) &&
                RegisterStages<=TK/(16*WarpsK));
  static_assert(Op!=ServingEpilogueOp::kPartial,"dense decode never splits K across tasks");
  static constexpr int kThreads=160,kIterations=K/TK;
  static constexpr int kOperandBytes=2*Slots*TK*(16+TN);
  static constexpr int kSharedBytes=1024+kOperandBytes+4*WarpsK*16*TN+128;
  using Atom=decltype(cute::composition(cute::Swizzle<3,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::_64>,cute::Stride<cute::_64,cute::_1>>{}));
  using LayoutA=decltype(cute::tile_to_shape(Atom{},cute::Shape<cute::_16,cute::Int<TK>>{}));
  using LayoutB=decltype(cute::tile_to_shape(Atom{},cute::Shape<cute::Int<TN>,cute::Int<TK>>{}));
  using Mma=decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      cute::Layout<cute::Shape<cute::_1,cute::Int<WarpsN>,cute::_1>>{},
      cute::Tile<cute::_16,cute::Int<TN>,cute::_16>{}));
  using CopyA=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>;
  using CopyB=std::conditional_t<(TN/WarpsN<=16 || (TN/WarpsN)%16!=0),
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N,Element>,CopyA>;
  static_assert(cute::cosize_v<LayoutA> ==16*TK && cute::cosize_v<LayoutB> ==TN*TK);

  template<int Bytes>
  // Raw operand pointers share the transfer loop across layer-specific bindings.
  __device__ static __forceinline__ void Preissue(Element const* weight,char* shared,
      Sm90DenseRingState& state) {
    static_assert(Bytes>0 && Bytes%16==0 &&
        Bytes<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    auto* b=reinterpret_cast<Element*>(shared+1024)+Slots*16*TK;
    int remaining=Bytes;
    #pragma unroll
    for(int slot=0;slot<Slots && slot<kIterations;++slot) {
      int part=min(remaining,2*TN*TK);remaining-=part;
      if(part) {
        // Preissue owns the alternate bank: its previous users joined before
        // the preceding V3 boundary. Current arena overlap is checked by Tail.
        Async::ExpectTx(state.full+slot,2*TN*TK);
        Async::Bulk(b+slot*TN*TK,weight+(std::size_t(blockIdx.x)*kIterations+slot)*TN*TK,
            part,state.full+slot);
      }
    }
  }
  template<int Bytes,int Prefix>
  __device__ static __forceinline__ void Prefetch(Element const* weight) {
    constexpr int bytes=Bytes<2*TN*K-Prefix?Bytes:2*TN*K-Prefix;
    if constexpr(bytes>0)Async::Prefetch(weight+std::size_t(blockIdx.x)*TN*K+Prefix/2,bytes);
  }

  template<int Prefix>
  __device__ static __noinline__ void Produce(Sm90DenseOperands p,
      Sm90DenseRingState& state,Element* a,Element* b,int tasks,unsigned phase) {
    int lane=threadIdx.x-128;
    for(int sequence=0;sequence<tasks*kIterations;++sequence) {
      int slot=sequence%Slots,task=int(blockIdx.x)+(sequence/kIterations)*int(gridDim.x);
      int tile=sequence%kIterations;
      Async::Wait(state.empty+slot,(phase>>slot)&1);
      if(lane==0) {
        // The first task starts at slot zero in every invocation. Its incoming
        // budget is fixed by the same Placement that issued the transaction.
        unsigned prefix=0;
        if constexpr(Prefix>0)
          if(sequence<(Prefix+2*TN*TK-1)/(2*TN*TK))
            prefix=min(Prefix-sequence*2*TN*TK,2*TN*TK);
        if(!prefix)Async::ExpectTx(state.full+slot,2*TN*TK);
        if(prefix<2*TN*TK)Async::Bulk(b+slot*TN*TK+prefix/2,
            p.weight+(std::size_t(task)*kIterations+tile)*TN*TK+prefix/2,
            2*TN*TK-prefix,state.full+slot);
      }
      for(int v=lane*8;v<Batch*TK;v+=32*8) {
        int row=v/TK,col=v%TK;
        auto* source=p.activation+std::size_t(row)*K+tile*TK+col;
        Async::Copy16Bytes(a+slot*16*TK+LayoutA{}(row,col),source,16);
      }
      // One bulk arrival plus 32 cp.async arrivals make the full barrier cover
      // both operands. Inactive rows remain zero throughout this invocation.
      Async::CompleteCopies(state.full+slot);
      phase^=1u<<slot;
    }
    __syncwarp();
    if(lane==0 && tasks)for(int slot=0;slot<Slots;++slot)
      state.producer_generations[slot]=(phase>>slot)&1;
  }

  template<int Split,class Accumulator>
  __device__ static __forceinline__ void MmaSlice(Element* a,Element* b,Accumulator& accum) {
    using namespace cute;
    static_assert(Split>=0 && Split<WarpsK);
    int lane=int(threadIdx.x)%(32*WarpsN);
    Mma mma;auto thread=mma.get_slice(lane);
    auto copy_a=make_tiled_copy_A(CopyA{},mma);
    auto copy_b=make_tiled_copy_B(CopyB{},mma);
    auto sa=make_tensor(make_smem_ptr(a),LayoutA{});
    auto sb=make_tensor(make_smem_ptr(b),LayoutB{});
    auto ra=thread.partition_fragment_A(sa);auto rb=thread.partition_fragment_B(sb);
    auto as=copy_a.get_slice(lane).partition_S(sa);auto bs=copy_b.get_slice(lane).partition_S(sb);
    auto ad=copy_a.get_slice(lane).retile_D(ra);auto bd=copy_b.get_slice(lane).retile_D(rb);
    for_each(make_int_sequence<RegisterStages>{},[&](auto i) {
      constexpr int k=Split+int(i)*WarpsK;
      copy(CopyA{},as(_,_,Int<k>{}),ad(_,_,Int<k>{}));
      copy(CopyB{},bs(_,_,Int<k>{}),bd(_,_,Int<k>{}));
    });
    for_each(make_int_sequence<TK/(16*WarpsK)>{},[&](auto i) {
      constexpr int k=Split+int(i)*WarpsK,next=k+RegisterStages*WarpsK;
      gemm(mma,ra(_,_,Int<k>{}),rb(_,_,Int<k>{}),accum);
      if constexpr(next<TK/16) {
        copy(CopyA{},as(_,_,Int<next>{}),ad(_,_,Int<next>{}));
        copy(CopyB{},bs(_,_,Int<next>{}),bd(_,_,Int<next>{}));
      }
    });
  }

  __device__ static __noinline__ void Consume(Sm90DenseOperands p,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,int tasks,
      unsigned phase,float epsilon) {
    using namespace cute;
    int tid=threadIdx.x,lane=tid%(32*WarpsN),split=tid/(32*WarpsN);
    Mma mma;auto thread=mma.get_slice(lane);
    auto owned=thread.partition_C(make_identity_tensor(Shape<_16,Int<TN>>{}));
    for(int local=0;local<tasks;++local) {
      auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
      for(int iteration=0;iteration<kIterations;++iteration) {
        int sequence=local*kIterations+iteration,slot=sequence%Slots;
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        auto* sa=a+slot*16*TK;auto* sb=b+slot*TN*TK;
        // A warp takes one uniform specialization. All 128-thread barriers
        // and shared reductions remain at common call sites after reconvergence.
        if constexpr(WarpsK==1)MmaSlice<0>(sa,sb,accum);
        else if constexpr(WarpsK==2) {
          if(split==0)MmaSlice<0>(sa,sb,accum);else MmaSlice<1>(sa,sb,accum);
        }else {
          if(split==0)MmaSlice<0>(sa,sb,accum);
          else if(split==1)MmaSlice<1>(sa,sb,accum);
          else if(split==2)MmaSlice<2>(sa,sb,accum);
          else MmaSlice<3>(sa,sb,accum);
        }
        // The empty barrier counts every reader; a separate CTA rendezvous
        // would serialize otherwise independent consumer warps on every slot.
        Async::Arrive(state.empty+slot);
      }
      for(int i=0;i<size(accum);++i)
        scratch[split*16*TN+get<0>(owned(i))*TN+get<1>(owned(i))]=accum(i);
      codegen::executor::ComputeSync();
      if constexpr(WarpsK>1) {
        for(int i=tid;i<16*TN;i+=128) {
          float sum=scratch[i];
          #pragma unroll
          for(int rank=1;rank<WarpsK;++rank)sum+=scratch[rank*16*TN+i];
          scratch[i]=sum;
        }
        codegen::executor::ComputeSync();
      }
      int task=int(blockIdx.x)+local*int(gridDim.x);
      constexpr int stride=Op==ServingEpilogueOp::kSwiGLU?N/2:
          Op==ServingEpilogueOp::kArgmaxPartial?N/TN:N;
      Sm90DenseEpilogue<Op,TN,U>::Run(scratch,task,Batch,N,stride,p.output,p.residual,
          p.argmax_index,p.norm_sums,p.output_sums,K,epsilon);
      codegen::executor::ComputeSync();
    }
    if(tid==0 && tasks)for(int slot=0;slot<Slots;++slot)
      state.consumer_generations[slot]=(phase>>slot)&1;
  }

  template<int Prefix=0,class Tail=Sm90DenseNoTail>
  __device__ static void Run(Sm90DenseOperands p,char* shared,float epsilon,
      Sm90DenseRingState* persistent=nullptr,Tail tail={}) {
    static_assert(Prefix>=0 && Prefix%16==0 &&
        Prefix<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    static_assert(!arch::kDevicePass || arch::kSm90AllAvailable,
                  "native dense GEMM requires Hopper sm_90a");
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Slots*16*TK;
    auto* scratch=reinterpret_cast<float*>(shared+1024+kOperandBytes);
    if constexpr(Batch<16) {
      // Only active rows are overwritten by the producer. Initializing the
      // other rows once avoids issuing fifteen zero-fill rows per K block at B1.
      for(int i=threadIdx.x*8;i<Slots*(16-Batch)*TK;i+=160*8) {
        int slot=i/((16-Batch)*TK),within=i%((16-Batch)*TK);
        *reinterpret_cast<uint4*>(a+slot*16*TK+LayoutA{}(Batch+within/TK,within%TK))=
            make_uint4(0,0,0,0);
      }
      __syncthreads();
    }
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    // Each role publishes only its own phases, after its readers have joined.
    // The previous stage's final CTA join makes the two snapshots consistent.
    int tasks=(N/TN)/int(gridDim.x)+(int(blockIdx.x)<(N/TN)%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,a,b,scratch,tasks,phase,epsilon);
    else {
      if constexpr(Tail::kBefore) {
        if(threadIdx.x==128)tail.Before();
      }
      Produce<Prefix>(p,state,a,b,tasks,phase);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
