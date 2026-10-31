// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseOperands.cuh>
#include <tilemega/Backend/Sm90DenseRoles.cuh>

#include <tilemega/Backend/Sm90DenseEpilogue.h>
#include <tilemega/Backend/Sm90DenseMark.cuh>
#include <tilemega/Backend/Sm90DenseStream.cuh>
#include <tilemega/Backend/Sm90DenseHead.cuh>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseLayout.h>
#include <cute/tensor.hpp>

namespace tilemega::backend {

// Multicast (D10c, clusters): the CTA pair (2j, 2j + 1) shares every A slot.
// Rank r issues the activation TMA of the slots s with s % 2 == r into both
// CTAs; it first waits for the peer's consumer releases on peer[s / 2].
template<int Batch,int N,int K,int TN,int TK,int Slots,int WarpsN,int WarpsK,
         ServingEpilogueOp Op,int U,bool InputNorm,int RegisterStages=1,bool Multicast=false>
struct Sm90DenseGemm {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  static_assert(Batch>0 && Batch<=16 && N%TN==0 && TN%8==0 && TK%64==0 && K%TK==0);
  static_assert(WarpsN*WarpsK==4 && TN%(8*WarpsN)==0 && Slots>=2 && Slots<=8);
  static_assert((RegisterStages==1 || RegisterStages==2 || RegisterStages==4) &&
                RegisterStages<=TK/(16*WarpsK));
  static_assert(Op!=ServingEpilogueOp::kPartial,"dense decode never splits K across tasks");
  static_assert(!Multicast || Slots%2==0,"a multicast slot keeps one issuing rank");
  __device__ static __forceinline__ unsigned& PeerGenerations(Sm90DenseRingState& state) {
    return Sm90DenseClusterState::OfRing(state).peer_generations[Sm90DenseClusterState::RingIndex(state)];
  }
  static constexpr bool kResidual=Op==ServingEpilogueOp::kResidual;
  // A residual task's tile is prefetched per local task; a grid of at least
  // 132 CTAs gives each CTA at most two of them.
  static_assert(!kResidual || N/TN<=264,"residual stages prefetch at most two local tasks");
  static constexpr int kThreads=160,kIterations=K/TK;
  static constexpr int kRows=Sm90DenseLayout::ActivationRows(Batch);
  static constexpr int kOperandBytes=2*Slots*TK*(kRows+TN);
  static constexpr int kSlotBytes=2*TK*(kRows+TN);
  static constexpr int kScratchBytes=4*WarpsK*16*TN;
  static constexpr auto kLayout=Sm90DenseLayout::Gemm(TN,TK,Slots,WarpsK,Batch,kResidual);
  static constexpr int kSharedBytes=kLayout.end;
  static_assert(kSharedBytes==1024+kOperandBytes+kScratchBytes+512+(kResidual?4*Batch*TN:0));
  // Store and residual write from registers when one warp holds all of K.
  static constexpr bool kRegisterEpilogue=WarpsK==1 &&
      (Op==ServingEpilogueOp::kStore || Op==ServingEpilogueOp::kResidual);
  using Epilogue=Sm90DenseEpilogue<Op,Batch,N,K,TN,U,InputNorm>;
  static_assert(kThreads==Sm90DenseRoleAbi::kThreads);
  using Atom=decltype(cute::composition(cute::Swizzle<3,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::_64>,cute::Stride<cute::_64,cute::_1>>{}));
  // The TMA box writes [TK/64][kRows][64] panels with SWIZZLE_128B, the same
  // 8x64 atom as Swizzle<3,3,3>. For B<=8 MMA rows 8..15 alias rows 0..7:
  // those duplicate output rows are never written.
  using LayoutA=decltype(cute::composition(cute::Swizzle<3,3,3>{},cute::Layout<
      cute::Shape<cute::Shape<cute::_8,cute::_2>,cute::Shape<cute::_64,cute::Int<TK/64>>>,
      cute::Stride<cute::Stride<cute::_64,cute::Int<kRows==16?512:0>>,
                   cute::Stride<cute::_1,cute::Int<kRows*64>>>>{}));
  using LayoutB=decltype(cute::tile_to_shape(Atom{},cute::Shape<cute::Int<TN>,cute::Int<TK>>{}));
  using Mma=decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{},
      cute::Layout<cute::Shape<cute::_1,cute::Int<WarpsN>,cute::_1>>{},
      cute::Tile<cute::_16,cute::Int<TN>,cute::_16>{}));
  using CopyA=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,Element>;
  using CopyB=std::conditional_t<(TN/WarpsN<=16 || (TN/WarpsN)%16!=0),
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N,Element>,CopyA>;
  static_assert(cute::cosize_v<LayoutB> ==TN*TK);

  template<int Bytes>
  // Raw operand pointers share the transfer loop across layer-specific bindings.
  __device__ static __forceinline__ void Preissue(Element const* weight,char* shared,
      Sm90DenseRingState& state) {
    static_assert(Bytes>0 && Bytes%16==0 &&
        Bytes<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    auto* b=reinterpret_cast<Element*>(shared+1024)+Slots*kRows*TK;
    int remaining=Bytes;
    #pragma unroll
    for(int slot=0;slot<Slots && slot<kIterations;++slot) {
      int part=min(remaining,2*TN*TK);remaining-=part;
      if(part) {
        // Preissue owns the alternate bank: its previous users joined before
        // the preceding V3 boundary. Current arena overlap is checked by Tail.
        // The activation depends on the current stage; its bytes are expected
        // here and its TMA is issued by the consuming stage after its fence.
        Async::ExpectTx(state.full+slot,kSlotBytes);
        Async::Bulk(b+slot*TN*TK,weight+(std::size_t(blockIdx.x)*kIterations+slot)*TN*TK,
            part,state.full+slot);
      }
    }
  }

  // streamed: the previous stage issued the Prefix bytes (false only after
  // the ring-less prologue, which shares this typed stage in a layer loop).
  template<int Prefix>
  __device__ static __noinline__ void Produce(Sm90DenseOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,Element* residual,int tasks,unsigned phase,
      bool streamed) {
    Sm90DenseRoles::Producer();
    if(threadIdx.x==128) {
      CUtensorMap const* const activation=p.activation;
      Element const* const weight=p.weight;
      // E1: one policy per producer, not one per copy.
      std::uint64_t const policy=Sm90DenseStream::WeightPolicy();
      // The activation (and residual) were written by other CTAs with generic
      // stores before the grid barrier; the async proxy must observe them.
      Async::ProxyAsyncGlobalFence();
      if constexpr(kResidual)
        for(int local=0;local<tasks;++local) {
          int task=int(blockIdx.x)+local*int(gridDim.x);
          Async::ExpectTx(state.side+local,2*Batch*TN);
          #pragma unroll
          for(int row=0;row<Batch;++row)
            Async::Bulk(residual+(local*Batch+row)*TN,p.residual+std::size_t(row)*N+task*TN,
                2*TN,state.side+local);
        }
      unsigned rank=0,peer=0;
      if constexpr(Multicast) {rank=Async::ClusterRank();peer=PeerGenerations(state);}
      for(int sequence=0;sequence<tasks*kIterations;++sequence) {
        int slot=sequence%Slots,task=int(blockIdx.x)+(sequence/kIterations)*int(gridDim.x);
        int tile=sequence%kIterations;
        Async::Wait(state.empty+slot,(phase>>slot)&1);
        bool issue=!Multicast || unsigned(slot&1)==(rank&1);
        if constexpr(Multicast)
          if(issue) {Async::Wait(state.peer+slot/2,(peer>>(slot/2))&1);peer^=1u<<(slot/2);}
        // The first task starts at slot zero in every invocation. Its incoming
        // budget is fixed by the same Placement that issued the transaction.
        unsigned prefix=0;
        if constexpr(Prefix>0)
          if(streamed && sequence<(Prefix+2*TN*TK-1)/(2*TN*TK))
            prefix=min(Prefix-sequence*2*TN*TK,2*TN*TK);
        if(!prefix)Async::ExpectTx(state.full+slot,kSlotBytes);
        if(prefix<2*TN*TK)Sm90DenseStream::WeightBulk(b+slot*TN*TK+prefix/2,
            weight+(std::size_t(task)*kIterations+tile)*TN*TK+prefix/2,
            2*TN*TK-prefix,state.full+slot,policy);
        // Rows past Batch fall outside the map and arrive as zeros.
        if constexpr(Multicast) {
          if(issue)Async::Tensor3DMulticast(a+slot*kRows*TK,activation,0,0,tile*(TK/64),state.full+slot,
              std::uint16_t(3u<<(rank&~1u)));
        }else Async::Tensor3D(a+slot*kRows*TK,activation,0,0,tile*(TK/64),state.full+slot);
        phase^=1u<<slot;
      }
      if(tasks) {
        for(int slot=0;slot<Slots;++slot)state.producer_generations[slot]=(phase>>slot)&1;
        if constexpr(Multicast)PeerGenerations(state)=peer;
      }
    }
    __syncwarp();
  }

  template<int Split,class Accumulator>
  __device__ static __forceinline__ void MmaSlice(Element* a,Element* b,Accumulator& accum,
                                                  float (&squares)[2]) {
    using namespace cute;
    static_assert(Split>=0 && Split<WarpsK);
    int lane=int(threadIdx.x)%(32*WarpsN);
    Mma mma;auto thread=mma.get_slice(lane);
    auto copy_a=make_tiled_copy_A(CopyA{},mma);
    auto copy_b=make_tiled_copy_B(CopyB{},mma);
    auto sa=make_tensor(make_smem_ptr(a),LayoutA{});
    auto sb=make_tensor(make_smem_ptr(b),LayoutB{});
    // Keep only the register lookahead, rather than an addressable fragment
    // for every k16 in a large shared-memory slot.
    auto ra=make_fragment_like<Element>(make_layout(append(
        take<0,2>(shape(thread.partition_A(sa))),Int<RegisterStages>{})));
    auto rb=make_fragment_like<Element>(make_layout(append(
        take<0,2>(shape(thread.partition_B(sb))),Int<RegisterStages>{})));
    auto as=copy_a.get_slice(lane).partition_S(sa);auto bs=copy_b.get_slice(lane).partition_S(sb);
    auto ad=copy_a.get_slice(lane).retile_D(ra);auto bd=copy_b.get_slice(lane).retile_D(rb);
    for_each(make_int_sequence<RegisterStages>{},[&](auto i) {
      constexpr int k=Split+int(i)*WarpsK;
      copy(CopyA{},as(_,_,Int<k>{}),ad(_,_,i));
      copy(CopyB{},bs(_,_,Int<k>{}),bd(_,_,i));
    });
    MmaSteps<Split,0>(mma,ra,rb,as,bs,ad,bd,accum,squares);
  }

  template<int Split,int I,class MA,class RA,class RB,class AS,class BS,class AD,class BD,class Acc>
  __device__ static __forceinline__ void MmaSteps(MA const& mma,RA& ra,RB& rb,
      AS const& as,BS const& bs,AD& ad,BD& bd,Acc& accum,float (&squares)[2]) {
    using namespace cute;
    constexpr int k=Split+I*WarpsK,next=k+RegisterStages*WarpsK,slot=I%RegisterStages;
    if constexpr(InputNorm) {
      // m16n8k16 A fragment: values 2..3 and 6..7 belong to row lane/4+8,
      // the others to row lane/4 (PTX ISA "mma.m16n8k16" fragment layout).
      auto fragment=ra(_,_,Int<slot>{});
      static_assert(decltype(size(fragment))::value==8);
      #pragma unroll
      for(int v=0;v<8;++v) {float x=float(fragment(v));squares[(v>>1)&1]+=x*x;}
    }
    gemm(mma,ra(_,_,Int<slot>{}),rb(_,_,Int<slot>{}),accum);
    if constexpr(next<TK/16) {
      copy(CopyA{},as(_,_,Int<next>{}),ad(_,_,Int<slot>{}));
      copy(CopyB{},bs(_,_,Int<next>{}),bd(_,_,Int<slot>{}));
    }
    // CuTe's large for_each fold can become an outlined closure with local
    // fragment arguments. Keep each owned k16 at an explicit inline call site.
    if constexpr(I+1<TK/(16*WarpsK))MmaSteps<Split,I+1>(mma,ra,rb,as,bs,ad,bd,accum,squares);
  }

  template<class Finish>
  __device__ static __noinline__ void Consume(Sm90DenseOperands const& p,
      Sm90DenseRingState& state,Element* a,Element* b,float* scratch,Element* residual,
      int tasks,unsigned phase,Finish finish) {
    Sm90DenseRoles::Consumer();
    using namespace cute;
    int tid=threadIdx.x,lane=tid%(32*WarpsN),split=tid/(32*WarpsN);
    float* norms=scratch+WarpsK*16*TN;
    // Argmax: the CTA's per-row (value, index) over its tasks (D9), after the
    // WK x 16 partial squares in the 512-byte statistics area.
    float* best=norms+64;int* arg=reinterpret_cast<int*>(norms+80);
    if constexpr(Op==ServingEpilogueOp::kArgmaxPartial)
      if(tid<16) {best[tid]=-INFINITY;arg[tid]=INT_MAX;}
    unsigned side=state.side_generations[0]|state.side_generations[1]<<1;
    unsigned rank=0;
    if constexpr(Multicast)rank=Async::ClusterRank();
    Mma mma;auto thread=mma.get_slice(lane);
    auto owned=thread.partition_C(make_identity_tensor(Shape<_16,Int<TN>>{}));
    for(int local=0;local<tasks;++local) {
      auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
      float squares[2]={0,0};
      for(int iteration=0;iteration<kIterations;++iteration) {
        int sequence=local*kIterations+iteration,slot=sequence%Slots;
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        if(tid==0)Sm90DenseMarkFirst(0);
        auto* sa=a+slot*kRows*TK;auto* sb=b+slot*TN*TK;
        // A warp takes one uniform specialization. All 128-thread barriers
        // and shared reductions remain at common call sites after reconvergence.
        if constexpr(WarpsK==1)MmaSlice<0>(sa,sb,accum,squares);
        else if constexpr(WarpsK==2) {
          if(split==0)MmaSlice<0>(sa,sb,accum,squares);else MmaSlice<1>(sa,sb,accum,squares);
        }else {
          if(split==0)MmaSlice<0>(sa,sb,accum,squares);
          else if(split==1)MmaSlice<1>(sa,sb,accum,squares);
          else if(split==2)MmaSlice<2>(sa,sb,accum,squares);
          else MmaSlice<3>(sa,sb,accum,squares);
        }
        // The empty barrier counts every reader; a separate CTA rendezvous
        // would serialize otherwise independent consumer warps on every slot.
        // The refill is an async-proxy write after these generic reads: the
        // mbarrier orders only the generic proxy, so each reader fences first
        // (PTX ISA 8.7 sec. 8.6 proxies; without it TN448 B1 lost 132/200 runs).
        Async::FenceProxyAsyncShared();
        Async::Arrive(state.empty+slot);
        // The peer issued this slot's A into this CTA too: release it there.
        if constexpr(Multicast)
          if(unsigned(slot&1)!=(rank&1))Async::RemoteArrive(Async::Mapa(state.peer+slot/2,rank^1u));
      }
      if(tid==0)Sm90DenseMark(1);
      if constexpr(InputNorm)
        #pragma unroll
        for(int r=0;r<2;++r) {
          squares[r]+=__shfl_xor_sync(0xffffffff,squares[r],1);
          squares[r]+=__shfl_xor_sync(0xffffffff,squares[r],2);
        }
      int task=int(blockIdx.x)+local*int(gridDim.x);
      Element const* tile_residual=nullptr;
      if constexpr(kResidual) {
        Async::Wait(state.side+local,(side>>local)&1);side^=1u<<local;
        tile_residual=residual+local*Batch*TN;
      }
      if constexpr(kRegisterEpilogue) {
        float inv[2]={1,1};
        if constexpr(InputNorm)
          for(int r=0;r<2;++r)inv[r]=Epilogue::Inverse(squares[r],p.epsilon);
        Epilogue::Registers(accum,owned,inv,tile_residual,p.output,task);
        if(tid==0)Sm90DenseMark(2);
      }else {
        for(int i=0;i<size(accum);++i)
          scratch[split*16*TN+get<0>(owned(i))*TN+get<1>(owned(i))]=accum(i);
        // One warp per split publishes its rows' partial squares.
        if constexpr(InputNorm)
          if((tid/32)%WarpsN==0 && (tid&3)==0) {
            int row=(tid&31)/4;
            norms[split*16+row]=squares[0];norms[split*16+row+8]=squares[1];
          }
        codegen::executor::ComputeSync();
        if constexpr(Op==ServingEpilogueOp::kArgmaxPartial)
          Epilogue::template Shared<WarpsK>(scratch,norms,tile_residual,
              reinterpret_cast<Element*>(best),arg,task,p.epsilon);
        else Epilogue::template Shared<WarpsK>(scratch,norms,tile_residual,p.output,p.argmax_index,
            task,p.epsilon);
        codegen::executor::ComputeSync();
        if(tid==0)Sm90DenseMark(2);
      }
    }
    // Every task has been consumed: the operand slots are free.
    if constexpr(Op==ServingEpilogueOp::kArgmaxPartial) {
      codegen::executor::ComputeSync();
      finish(p,state,best,arg,reinterpret_cast<char*>(a),kOperandBytes);
    }
    if(tid==0 && tasks) {
      for(int slot=0;slot<Slots;++slot)state.consumer_generations[slot]=(phase>>slot)&1;
      if constexpr(kResidual)for(int i=0;i<2;++i)state.side_generations[i]=(side>>i)&1;
    }
  }

  template<int Prefix=0,class Tail=Sm90DenseNoTail,class Finish=Sm90DenseNoFinish>
  __device__ static __forceinline__ void Run(Sm90DenseOperands const& p,char* shared,
      Sm90DenseRingState* persistent=nullptr,Tail tail={},bool streamed=true,Finish finish={}) {
    static_assert(Op==ServingEpilogueOp::kArgmaxPartial ||
        std::is_same_v<Finish,Sm90DenseNoFinish>,"only lm_head finishes with the folded argmax");
    static_assert(Prefix>=0 && Prefix%16==0 &&
        Prefix<=(Slots<kIterations?Slots:kIterations)*2*TN*TK);
    static_assert(!Multicast || Prefix==0,"multicast stages receive no preissued slots");
    static_assert(!arch::kDevicePass || arch::kSm90AllAvailable,
                  "native dense GEMM requires Hopper sm_90a");
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto* a=reinterpret_cast<Element*>(shared+1024);
    auto* b=a+Slots*kRows*TK;
    auto* scratch=reinterpret_cast<float*>(shared+kLayout.scratch);
    auto* residual=reinterpret_cast<Element*>(shared+kLayout.scratch+kScratchBytes+512);
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    // Each role publishes only its own phases, after its readers have joined.
    // The previous stage's final CTA join makes the two snapshots consistent.
    int tasks=(N/TN)/int(gridDim.x)+(int(blockIdx.x)<(N/TN)%int(gridDim.x));
    if(threadIdx.x<128)Consume(p,state,a,b,scratch,residual,tasks,phase,finish);
    else {
      Produce<Prefix>(p,state,a,b,residual,tasks,phase,streamed);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
