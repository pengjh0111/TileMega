// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/Sm90F3Gemm.cuh>

// The F3 swap-AB GEMM (DNN3 prompt §6.7). A stage with at most sixteen rows
// would leave a 64-row WGMMA tile almost empty, so the weight becomes the A
// operand and the activation the B operand: the CTA tile is 128 output
// features by all M rows, and the task count follows N instead of M.
//
// Used by ResNet18/50's classifier, VGG16's three fully connected layers and
// BERT's pooler, whose activation is one row per sequence.
namespace tilemega::backend {

template<int Slots,int SplitK,int M,int N,int K,int NStore,class Chain>
struct Sm90F3GemmSwapAB {
  using Element=cutlass::bfloat16_t;
  using Async=Sm90F3Async;
  using Base=Sm90F3Async::Base;
  using Side=typename Chain::Layout;

  static constexpr int kThreads=288;
  // The WGMMA's n extent: eight rows when the stage has at most eight, else
  // sixteen. Rows past M fall outside the activation box and arrive as zeros.
  static constexpr int kRows=M<=8?8:16;
  static_assert(M>=1 && M<=16,"swap-AB is for a stage of at most sixteen rows");
  static constexpr int kTileN=128;
  static constexpr int kWgRows=64;            // each warpgroup's weight rows
  static constexpr int kIterations=K/(SplitK*64);
  static_assert(K%(SplitK*64)==0,"split-K divides the contraction into whole k blocks");
  static_assert(Slots>=2 && Slots<=8);
  static_assert(SplitK>=1 && SplitK<=32,"swap-AB admits a deeper split than normal mode");
  static constexpr int kTiles=(N+kTileN-1)/kTileN;
  static constexpr int kTasks=kTiles*SplitK;
  static constexpr int kSlotBytes=Sm90F3SwapSlotBytes(kRows);

  using Gmma=Sm90Gmma<arch::Sm90>;
  using Mma=decltype(cute::make_tiled_mma(typename Gmma::template SS<kRows>{}));
  using Fragment=decltype(cute::partition_fragment_C(Mma{},
      cute::Shape<cute::_64,cute::Int<kRows>>{}));
  struct Accumulators {Fragment tile[1];};

  __device__ static __forceinline__ int LocalTasks() {
    return kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
  }
  __device__ static __forceinline__ int Task(int local) {
    return int(blockIdx.x)+local*int(gridDim.x);
  }
  __device__ static __forceinline__ int NTile(int task) {return task%kTiles;}
  __device__ static __forceinline__ int Slice(int task) {return task/kTiles;}

  // A slot holds the weight page first, then the activation panel.
  __device__ static __forceinline__ Element* SlotWeight(Element* arena,int slot) {
    return arena+std::size_t(slot)*(kSlotBytes/2);
  }
  __device__ static __forceinline__ Element* SlotActivation(Element* arena,int slot) {
    return SlotWeight(arena,slot)+std::size_t(kTileN)*64;
  }

  __device__ static __noinline__ void Produce(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase,int) {
    if(int(threadIdx.x)!=256) {__syncwarp();return;}
    Base::ProxyAsyncGlobalFence();
    int tasks=LocalTasks();
    for(int local=0;local<tasks;++local) {
      int task=Task(local),n_tile=NTile(task),slice=Slice(task);
      for(int k=0;k<kIterations;++k) {
        int slot=(local*kIterations+k)%Slots;
        int block=slice*kIterations+k;
        Base::Wait(ring.empty+slot,phase.Bit(phase.empty,slot));
        phase.empty^=1u<<slot;
        Base::ExpectTx(ring.full+slot,kSlotBytes);
        Base::Bulk(SlotWeight(arena,slot),
            p.weight+(std::size_t(n_tile)*(K/64)+block)*kTileN*64,2*kTileN*64,
            ring.full+slot);
        // The activation's own row mapping is in its tensor map, so BERT's
        // pooler reads one row per sequence without a separate code path.
        Base::Tensor2D(SlotActivation(arena,slot),p.activation,block*64,0,ring.full+slot);
      }
      ProduceSide(p,ring,side,phase,n_tile);
    }
    __syncwarp();
  }

  static constexpr int kVectors=Side::kVectorBytes/(4*kTileN);
  static constexpr Sm90F3GemmLayout kLayout=Sm90F3MakeGemmLayout(kRows,kTileN,Slots,
      true,kRows,false,kVectors,false,0,0);
  static constexpr int kLayoutBytes=kLayout.end;
  static constexpr int kSideOffset=kLayout.side_base;
  static_assert(Sm90F3Budget::Fits(kLayoutBytes),
      "an F3 swap-AB stage must fit the shared budget");
  __device__ static __forceinline__ int SideBytes() {return 4*kTileN*kVectors;}

  __device__ static __forceinline__ void ProduceSide(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,int n_tile) {
    if constexpr(kVectors>0) {
      Base::Wait(&ring.side_empty,phase.side_empty&1u);
      phase.side_empty^=1u;
      Base::ExpectTx(&ring.side_full,unsigned(SideBytes()));
      #pragma unroll
      for(int v=0;v<kVectors;++v)
        if(p.epilogue.vectors[v])
          Base::Bulk(side+Side::kVectorBase+std::size_t(v)*kTileN*4,
              p.epilogue.vectors[v]+std::size_t(n_tile)*kTileN,4*kTileN,&ring.side_full);
    }
  }

  __device__ static __noinline__ void Consume(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,Element* arena,char* side,Sm90F3Phase& phase,
      Sm90F3Header* header,int wg) {
    using namespace cute;
    Mma mma;
    auto thread=mma.get_slice(int(threadIdx.x)%128);
    auto owned=thread.partition_C(make_identity_tensor(
        cute::Shape<cute::_64,cute::Int<kRows>>{}));
    int tasks=LocalTasks();
    Accumulators accum;
    for(int local=0;local<tasks;++local) {
      int task=Task(local),n_tile=NTile(task);
      cute::clear(accum.tile[0]);
      int previous=-1;
      for(int k=0;k<kIterations;++k) {
        int slot=(local*kIterations+k)%Slots;
        Base::Wait(ring.full+slot,phase.Bit(phase.full,slot));
        phase.full^=1u<<slot;
        auto weight=make_tensor(make_smem_ptr(
            SlotWeight(arena,slot)+std::size_t(wg)*kWgRows*64),Sm90KLayout<kWgRows,64>{});
        auto activation=make_tensor(make_smem_ptr(
            SlotActivation(arena,slot)),Sm90KLayout<kRows,64>{});
        auto ra=thread.partition_fragment_A(weight);
        auto rb=thread.partition_fragment_B(activation);
        cute::warpgroup_fence_operand(accum.tile[0]);
        cute::warpgroup_arrive();
        CUTE_UNROLL
        for(int step=0;step<size<2>(ra);++step)
          cute::gemm(mma,ra(cute::_,cute::_,step),rb(cute::_,cute::_,step),accum.tile[0]);
        cute::warpgroup_commit_batch();
        cute::warpgroup_wait<1>();
        cute::warpgroup_fence_operand(accum.tile[0]);
        if(previous>=0) {
          __syncwarp();
          if((threadIdx.x&31)==0)Base::Arrive(ring.empty+previous);
        }
        previous=slot;
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(accum.tile[0]);
      if(previous>=0) {
        __syncwarp();
        if((threadIdx.x&31)==0)Base::Arrive(ring.empty+previous);
      }
      if constexpr(SplitK==1)Epilogue(p,ring,side,phase,accum,owned,wg,n_tile);
      else SplitKEpilogue(p,ring,side,phase,accum,owned,wg,task,n_tile,header);
    }
  }

  // The accumulator element (n_row, m) is output row m, column n: the chain's
  // per-column vectors index n, and each element is its own output address.
  template<class Accum,class Owned>
  __device__ static __forceinline__ void Epilogue(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,Accum& accumulators,
      Owned const& owned,int wg,int n_tile) {
    using namespace cute;
    if constexpr(kVectors>0) {
      Base::Wait(&ring.side_full,phase.side_full&1u);
      phase.side_full^=1u;
    }
    auto const& e=p.epilogue;
    auto& tile=accumulators.tile[0];
    CUTE_UNROLL
    for(int i=0;i<int(cute::size(Fragment{}));++i) {
      int feature=wg*kWgRows+get<0>(owned(i)),row=get<1>(owned(i));
      int column=n_tile*kTileN+feature;
      if(row>=M || column>=NStore)continue;
      float value=Chain::Apply(tile(i),row,feature,side,0.f,0.f,e);
      e.output[std::size_t(row)*e.output_stride+column]=Element(value);
    }
    __syncwarp();
    if((threadIdx.x&31)==0)Base::Arrive(&ring.side_empty);
  }

  // The partial layout is [tile][slice][kRows][128]: the same fixed ascending
  // slice order the normal mode reduces in.
  template<class Accum,class Owned>
  __device__ static __forceinline__ void SplitKEpilogue(Sm90F3GemmOperands const& p,
      Sm90F3Ring& ring,char* side,Sm90F3Phase& phase,Accum& accumulators,
      Owned const& owned,int wg,int task,int n_tile,Sm90F3Header* header) {
    using namespace cute;
    int slice=Slice(task);
    auto* base=p.epilogue.partial+(std::size_t(n_tile)*SplitK+slice)*kRows*kTileN;
    auto& tile=accumulators.tile[0];
    CUTE_UNROLL
    for(int i=0;i<int(cute::size(Fragment{}));++i) {
      int feature=wg*kWgRows+get<0>(owned(i)),row=get<1>(owned(i));
      Async::StoreRelaxed(base+std::size_t(row)*kTileN+feature,tile(i));
    }
    Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
    if(threadIdx.x==0) {
      auto old=Async::CounterAdd(p.epilogue.counters+n_tile,1);
      header->last_arriver=int(old+1)==SplitK;
      if(header->last_arriver)p.epilogue.counters[n_tile]=0;
    }
    Async::template NamedBarrier<Sm90F3Async::kBarConsumers,256>();
    if(!header->last_arriver) {
      if constexpr(kVectors>0) {
        Base::Wait(&ring.side_full,phase.side_full&1u);
        phase.side_full^=1u;
        __syncwarp();
        if((threadIdx.x&31)==0)Base::Arrive(&ring.side_empty);
      }
      return;
    }
    CUTE_UNROLL
    for(int i=0;i<int(cute::size(Fragment{}));++i) {
      int feature=wg*kWgRows+get<0>(owned(i)),row=get<1>(owned(i));
      float total=0;
      #pragma unroll 1
      for(int sl=0;sl<SplitK;++sl)
        total+=Async::LoadRelaxed(p.epilogue.partial+
            (std::size_t(n_tile)*SplitK+sl)*kRows*kTileN+std::size_t(row)*kTileN+feature);
      tile(i)=total;
    }
    Epilogue(p,ring,side,phase,accumulators,owned,wg,n_tile);
  }
};

}  // namespace tilemega::backend
