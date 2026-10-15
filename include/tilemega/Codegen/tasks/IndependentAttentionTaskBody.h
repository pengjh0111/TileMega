// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/PagedAttentionTaskBody.h>
#ifndef TILEMEGA_ATTENTION_BUFFERS
#define TILEMEGA_ATTENTION_BUFFERS 2
#endif
namespace tilemega::codegen {
// The AT-3 arithmetic is shared with the page consumer. With PG disabled,
// each warp supplies the same KV stream through a local cp.async double buffer.
// This separates transport/protocol equivalence from changed softmax ordering.
template<class Arch,int D,int Q,bool Norm,int PageBytes=(TILEMEGA_ATTENTION_IMPL==4?4*D*64:8192),
    AttentionPagePolicy Policy=AttentionPagePolicy::WarpPrivate,
    int Buffers=TILEMEGA_ATTENTION_BUFFERS>
struct IndependentAttentionTaskBody {
  static_assert(Buffers==1 || Buffers==2);
  using Body=PagedAttentionTaskBody<Arch,D,Q,Norm,PageBytes,3,false,Q,Policy>;
  using Element=typename Body::Element;
  using Copy=executor::Async<Arch>;
  struct LegacySharedStorage {
    alignas(1024) char pages[4][Buffers][PageBytes];
    typename Body::SharedStorage compute;
  };
  struct HopperSharedStorage {
    alignas(1024) char pages[Buffers][PageBytes];
    alignas(16) std::uint64_t full[Buffers];
    typename Body::SharedStorage compute;
  };
  using SharedStorage=std::conditional_t<Body::kWgmma,HopperSharedStorage,LegacySharedStorage>;
  struct LocalPages {
    ServingAttentionOperands const& p;
    SharedStorage& shared;
    int batch,group,block;
    AttentionPageLayout layout;
    mutable int releases=0;
    __device__ char* Page(std::uint64_t cursor) const {
      return shared.pages[cursor%layout.pages_per_wave][(cursor/layout.pages_per_wave)%Buffers];
    }
    __device__ void Issue(int wave) const {
      if(wave>=layout.waves)return;
      int warp=ComputeThread()/32,lane=ComputeThread()%32;
      auto* page=reinterpret_cast<Element*>(Page(layout.PageOf(warp,wave)));
      int begin=Body::Begin(p,block),end=Body::End(p,block);
      for(int i=lane*8;i<layout.RowsPerOwner()*D;i+=32*8) {
        int local=i/D,row=layout.RowOffset(warp)+local,d=i%D;
        int position=layout.Position(warp,wave,local);
        bool valid=layout.Valid(warp,wave,local,end,p.past);
        std::size_t offset=((std::size_t(batch)*p.heads_kv+group)*p.capacity+position)*D+d;
        auto* key=page+(row/16)*16*D+typename Body::QK::LayoutB{}(row%16,d);
        auto* value=page+Body::kPageRows*D+(row/16)*16*D+typename Body::PV::LayoutB{}(d,row%16);
        Copy::Copy16(key,valid?p.key_cache+offset:p.key_cache,valid);
        Copy::Copy16(value,valid?p.value_cache+offset:p.value_cache,valid);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    __device__ void AwaitFull(std::uint64_t cursor) const {
      if(Buffers==2 && int(cursor/layout.pages_per_wave)+1<layout.waves)asm volatile("cp.async.wait_group 1;" ::: "memory");
      else asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncwarp();
      if(layout.needs_cta_sync)ComputeSync();
    }
    __device__ void Release(std::uint64_t cursor) const {
      if(++releases%layout.ReleaseArrivals(int(cursor))==0) {
        __syncwarp();
        if(layout.needs_cta_sync)ComputeSync();
        // All consumers have finished this slot before the next wave writes
        // it. Packed ownership uses the same CTA rendezvous as double buffering.
        Issue(int(cursor/layout.pages_per_wave)+Buffers);
      }
    }
  };
  struct HopperPages {
    ServingAttentionOperands const& p;
    SharedStorage& shared;
    int batch,group,block;
    AttentionPageLayout layout;
    __device__ char* Page(std::uint64_t cursor) const {return shared.pages[cursor%Buffers];}
    __device__ void Issue(int wave) const {
      if(wave>=layout.waves || ComputeThread()>=32)return;
      int lane=ComputeThread()%32;
      auto* page=reinterpret_cast<Element*>(Page(wave));
      auto* barrier=&shared.full[wave%Buffers];
      int begin=Body::Begin(p,block),end=Body::End(p,block);
      int start=begin+wave*Body::kPageRows;
      if(layout.Full(wave,end,p.past) && p.key_tensor_map && p.value_tensor_map) {
        Copy::ProxyAsyncGlobalFence();
        if(Copy::ElectOne()) {
          Copy::ExpectTx(barrier,PageBytes);
          for(int d=0;d<D;d+=64) {
            int y=(batch*p.heads_kv+group)*p.capacity+start;
            Copy::Tensor2D(page+Body::PageCoordinates::Offset(0,d),p.key_tensor_map,d,y,barrier);
            Copy::Tensor2D(page+Body::kPageRows*D+Body::PageCoordinates::Offset(0,d),p.value_tensor_map,d,y,barrier);
          }
        }else Copy::Arrive(barrier);
      }else {
        for(int i=lane*8;i<Body::kPageRows*D;i+=32*8) {
          int row=i/D,d=i%D,position=start+row;
          bool valid=layout.Valid(0,wave,row,end,p.past);
          std::size_t offset=((std::size_t(batch)*p.heads_kv+group)*p.capacity+position)*D+d;
          auto page_offset=Body::PageCoordinates::Offset(row,d);
          Copy::Copy16(page+page_offset,valid?p.key_cache+offset:p.key_cache,valid);
          Copy::Copy16(page+Body::kPageRows*D+page_offset,valid?p.value_cache+offset:p.value_cache,valid);
        }
        Copy::CompleteCopies(barrier);
      }
    }
    __device__ void AwaitFull(std::uint64_t cursor) const {
      Copy::Wait(&shared.full[cursor%Buffers],(cursor/Buffers)&1);
      ComputeSync();
    }
    __device__ void Release(std::uint64_t cursor) const {
      ComputeSync();Issue(int(cursor)+Buffers);
    }
  };
  __device__ static TILEMEGA_HOPPER_TASK_INLINE void Run(ServingAttentionOperands const& p,SharedStorage& shared,
                            int batch,int group,int query_block,int block) {
    (void)query_block;
    int count=Body::End(p,block)-Body::Begin(p,block);if(count<=0)return;
    auto layout=Body::Layout(count,Body::Begin(p,block));
    if constexpr(Body::kWgmma) {
      if(ComputeThread()==0) {
        for(int slot=0;slot<Buffers;++slot)Copy::Init(&shared.full[slot],32);
        Copy::InitFence();
      }
      ComputeSync();
      HopperPages pages{p,shared,batch,group,block,layout};
      pages.Issue(0);if constexpr(Buffers==2)pages.Issue(1);std::uint64_t sequence=0;
      Body::Run(p,batch,group,block,pages,sequence,shared.compute);
    }else {
      LocalPages pages{p,shared,batch,group,block,layout};
      pages.Issue(0);if constexpr(Buffers==2)pages.Issue(1);std::uint64_t sequence=0;
      Body::Run(p,batch,group,block,pages,sequence,shared.compute);
    }
  }
};
} // namespace tilemega::codegen
