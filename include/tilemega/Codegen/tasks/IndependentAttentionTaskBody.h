// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/PagedAttentionTaskBody.h>
namespace tilemega::codegen {
// The AT-3 arithmetic is shared with the page consumer. With PG disabled,
// each warp supplies the same KV stream through a local cp.async double buffer.
// This separates transport/protocol equivalence from changed softmax ordering.
template<class Arch,int D,int Q,bool Norm,bool BoundLayout=(Q==8),
         AttentionPagePolicy Policy=AttentionPagePolicy::kWarpPrivate>
struct IndependentAttentionTaskBody {
  static constexpr int PageBytes=8192;
  using Body=PagedAttentionTaskBody<Arch,D,Q,Norm,PageBytes,3,false,Q,BoundLayout,Policy>;
  using Element=typename Body::Element;
  using Copy=executor::Async<Arch>;
  struct SharedStorage {
    alignas(1024) char pages[4][2][PageBytes];
    typename Body::SharedStorage compute;
  };
  struct LocalPages {
    ServingAttentionOperands const& p;
    SharedStorage& shared;
    int batch,group,block,waves,extent;
    typename Body::PageLayout layout;
    mutable int releases=0;
    __device__ char* Page(std::uint64_t cursor) const {
      if constexpr(BoundLayout)return shared.pages[cursor%layout.pages_per_wave()][(cursor/layout.pages_per_wave())%2];
      else return shared.pages[cursor%4][(cursor/4)%2];
    }
    __device__ void Issue(int wave) const {
      if(wave>=waves)return;
      int warp=ComputeThread()/32,lane=ComputeThread()%32;
      auto* page=reinterpret_cast<Element*>(Page(BoundLayout?layout.PageOf(warp,wave):wave*4+warp));
      int begin=Body::Begin(p,block),end=Body::End(p,block);
      int limit=BoundLayout?layout.RowsPerOwner():Body::kPageRows;
      for(int i=lane*8;i<limit*D;i+=32*8) {
        int local=i/D,d=i%D;
        int row=local+(BoundLayout?layout.RowOffset(warp):0);
        int position=begin+(BoundLayout?layout.Position(warp,wave,local):warp*extent+wave*Body::kPageRows+local);
        bool valid=(BoundLayout?layout.Valid(warp,wave,local):local+wave*Body::kPageRows<extent && position<end) && position<p.past;
        std::size_t offset=((std::size_t(batch)*p.heads_kv+group)*p.capacity+position)*D+d;
        auto* key=page+(row/16)*16*D+typename Body::QK::LayoutB{}(row%16,d);
        auto* value=page+Body::kPageRows*D+(row/16)*16*D+typename Body::PV::LayoutB{}(d,row%16);
        Copy::Copy16(key,valid?p.key_cache+offset:p.key_cache,valid);
        Copy::Copy16(value,valid?p.value_cache+offset:p.value_cache,valid);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    __device__ void AwaitFull(std::uint64_t cursor) const {
      int wave=BoundLayout?int(cursor/layout.pages_per_wave()):int(cursor/4);
      if(wave+1<waves)asm volatile("cp.async.wait_group 1;" ::: "memory");
      else asm volatile("cp.async.wait_group 0;" ::: "memory");
      if constexpr(BoundLayout) {if(layout.needs_cta_sync())ComputeSync();else __syncwarp();}
      else __syncwarp();
    }
    __device__ void Release(std::uint64_t cursor) const {
      int arrivals=BoundLayout?layout.ReleaseIterations(int(cursor)):4;
      if(++releases%arrivals==0) {
        if constexpr(BoundLayout) {if(layout.needs_cta_sync())ComputeSync();else __syncwarp();}
        else __syncwarp();
        Issue((BoundLayout?int(cursor/layout.pages_per_wave()):int(cursor/4))+2);
      }
    }
  };
  __device__ static void Run(ServingAttentionOperands const& p,SharedStorage& shared,
                            int batch,int group,int query_block,int block) {
    (void)query_block;
    int count=Body::End(p,block)-Body::Begin(p,block);if(count<=0)return;
    int extent=Body::WarpExtent(count),waves=(extent+Body::kPageRows-1)/Body::kPageRows;
    LocalPages pages{p,shared,batch,group,block,waves,extent,{count}};
    if constexpr(BoundLayout) {extent=pages.layout.extent();waves=pages.layout.waves();pages.extent=extent;pages.waves=waves;}
    pages.Issue(0);pages.Issue(1);std::uint64_t sequence=0;
    Body::Run(p,batch,group,block,pages,sequence,shared.compute);
  }
};
} // namespace tilemega::codegen
