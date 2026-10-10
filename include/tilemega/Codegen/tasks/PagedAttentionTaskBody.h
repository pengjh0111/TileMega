// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <tilemega/Codegen/executor/PageRing.cuh>
#include <tilemega/Codegen/tasks/AttentionPageLayout.h>
namespace tilemega::codegen {
template<class Arch,int D,int Q,bool QkNorm,int PageBytes,int Pages,bool ForceSm80=false,int PartialRows=Q,
         bool BoundLayout=(Q==8),AttentionPagePolicy Policy=AttentionPagePolicy::kPacked>
struct PagedAttentionTaskBody {
  static_assert(Q<=16);
  using Element=cutlass::bfloat16_t;
  using Base=FusedAttentionTaskBody<Arch,D,Q,1,16,32,QkNorm>;
  using Atom=decltype(cute::composition(cute::Swizzle<3,3,3>{},
      cute::Layout<cute::Shape<cute::_8,cute::_64>,cute::Stride<cute::_64,cute::_1>>{}));
  using KeyLayout=decltype(cute::tile_to_shape(Atom{},cute::Shape<cute::_16,cute::Int<D>>{}));
  using ValueLayout=decltype(cute::composition(KeyLayout{},
      cute::Layout<cute::Shape<cute::Int<D>,cute::_16>,cute::Stride<cute::_16,cute::_1>>{}));
  using QK=backend::ServingAttentionWarp<Arch,16,D,false,KeyLayout>;
  using PV=std::conditional_t<Q==8,backend::ServingAttentionPvSwap<Arch,D,ValueLayout>,
      backend::ServingAttentionWarp<Arch,D,16,true,ValueLayout>>;
  using Ring=executor::PageRing<PageBytes,Pages,Arch,ForceSm80>;
  using Copy=typename Ring::Copy;
  using PageLayout=AttentionPageLayout<D,PageBytes,Policy>;
  static constexpr int kPageRows=PageBytes/(4*D);
  static_assert(kPageRows%16==0);
  struct SharedStorage {
    alignas(16) Element query[16*D];
    alignas(16) float partial[4*PartialRows*D];
    float lse[4][PartialRows];
  };
  __device__ static int Begin(ServingAttentionOperands const& p,int c){return c*p.block_extent;}
  __device__ static int End(ServingAttentionOperands const& p,int c){return min((c+1)*p.block_extent,p.past+1);}
  __device__ static int WarpExtent(int count){return ((count+63)/64)*16;}
  template<class BeforePage=executor::NoPageHook>
  __device__ static void Load(ServingAttentionOperands const& p,int b,int g,int c,
                              Ring const& ring,std::uint64_t& sequence,
                              BeforePage const& before_page=BeforePage{}) {
    int begin=Begin(p,c),end=End(p,c);if(begin>=end)return;
    unsigned remaining=2*max(0,min(end,p.past)-begin)*D*sizeof(Element);
    PageLayout layout{end-begin};
    int extent=BoundLayout?layout.extent():WarpExtent(end-begin);
    int waves=BoundLayout?layout.waves():(extent+kPageRows-1)/kPageRows;
    int warps_per_page=BoundLayout?layout.owners_per_page():(extent<kPageRows?min(4,kPageRows/extent):1);
    int page_count=BoundLayout?layout.pages_per_wave():4/warps_per_page;
    constexpr int vectors_per_lane=kPageRows*D/(executor::kLoaderThreads*8);
    int key_offsets[vectors_per_lane],value_offsets[vectors_per_lane];
    int rows[vectors_per_lane],columns[vectors_per_lane];
    #pragma unroll
    for(int step=0;step<vectors_per_lane;++step) {
      int i=(executor::LoaderLane()+step*executor::kLoaderThreads)*8;
      int row=i/D,d=i%D;
      rows[step]=row;columns[step]=d;
      key_offsets[step]=(row/16)*16*D+typename QK::LayoutB{}(row%16,d);
      value_offsets[step]=(row/16)*16*D+typename PV::LayoutB{}(d,row%16);
    }
    for(int wave=0;wave<waves;++wave)for(int page_index=0;page_index<page_count;++page_index) {
      unsigned stream_bytes=min(remaining,unsigned(PageBytes));
      before_page(stream_bytes);remaining-=stream_bytes;
      ring.AcquireEmpty(sequence);
      auto* page=reinterpret_cast<Element*>(ring.Page(sequence));
      int start=begin+page_index*warps_per_page*extent+wave*kPageRows;
      bool full=wave*kPageRows+kPageRows<=extent || warps_per_page>1;
      full=full && start+kPageRows<=min(end,p.past);
      if constexpr(Copy::Caps::kTma) {
        if(full && p.key_tensor_map && p.value_tensor_map) {
          auto* barrier=&ring.slots[Ring::SlotIndex(sequence)].full;
          if(executor::LoaderLane()==0) {
            Copy::ExpectTx(barrier,PageBytes);
            for(int row=0;row<kPageRows;row+=16)for(int d=0;d<D;d+=64) {
              int y=(b*p.heads_kv+g)*p.capacity+start+row;
              Copy::Tensor2D(page+row*D+KeyLayout{}(0,d),p.key_tensor_map,d,y,barrier);
              Copy::Tensor2D(page+kPageRows*D+row*D+ValueLayout{}(d,0),p.value_tensor_map,d,y,barrier);
            }
          }else Copy::Arrive(barrier);
          ++sequence;continue;
        }
      }
      #pragma unroll
      for(int step=0;step<vectors_per_lane;++step) {
        int row=rows[step],d=columns[step];
        int owner=page_index*warps_per_page+
            (warps_per_page==1?0:row/extent);
        if constexpr(BoundLayout)owner=layout.OwnerOf(page_index,row);
        int local=BoundLayout?layout.LocalRow(row):(warps_per_page==1?row:row%extent);
        int position=begin+(BoundLayout?layout.Position(owner,wave,local):owner*extent+wave*kPageRows+local);
        bool valid=full || ((BoundLayout?layout.Valid(owner,wave,local):
            local+wave*kPageRows<extent && position<end) && position<p.past);
        std::size_t offset=((std::size_t(b)*p.heads_kv+g)*p.capacity+position)*D+d;
        auto* key=page+key_offsets[step];
        auto* value=page+kPageRows*D+value_offsets[step];
        Copy::Copy16(key,valid?p.key_cache+offset:p.key_cache,valid);
        Copy::Copy16(value,valid?p.value_cache+offset:p.value_cache,valid);
      }
      ring.PublishCopies(sequence++);
    }
  }
  __device__ static void Query(ServingAttentionOperands const& p,SharedStorage& s,int b,int g) {
    int warp=ComputeThread()/32,lane=ComputeThread()%32;
    constexpr int width=(Q+2)*D;
    for(int row=warp;row<16;row+=4) {
      if(row<Q)Base::RotateRow(p.qkv+(b*p.heads_kv+g)*width+row*D,p.q_norm,
          p.cosine+p.past*D,p.sine+p.past*D,p.epsilon,[&](int d,uint4 v){
            *reinterpret_cast<uint4*>(s.query+typename QK::LayoutA{}(row,d))=v;});
      else if(lane*8<D)*reinterpret_cast<uint4*>(s.query+typename QK::LayoutA{}(row,lane*8))=make_uint4(0,0,0,0);
    }
  }
  __device__ static void NewRow(ServingAttentionOperands const& p,Element* page,int row,int b,int g) {
    constexpr int width=(Q+2)*D;
    auto* x=p.qkv+(b*p.heads_kv+g)*width+Q*D;
    Base::RotateRow(x,p.k_norm,p.cosine+p.past*D,p.sine+p.past*D,p.epsilon,[&](int d,uint4 v){
      *reinterpret_cast<uint4*>(page+(row/16)*16*D+typename QK::LayoutB{}(row%16,d))=v;
      backend::StoreGlobal16(p.key_cache+((std::size_t(b)*p.heads_kv+g)*p.capacity+p.past)*D+d,v);
    });
    int d=(ComputeThread()%32)*8;
    if(d<D) {
      uint4 v=backend::LoadGlobal16(x+D+d);
      *reinterpret_cast<uint4*>(page+kPageRows*D+(row/16)*16*D+typename PV::LayoutB{}(d,row%16))=v;
      backend::StoreGlobal16(p.value_cache+((std::size_t(b)*p.heads_kv+g)*p.capacity+p.past)*D+d,v);
    }
  }
  template<class PageSource>
  __device__ static void Run(ServingAttentionOperands const& p,int b,int g,int c,
      PageSource const& ring,std::uint64_t& sequence,SharedStorage& s) {
    using namespace cute;
    int begin=Begin(p,c),end=End(p,c);if(begin>=end)return;
    PageLayout layout{end-begin};
    int extent=BoundLayout?layout.extent():WarpExtent(end-begin);
    int waves=BoundLayout?layout.waves():(extent+kPageRows-1)/kPageRows;
    int warps_per_page=BoundLayout?layout.owners_per_page():(extent<kPageRows?min(4,kPageRows/extent):1);
    int page_count=BoundLayout?layout.pages_per_wave():4/warps_per_page;
    int warp=ComputeThread()/32,lane=ComputeThread()%32;
    Query(p,s,b,g);ComputeSync();
    auto score_coords=typename QK::Mma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_16,_16>{}));
    auto out_coords=PV::OutputCoordinates(lane);
    auto output=PV::Accumulator();float maximum[2]={-INFINITY,-INFINITY},sum[2]={0,0};
    // Each warp advances independently. A page belongs to one warp, which
    // supplies the ring's 128 release arrivals (four per lane).
    for(int wave=0;wave<waves;++wave) {
      auto cursor=sequence+(BoundLayout?layout.PageOf(warp,wave):wave*page_count+warp/warps_per_page);
      ring.AwaitFull(cursor);
      auto* page=reinterpret_cast<Element*>(ring.Page(cursor));
      int start=begin+(BoundLayout?layout.Position(warp,wave,0):warp*extent+wave*kPageRows);
      int page_row=BoundLayout?layout.RowOffset(warp):(warp%warps_per_page)*extent;
      if(p.past>=start && p.past<start+kPageRows && p.past<begin+(warp+1)*extent && p.past<end)
        NewRow(p,page,page_row+p.past-start,b,g);
      __syncwarp();
      int tile_limit=BoundLayout?layout.RowsPerOwner():(warps_per_page>1?extent:kPageRows);
      for(int tile=0;tile<tile_limit;tile+=16) {
        auto score=QK::Accumulator();QK::QK(s.query,page+(page_row+tile)*D,score);
        float next[2]={maximum[0],maximum[1]},alpha[2];
        #pragma unroll
        for(int i=0;i<size(score);++i) {
          int row=get<0>(score_coords(i)),col=get<1>(score_coords(i));
          int position=start+tile+col;
          score(i)=row<Q && position<end && wave*kPageRows+tile+col<extent
              ?score(i)*(1.4426950408889634f/sqrtf(float(D))):-INFINITY;
          next[row/8]=fmaxf(next[row/8],score(i));
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
          next[r]=fmaxf(next[r],__shfl_xor_sync(0xffffffffu,next[r],1));
          next[r]=fmaxf(next[r],__shfl_xor_sync(0xffffffffu,next[r],2));
          alpha[r]=isfinite(maximum[r])?exp2f(maximum[r]-next[r]):0;
          maximum[r]=next[r];
        }
        float added[2]={0,0};
        #pragma unroll
        for(int i=0;i<size(score);++i) {
          int r=int(get<0>(score_coords(i)))/8;
          score(i)=isfinite(score(i))?exp2f(score(i)-maximum[r]):0;
          added[r]+=score(i);
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
          added[r]+=__shfl_xor_sync(0xffffffffu,added[r],1);
          added[r]+=__shfl_xor_sync(0xffffffffu,added[r],2);
          sum[r]=sum[r]*alpha[r]+added[r];
        }
        #pragma unroll
        for(int i=0;i<size(output);++i)output(i)*=PV::RowStatistic(alpha,PV::QueryRow(out_coords(i)));
        PV::PV(score,score_coords,page+kPageRows*D+(page_row+tile)*D,output);
      }
      int arrivals=BoundLayout?layout.ReleaseIterations(int(cursor-sequence)):4/warps_per_page;
      for(int arrival=0;arrival<arrivals;++arrival)ring.Release(cursor);
    }
    sequence+=waves*page_count;
    #pragma unroll
    for(int i=0;i<size(output);++i) {
      int row=PV::QueryRow(out_coords(i)),d=PV::OutputDim(out_coords(i));
      float normalizer=PV::RowStatistic(sum,row);
      if(row<PartialRows)s.partial[(warp*PartialRows+row)*D+d]=normalizer>0?output(i)/normalizer:0;
    }
    if((lane&3)==0)for(int r=0;r<2;++r)
      if(lane/4+8*r<PartialRows)s.lse[warp][lane/4+8*r]=sum[r]>0?maximum[r]+log2f(sum[r]):-INFINITY;
    ComputeSync();
    int cmax=(p.capacity+p.block_extent-1)/p.block_extent;
    for(int vector=ComputeThread();vector<Q*D/8;vector+=kComputeThreads) {
      int row=vector/(D/8),d=(vector%(D/8))*8;
      float m=-INFINITY,normalizer=0,values[8]={};
      for(int w=0;w<4;++w)m=fmaxf(m,s.lse[w][row]);
      for(int w=0;w<4;++w) {
        float weight=isfinite(s.lse[w][row])?exp2f(s.lse[w][row]-m):0;
        normalizer+=weight;
        float4 a=*reinterpret_cast<float4 const*>(s.partial+(w*PartialRows+row)*D+d);
        float4 v=*reinterpret_cast<float4 const*>(s.partial+(w*PartialRows+row)*D+d+4);
        float incoming[8]={a.x,a.y,a.z,a.w,v.x,v.y,v.z,v.w};
        for(int i=0;i<8;++i)values[i]+=weight*incoming[i];
      }
      for(int i=0;i<8;++i)values[i]=normalizer>0?values[i]/normalizer:0;
      if(cmax==1) {
        alignas(16) Element result[8];for(int i=0;i<8;++i)result[i]=Element(values[i]);
        backend::StoreGlobal16(p.context+((std::size_t(b)*p.heads_kv+g)*Q+row)*D+d,*reinterpret_cast<uint4 const*>(result));
      }else {
        std::size_t base=((std::size_t(b)*p.heads_kv+g)*cmax+c)*Q+row;
        *reinterpret_cast<float4*>(p.partial+base*D+d)=make_float4(values[0],values[1],values[2],values[3]);
        *reinterpret_cast<float4*>(p.partial+base*D+d+4)=make_float4(values[4],values[5],values[6],values[7]);
        if(d==0)p.lse[base]=m+log2f(normalizer);
      }
    }
    ComputeSync();
  }
};
} // namespace tilemega::codegen
