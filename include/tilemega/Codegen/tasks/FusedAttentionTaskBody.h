// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingAttentionWarp.h>
#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace tilemega::codegen {
struct ServingAttentionOperands {
  cutlass::bfloat16_t const* qkv;
  cutlass::bfloat16_t* key_cache;
  cutlass::bfloat16_t* value_cache;
  cutlass::bfloat16_t const* cosine;
  cutlass::bfloat16_t const* sine;
  cutlass::bfloat16_t const* q_norm;
  cutlass::bfloat16_t const* k_norm;
  cutlass::bfloat16_t* context;
  float* partial;
  float* lse;
  int batch, heads_kv, capacity, past, block_extent;
  float epsilon;
};
template<class Arch,int kHeadDim,int kQPerKV,int kTokens,
         int kQRows,int kKvTile,bool kQkNorm>
struct FusedAttentionTaskBody {
  static_assert(kHeadDim==64 || kHeadDim==128);
  static_assert(kQPerKV==2 || kQPerKV==4);
  static_assert(kTokens==1 || kTokens==64);
  static_assert((kQRows>0 && kQRows%16==0) || kTokens==1);
  static_assert(kKvTile==32 || kKvTile==64);
  using Element=cutlass::bfloat16_t;
  using QK=backend::ServingAttentionWarp<Arch,16,kHeadDim>;
  using PV=backend::ServingAttentionWarp<Arch,kHeadDim,16,true>;
  static constexpr int kQueryExtent=kTokens*kQPerKV;
  struct SharedStorage {
    alignas(16) Element query[16*kHeadDim];
    union {
      struct {
        alignas(16) Element key[2][kKvTile*kHeadDim];
        alignas(16) Element value[2][kKvTile*kHeadDim];
      } pipeline;
      alignas(16) float partial[4*16*kHeadDim];
    } storage;
    float maxima[4][16],sums[4][16];
    float row_max[16],row_sum[16],alpha[16];
  };
  static_assert(sizeof(SharedStorage)==ServingAttentionSharedBytes(kHeadDim,kKvTile));

  __device__ static float WarpSum(float value) {
    for(int shift=16;shift;shift>>=1)
      value+=__shfl_down_sync(0xffffffffu,value,shift);
    return __shfl_sync(0xffffffffu,value,0);
  }
  // Eight BF16 values per active lane; all 32 lanes participate in norm.
  template<class Store>
  __device__ static void RotateRow(Element const* x,Element const* norm,
      Element const* cosine,Element const* sine,float epsilon,Store store) {
    int lane=int(threadIdx.x)&31,d=lane*8;
    alignas(16) Element a[8],b[8],w[8],partner_w[8],cos[8],sin[8];
    float square=0;
    if(d<kHeadDim) {
      *reinterpret_cast<uint4*>(a)=backend::LoadGlobal16(x+d);
      int partner=(d+kHeadDim/2)%kHeadDim;
      *reinterpret_cast<uint4*>(b)=backend::LoadGlobal16(x+partner);
      *reinterpret_cast<uint4*>(cos)=backend::LoadGlobal16(cosine+d);
      *reinterpret_cast<uint4*>(sin)=backend::LoadGlobal16(sine+d);
      if constexpr(kQkNorm) {
        *reinterpret_cast<uint4*>(w)=backend::LoadGlobal16(norm+d);
        *reinterpret_cast<uint4*>(partner_w)=backend::LoadGlobal16(norm+partner);
        #pragma unroll
        for(int i=0;i<8;++i)square+=float(a[i])*float(a[i]);
      }
    }
    float inverse=1;
    if constexpr(kQkNorm)inverse=rsqrtf(WarpSum(square)/kHeadDim+epsilon);
    if(d<kHeadDim) {
      #pragma unroll
      for(int i=0;i<8;++i) {
        if constexpr(kQkNorm) {
          a[i]=Element(float(Element(float(a[i])*inverse))*float(w[i]));
          b[i]=Element(float(Element(float(b[i])*inverse))*float(partner_w[i]));
        }
        auto first=Element(float(a[i])*float(cos[i]));
        auto second=Element((d<kHeadDim/2?-float(b[i]):float(b[i]))*float(sin[i]));
        a[i]=Element(float(first)+float(second));
      }
      store(d,*reinterpret_cast<uint4 const*>(a));
    }
  }
  __device__ static void Query(ServingAttentionOperands const& p,SharedStorage& s,
                               int b,int g,int begin) {
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    constexpr int width=(kQPerKV+2)*kHeadDim;
    for(int row=warp;row<16;row+=4) {
      int r=begin+row,token=r/kQPerKV,head=r%kQPerKV;
      if(r<kQueryExtent) {
        auto x=p.qkv+(b*kTokens+token)*p.heads_kv*width+g*width+head*kHeadDim;
        RotateRow(x,p.q_norm,p.cosine+(p.past+token)*kHeadDim,
            p.sine+(p.past+token)*kHeadDim,p.epsilon,[&](int d,uint4 v){
              *reinterpret_cast<uint4*>(s.query+typename QK::LayoutA{}(row,d))=v;
            });
      } else if(lane*8<kHeadDim)
        *reinterpret_cast<uint4*>(s.query+typename QK::LayoutA{}(row,lane*8))=make_uint4(0,0,0,0);
    }
  }
  __device__ static void LoadKV(ServingAttentionOperands const& p,SharedStorage& s,
      int slot,int b,int g,int begin,int limit,int query_begin,int q_begin) {
    constexpr int width=(kQPerKV+2)*kHeadDim;
    auto* key=s.storage.pipeline.key[slot];
    auto* value=s.storage.pipeline.value[slot];
    for(int i=int(threadIdx.x)*8;i<kKvTile*kHeadDim;i+=128*8) {
      int row=i/kHeadDim,d=i%kHeadDim,position=begin+row;
      auto* k=key+(row/16)*16*kHeadDim+typename QK::LayoutB{}(row%16,d);
      auto* v=value+(row/16)*16*kHeadDim+typename PV::LayoutB{}(d,row%16);
      if(position<limit && position<p.past) {
        std::size_t offset=((std::size_t(b)*p.heads_kv+g)*p.capacity+position)*kHeadDim+d;
        unsigned ka=unsigned(__cvta_generic_to_shared(k)),va=unsigned(__cvta_generic_to_shared(v));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(ka),"l"(p.key_cache+offset));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(va),"l"(p.value_cache+offset));
      } else if(position>=limit) {
        *reinterpret_cast<uint4*>(k)=make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(v)=make_uint4(0,0,0,0);
      }
    }
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int row=warp;row<kKvTile;row+=4) {
      int position=begin+row;
      if(position<p.past || position>=limit)continue;
      int token=position-p.past;
      auto* x=p.qkv+(b*kTokens+token)*p.heads_kv*width+g*width+kQPerKV*kHeadDim;
      bool writer=query_begin==q_begin && token*kQPerKV>=q_begin &&
                  token*kQPerKV<q_begin+kQRows;
      RotateRow(x,p.k_norm,p.cosine+position*kHeadDim,p.sine+position*kHeadDim,
          p.epsilon,[&](int d,uint4 v){
            *reinterpret_cast<uint4*>(key+(row/16)*16*kHeadDim+typename QK::LayoutB{}(row%16,d))=v;
            if(writer)*reinterpret_cast<uint4*>(p.key_cache+
                ((std::size_t(b)*p.heads_kv+g)*p.capacity+position)*kHeadDim+d)=v;
          });
      if(lane*8<kHeadDim) {
        int d=lane*8;uint4 v=backend::LoadGlobal16(x+kHeadDim+d);
        *reinterpret_cast<uint4*>(value+(row/16)*16*kHeadDim+typename PV::LayoutB{}(d,row%16))=v;
        if(writer)*reinterpret_cast<uint4*>(p.value_cache+
            ((std::size_t(b)*p.heads_kv+g)*p.capacity+position)*kHeadDim+d)=v;
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  __device__ static void Run(ServingAttentionOperands const& p,SharedStorage& s,
                             int b,int g,int qb,int c) {
    using namespace cute;
    int block_begin=c*p.block_extent,limit=min((c+1)*p.block_extent,p.past+kTokens);
    int q_begin=qb*kQRows,warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    if constexpr(kTokens>1)
      limit=min(limit,p.past+(min(q_begin+kQRows,kQueryExtent)-1)/kQPerKV+1);
    if(block_begin>=limit)return;
    int cmax=(p.capacity+p.block_extent-1)/p.block_extent;
    auto score_coords=typename QK::Mma{}.get_slice(lane).partition_C(
        make_identity_tensor(Shape<_16,_16>{}));
    auto out_coords=typename PV::Mma{}.get_slice(lane).partition_C(
        make_identity_tensor(Shape<_16,Int<kHeadDim>>{}));
    for(int query_begin=q_begin;query_begin<min(q_begin+kQRows,kQueryExtent);query_begin+=16) {
      Query(p,s,b,g,query_begin);
      if(threadIdx.x<16){s.row_max[threadIdx.x]=-INFINITY;s.row_sum[threadIdx.x]=0;}
      auto output=PV::Accumulator();
      LoadKV(p,s,0,b,g,block_begin,limit,query_begin,q_begin);
      for(int begin=block_begin,step=0;begin<limit;begin+=kKvTile,++step) {
        int slot=step&1;
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        // Next K/V writes target the other buffer while this buffer feeds MMA.
        if(begin+kKvTile<limit)
          LoadKV(p,s,slot^1,b,g,begin+kKvTile,limit,query_begin,q_begin);
        auto score=QK::Accumulator();
        if(warp*16<kKvTile)
          QK::QK(s.query,s.storage.pipeline.key[slot]+warp*16*kHeadDim,score);
        float maximum[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int i=0;i<size(score);++i) {
          int row=get<0>(score_coords(i)),col=warp*16+get<1>(score_coords(i));
          int token=(query_begin+row)/kQPerKV,position=begin+col;
          score(i)=query_begin+row<kQueryExtent && col<kKvTile &&
              position<limit && position<=p.past+token
              ?score(i)*(1.4426950408889634f/sqrtf(float(kHeadDim))):-INFINITY;
          maximum[row/8]=fmaxf(maximum[row/8],score(i));
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
          maximum[r]=fmaxf(maximum[r],__shfl_xor_sync(0xffffffffu,maximum[r],1));
          maximum[r]=fmaxf(maximum[r],__shfl_xor_sync(0xffffffffu,maximum[r],2));
          if((lane&3)==0)s.maxima[warp][lane/4+8*r]=maximum[r];
        }
        __syncthreads();
        if(threadIdx.x<16) {
          int row=threadIdx.x;float m=s.row_max[row];
          for(int w=0;w<4;++w)m=fmaxf(m,s.maxima[w][row]);
          s.alpha[row]=isfinite(s.row_max[row])?exp2f(s.row_max[row]-m):0;
          s.row_max[row]=m;
        }
        __syncthreads();
        float sum[2]={0,0};
        #pragma unroll
        for(int i=0;i<size(score);++i) {
          int row=get<0>(score_coords(i));
          score(i)=isfinite(score(i))?exp2f(score(i)-s.row_max[row]):0;
          sum[row/8]+=score(i);
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
          sum[r]+=__shfl_xor_sync(0xffffffffu,sum[r],1);
          sum[r]+=__shfl_xor_sync(0xffffffffu,sum[r],2);
          if((lane&3)==0)s.sums[warp][lane/4+8*r]=sum[r];
        }
        __syncthreads();
        if(threadIdx.x<16) {
          int row=threadIdx.x;float total=s.row_sum[row]*s.alpha[row];
          for(int w=0;w<4;++w)total+=s.sums[w][row];
          s.row_sum[row]=total;
        }
        #pragma unroll
        for(int i=0;i<size(output);++i)output(i)*=s.alpha[int(get<0>(out_coords(i)))];
        if(warp*16<kKvTile)
          PV::PV(score,score_coords,s.storage.pipeline.value[slot]+warp*16*kHeadDim,output);
        __syncthreads();
      }
      asm volatile("cp.async.wait_group 0;");
      __syncthreads();
      // Reuse the finished K/V pipeline allocation for the warp partial sum.
      #pragma unroll
      for(int i=0;i<size(output);++i)
        s.storage.partial[(warp*16+int(get<0>(out_coords(i))))*kHeadDim+int(get<1>(out_coords(i)))]=output(i);
      __syncthreads();
      for(int index=int(threadIdx.x)*8;index<16*kHeadDim;index+=128*8) {
        int local=index/kHeadDim,row=query_begin+local,d=index%kHeadDim;
        if(row>=kQueryExtent)continue;
        alignas(16) float values[8]={};
        for(int w=0;w<4;++w) {
          auto* src=s.storage.partial+(w*16+local)*kHeadDim+d;
          float4 a=*reinterpret_cast<float4 const*>(src),v=*reinterpret_cast<float4 const*>(src+4);
          values[0]+=a.x;values[1]+=a.y;values[2]+=a.z;values[3]+=a.w;
          values[4]+=v.x;values[5]+=v.y;values[6]+=v.z;values[7]+=v.w;
        }
        #pragma unroll
        for(int i=0;i<8;++i)values[i]=s.row_sum[local]>0?values[i]/s.row_sum[local]:0;
        if(cmax==1 || kTokens>1) {
          alignas(16) Element result[8];
          #pragma unroll
          for(int i=0;i<8;++i)result[i]=Element(values[i]);
          auto* dst=p.context+((std::size_t(b)*kTokens+row/kQPerKV)*p.heads_kv*kQPerKV+
              g*kQPerKV+row%kQPerKV)*kHeadDim+d;
          backend::StoreGlobal16(dst,*reinterpret_cast<uint4 const*>(result));
        } else {
          std::size_t base=((std::size_t(b)*p.heads_kv+g)*cmax+c)*kQueryExtent+row;
          *reinterpret_cast<float4*>(p.partial+base*kHeadDim+d)=*reinterpret_cast<float4 const*>(values);
          *reinterpret_cast<float4*>(p.partial+base*kHeadDim+d+4)=*reinterpret_cast<float4 const*>(values+4);
          if(d==0)p.lse[base]=s.row_max[local]+log2f(s.row_sum[local]);
        }
      }
      __syncthreads();
    }
  }
};
}
