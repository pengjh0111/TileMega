// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/MoeBinding.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <tilemega/Target/ArchDispatch.h>
#include <cutlass/bfloat16.h>

namespace tilemega::codegen {
struct MoeDispatchOperands {
  std::int32_t const* indices=nullptr;
  cutlass::bfloat16_t const* weights=nullptr;
  MoeBindingRecord* bindings=nullptr;
  MoeBindingRow* rows=nullptr;
  // Histogram changes in place to the exclusive prefix per expert/chunk.
  unsigned* histogram=nullptr;
  unsigned* expert_offsets=nullptr;
  unsigned* block_offsets=nullptr;
  unsigned tokens=0,experts=0,capacity=0;
};

template<class Arch,int TopK=8,int BlockRows=16,bool Grouped=true,
         int ChunkTokens=128,int MaxExperts=128>
struct MoEDispatchTaskBody {
  static_assert(arch::Caps<Arch>::kBf16TensorCore);
  static_assert(TopK>0 && TopK<=32 && BlockRows>0 && ChunkTokens>0);
  static_assert(MaxExperts>0 && MaxExperts<=128);
  static constexpr int kThreads=128,kSharedBytes=5*MaxExperts*sizeof(unsigned);
  struct SharedStorage {unsigned seen[MaxExperts],warps[4][MaxExperts];};
  __host__ __device__ static unsigned Count(MoeDispatchOperands const& p) {
    return (p.tokens+ChunkTokens-1)/ChunkTokens;
  }
  __device__ static void Validate(MoeDispatchOperands const& p) {
    unsigned capacity=0;
    if(!p.indices || !p.weights || !p.bindings || !p.rows || p.experts>MaxExperts ||
       !MoeVirtualCapacity(p.tokens,TopK,p.experts,BlockRows,Grouped,&capacity) ||
       capacity!=p.capacity || (Grouped && (!p.histogram || !p.expert_offsets || !p.block_offsets)))
      asm volatile("trap;");
  }
  __device__ static void Histogram(MoeDispatchOperands const& p,unsigned chunk,SharedStorage& s) {
    Validate(p);
    if constexpr(Grouped) {
      for(unsigned e=executor::ComputeThread();e<p.experts;e+=kThreads)s.seen[e]=0;
      executor::ComputeSync();
      unsigned begin=chunk*ChunkTokens*TopK,end=min(p.tokens,(chunk+1)*ChunkTokens)*TopK;
      for(unsigned row=begin+executor::ComputeThread();row<end;row+=kThreads) {
        auto e=p.indices[row];if(e<0 || unsigned(e)>=p.experts)asm volatile("trap;");
        atomicAdd(s.seen+e,1u);
      }
      executor::ComputeSync();
      for(unsigned e=executor::ComputeThread();e<p.experts;e+=kThreads)
        p.histogram[chunk*p.experts+e]=s.seen[e];
      executor::ComputeSync();
    }
  }
  __device__ static void Prefix(MoeDispatchOperands const& p,SharedStorage& s) {
    Validate(p);
    if constexpr(Grouped) {
      unsigned thread=executor::ComputeThread(),chunks=Count(p);
      if(thread<p.experts) {
        unsigned total=0;
        for(unsigned chunk=0;chunk<chunks;++chunk) {
          auto offset=chunk*p.experts+thread,value=p.histogram[offset];
          p.histogram[offset]=total;total+=value;
        }
        s.seen[thread]=total;s.warps[0][thread]=(total+BlockRows-1)/BlockRows;
      }
      executor::ComputeSync();
      for(unsigned stride=1;stride<p.experts;stride*=2) {
        unsigned rows=0,blocks=0;
        if(thread<p.experts) {
          rows=s.seen[thread];blocks=s.warps[0][thread];
          if(thread>=stride) {rows+=s.seen[thread-stride];blocks+=s.warps[0][thread-stride];}
        }
        executor::ComputeSync();
        if(thread<p.experts) {s.seen[thread]=rows;s.warps[0][thread]=blocks;}
        executor::ComputeSync();
      }
      if(thread<p.experts) {
        p.expert_offsets[thread]=thread?s.seen[thread-1]:0;
        p.block_offsets[thread]=thread?s.warps[0][thread-1]:0;
      }
      if(!thread) {
        p.expert_offsets[p.experts]=s.seen[p.experts-1];
        p.block_offsets[p.experts]=s.warps[0][p.experts-1];
        if(s.seen[p.experts-1]!=p.tokens*TopK || s.warps[0][p.experts-1]>p.capacity)
          asm volatile("trap;");
      }
      // Every launch invalidates the entire tail before publishing active
      // records. Their other fields may retain bytes from an earlier step.
      for(unsigned v=thread;v<p.capacity;v+=kThreads)p.bindings[v].valid=0;
      executor::ComputeSync();
      if(thread<p.experts) {
        unsigned row_begin=thread?s.seen[thread-1]:0;
        unsigned row_end=s.seen[thread],block_begin=thread?s.warps[0][thread-1]:0;
        for(unsigned offset=0;row_begin+offset<row_end;offset+=BlockRows)
          p.bindings[block_begin+offset/BlockRows]={thread,row_begin+offset,
              min(unsigned(BlockRows),row_end-row_begin-offset),1};
      }
      executor::ComputeSync();
    }
  }
  __device__ static MoeBindingRow Row(MoeDispatchOperands const& p,unsigned assignment) {
    MoeBindingRow result;result.token=assignment/TopK;result.rank=assignment%TopK;
    result.weight_bf16=reinterpret_cast<std::uint16_t const*>(p.weights)[assignment];return result;
  }
  __device__ static void Scatter(MoeDispatchOperands const& p,unsigned chunk,SharedStorage& s) {
    Validate(p);
    unsigned thread=executor::ComputeThread();
    unsigned begin=chunk*ChunkTokens*TopK,end=min(p.tokens,(chunk+1)*ChunkTokens)*TopK;
    if constexpr(!Grouped) {
      for(unsigned row=begin+thread;row<end;row+=kThreads) {
        auto e=p.indices[row];if(e<0 || unsigned(e)>=p.experts)asm volatile("trap;");
        p.rows[row]=Row(p,row);p.bindings[row]={unsigned(e),row,1,1};
      }
    }else {
      for(unsigned e=thread;e<p.experts;e+=kThreads)s.seen[e]=0;
      executor::ComputeSync();
      // Four warp histograms per wave give a stable rank without atomically
      // allocating output rows. Earlier chunks, waves, warps and lanes retain
      // their token/rank order, independently of CTA execution order.
      for(unsigned wave=begin;wave<end;wave+=kThreads) {
        for(unsigned i=thread;i<4*p.experts;i+=kThreads)
          s.warps[i/p.experts][i%p.experts]=0;
        executor::ComputeSync();
        unsigned assignment=wave+thread,e=p.experts;
        if(assignment<end) {
          auto index=p.indices[assignment];if(index<0 || unsigned(index)>=p.experts)asm volatile("trap;");
          e=index;
        }
        unsigned mask=__match_any_sync(0xffffffff,e),lane=thread%32,warp=thread/32;
        unsigned earlier=mask&((1u<<lane)-1);
        if(e<p.experts && !earlier)s.warps[warp][e]=__popc(mask);
        executor::ComputeSync();
        if(e<p.experts) {
          unsigned rank=s.seen[e]+__popc(earlier);
          for(unsigned w=0;w<warp;++w)rank+=s.warps[w][e];
          auto row=p.expert_offsets[e]+p.histogram[chunk*p.experts+e]+rank;
          p.rows[row]=Row(p,assignment);
        }
        executor::ComputeSync();
        for(unsigned e=thread;e<p.experts;e+=kThreads)
          for(unsigned w=0;w<4;++w)s.seen[e]+=s.warps[w][e];
        executor::ComputeSync();
      }
    }
    executor::ComputeSync();
  }
  __device__ static void RunSmall(MoeDispatchOperands const& p,SharedStorage& s) {
    if(std::uint64_t(p.tokens)*TopK>4096) {asm volatile("trap;");return;}
    if constexpr(Grouped) {
      for(unsigned chunk=0;chunk<Count(p);++chunk)Histogram(p,chunk,s);
      Prefix(p,s);
    }
    for(unsigned chunk=0;chunk<Count(p);++chunk)Scatter(p,chunk,s);
  }
};
} // namespace tilemega::codegen
