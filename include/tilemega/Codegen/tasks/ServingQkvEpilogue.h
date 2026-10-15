// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/FusedAttentionTaskBody.h>
#include <stdexcept>
#ifndef TILEMEGA_QKV_EPILOGUE
#define TILEMEGA_QKV_EPILOGUE 0
#endif
static_assert(TILEMEGA_QKV_EPILOGUE==0 || TILEMEGA_QKV_EPILOGUE==1);
namespace tilemega::codegen {
struct QkvEpilogueDesc {
  using Element=cutlass::bfloat16_t;
  Element* key_cache=nullptr;
  Element* value_cache=nullptr;
  Element const* cosine=nullptr;
  Element const* sine=nullptr;
  Element const* q_norm=nullptr;
  Element const* k_norm=nullptr;
  int heads_kv=0,head_dim=0,q_per_kv=0,capacity=0;
  float epsilon=0;
  bool Enabled() const {return key_cache!=nullptr;}
  void Validate(int columns,int tile_columns,int chunks,int sequence) const {
    if(!Enabled())return;
    if(!value_cache || !cosine || !sine || (bool(q_norm)!=bool(k_norm)) ||
       heads_kv<=0 || (head_dim!=64 && head_dim!=128) || q_per_kv<1 || q_per_kv>8 ||
       capacity<=0 || columns!=heads_kv*(q_per_kv+2)*head_dim ||
       tile_columns<=0 || tile_columns%head_dim || chunks!=1 || sequence!=1)
      throw std::invalid_argument("qkv epilogue requires unsplit decode, complete aligned heads and exact QKV/cache geometry");
  }
};

template<class Arch>
struct ServingQkvEpilogue {
  using Element=cutlass::bfloat16_t;
  template<int D,bool Norm>
  __device__ static __forceinline__ void RunHeads(QkvEpilogueDesc const& p,Element* qkv,
      int rows,int columns,int tile_m,int tile_n,int tile_rows,int tile_columns,int past) {
    static_assert(arch::Caps<Arch>::kWgmma,"qkv epilogue candidate requires native Hopper");
    using Base=FusedAttentionTaskBody<Arch,D,1,1,16,32,Norm>;
    int warp=executor::ComputeThread()/32,lane=executor::ComputeThread()%32;
    int heads=tile_columns/D;
    executor::ComputeSync();
    for(int item=warp;item<tile_rows*heads;item+=4) {
      int row=tile_m*tile_rows+item/heads,head=tile_n*heads+item%heads;
      if(row>=rows || head>=columns/D)continue;
      int group=head/(p.q_per_kv+2),kind=head%(p.q_per_kv+2);
      auto* source=qkv+std::size_t(row)*columns+head*D;
      auto offset=((std::size_t(row)*p.heads_kv+group)*p.capacity+past)*D;
      if(kind<p.q_per_kv) {
        // The existing epilogue already rounded the complete head to BF16.
        // One warp owns it; all source/partner loads precede in-place stores.
        Base::RotateRow(source,p.q_norm,p.cosine+past*D,p.sine+past*D,p.epsilon,
            [&](int d,uint4 value){backend::StoreGlobal16(source+d,value);});
      }else if(kind==p.q_per_kv) {
        Base::RotateRow(source,p.k_norm,p.cosine+past*D,p.sine+past*D,p.epsilon,
            [&](int d,uint4 value){backend::StoreGlobal16(p.key_cache+offset+d,value);});
      }else if(lane*8<D) {
        backend::StoreGlobal16(p.value_cache+offset+lane*8,backend::LoadGlobal16(source+lane*8));
      }
    }
    executor::ComputeSync();
  }
  __device__ static __forceinline__ void Run(QkvEpilogueDesc const& p,Element* qkv,
      int rows,int columns,int tile_m,int tile_n,int tile_rows,int tile_columns,int past) {
    if(!p.key_cache)return;
    if((p.head_dim!=64 && p.head_dim!=128) || past<0 || past>=p.capacity || tile_columns%p.head_dim){asm volatile("trap;");return;}
#if defined(TILEMEGA_SERVING_HEAD_DIM) && defined(TILEMEGA_SERVING_QK_NORM)
    // A compiled decoder has one head geometry. Avoid keeping four runtime
    // norm/dimension branches live alongside the megakernel's MMA state.
    if(p.head_dim!=TILEMEGA_SERVING_HEAD_DIM || bool(p.q_norm)!=bool(TILEMEGA_SERVING_QK_NORM)) {
      asm volatile("trap;");return;
    }
    RunHeads<TILEMEGA_SERVING_HEAD_DIM,bool(TILEMEGA_SERVING_QK_NORM)>(p,qkv,rows,columns,
        tile_m,tile_n,tile_rows,tile_columns,past);
#else
    if(p.head_dim==64) {
      if(p.q_norm)RunHeads<64,true>(p,qkv,rows,columns,tile_m,tile_n,tile_rows,tile_columns,past);
      else RunHeads<64,false>(p,qkv,rows,columns,tile_m,tile_n,tile_rows,tile_columns,past);
    }else if(p.head_dim==128) {
      if(p.q_norm)RunHeads<128,true>(p,qkv,rows,columns,tile_m,tile_n,tile_rows,tile_columns,past);
      else RunHeads<128,false>(p,qkv,rows,columns,tile_m,tile_n,tile_rows,tile_columns,past);
    }else asm volatile("trap;");
#endif
  }
};
} // namespace tilemega::codegen
