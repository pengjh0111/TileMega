// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/mx/DecodeProgramV2.cuh>
#include <tilemega/Backend/Mx/RouterGemmV2.h>
#include <tilemega/Backend/Mx/GroupedGemmV2.h>
namespace tilemega::codegen::mx {
using backend::mx::MxMoeGlobalTicketV2;
struct MxMoeLayerBindingV2 {
  Bf16 const *w_qkv,*w_o,*w_gu,*w_down,*q_norm,*k_norm;
  Bf16 *k_cache,*vt_cache;
  Bf16 const* w_router;
};
struct MxMoeScratchV2 {
  Bf16 *x, *qkv, *ctx, *h;
  float *gemm_partial, *attn_partial;
  backend::mx::MxHeadPartialV2* head_partial;
  unsigned *t_qkv, *t_o, *t_gu, *t_down, *t_attn, *t_head;
  // One 64-byte line per program stage, monotonic across launches/steps.
  unsigned long long* barrier;
  WatchdogRecord* watchdog;
  unsigned long long watchdog_ns;
  unsigned diag_skip;
  Bf16 *router_logits,*expert_partials;
  unsigned *n_active,*active,*row_count,*u_gate_up,*u_down;
  uint2* rows;
  void* routing_weights;
  MxMoeGlobalTicketV2 *router_tile_tickets,*router_ticket,*combine_tickets;
#if TILEMEGA_MX_DIAG
  unsigned* route_debug;
#endif
};
template<int Layers>
struct MxMoeBindingV2 {
  MxMoeLayerBindingV2 layer[Layers];
  Bf16 const *embed,*w_head,*rope_cos,*rope_sin;
  int* tokens;
  MxMoeScratchV2 s;
};
using MxMoeSharedMetadataV2=__attribute__((address_space(3))) unsigned const;
template <class P>
struct MoeDenseOperandsV2 {
  using Binding = typename P::Binding;
  using Tokens = backend::mx::MxSharedTokensV2;
  using GemmOps = backend::mx::MxGemmOperandsV2;

  __device__ __forceinline__ static MxMoeScratchV2 const& Scratch(Binding const& b) {
    // Keep scratch-field reloads at their stage; hoisting them spilled the 576-thread kernel.
    return (&b.s)[backend::mx::UniformOpaque(0)];
  }
  __device__ __forceinline__ static Binding const& Top(Binding const& b) {
    // Keep top-level binding fields at their stage, as for scratch fields.
    return (&b)[backend::mx::UniformOpaque(0)];
  }
  __device__ __forceinline__ static GemmOps Gemm(Binding const& b, Bf16 const* a, Bf16 const* w,
                                                Bf16 const* residual, Bf16* out, unsigned* ticket,
                                                int a_stride, int residual_stride,
                                                Tokens* a_tokens = nullptr, Tokens* residual_tokens = nullptr) {
    auto const& scratch = Scratch(b);
    return {a, w, residual, out, scratch.gemm_partial, ticket, a_tokens, residual_tokens,
            a_stride, residual_stride, P::kEpsilon};
  }
  __device__ __forceinline__ static GemmOps QkvWeights(Binding const& b, int l) {
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, nullptr, b.layer[l].w_qkv, nullptr, nullptr, nullptr, P::kHidden, 0);
  }
  __device__ __forceinline__ static GemmOps Qkv(Binding const& b, int l, int past, Tokens* tokens) {
    auto const& top = Top(b);
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    (void)past;
    return Gemm(b, l == 0 ? top.embed : scratch.x, b.layer[l].w_qkv, nullptr, scratch.qkv, scratch.t_qkv,
                P::kHidden, 0, l == 0 ? tokens : nullptr);
  }
  __device__ __forceinline__ static GemmOps O(Binding const& b, int l, Tokens* tokens) {
    auto const& top = Top(b);
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, scratch.ctx, b.layer[l].w_o, l == 0 ? top.embed : scratch.x, scratch.x, scratch.t_o,
                P::OShape::k, P::kHidden, nullptr, l == 0 ? tokens : nullptr);
  }
  __device__ __forceinline__ static GemmOps GateUp(Binding const& b, int l) {
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, scratch.x, b.layer[l].w_gu, nullptr, scratch.h, scratch.t_gu, P::kHidden, 0);
  }
  __device__ __forceinline__ static GemmOps Down(Binding const& b, int l) {
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l);
    return Gemm(b, scratch.h, b.layer[l].w_down, scratch.x, scratch.x, scratch.t_down, P::DownShape::k, P::kHidden);
  }
  __device__ __forceinline__ static backend::mx::MxAttentionOperandsV2 Attn(Binding const& b, int l, int past) {
    auto const& top = Top(b);
    auto const& scratch = Scratch(b);
    l = backend::mx::UniformOpaque(l), past = backend::mx::UniformOpaque(past);
    // set_steps validates this range before copying the private past array.
    __builtin_assume(past >= P::kPastLo && past <= P::kPastHi);
    auto const& layer = b.layer[l];
    return {scratch.qkv, layer.q_norm, layer.k_norm, top.rope_cos, top.rope_sin,
            layer.k_cache, layer.vt_cache, scratch.ctx, scratch.attn_partial, scratch.t_attn,
            P::kCapacity, past, P::kEpsilon};
  }
  __device__ __forceinline__ static backend::mx::MxHeadOperandsV2 Head(Binding const& b, int past) {
    auto const& top = Top(b);
    auto const& scratch = Scratch(b);
    past = backend::mx::UniformOpaque(past);
    __builtin_assume(past >= P::kPastLo && past <= P::kPastHi);
    return {Gemm(b, scratch.x, top.w_head, nullptr, nullptr, nullptr, P::kHidden, 0),
            scratch.head_partial, scratch.t_head, top.tokens, P::kCapacity, past};
  }
};

template<class P>
struct MoeOperandsV2 : MoeDenseOperandsV2<P> {
  using Binding=typename P::Binding;
  using Base=MoeDenseOperandsV2<P>;
  using Tokens=backend::mx::MxSharedTokensV2;
  __device__ __forceinline__ static MxMoeScratchV2 const& Extra(Binding const& b) {
    return (&b.s)[backend::mx::UniformOpaque(0)];
  }
  __device__ __forceinline__ static backend::mx::MxMoeRouterOperandsV2 Router(Binding const& b,int layer,int past) {
    layer=backend::mx::UniformOpaque(layer);
    auto const& sc=Extra(b);auto const& w=b.layer[layer];
    backend::mx::MxMoeDispatchOperands dispatch{sc.router_logits,sc.n_active,sc.active,sc.row_count,
      sc.rows,sc.routing_weights,sc.u_gate_up,sc.u_down
#if TILEMEGA_MX_DIAG
      ,sc.route_debug,unsigned(past-64),unsigned(layer),P::kLayers
#endif
    };
    return {sc.x,w.w_router,nullptr,sc.router_logits,sc.gemm_partial,sc.router_tile_tickets,
      nullptr,nullptr,P::kHidden,0,P::kEpsilon,sc.router_ticket,dispatch};
  }
  template<bool Gate>
  __device__ __forceinline__ static backend::mx::MxGroupedGemmOperandsV2 Grouped(
      Binding const& b,int layer,MxMoeSharedMetadataV2* metadata) {
    layer=backend::mx::UniformOpaque(layer);
    auto const& sc=Extra(b);auto const& w=b.layer[layer];
    unsigned const units=__builtin_mxc_readfirstlane(metadata[Gate?1:2]);
    backend::mx::MxMoeCombineOperandsV2 combine{sc.expert_partials,sc.routing_weights,sc.x,sc.x,sc.combine_tickets};
    return {Gate?sc.x:sc.h,Gate?w.w_gu:w.w_down,Gate?sc.h:sc.expert_partials,
      sc.active,sc.row_count,sc.rows,units,unsigned(Gate?2*P::kIntermediate*P::kHidden:P::kHidden*P::kIntermediate),
      P::kEpsilon,{},combine};
  }
};
template<class P>
struct MoeControlV2 {
  using Sync=StageSyncV2<P::kThreads>;
  static constexpr int kStages=P::kStageCount;
  static constexpr int kTokenOffset=P::kBodySmemBytes;
  static constexpr int kMetadataOffset=StageAlign16V2(kTokenOffset+P::kBatch*4);
  static constexpr int kGenerationOffset=kMetadataOffset+16;
  static constexpr int kSharedBytes=kGenerationOffset+16;
  static_assert(kStages==6*P::kLayers+1&&kSharedBytes<=65536);
  __device__ __forceinline__ static int Past(int const* pasts,unsigned index) {
    using Global=__attribute__((address_space(1))) int const;
    return __builtin_mxc_readfirstlane(*((Global*)pasts+index));
  }
  __device__ __forceinline__ static void Mark(MxTraceV2 trace,unsigned long long iteration,unsigned stage,int which) {
#if TILEMEGA_TRACE_STAGE
    if(Sync::IsRole() && trace.data && trace.slots)
      trace.data[((std::size_t(iteration%trace.slots)*kStages+stage)*gridDim.x+blockIdx.x)*3+which]=maca::GlobalTimerNs();
#else
    (void)trace;(void)iteration;(void)stage;(void)which;
#endif
  }
  __device__ __forceinline__ static void BroadcastTokens(typename P::Binding const& b,int past,int* tokens) {
    using Global=__attribute__((address_space(1))) int const;
    auto const& top=MoeOperandsV2<P>::Top(b);
#pragma unroll
    for(int r=0;r<P::kBatch;++r)tokens[r]=*((Global*)top.tokens+r*P::kCapacity+past);
  }
  __device__ __forceinline__ static void BroadcastRouting(typename P::Binding const& b,unsigned* values) {
    using Global=__attribute__((address_space(1))) unsigned const;
    auto const& sc=MoeOperandsV2<P>::Extra(b);
    values[0]=*((Global*)sc.n_active);values[1]=*((Global*)sc.u_gate_up);values[2]=*((Global*)sc.u_down);
  }
};
template<class P>
__global__ void __launch_bounds__(P::kThreads) mx_moe_kernel_v2(typename P::Binding b,StageLaunchV2 launch) {
  unsigned const grid=gridDim.x;__builtin_assume(grid==104);
  extern __shared__ __align__(16) char smem[];
  using G=MoeControlV2<P>;using Sync=typename G::Sync;
  auto* tokens=reinterpret_cast<int*>(smem+G::kTokenOffset);
  auto* shared_tokens=(backend::mx::MxSharedTokensV2*)tokens;
  auto* metadata=reinterpret_cast<unsigned*>(smem+G::kMetadataOffset);
  auto* shared_metadata=(MxMoeSharedMetadataV2*)metadata;
  auto* generation=reinterpret_cast<unsigned*>(smem+G::kGenerationOffset);
  unsigned const steps=P::kPrompt?64:LaunchControlV2(launch).steps;
  if(!steps)return;
  int initial=0;
  if constexpr(!P::kPrompt){auto const& c=LaunchControlV2(launch);initial=G::Past(c.pasts,c.first);}
  if(Sync::IsRole()){*generation=0;G::BroadcastTokens(b,initial,tokens);}
  Sync::CtaSync();
  auto pf_q=P::template EntryPrefetch<P>(b,initial,shared_tokens,shared_metadata);
#if TILEMEGA_MX_DIAG
  if(blockIdx.x==0&&Sync::IsRole()){auto const& c=LaunchControlV2(launch);if(c.step_ns)c.step_ns[0]=maca::GlobalTimerNs();}
#endif
#pragma unroll 1
  for(unsigned s=0;s<steps;++s) {
    int past=int(s);
    if constexpr(!P::kPrompt){auto const& c=LaunchControlV2(launch);past=G::Past(c.pasts,c.first+s);}
    unsigned stage=0;
#if TILEMEGA_TRACE_STAGE
    {auto const& c=LaunchControlV2(launch);G::Mark(c.trace,P::kPrompt?c.iteration*64+s:c.iteration+s,0,0);}
#endif
    auto barrier=[&] {
      auto const& c=LaunchControlV2(launch);
#if TILEMEGA_TRACE_STAGE
      G::Mark(c.trace,P::kPrompt?c.iteration*64+s:c.iteration+s,stage,1);
#endif
      unsigned long long it;
      if constexpr(P::kPrompt)it=stage==G::kStages-1?c.iteration:c.iteration*64+s;
      else it=c.iteration+s;
      auto const& sc=MoeOperandsV2<P>::Extra(b);
      unsigned const value=P::kPrompt?s*(G::kStages-1)+stage+1:s*G::kStages+stage+1;
      StageBarrierV2<Sync>(sc.barrier+8*std::size_t(stage),104ull*(it+1),generation,value,[&] {
        if(stage<G::kStages-1&&(stage%6==3||stage%6==4))G::BroadcastRouting(b,metadata);
        if constexpr(P::kPrompt) {
          if(stage==G::kStages-2&&s+1<64)G::BroadcastTokens(b,int(s+1),tokens);
        } else if(stage+1==G::kStages&&s+1<c.steps)
          G::BroadcastTokens(b,G::Past(c.pasts,c.first+s+1),tokens);
      },StageWatchV2{sc.watchdog,sc.watchdog_ns,stage,it});
#if TILEMEGA_TRACE_STAGE
      G::Mark(c.trace,P::kPrompt?c.iteration*64+s:c.iteration+s,stage,2);
#endif
      ++stage;
#if TILEMEGA_TRACE_STAGE
      if(stage<G::kStages&&(!P::kPrompt||stage<G::kStages-1||past==63))
        G::Mark(c.trace,P::kPrompt?c.iteration*64+s:c.iteration+s,stage,0);
#endif
    };
    auto run=[&](int stage_class) {
#if TILEMEGA_MX_DIAG
      return !((b.s.diag_skip>>stage_class)&1u);
#else
      (void)stage_class;return true;
#endif
    };
    P::template RunStep<P>(b,past,smem,shared_tokens,shared_metadata,pf_q,barrier,run);
#if TILEMEGA_MX_DIAG
    if(blockIdx.x==0&&Sync::IsRole()) {
      auto const& c=LaunchControlV2(launch);
      if(c.step_ns){if constexpr(P::kPrompt){if(s==63)c.step_ns[1]=maca::GlobalTimerNs();}else c.step_ns[s+1]=maca::GlobalTimerNs();}
    }
#endif
  }
}
}
