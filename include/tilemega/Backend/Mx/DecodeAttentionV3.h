// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Mx/DecodeAttentionV2.h>
#include <tilemega/Codegen/tasks/mx/StageSyncV3.cuh>

namespace tilemega::backend::mx {
struct MxAttentionGeometryV3 {
  int blocks, head_begin, head_end, block_begin, block_end, split_index;
  TM_MX_HD constexpr MxAttentionGeometryV3(int heads,int past,int grid,int cta,int split)
      :blocks((past+32)/32),head_begin(heads),head_end(heads),block_begin(0),block_end(0),split_index(0) {
#if defined(__CUDA_ARCH__) || defined(__MACA_ARCH__)
    __builtin_assume(cta>=0&&cta<grid);
#endif
    if(split>1) {
      if(cta<heads*split) {
        head_begin=cta/split;head_end=head_begin+1;split_index=cta%split;
        block_begin=split_index*blocks/split;block_end=(split_index+1)*blocks/split;
      }
    } else if(heads<=grid) {
      if(cta<heads){head_begin=cta;head_end=cta+1;block_end=blocks;}
    } else {
      head_begin=cta*heads/grid;head_end=(cta+1)*heads/grid;block_end=blocks;
    }
  }
  TM_MX_HD constexpr int Heads()const{return head_end-head_begin;}
  TM_MX_HD constexpr int FirstWave(int i)const{return (8*i+Heads()-1)/Heads();}
  TM_MX_HD constexpr int HeadIndex(int wave)const{return wave*Heads()/8;}
  TM_MX_HD constexpr int Waves(int i)const{return FirstWave(i+1)-FirstWave(i);}
};

template<int D,int Heads,int Q>
struct MxProjectionLoadsV3 {
  static_assert(Q==1&&(Heads*D)%512==0);
  static constexpr int kPackets=Heads*D/512;
  uint4 values[kPackets];
  __device__ __forceinline__ void Load(Bf16 const* qkv,int batch,int component,int lane) {
#pragma unroll
    for(int i=0;i<kPackets;++i) {
      int const at=i*512+lane*8;
      values[i]=LoadGlobal16Gvm(qkv+((batch*Heads+at/D)*(Q+2)+component)*D+at%D);
    }
  }
  __device__ __forceinline__ float Inverse(float epsilon,float* exchange,int lane)const {
    float square=0;
#pragma unroll
    for(int i=0;i<kPackets;++i) {
      alignas(16) Bf16 elements[8];*reinterpret_cast<uint4*>(elements)=values[i];
      square+=codegen::mx::SquareEight(elements);
    }
    exchange[lane]=square;
    __syncwarp();
#pragma unroll
    for(int distance=32;distance;distance/=2) {
      if(lane<distance)exchange[lane]=exchange[lane]+exchange[lane+distance];
      __syncwarp();
    }
    return maca::Rsqrt(exchange[0]/float(Heads*D)+epsilon);
  }
};

template<int D,int Q,bool QkNorm,int PrefetchKind=2,int Capacity=1088,int RowNormHeads=0,int Split=1>
struct MxAttentionV3 : MxAttentionV2<D,Q,1,QkNorm,PrefetchKind,Capacity,RowNormHeads> {
  using Base=MxAttentionV2<D,Q,1,QkNorm,PrefetchKind,Capacity,RowNormHeads>;
  using typename Base::Prefetched;
  using typename Base::Block;
  using typename Base::Layout;
  using Base::kChunks;using Base::kAtoms;using Base::kKLoads;using Base::kVLoads;
  using Base::kWaveFloats;using Base::kSharedBytes;
  using Base::LoadK;using Base::LoadV;using Base::MaskV;
  static constexpr int kSplit=Split;
  static_assert(Split==1||Split==2||Split==4);

  template<class Shape>
  __device__ __forceinline__ static Prefetched Prefetch(MxAttentionOperandsV2 const& p,Shape const& s) {
    Prefetched pf{};
    unsigned const t=StageThread();__builtin_assume(t<unsigned(Shape::kThreads));
    int const wave=__builtin_mxc_readfirstlane(int(t/64)),lane=t%64;
    MxAttentionGeometryV3 const geo(s.b*s.hkv,p.past,int(gridDim.x),StageBlock(),Split);
    if(wave<8&&geo.Heads()) {
      int const i=geo.HeadIndex(wave),j=geo.block_begin+wave-geo.FirstWave(i),head=geo.head_begin+i;
      if(j<geo.block_end) {
        Block block;
        if constexpr(PrefetchKind) {
          LoadK(block,p,head,j,lane);
#pragma unroll
          for(int a=0;a<2;++a)
#pragma unroll
            for(int dc=0;dc<kChunks;++dc)pf.k[a][dc]=block.k[a][dc];
        }
        if constexpr(PrefetchKind==2) {
          LoadV(block,p,head,j,lane);
#pragma unroll
          for(int n=0;n<kAtoms;++n)pf.v[n]=block.v[n];
        }
      }
    }
    return pf;
  }

  template<class Shape>
  __device__ __forceinline__ static void Wave(Prefetched const& pf,MxAttentionOperandsV2 const& p,
      Shape const&,MxAttentionGeometryV3 const& geo,float* shared) {
    int const t=StageThread();__builtin_assume(unsigned(t)<unsigned(Shape::kThreads));
    int const wave=__builtin_mxc_readfirstlane(t/64),lane=t%64;
    if(wave>=8||!geo.Heads())return;
    int const i=geo.HeadIndex(wave),head=geo.head_begin+i,waves=geo.Waves(i);
    int const first=geo.block_begin+wave-geo.FirstWave(i),end=geo.block_end;
    Block block{};
    int const r = lane & 15, g = lane >> 4;
    float* const workspace = shared + min(wave, 7) * kWaveFloats;
    float* const exchange = workspace;
    auto* const new_value = reinterpret_cast<Bf16*>(workspace + 64);
    float* const wave_result = workspace + 64 + D / 2;
    int const count=max(0,(end-first+waves-1)/waves);
    uint4 query[kChunks];
    uint4 new_key[kChunks];
    unsigned new_values[kAtoms];
    if (wave < 8 && count) {
      float query_inverse=1.f,key_inverse=1.f;
      using Projection=MxProjectionLoadsV3<D,RowNormHeads?RowNormHeads:8,RowNormHeads?Q:1>;
      Projection row_q{},row_k{};
      uint4 cosine[kChunks], sine[kChunks], weights[kChunks], key_weights[kChunks];
      uint4 new_v{};
      int const offset = p.past % 32, owner_r = 4 * (offset / 8) + offset % 4;
      bool const owner=geo.blocks-1>=first&&(geo.blocks-1-first)%waves==0&&end==geo.blocks;
#pragma unroll
      for (int dc = 0; dc < kChunks; ++dc) {
        int const at = dc * 32 + 8 * g;
        cosine[dc] = LoadGlobal16Gvm(p.cosine + p.past * D + at);
        sine[dc] = LoadGlobal16Gvm(p.sine + p.past * D + at);
        query[dc] = LoadGlobal16Gvm(p.qkv + (head * (Q + 2) + min(r, Q - 1)) * D + at);
        weights[dc] = uint4{};
        if constexpr (QkNorm) weights[dc] = LoadGlobal16Gvm(p.q_norm + (RowNormHeads?head%RowNormHeads*Q*D:0) + at);
        new_key[dc] = uint4{};
        key_weights[dc] = uint4{};
      }
      if (owner) {
#pragma unroll
        for (int dc = 0; dc < kChunks; ++dc) {
          int const at = dc * 32 + 8 * g;
          if (r == owner_r) new_key[dc] = LoadGlobal16Gvm(p.qkv + (head * (Q + 2) + Q) * D + at);
          if constexpr (QkNorm) key_weights[dc] = LoadGlobal16Gvm(p.k_norm + (RowNormHeads?head%RowNormHeads*D:0) + at);
        }
        if (lane < D / 8)
          new_v = LoadGlobal16Gvm(p.qkv + (head * (Q + 2) + Q + 1) * D + lane * 8);
      }
      if constexpr(RowNormHeads) {
        row_q.Load(p.qkv,head/RowNormHeads,0,lane);
        row_k.Load(p.qkv,head/RowNormHeads,Q,lane);
      }
      if constexpr(PrefetchKind) {
#pragma unroll
        for(int a=0;a<2;++a)
#pragma unroll
          for(int dc=0;dc<kChunks;++dc)block.k[a][dc]=pf.k[a][dc];
      } else LoadK(block,p,head,first,lane);
      if constexpr(PrefetchKind==2) {
#pragma unroll
        for(int n=0;n<kAtoms;++n)block.v[n]=pf.v[n];
      } else LoadV(block,p,head,first,lane);
      asm volatile("" ::: "memory");
      if constexpr(PrefetchKind==0)__builtin_mxc_arrive_gvmcnt(kKLoads+kVLoads);
      else if constexpr(PrefetchKind==1)__builtin_mxc_arrive_gvmcnt(kVLoads);
      else __builtin_mxc_arrive_gvmcnt(0);
      if constexpr(RowNormHeads) {
        query_inverse=row_q.Inverse(p.epsilon,exchange,lane);
        key_inverse=row_k.Inverse(p.epsilon,exchange,lane);
      }
      if (owner) {
        MxRowTransformV2<D, QkNorm, (RowNormHeads>0)>(new_key, key_weights, cosine, sine, p.epsilon, exchange, lane,key_inverse);
        __syncwarp();
      }
      MxRowTransformV2<D, QkNorm, (RowNormHeads>0)>(query, weights, cosine, sine, p.epsilon, exchange, lane,query_inverse);
      __syncwarp();
      if (owner && lane < D / 8)
        *reinterpret_cast<uint4*>(new_value + lane * 8) = new_v;
      __syncwarp();
#pragma unroll
      for (int n = 0; n < kAtoms; ++n) {
        unsigned value = owner ? unsigned(new_value[16 * n + r].raw()) : 0u;
        // Retire this LDS value before K_next is issued. Combining a fresh
        // LDS load with a V fragment made MACA insert ARRIVE 4160 (MIR).
        asm volatile("" : "+r"(value));
        new_values[n] = value;
      }
      unsigned mask = 0u - unsigned(r < Q);
      asm volatile("" : "+r"(mask));
#pragma unroll
      for (int dc = 0; dc < kChunks; ++dc)
        query[dc] = uint4{query[dc].x & mask, query[dc].y & mask, query[dc].z & mask, query[dc].w & mask};

      MmaAccum out[kAtoms];
#pragma unroll
      for (int n = 0; n < kAtoms; ++n) out[n] = MmaAccum{0.f, 0.f, 0.f, 0.f};
      float maximum = -INFINITY, lambda = 0;
#pragma unroll 1
      for (int index = 0; index < count; ++index) {
        int const j=first+waves*index;
        int const next_head=head;
        int const next_j=j+waves;
        bool const has_next=index+1<count;
        block.valid = has_next;
        __builtin_mxc_arrive_gvmcnt(kVLoads);
        if (j == geo.blocks - 1) {
#pragma unroll
          for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int dc = 0; dc < kChunks; ++dc)
              if (a == offset % 8 / 4 && r == owner_r) block.k[a][dc] = new_key[dc];
          // DT-KV: the owner writes the complete block, including unchanged rows.
          // Keep whole-block store addresses out of the MMA live range.
          int const store_cap = Opaque(p.capacity);
#pragma unroll
          for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int dc = 0; dc < kChunks; ++dc)
              MxStore16V2(p.k_cache + Layout::KWave(head, j, dc, a, lane, store_cap), block.k[a][dc]);
        }
        float score[8];
#pragma unroll
        for (int a = 0; a < 2; ++a) {
          MmaAccum acc{0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int dc = 0; dc < kChunks; ++dc) {
            acc = Mma(Lo(block.k[a][dc]), Lo(query[dc]), acc);
            acc = Mma(Hi(block.k[a][dc]), Hi(query[dc]), acc);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i)
            score[4 * a + i] = j * 32 + 8 * g + 4 * a + i <= p.past
                               ? acc[i] * (codegen::mx::kLog2e / sqrtf(float(D))) : -INFINITY;
        }
        // X-A: one K slot is reused only after its S^T consumer.
        asm volatile("" :: "r"(score[7]) : "memory");
        if (has_next) LoadK(block, p, next_head, next_j, lane);
        float block_max = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) block_max = fmaxf(block_max, score[i]);
        exchange[g * 16 + r] = block_max;
        __syncwarp();
        block_max = fmaxf(fmaxf(exchange[r], exchange[16 + r]), fmaxf(exchange[32 + r], exchange[48 + r]));
        float const new_max = fmaxf(maximum, block_max), alpha = maca::Exp2(maximum - new_max);
        // All groups agree; only group zero publishes the O-layout scalars.
        __syncwarp();
        if (!g) exchange[r] = alpha;
        __syncwarp();
        float output_alpha[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) output_alpha[i] = exchange[4 * g + i];
        alignas(16) Bf16 probability[8];
        float block_sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float const e = maca::Exp2(score[i] - new_max);
          probability[i] = Bf16(e); block_sum += e;
        }
        uint4 const prob = *reinterpret_cast<uint4 const*>(probability);
        if (has_next) __builtin_mxc_arrive_gvmcnt(kKLoads);
        else __builtin_mxc_arrive_gvmcnt(0);
        if (j == geo.blocks - 1) {
#pragma unroll
          for (int n = 0; n < kAtoms; ++n) {
            alignas(16) Bf16 v[8];
            *reinterpret_cast<uint4*>(v) = block.v[n];
#pragma unroll
            for (int e = 0; e < 8; ++e)
              v[e] = Bf16::bitcast(g == offset / 8 && e == offset % 8 ? new_values[n] : unsigned(v[e].raw()));
            block.v[n] = *reinterpret_cast<uint4 const*>(v);
          }
        }
#pragma unroll
        for (int n = 0; n < kAtoms; ++n) {
#pragma unroll
          for (int i = 0; i < 4; ++i) out[n][i] *= output_alpha[i];
          uint4 const value = MaskV(block.v[n], j, p.past, g);
          out[n] = Mma(Lo(prob), Lo(value), out[n]);
          out[n] = Mma(Hi(prob), Hi(value), out[n]);
        }
        if (j == geo.blocks - 1) {
          int const store_cap = Opaque(p.capacity);
#pragma unroll
          for (int n = 0; n < kAtoms; ++n)
            MxStore16V2(p.v_cache + Layout::VWave(head, j, n, lane, store_cap), block.v[n]);
        }
        asm volatile("" :: "r"(out[kAtoms - 1][3]) : "memory");
        if (has_next) LoadV(block, p, next_head, next_j, lane);
        lambda = alpha * lambda + block_sum; maximum = new_max;
      }
      exchange[g * 16 + r] = lambda;
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int const q = 4 * g + i;
        if (q < Q) {
#pragma unroll
          for (int n = 0; n < kAtoms; ++n) wave_result[q * (D + 2) + 16 * n + r] = out[n][i];
        }
      }
      if (!g && r < Q) {
        wave_result[r * (D + 2) + D] = maximum;
        wave_result[r * (D + 2) + D + 1] = ((exchange[r] + exchange[16 + r]) + exchange[32 + r]) + exchange[48 + r];
      }

    }
  }

  __device__ __forceinline__ static void LocalMerge(MxAttentionOperandsV2 const& p,
      MxAttentionGeometryV3 const& geo,int t,float* shared) {
    if(t>=512)return;
    constexpr int columns=Split==1?D:D+2;
    for(int item=t;item<geo.Heads()*Q*columns;item+=512) {
      int const i=item/(Q*columns),within=item%(Q*columns),q=within/columns,d=within%columns;
      int const first=geo.FirstWave(i),end=geo.FirstWave(i+1),head=geo.head_begin+i;
      float maximum=-INFINITY;
#pragma unroll
      for(int w=0;w<8;++w)
        if(w>=first&&w<end&&geo.block_begin+w-first<geo.block_end)
          maximum=fmaxf(maximum,shared[w*kWaveFloats+64+D/2+q*(D+2)+D]);
      float result=0,lambda=0;
#pragma unroll
      for(int w=0;w<8;++w) {
        if(w>=first&&w<end&&geo.block_begin+w-first<geo.block_end) {
          float const* values=shared+w*kWaveFloats+64+D/2+q*(D+2);
          float const scale=maca::Exp2(values[D]-maximum);
          result+=values[d]*scale;
          if constexpr(Split==1)lambda+=values[D+1]*scale;
        }
      }
      if constexpr(Split==1)p.ctx[(head*Q+q)*D+d]=Bf16(result/lambda);
      else p.partial[((head*Split+geo.split_index)*Q+q)*(D+2)+d]=d==D?maximum:result;
    }
  }

  __device__ __forceinline__ static void SplitMerge(MxAttentionOperandsV2 const& p,int head,int t) {
    constexpr int items=(Q*D+511)/512;
    float maxima[items][Split],lambdas[items][Split],outputs[items][Split];
    if(t<512) {
#pragma unroll
      for(int i=0;i<items;++i) {
        int const item=t+i*512,q=item/D,d=item%D;
        if(item<Q*D) {
#pragma unroll
          for(int split=0;split<Split;++split) {
            float const* base=p.partial+((head*Split+split)*Q+q)*(D+2);
            maxima[i][split]=LoadFloatGvm(base+D);
            lambdas[i][split]=LoadFloatGvm(base+D+1);
            outputs[i][split]=LoadFloatGvm(base+d);
          }
        }
      }
      // All partial fields enter one batch; merge arithmetic has no CTA exchange.
      asm volatile("" ::: "memory");
      __builtin_mxc_arrive_gvmcnt(0);
#pragma unroll
      for(int i=0;i<items;++i) {
        int const item=t+i*512;
        if(item<Q*D) {
          float maximum=-INFINITY;
#pragma unroll
          for(int split=0;split<Split;++split)maximum=fmaxf(maximum,maxima[i][split]);
          float result=0,lambda=0;
#pragma unroll
          for(int split=0;split<Split;++split) {
            float const scale=maca::Exp2(maxima[i][split]-maximum);
            result+=outputs[i][split]*scale;lambda+=lambdas[i][split]*scale;
          }
          p.ctx[head*Q*D+item]=Bf16(result/lambda);
        }
      }
    }
  }

  template<class Shape,class Next>
  __device__ __forceinline__ static auto Run(Prefetched pf,MxAttentionOperandsV2 const& p,
      Shape const& s,char* smem,Next next) {
    using Sync=typename Shape::Sync;
    int const t=StageThread();__builtin_assume(unsigned(t)<unsigned(Shape::kThreads));
    MxAttentionGeometryV3 const geo(s.b*s.hkv,p.past,int(gridDim.x),StageBlock(),Split);
    float* const shared=reinterpret_cast<float*>(smem);
    Wave(pf,p,s,geo,shared);
    Sync::LdsSync();
    LocalMerge(p,geo,t,shared);
    if constexpr(Split>1) {
      if(geo.Heads()) {
        auto* flag=reinterpret_cast<unsigned*>(shared+kSharedBytes/4-4);
        codegen::executor::LastArriverV2<Sync>::Run(p.tickets+geo.head_begin,Split,flag,[&] {
          SplitMerge(p,geo.head_begin,t);
        });
      }
    }
    Sync::CtaSync();
    using Result=decltype(next());
    if(Sync::IsCompute())return next();
    return Result{};
  }
};
}  // namespace tilemega::backend::mx
