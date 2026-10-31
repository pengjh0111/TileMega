// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/Sm90DenseMark.cuh>
#include <tilemega/Backend/Sm90DenseStream.cuh>
#include <tilemega/Backend/Sm90DenseOperands.cuh>
#include <tilemega/Backend/Sm90DenseRoles.cuh>
#include <tilemega/Backend/Sm90DenseRing.cuh>
#include <tilemega/Backend/Sm90DenseAttentionMath.cuh>
#include <tilemega/Backend/Sm90DenseAttentionPartition.h>
#include <tilemega/Backend/Sm90DenseLastArriver.cuh>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cuda_bf16.h>

namespace tilemega::backend {

// Decode attention (prompt §5 D2). Task t = (g*Batch + b)*Chunks + c; its
// cache/qkv group is bg = b*Hkv + g. The producer streams each chunk's
// history in BlockKeys blocks by TMA ([D/64][BlockKeys][64] SWIZZLE_128B
// panels); four consumer warps each take 16-key sub-blocks with ldmatrix +
// mma.sync and an online softmax, then merge their states. Keys at or past
// the history bound are masked by selection, so stale or zero rows beyond it
// (the cache is zero-initialized) contribute exactly nothing.
// Dsmem (D10b, clusters): the S chunks of one (b, g) run on consecutive ranks
// of one cluster (S divides the cluster size, one task wave). Each chunk
// stores its normalized partial into the inbox of the chunk-0 CTA through
// DSMEM, then arrives on its inbox barrier; that leader merges and writes
// the context. No global partials, lse or arrival counters are used.
// Norm: 0 none, 1 per head (Qwen3), 2 per projected row (OLMoE).
template<int Batch,int Hkv,int D,int Q,int Norm,int Chunks,int BlockKeys,int Slots,
         int NewRowWork,int Capacity,bool Dsmem=false>
struct Sm90DenseAttentionBody {
  using Element=cutlass::bfloat16_t;
  using Async=codegen::executor::Async<arch::Sm90>;
  using Math=Sm90DenseAttentionMath;
  using Partition=Sm90DenseAttentionPartition<Chunks>;
  // A3: OLMoE is MHA (Q = 1) and normalizes the projected row, not the head.
  static_assert(Batch>=1 && Batch<=16 && (D==64 || D==128) && Q>=1 && Q<=8);
  static_assert(Norm>=0 && Norm<=2,"norm mode is none, head or row");
  static constexpr bool kNormalized=Norm!=0,kRowNorm=Norm==2;
  static_assert(BlockKeys==64 || BlockKeys==128);
  static_assert(Slots>=2 && Slots<=4 && Capacity>=1087);
  static_assert(!Dsmem || Chunks>1,"a DSMEM merge needs at least two chunks");
  static constexpr int kThreads=160,kTasks=Batch*Hkv*Chunks,kPanels=D/64;
  static constexpr float kLog2E=1.4426950408889634f;
  static_assert(kThreads==Sm90DenseRoleAbi::kThreads);
  struct alignas(1024) Storage {
    Element kv[Slots][2][BlockKeys*D];
    Element query[16*D],new_key[16*D],new_value[16*D];
    float output[4*Q*D],maximum[4*Q],sum[4*Q];
    float weight[(Chunks>4?Chunks:4)*Q];
    unsigned last;
    float inbox[Dsmem?Chunks*Q*D:1],inbox_lse[Dsmem?Chunks*Q:1];
  };
  static constexpr int kSharedBytes=1024+int(sizeof(Storage));

  // Byte offset of (row, col) in a [D/64][Rows][64] SWIZZLE_128B tile.
  template<int Rows>
  __device__ static __forceinline__ unsigned Swizzled(int row,int col) {
    return (col/64)*Rows*128+row*128+((((col%64)/8)^(row%8))*16)+(col%8)*2;
  }
  __device__ static __forceinline__ void LoadX4(unsigned (&r)[4],unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
  }
  __device__ static __forceinline__ void LoadX4T(unsigned (&r)[4],unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
  }
  __device__ static __forceinline__ void Mma(float& c0,float& c1,float& c2,float& c3,
      unsigned const (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
  }
  __device__ static __forceinline__ void Mma(float (&c)[4],unsigned const (&a)[4],unsigned b0,unsigned b1) {
    Mma(c[0],c[1],c[2],c[3],a,b0,b1);
  }
  __device__ static __forceinline__ unsigned Pack(float x,float y) {
    __nv_bfloat162 v=__floats2bfloat162_rn(x,y);return *reinterpret_cast<unsigned*>(&v);
  }
  __device__ static __forceinline__ void Decompose(int task,int& bg,int& chunk) {
    int pair=task/Chunks;chunk=task%Chunks;
    int g=pair/Batch,b=pair%Batch;bg=b*Hkv+g;
  }

  // Blocks of the CTA's first task issued by the previous stage (D6): the
  // history predates the current step, so whole blocks can be preissued.
  template<int Prefix>
  __device__ static __forceinline__ int Preissued(int past) {
    if constexpr(Prefix==0)return 0;
    else {
      int bg,chunk;Decompose(int(blockIdx.x),bg,chunk);
      auto range=Partition::Range(past,chunk,NewRowWork);
      int blocks=(max(0,min(range.end,past)-range.begin)+BlockKeys-1)/BlockKeys;
      return min(blocks,Prefix/(4*BlockKeys*D));
    }
  }
  static constexpr int kMaxPreissue=Slots*4*BlockKeys*D;
  template<int Bytes>
  __device__ static __forceinline__ void Preissue(CUtensorMap const* key,CUtensorMap const* value,
      int layer,int past,char* shared,Sm90DenseRingState& state) {
    static_assert(Bytes>0 && Bytes%(4*BlockKeys*D)==0 && Bytes<=kMaxPreissue);
    auto& s=*reinterpret_cast<Storage*>(shared+1024);
    int bg,chunk;Decompose(int(blockIdx.x),bg,chunk);
    int begin=Partition::Range(past,chunk,NewRowWork).begin,count=Preissued<Bytes>(past);
    // The alternate arena and ring retired before the preceding boundary.
    Async::ProxyAsyncGlobalFence();
    // E1: the D6 prefix reads the same history stream, so it takes the same policy.
    std::uint64_t const policy=Sm90DenseStream::WeightPolicy();
    for(int slot=0;slot<count;++slot) {
      Async::ExpectTx(state.full+slot,2*BlockKeys*D*2);
      #pragma unroll
      for(int panel=0;panel<kPanels;++panel) {
        Sm90DenseStream::HistoryTensor4D(&s.kv[slot][0][panel*BlockKeys*64],key,
            panel*64,begin+slot*BlockKeys,bg,layer,state.full+slot,policy);
        Sm90DenseStream::HistoryTensor4D(&s.kv[slot][1][panel*BlockKeys*64],value,
            panel*64,begin+slot*BlockKeys,bg,layer,state.full+slot,policy);
      }
    }
  }

  template<int Prefix>
  __device__ static __forceinline__ void Produce(Sm90DenseAttentionOperands const& p,
      Sm90DenseRingState& state,Storage& s,int tasks,unsigned phase,bool streamed) {
    Sm90DenseRoles::Producer();
    if(threadIdx.x==128) {
      int const past=p.past;int sequence=0,preissued=streamed?Preissued<Prefix>(past):0;
      for(int local=0;local<tasks;++local) {
        int bg,chunk;Decompose(int(blockIdx.x)+local*int(gridDim.x),bg,chunk);
        auto range=Partition::Range(past,chunk,NewRowWork);
        // Earlier steps wrote the history with generic stores (appendix C.1).
        Async::ProxyAsyncGlobalFence();
        for(int begin=range.begin;begin<range.end && begin<past;begin+=BlockKeys,++sequence) {
          int slot=sequence%Slots;
          Async::Wait(state.empty+slot,(phase>>slot)&1);phase^=1u<<slot;
          if(sequence<preissued)continue;
          // A whole box is read: rows past the history bound are masked and
          // rows past Capacity arrive as zeros.
          Async::ExpectTx(state.full+slot,2*BlockKeys*D*2);
          // E1: the history is read once per step, so it takes the same
          // evict_first policy as the weight streams.
          std::uint64_t const policy=Sm90DenseStream::WeightPolicy();
          #pragma unroll
          for(int panel=0;panel<kPanels;++panel) {
            Sm90DenseStream::HistoryTensor4D(&s.kv[slot][0][panel*BlockKeys*64],p.key_map,
                panel*64,begin,bg,p.layer,state.full+slot,policy);
            Sm90DenseStream::HistoryTensor4D(&s.kv[slot][1][panel*BlockKeys*64],p.value_map,
                panel*64,begin,bg,p.layer,state.full+slot,policy);
          }
        }
      }
      if(tasks)for(int i=0;i<Slots;++i)state.producer_generations[i]=(phase>>i)&1;
    }
    __syncwarp();
  }

  // A row norm reduces the token's whole q (Hkv Q D) or k (Hkv D) row, which
  // the qkv buffer interleaves across this token's Hkv groups. All 128
  // consumer threads participate; `scratch` holds one float per warp.
  __device__ static __forceinline__ float RowInverse(Element const* base,int head_rows,
      float* scratch,float epsilon) {
    int tid=threadIdx.x;
    float square=0;
    for(int i=tid*8;i<Hkv*head_rows*D;i+=128*8) {
      int group=i/(head_rows*D),rest=i%(head_rows*D);
      square+=Math::Squares(*reinterpret_cast<uint4 const*>(base+std::size_t(group)*(Q+2)*D+rest));
    }
    #pragma unroll
    for(int offset=16;offset;offset>>=1)square+=__shfl_xor_sync(0xffffffff,square,offset);
    if((tid&31)==0)scratch[tid>>5]=square;
    codegen::executor::ComputeSync();
    float total=scratch[0]+scratch[1]+scratch[2]+scratch[3];
    codegen::executor::ComputeSync();
    return rsqrtf(total/float(Hkv*head_rows*D)+epsilon);
  }

  // MR4: q_norm + RoPE of the Q query heads and, in the chunk holding past,
  // the new key, as items of D/8-lane groups (D128: 8 groups, Q = 8's key in
  // a second round on warp 0's first group). Every global read of the task is
  // issued before any rotation; the new key also writes cache row `past` and
  // the new-row panels.
  static constexpr int kGroupLanes=D/8,kGroups=4*(32/kGroupLanes),kRounds=(Q+1+kGroups-1)/kGroups;
  __device__ static __forceinline__ void Prepare(Sm90DenseAttentionOperands const& p,Storage& s,
      int bg,Sm90DenseKeyRange range) {
    int lane=threadIdx.x&31,group=(threadIdx.x>>5)*(32/kGroupLanes)+lane/kGroupLanes;
    int d=(lane%kGroupLanes)*8,partner=(d+D/2)%D;
    auto const* row=p.qkv+std::size_t(bg)*(Q+2)*D;
    bool fresh=range.Contains(p.past);
    float row_q=1,row_k=1;
    if constexpr(kRowNorm) {
      // bg = b Hkv + g, so this token's group 0 starts (bg mod Hkv) groups back.
      auto const* token=p.qkv+std::size_t(bg-bg%Hkv)*(Q+2)*D;
      row_q=RowInverse(token,Q,s.weight,p.epsilon);
      row_k=RowInverse(token+std::size_t(Q)*D,1,s.weight,p.epsilon);
    }
    uint4 cos=*reinterpret_cast<uint4 const*>(p.cosine+p.past*D+d);
    uint4 sin=*reinterpret_cast<uint4 const*>(p.sine+p.past*D+d);
    uint4 x[kRounds],y[kRounds],w[kRounds],pw[kRounds],value{};
    #pragma unroll
    for(int r=0;r<kRounds;++r) {
      int item=r*kGroups+group;
      x[r]=y[r]=w[r]=pw[r]=uint4{};
      if(item<Q || (item==Q && fresh)) {
        x[r]=*reinterpret_cast<uint4 const*>(row+item*D+d);
        y[r]=*reinterpret_cast<uint4 const*>(row+item*D+partner);
        if constexpr(kNormalized) {
          // A row norm's gamma slice follows the element's position in the
          // whole row; a head norm repeats one head's gamma.
          auto const* gamma=item<Q?p.q_norm:p.k_norm;
          std::size_t head=0;
          if constexpr(kRowNorm)head=item<Q?std::size_t(bg%Hkv)*Q+item:std::size_t(bg%Hkv);
          w[r]=*reinterpret_cast<uint4 const*>(gamma+head*D+d);
          pw[r]=*reinterpret_cast<uint4 const*>(gamma+head*D+partner);
        }
      }
      if(item==Q && fresh)value=*reinterpret_cast<uint4 const*>(row+(Q+1)*D+d);
    }
    #pragma unroll
    for(int r=0;r<kRounds;++r) {
      int item=r*kGroups+group;
      bool key=item==Q;
      float inverse=1;
      if constexpr(kRowNorm)inverse=key?row_k:row_q;
      else if constexpr(kNormalized)inverse=Math::template HeadInverse<D>(x[r],p.epsilon);
      uint4 rotated=Math::template RotateGroup<D,kNormalized>(x[r],y[r],w[r],pw[r],cos,sin,d,inverse);
      if(item<Q)*reinterpret_cast<uint4*>(reinterpret_cast<char*>(s.query)+Swizzled<16>(item,d))=rotated;
      else if(key && fresh) {
        std::size_t cache=(std::size_t(bg)*Capacity+p.past)*D;
        *reinterpret_cast<uint4*>(p.key+cache+d)=rotated;*reinterpret_cast<uint4*>(p.value+cache+d)=value;
        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(s.new_key)+Swizzled<16>(0,d))=rotated;
        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(s.new_value)+Swizzled<16>(0,d))=value;
      }
    }
  }

  struct State {
    float o[D/8][4];
    float maximum[2],sum[2];
  };
  // One 16-key sub-block [first, first+16) of a K/V tile with `rows` rows;
  // keys at index >= bound are masked.
  // U5 remedy (prompt §12): the Q fragment is reloaded from its swizzled tile
  // for every k16 instead of pinning 2*D/4 registers across the chunk.
  template<int Rows>
  __device__ static __forceinline__ void Block(unsigned query,unsigned key,unsigned value,
      int first,int bound,State& st) {
    int lane=threadIdx.x&31,q=lane%4;
    // s[0..3]: keys first+2q+{0,1} (rows r, r+8); s[4..7]: the same 8 keys later.
    float sc[8]={0,0,0,0,0,0,0,0};
    #pragma unroll
    for(int k=0;k<D/16;++k) {
      unsigned a[4],b[4];
      LoadX4(a,query+Swizzled<16>(lane%16,k*16+(lane/16)*8));
      int j=lane/8,row=first+(j/2)*8+lane%8,col=k*16+(j%2)*8;
      LoadX4(b,key+Swizzled<Rows>(row,col));
      Mma(sc[0],sc[1],sc[2],sc[3],a,b[0],b[1]);Mma(sc[4],sc[5],sc[6],sc[7],a,b[2],b[3]);
    }
    float const scale=kLog2E/sqrtf(float(D));
    float block_max[2]={-INFINITY,-INFINITY};
    #pragma unroll
    for(int e=0;e<8;++e) {
      int index=first+(e/4)*8+2*q+(e&1);
      sc[e]=index<bound?sc[e]*scale:-INFINITY;
      block_max[(e>>1)&1]=fmaxf(block_max[(e>>1)&1],sc[e]);
    }
    float alpha[2];
    #pragma unroll
    for(int r=0;r<2;++r) {
      float m=block_max[r];
      m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));
      float next=fmaxf(st.maximum[r],m);
      alpha[r]=isfinite(st.maximum[r])?exp2f(st.maximum[r]-next):0.0f;
      st.maximum[r]=next;
    }
    float increment[2]={0,0};
    #pragma unroll
    for(int e=0;e<8;++e) {
      float v=isfinite(sc[e])?exp2f(sc[e]-st.maximum[(e>>1)&1]):0.0f;
      sc[e]=v;increment[(e>>1)&1]+=v;
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
      float v=increment[r];v+=__shfl_xor_sync(0xffffffff,v,1);v+=__shfl_xor_sync(0xffffffff,v,2);
      st.sum[r]=st.sum[r]*alpha[r]+v;
    }
    #pragma unroll
    for(int n=0;n<D/8;++n) {
      st.o[n][0]*=alpha[0];st.o[n][1]*=alpha[0];st.o[n][2]*=alpha[1];st.o[n][3]*=alpha[1];
    }
    // The S accumulator of two n8 key tiles is the m16k16 A fragment of P.
    unsigned pa[4]={Pack(sc[0],sc[1]),Pack(sc[2],sc[3]),Pack(sc[4],sc[5]),Pack(sc[6],sc[7])};
    #pragma unroll
    for(int n=0;n<D/16;++n) {
      unsigned b[4];
      int j=lane/8,row=first+(j%2)*8+lane%8,col=n*16+(j/2)*8;
      LoadX4T(b,value+Swizzled<Rows>(row,col));
      Mma(st.o[2*n],pa,b[0],b[1]);Mma(st.o[2*n+1],pa,b[2],b[3]);
    }
  }

  __device__ static __forceinline__ void Consume(Sm90DenseAttentionOperands const& p,
      Sm90DenseRingState& state,Storage& s,int tasks,unsigned phase) {
    Sm90DenseRoles::Consumer();
    int const past=p.past;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,sequence=0;
    unsigned query=Async::Shared(s.query);
    // Q rows Q..15 and new-row panel rows 1..15 stay zero for the stage.
    for(int i=threadIdx.x;i<16*D/8;i+=128) {
      int row=i/(D/8),col=(i%(D/8))*8;
      if(row>=Q)*reinterpret_cast<uint4*>(reinterpret_cast<char*>(s.query)+Swizzled<16>(row,col))=uint4{};
      if(row>=1) {
        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(s.new_key)+Swizzled<16>(row,col))=uint4{};
        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(s.new_value)+Swizzled<16>(row,col))=uint4{};
      }
    }
    for(int local=0;local<tasks;++local) {
      int task=int(blockIdx.x)+local*int(gridDim.x),bg,chunk;Decompose(task,bg,chunk);
      auto range=Partition::Range(past,chunk,NewRowWork);
      Prepare(p,s,bg,range);
      codegen::executor::ComputeSync();
      if(threadIdx.x==0)Sm90DenseMarkFirst(0);
      State st;
      #pragma unroll
      for(int n=0;n<D/8;++n)st.o[n][0]=st.o[n][1]=st.o[n][2]=st.o[n][3]=0;
      st.maximum[0]=st.maximum[1]=-INFINITY;st.sum[0]=st.sum[1]=0;
      int bound=min(range.end,past);
      for(int begin=range.begin;begin<bound;begin+=BlockKeys,++sequence) {
        int slot=sequence%Slots;
        Async::Wait(state.full+slot,(phase>>slot)&1);phase^=1u<<slot;
        if(threadIdx.x==0)Sm90DenseMarkFirst(1);
        unsigned key=Async::Shared(s.kv[slot][0]),value=Async::Shared(s.kv[slot][1]);
        #pragma unroll
        for(int sub=0;sub<BlockKeys/64;++sub)
          Block<BlockKeys>(query,key,value,warp*(BlockKeys/4)+sub*16,bound-begin,st);
        // Generic reads before the async-proxy refill (see Sm90DenseGemm).
        Async::FenceProxyAsyncShared();
        Async::Arrive(state.empty+slot);
      }
      if(range.Contains(past) && warp==0)
        Block<16>(query,Async::Shared(s.new_key),Async::Shared(s.new_value),0,1,st);
      // Only rows < Q (all in the first eight rows) carry queries.
      if(lane/4<Q) {
        int row=lane/4;
        #pragma unroll
        for(int n=0;n<D/8;++n) {
          int col=n*8+2*(lane%4);
          s.output[(warp*Q+row)*D+col]=st.o[n][0];s.output[(warp*Q+row)*D+col+1]=st.o[n][1];
        }
        if(lane%4==0) {s.maximum[warp*Q+row]=st.maximum[0];s.sum[warp*Q+row]=st.sum[0];}
      }
      codegen::executor::ComputeSync();
      if(threadIdx.x==0)Sm90DenseMark(2);
      Finish(p,s,bg,chunk);
      if(threadIdx.x==0)Sm90DenseMark(3);
      if constexpr(Dsmem)DsmemMerge(p,state,s,bg,chunk);
      else if constexpr(Chunks>1) {
        bool last=Sm90DenseLastArriver::Publish<Chunks>(p.arrivals+bg,&s.last);
        if(threadIdx.x==0)Sm90DenseMark(4);
        if(last)Merge(p,s,bg);
        if(threadIdx.x==0 && last)Sm90DenseMark(5);
      }
      codegen::executor::ComputeSync();
    }
    if(threadIdx.x==0 && tasks)for(int i=0;i<Slots;++i)state.consumer_generations[i]=(phase>>i)&1;
  }

  // Merge the four warps; S = 1 writes the context, otherwise a normalized
  // partial and its log2-sum-exp for the last-arriver merge (R2 protocol).
  __device__ static __forceinline__ void Finish(Sm90DenseAttentionOperands const& p,Storage& s,
      int bg,int chunk) {
    for(int i=threadIdx.x;i<Q*D;i+=128) {
      int q=i/D;
      float m=-INFINITY;
      #pragma unroll
      for(int w=0;w<4;++w)m=fmaxf(m,s.maximum[w*Q+q]);
      float total=0,value=0;
      #pragma unroll
      for(int w=0;w<4;++w) {
        float weight=isfinite(m) && isfinite(s.maximum[w*Q+q])?exp2f(s.maximum[w*Q+q]-m):0.0f;
        total+=weight*s.sum[w*Q+q];value+=weight*s.output[(w*Q+q)*D+i%D];
      }
      float result=total?value/total:0.0f;
      if constexpr(Chunks==1)p.output[std::size_t(bg)*Q*D+i]=Element(result);
      else if constexpr(Dsmem) {
        // Chunk c of a group runs on cluster rank (rank of chunk 0) + c.
        unsigned leader=Async::ClusterRank()-unsigned(chunk);
        float lse=total?m+log2f(total):-INFINITY;
        if(chunk==0) {
          s.inbox[i]=result;
          if(i%D==0)s.inbox_lse[q]=lse;
        }else {
          Async::StoreCluster(Async::Mapa(s.inbox+chunk*Q*D+i,leader),result);
          if(i%D==0)Async::StoreCluster(Async::Mapa(s.inbox_lse+chunk*Q+q,leader),lse);
        }
      }else {
        p.partial[(std::size_t(bg)*Chunks+chunk)*Q*D+i]=result;
        if(i%D==0)p.lse[(bg*Chunks+chunk)*Q+q]=total?m+log2f(total):-INFINITY;
      }
    }
  }

  // MR4 merge of S chunk partials, shared by the last-arriver and DSMEM forms:
  // `weight` holds all S x Q log2-sum-exps in shared memory; one warp per
  // query turns them into normalized weights in place, then each thread sums
  // one float4 of the context per round, reading 16 partials per batch.
  template<class Partial>
  __device__ static __forceinline__ void Combine(Sm90DenseAttentionOperands const& p,float* weight,
      int bg,Partial partial) {
    // One warp per query, its lanes over the chunks: S Q exp2 for the whole
    // merge, and the reduction stays parallel when Q is small.
    int warp=int(threadIdx.x)>>5,lane=int(threadIdx.x)&31;
    for(int q=warp;q<Q;q+=4) {
      float m=-INFINITY;
      for(int c=lane;c<Chunks;c+=32)m=fmaxf(m,weight[c*Q+q]);
      m=Math::WarpMax(m);
      float total=0;
      for(int c=lane;c<Chunks;c+=32) {
        float w=isfinite(m)?exp2f(weight[c*Q+q]-m):0.0f;weight[c*Q+q]=w;total+=w;
      }
      total=Math::WarpSum(total);
      for(int c=lane;c<Chunks;c+=32)weight[c*Q+q]=total?weight[c*Q+q]/total:0.0f;
    }
    codegen::executor::ComputeSync();
    for(int i=int(threadIdx.x)*4;i<Q*D;i+=128*4) {
      int q=i/D;
      float4 result=make_float4(0,0,0,0);
      #pragma unroll
      for(int first=0;first<Chunks;first+=16) {
        constexpr int kBatch=Chunks<16?Chunks:16;
        float4 v[kBatch];
        #pragma unroll
        for(int j=0;j<kBatch;++j)if(first+j<Chunks)v[j]=partial(first+j,i);
        #pragma unroll
        for(int j=0;j<kBatch;++j)if(first+j<Chunks) {
          float w=weight[(first+j)*Q+q];
          result.x+=w*v[j].x;result.y+=w*v[j].y;result.z+=w*v[j].z;result.w+=w*v[j].w;
        }
      }
      __nv_bfloat162 lo=__floats2bfloat162_rn(result.x,result.y),hi=__floats2bfloat162_rn(result.z,result.w);
      uint2 packed{*reinterpret_cast<unsigned*>(&lo),*reinterpret_cast<unsigned*>(&hi)};
      *reinterpret_cast<uint2*>(p.output+std::size_t(bg)*Q*D+i)=packed;
    }
  }

  // After every consumer's inbox stores (CTA join), one release per peer
  // chunk; the leader acquires all S - 1 and merges in fixed chunk order.
  __device__ static __forceinline__ void DsmemMerge(Sm90DenseAttentionOperands const& p,
      Sm90DenseRingState& state,Storage& s,int bg,int chunk) {
    codegen::executor::ComputeSync();
    auto& cluster=Sm90DenseClusterState::OfRing(state);
    if(chunk) {
      if(threadIdx.x==0)Async::RemoteArrive(Async::Mapa(&cluster.inbox,Async::ClusterRank()-unsigned(chunk)));
      return;
    }
    Async::WaitCluster(&cluster.inbox,cluster.inbox_generation&1);
    // Combine normalizes its lse in place, so copy the inbox's out of the way.
    for(int i=int(threadIdx.x);i<Chunks*Q;i+=128)s.weight[i]=s.inbox_lse[i];
    codegen::executor::ComputeSync();
    // The inbox follows `last` and is only 4-byte aligned.
    Combine(p,s.weight,bg,[&](int c,int i) {
      float const* v=s.inbox+c*Q*D+i;return make_float4(v[0],v[1],v[2],v[3]);
    });
    codegen::executor::ComputeSync();
    if(threadIdx.x==0)cluster.inbox_generation^=1u;
  }

  __device__ static __forceinline__ void Merge(Sm90DenseAttentionOperands const& p,Storage& s,int bg) {
    // s.weight holds max(S, 4) x Q floats: the S x Q lse fit.
    for(int i=threadIdx.x;i<Chunks*Q;i+=128)s.weight[i]=p.lse[std::size_t(bg)*Chunks*Q+i];
    codegen::executor::ComputeSync();
    Combine(p,s.weight,bg,[&](int c,int i) {
      return *reinterpret_cast<float4 const*>(p.partial+(std::size_t(bg)*Chunks+c)*Q*D+i);
    });
  }

  // One outlined call per stage: under the call ABI separate role functions
  // spilled 8-32 B; inlined roles in one callee keep only an 8 B save.
  template<int Prefix=0,class Tail=Sm90DenseNoTail>
  __device__ static __noinline__ void Run(Sm90DenseAttentionOperands const& p,char* shared,
      Sm90DenseRingState* persistent=nullptr,Tail tail={},bool streamed=true) {
    auto& state=persistent?*persistent:*reinterpret_cast<Sm90DenseRingState*>(shared);
    auto& s=*reinterpret_cast<Storage*>(shared+1024);
    int tasks=kTasks/int(gridDim.x)+(int(blockIdx.x)<kTasks%int(gridDim.x));
    unsigned phase=0;
    #pragma unroll
    for(int i=0;i<Slots;++i)phase|=(threadIdx.x<128?
        state.consumer_generations[i]:state.producer_generations[i])<<i;
    if(threadIdx.x<128)Consume(p,state,s,tasks,phase);
    else {
      Produce<Prefix>(p,state,s,tasks,phase,streamed);
      if(threadIdx.x==128)tail();
    }
    __syncthreads();
  }
};

}  // namespace tilemega::backend
