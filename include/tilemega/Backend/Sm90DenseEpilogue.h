// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Backend/ServingEpilogue.h>

namespace tilemega::backend {
template<ServingEpilogueOp Op,int TN,int U,bool InputNorm,bool OutputSums>
struct Sm90DenseEpilogue {
  static_assert(Op!=ServingEpilogueOp::kPartial,"native decode does not split K across tasks");
  static_assert(Op!=ServingEpilogueOp::kSwiGLU || TN%(2*U)==0);
 private:
  static constexpr int TileM=16,TileN=TN;
  template<bool Swizzled>
  __device__ static __forceinline__ int Index(int row,int col) {
    return ServingEpilogue<Op,16,TN,U>::template Index<Swizzled>(row,col);
  }
  template <bool Swizzled = false>
  __device__ static __forceinline__ void RunFromTile(
      float* tile, int tile_m, int tile_n, int M, int N, int output_stride,
      int partial_stride,
      cutlass::bfloat16_t* output,
      cutlass::bfloat16_t const* residual = nullptr,
      float* partial = nullptr, float* argmax_value = nullptr,
      int* argmax_index = nullptr,float const* norm_ss = nullptr,
      float* ss_out = nullptr,int norm_k = 0,float norm_eps = 0.0f) {
    static_assert(!Swizzled,"native accumulator tiles are unswizzled");

    if constexpr(Op!=ServingEpilogueOp::kPartial) {
      if constexpr(InputNorm) {
        float* inv=tile+TileM*TileN;
        for(int row=ComputeThread()/8;row<TileM;row+=16) {
          int part=ComputeThread()%8;
          int global_row=tile_m*TileM+row;
          float sum=0.0f;
          if(global_row<M)for(int j=part;j<norm_k/32;j+=8)
            sum+=DeferredNormBlock(norm_ss,global_row,norm_k,j);
          sum+=__shfl_xor_sync(0xffffffff,sum,1);
          sum+=__shfl_xor_sync(0xffffffff,sum,2);
          sum+=__shfl_xor_sync(0xffffffff,sum,4);
          if(part==0)inv[row]=global_row<M?rsqrtf(sum/float(norm_k)+norm_eps):0.0f;
        }
        ComputeSync();
        for(int i=ComputeThread();i<TileM*TileN;i+=kComputeThreads) {
          int row=i/TileN,col=i%TileN;
          tile[Index<Swizzled>(row,col)]*=inv[row];
        }
        ComputeSync();
      }
    }

    if constexpr (Op == ServingEpilogueOp::kArgmaxPartial) {
#if TILEMEGA_EP_PARALLEL_ARGMAX
      float* scratch_values=tile+TileM*TileN;
      int* scratch_indices=reinterpret_cast<int*>(scratch_values+TileM*4);
      int lane=ComputeThread()&31,warp=ComputeThread()/32;
      for(int row=0;row<TileM;++row) {
        float best=-INFINITY;int index=INT32_MAX;
        if(tile_m*TileM+row<M)
          for(int col=ComputeThread();col<TileN;col+=kComputeThreads) {
            int global=tile_n*TileN+col;if(global>=N)continue;
            float value=float(cutlass::bfloat16_t(tile[Index<Swizzled>(row,col)]));
            if(value>best || (value==best && global<index)){best=value;index=global;}
          }
        for(int offset=16;offset;offset>>=1) {
          float other=__shfl_down_sync(0xffffffff,best,offset);
          int oi=__shfl_down_sync(0xffffffff,index,offset);
          if(other>best || (other==best && oi<index)){best=other;index=oi;}
        }
        if(lane==0){scratch_values[row*4+warp]=best;scratch_indices[row*4+warp]=index;}
      }
      ComputeSync();
      for(int row=ComputeThread();row<TileM;row+=kComputeThreads) {
        int global_row=tile_m*TileM+row;if(global_row>=M)continue;
        float best=scratch_values[row*4];int index=scratch_indices[row*4];
        for(int w=1;w<4;++w) {
          float other=scratch_values[row*4+w];int oi=scratch_indices[row*4+w];
          if(other>best || (other==best && oi<index)){best=other;index=oi;}
        }
        argmax_value[global_row*output_stride+tile_n]=best;
        argmax_index[global_row*output_stride+tile_n]=index;
      }
      ComputeSync();
#else
      // Descending rows leave the last row free as four-warp reduction scratch.
      // This avoids adding storage beyond ServingBF16SmemBytes.
      float* scratch_values = tile + (TileM - 1) * TileN;
      int* scratch_indices = reinterpret_cast<int*>(scratch_values + 4);
      for (int row = TileM - 1; row >= 0; --row) {
        int global_row = tile_m * TileM + row;
        if (global_row >= M) continue;
        float best = -INFINITY;
        int best_index = INT32_MAX;
        for (int column = ComputeThread(); column < TileN; column += kComputeThreads) {
          int global_column = tile_n * TileN + column;
          if (global_column >= N) continue;
          float value = float(cutlass::bfloat16_t(tile[Index<Swizzled>(row, column)]));
          if (value > best || (value == best && global_column < best_index)) {
            best = value;
            best_index = global_column;
          }
        }
        for (int offset = 16; offset != 0; offset >>= 1) {
          float other = __shfl_down_sync(0xffffffff, best, offset);
          int other_index = __shfl_down_sync(0xffffffff, best_index, offset);
          if (other > best || (other == best && other_index < best_index)) {
            best = other;
            best_index = other_index;
          }
        }
        ComputeSync();
        if ((ComputeThread() & 31) == 0) {
          scratch_values[ComputeThread() >> 5] = best;
          scratch_indices[ComputeThread() >> 5] = best_index;
        }
        ComputeSync();
        if (ComputeThread() == 0) {
          best = scratch_values[0];
          best_index = scratch_indices[0];
          for (int warp = 1; warp < 4; ++warp) {
            float other = scratch_values[warp];
            int other_index = scratch_indices[warp];
            if (other > best || (other == best && other_index < best_index)) {
              best = other;
              best_index = other_index;
            }
          }
          argmax_value[global_row * output_stride + tile_n] = best;
          argmax_index[global_row * output_stride + tile_n] = best_index;
        }
        ComputeSync();
      }
#endif
    } else if constexpr(Op==ServingEpilogueOp::kSwiGLU && TileN==8) {
      for(int i=ComputeThread();i<TileM*(TileN/2);i+=kComputeThreads) {
        int row=i/(TileN/2),col=i%(TileN/2),gr=tile_m*TileM+row;
        int gc=tile_n*(TileN/2)+col;
        if(gr<M && gc<N/2) {
          int input=2*U*(col/U)+col%U;
          output[gr*output_stride+gc]=ServingEpilogueValue<Op>::Apply(
              tile[Index<Swizzled>(row,input)],tile[Index<Swizzled>(row,input+U)]);
        }
      }
    } else {
      constexpr int kOutputColumns =
          Op == ServingEpilogueOp::kSwiGLU ? TileN / 2 : TileN;
      constexpr int kVectors=TileM*kOutputColumns/8;
      constexpr int kPasses=(kVectors+kComputeThreads-1)/kComputeThreads;
      float square[kPasses]={};
      #pragma unroll
      for (int pass=0;pass<kPasses;++pass) {
        int vector=ComputeThread()+kComputeThreads*pass;
        if(vector>=kVectors)continue;
        int row = vector / (kOutputColumns / 8);
        int out_col = (vector % (kOutputColumns / 8)) * 8;
        int global_row = tile_m * TileM + row;
        int global_col = tile_n * (Op == ServingEpilogueOp::kSwiGLU ? TileN / 2 : TileN)
                         + out_col;
        if (global_row >= M) continue;
        alignas(16) cutlass::bfloat16_t values[8], residual_values[8] = {};
        bool vector_residual = residual && global_col + 8 <= N &&
            (reinterpret_cast<std::uintptr_t>(residual + global_row * output_stride + global_col) & 15) == 0;
        if constexpr (Op == ServingEpilogueOp::kResidual)
          if (vector_residual)
            *reinterpret_cast<uint4*>(residual_values) =
                LoadGlobal16(residual + global_row * output_stride + global_col);
        alignas(16) float acc[8], up_acc[8];
        int input_col = out_col;
        if constexpr (Op == ServingEpilogueOp::kSwiGLU)
          input_col = 2 * U * (out_col / U) + out_col % U;
        if constexpr(Op==ServingEpilogueOp::kSwiGLU && U<8) {
          #pragma unroll
          for(int e=0;e<8;++e) {
            int input=2*U*((out_col+e)/U)+(out_col+e)%U;
            acc[e]=tile[Index<Swizzled>(row,input)];
            up_acc[e]=tile[Index<Swizzled>(row,input+U)];
          }
        }else {
        *reinterpret_cast<float4*>(acc) = *reinterpret_cast<float4 const*>(tile + Index<Swizzled>(row, input_col));
        *reinterpret_cast<float4*>(acc + 4) = *reinterpret_cast<float4 const*>(tile + Index<Swizzled>(row, input_col + 4));
        if constexpr (Op == ServingEpilogueOp::kSwiGLU) {
          *reinterpret_cast<float4*>(up_acc) = *reinterpret_cast<float4 const*>(tile + Index<Swizzled>(row, input_col + U));
          *reinterpret_cast<float4*>(up_acc + 4) = *reinterpret_cast<float4 const*>(tile + Index<Swizzled>(row, input_col + U + 4));
        }
        }
        #pragma unroll
        for (int element = 0; element < 8; ++element) {
          int local_col = out_col + element;
          int col = tile_n * TileN + local_col;
          if constexpr (Op == ServingEpilogueOp::kSwiGLU) {
            values[element] = ServingEpilogueValue<Op>::Apply(acc[element], up_acc[element]);
          } else {
            auto r = residual && col < N
                         ? (vector_residual ? residual_values[element] : residual[global_row * output_stride + col])
                         : cutlass::bfloat16_t{};
            values[element] = ServingEpilogueValue<Op>::Apply(
                acc[element], r);
          }
        }
        if constexpr(Op==ServingEpilogueOp::kResidual) {
          if constexpr(OutputSums) {
            #pragma unroll
            for(int element=0;element<8;++element)if(global_col+element<N) {
              float value=float(values[element]);square[pass]+=value*value;
            }
          }
        }
        int output_n = Op == ServingEpilogueOp::kSwiGLU ? N / 2 : N;
        if (global_col + 8 <= output_n &&
            (reinterpret_cast<std::uintptr_t>(output + global_row * output_stride + global_col) & 15) == 0) {
          // A full vector is one coalesced 16-byte global store.
          StoreGlobal16(output + global_row * output_stride + global_col,
                        *reinterpret_cast<uint4 const*>(values));
        } else {
          for (int element = 0; element < 8; ++element)
            if (global_col + element < output_n)
              output[global_row * output_stride + global_col + element] =
                  values[element];
        }
      }
      if constexpr(Op==ServingEpilogueOp::kResidual) {
#if TILEMEGA_DN_VECTOR_SUMS
        if constexpr(OutputSums) {
          #pragma unroll
          for(int pass=0;pass<kPasses;++pass) {
            int vector=ComputeThread()+kComputeThreads*pass;
            int row=vector/(kOutputColumns/8),col=(vector%(kOutputColumns/8))*8;
            int gr=tile_m*TileM+row,gc=tile_n*kOutputColumns+col;
            if(vector<kVectors && gr<M && gc<N)
              ss_out[gr*((N+7)/8)+gc/8]=square[pass];
          }
        }
#else
        if constexpr(OutputSums) {
          static_assert(kOutputColumns%32==0,
              "native legacy sums require a complete 32-column tile");
          // All lanes participate, including predicated rows/vectors. This
          // matches the fixed xor-1/xor-2 order without rereading output.
          #pragma unroll
          for(int pass=0;pass<kPasses;++pass) {
            float sum=square[pass];
            sum+=__shfl_xor_sync(0xffffffff,sum,1);
            sum+=__shfl_xor_sync(0xffffffff,sum,2);
            int vector=ComputeThread()+kComputeThreads*pass;
            int row=vector/(kOutputColumns/8),col=(vector%(kOutputColumns/8))*8;
            int gr=tile_m*TileM+row,gc=tile_n*kOutputColumns+col;
            if(ComputeThread()%4==0 && vector<kVectors && gr<M && gc<N)
              ss_out[gr*(N/32)+gc/32]=sum;
          }
        }
#endif
      }
    }
  }

 public:
  __device__ static __forceinline__ void Run(float* tile,int task,int batch,int n,int stride,
      cutlass::bfloat16_t* output,cutlass::bfloat16_t const* residual,int* index,
      float const* norm,float* sums,int norm_k,float epsilon) {
    if constexpr(Op==ServingEpilogueOp::kSwiGLU && TN%16) {
      static_assert(TN%(2*U)==0);
      if constexpr(InputNorm) {
        float* inverse=tile+16*TN;
        int row=threadIdx.x/8,part=threadIdx.x%8;
        float sum=0;
        if(row<batch)for(int j=part;j<norm_k/32;j+=8)
          sum+=DeferredNormBlock(norm,row,norm_k,j);
        sum+=__shfl_xor_sync(0xffffffff,sum,1);
        sum+=__shfl_xor_sync(0xffffffff,sum,2);
        sum+=__shfl_xor_sync(0xffffffff,sum,4);
        if(part==0)inverse[row]=row<batch?rsqrtf(sum/float(norm_k)+epsilon):0;
        codegen::executor::ComputeSync();
        for(int i=threadIdx.x;i<16*TN;i+=128)tile[i]*=inverse[i/TN];
        codegen::executor::ComputeSync();
      }
      // TN=8 mod 16 has a half-width that is not an integral eight-element
      // vector. Every logical output must still be written, including its tail.
      for(int i=threadIdx.x;i<16*(TN/2);i+=128) {
        int row=i/(TN/2),col=i%(TN/2),gate=2*U*(col/U)+col%U;
        if(row<batch)output[row*stride+task*(TN/2)+col]=
            ServingEpilogueValue<Op>::Apply(tile[row*TN+gate],tile[row*TN+gate+U]);
      }
    }else RunFromTile<false>(tile,0,task,batch,n,
        stride,n,output,residual,nullptr,reinterpret_cast<float*>(output),index,
        norm,sums,norm_k,epsilon);
  }
};
}  // namespace tilemega::backend
