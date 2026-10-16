// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/executor/ComputeGroup.cuh>

#include <tilemega/Backend/ServingVectorIO.h>
#include <tilemega/Backend/ServingDeferredNorm.h>
#include <tilemega/Backend/ServingEpilogueScratch.h>
#include <tilemega/Target/Maca/Bf16.h>
#include <tilemega/Target/Maca/DevicePrimitives.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#ifndef TILEMEGA_EP_DIRECT
#define TILEMEGA_EP_DIRECT 0
#endif

namespace tilemega::backend {

using codegen::executor::ComputeThread;
using codegen::executor::ComputeSync;
using codegen::executor::kComputeThreads;

enum class ServingEpilogueOp {
  kStore, kResidual, kSwiGLU, kArgmaxPartial, kPartial
};

/// The same scalar operation is called by an unsplit GEMM and by its combine
/// task.  Only the input accumulation differs; the BF16 materialization point
/// is therefore independent of split-K.
template <ServingEpilogueOp Op>
struct ServingEpilogueValue {
  __device__ static tilemega::Bf16 Apply(float acc,
                                               tilemega::Bf16 residual = {}) {
    if constexpr (Op == ServingEpilogueOp::kResidual) {
      auto rounded = tilemega::Bf16(acc);
      return tilemega::Bf16(float(rounded) + float(residual));
    } else {
      return tilemega::Bf16(acc);
    }
  }
};

template <>
struct ServingEpilogueValue<ServingEpilogueOp::kSwiGLU> {
  __device__ static tilemega::Bf16 Apply(float gate, float up) {
    float g = float(tilemega::Bf16(gate));
    float v = float(tilemega::Bf16(up));
    auto activated = tilemega::Bf16(g / (1.0f + expf(-g)));
    return tilemega::Bf16(float(activated) * v);
  }
};

template <ServingEpilogueOp Op, int TileM, int TileN, int U = TILEMEGA_SWIGLU_U>
struct ServingEpilogue {
  static_assert(TileN % (2 * U) == 0 || Op != ServingEpilogueOp::kSwiGLU,
                "SwiGLU interleave must end on a gate/up pair");

  // XOR 8-column groups by row: preserve aligned vectors while distributing
  // the MMA accumulator's row groups across shared-memory banks.
  __host__ __device__ static constexpr int SharedIndex(int row, int col) {
    if constexpr(TileN<32)return row*TileN+col;
    else return row * TileN + (col ^ ((row & 3) * 8));
  }

  template <bool Swizzled>
  __device__ static int Index(int row, int col) {
    return Swizzled ? SharedIndex(row, col) : row * TileN + col;
  }

  // The MMA bodies materialize their accumulators into the FP32 tile
  // themselves (no CuTe partitions on xcore1000) and then call RunFromTile.

  template <bool Swizzled = false>
  __device__ static void RunFromTile(
      float* tile, int tile_m, int tile_n, int M, int N, int output_stride,
      int partial_stride,
      tilemega::Bf16* output,
      tilemega::Bf16 const* residual = nullptr,
      float* partial = nullptr, float* argmax_value = nullptr,
      int* argmax_index = nullptr,float const* norm_ss = nullptr,
      float* ss_out = nullptr,int norm_k = 0,float norm_eps = 0.0f) {

    if constexpr(Op!=ServingEpilogueOp::kPartial) {
      if(norm_ss) {
        float* inv=tile+TileM*TileN;
        for(int row=ComputeThread()/8;row<TileM;row+=16) {
          int part=ComputeThread()%8;
          int global_row=tile_m*TileM+row;
          float sum=0.0f;
          if(global_row<M)for(int j=part;j<norm_k/32;j+=8)
            sum+=DeferredNormBlock(norm_ss,global_row,norm_k,j);
          sum+=__shfl_xor_sync(maca::kFullWaveMask,sum,1);
          sum+=__shfl_xor_sync(maca::kFullWaveMask,sum,2);
          sum+=__shfl_xor_sync(maca::kFullWaveMask,sum,4);
          if(part==0)inv[row]=global_row<M?maca::Rsqrt(sum/float(norm_k)+norm_eps):0.0f;
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
            float value=float(tilemega::Bf16(tile[Index<Swizzled>(row,col)]));
            if(value>best || (value==best && global<index)){best=value;index=global;}
          }
        for(int offset=16;offset;offset>>=1) {
          float other=__shfl_down_sync(maca::kFullWaveMask,best,offset,32);
          int oi=__shfl_down_sync(maca::kFullWaveMask,index,offset,32);
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
          float value = float(tilemega::Bf16(tile[Index<Swizzled>(row, column)]));
          if (value > best || (value == best && global_column < best_index)) {
            best = value;
            best_index = global_column;
          }
        }
        for (int offset = 16; offset != 0; offset >>= 1) {
          float other = __shfl_down_sync(maca::kFullWaveMask, best, offset, 32);
          int other_index = __shfl_down_sync(maca::kFullWaveMask, best_index, offset, 32);
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
    } else if constexpr (Op == ServingEpilogueOp::kPartial) {
      if (partial_stride < N) { maca::Trap(); return; }
      for (int index = ComputeThread() * 4; index < TileM * TileN; index += kComputeThreads * 4) {
        int row = index / TileN, col = index % TileN;
        int global_row = tile_m * TileM + row;
        int global_col = tile_n * TileN + col;
        if (global_row >= M || global_col >= N) continue;
        float4 values = *reinterpret_cast<float4 const*>(tile + Index<Swizzled>(row, col));
        float* dst = partial + global_row * partial_stride + global_col;
        if (global_col + 4 <= N && (reinterpret_cast<std::uintptr_t>(dst) & 15) == 0)
          *reinterpret_cast<float4*>(dst) = values;
        else {
          float lane[4] = {values.x, values.y, values.z, values.w};
          for (int i = 0; i < 4 && global_col + i < N; ++i) dst[i] = lane[i];
        }
      }
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
        alignas(16) tilemega::Bf16 values[8], residual_values[8] = {};
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
                         : tilemega::Bf16{};
            values[element] = ServingEpilogueValue<Op>::Apply(
                acc[element], r);
          }
        }
        if constexpr(Op==ServingEpilogueOp::kResidual) {
          if(ss_out) {
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
        if(ss_out) {
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
        if constexpr(kOutputColumns%32!=0) {
          // DN's producer owns a complete 32-column sum-of-squares block.
          // Small GEMV tiles may use residual only when no DN writer is attached.
          if(ss_out)maca::Trap();
        }else if(ss_out) {
          // All lanes participate, including predicated rows/vectors. This
          // matches the fixed xor-1/xor-2 order without rereading output.
          #pragma unroll
          for(int pass=0;pass<kPasses;++pass) {
            float sum=square[pass];
            sum+=__shfl_xor_sync(maca::kFullWaveMask,sum,1);
            sum+=__shfl_xor_sync(maca::kFullWaveMask,sum,2);
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
};

}  // namespace tilemega::backend
