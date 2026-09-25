// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/ServingAttentionMma.h>

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

template <class Arch, int kHeadDim, int kQPerKV, int kTokens,
          int kQRows, int kKvTile, bool kQkNorm>
struct FusedAttentionTaskBody {
  static_assert(kHeadDim == 64 || kHeadDim == 128);
  static_assert(kQPerKV == 2 || kQPerKV == 4);
  static_assert(kTokens == 1 || kTokens == 64);
  static_assert(kQRows > 0 && kQRows % 16 == 0 || kTokens == 1);
  static_assert(kKvTile == 32 || kKvTile == 64);
  using Element = cutlass::bfloat16_t;
  using QkMma = backend::ServingAttentionMma<Arch, 16, 64, kHeadDim>;
  using PvMma = backend::ServingAttentionMma<Arch, 16, kHeadDim, 64, true>;
  static constexpr int kQueryExtent = kTokens * kQPerKV;

  struct SharedStorage {
    alignas(16) Element query[QkMma::kAElements];
    alignas(16) Element key[QkMma::kBElements];
    alignas(16) Element probability[PvMma::kAElements];
    alignas(16) Element value[PvMma::kBElements];
    alignas(16) Element raw[64 * kHeadDim];
    alignas(16) float score[16 * 64];
    alignas(16) float product[16 * kHeadDim];
    alignas(16) float accumulated[16 * kHeadDim];
    float row_max[16], row_sum[16];
  };

  // The cache path uses one 16-byte cp.async.cg per lane and never issues an
  // out-of-bounds transaction. New positions are formed directly from QKV.
  __device__ static void CopyCached(Element const* source, Element* target,
                                    int elements) {
    for (int vector = int(threadIdx.x); vector < elements / 8;
         vector += int(blockDim.x)) {
      unsigned address = static_cast<unsigned>(
          __cvta_generic_to_shared(target + 8 * vector));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                   "r"(address), "l"(source + 8 * vector));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
  }

  __device__ static float WarpSum(float value) {
    for (int shift = 16; shift; shift >>= 1)
      value += __shfl_down_sync(0xffffffffu, value, shift);
    return __shfl_sync(0xffffffffu, value, 0);
  }

  __device__ static Element NormalizeRotate(Element const* x,
                                             Element const* norm,
                                             Element const* cosine,
                                             Element const* sine,
                                             float inverse, int dimension) {
    int partner = dimension < kHeadDim / 2
                      ? dimension + kHeadDim / 2 : dimension - kHeadDim / 2;
    Element a = x[dimension], b = x[partner];
    if constexpr (kQkNorm) {
      a = Element(float(Element(float(a) * inverse)) * float(norm[dimension]));
      b = Element(float(Element(float(b) * inverse)) * float(norm[partner]));
    }
    float rotated = dimension < kHeadDim / 2 ? -float(b) : float(b);
    Element first = Element(float(a) * float(cosine[dimension]));
    Element second = Element(rotated * float(sine[dimension]));
    return Element(float(first) + float(second));
  }

  __device__ static void PrepareRows(ServingAttentionOperands const& p,
                                     SharedStorage& smem, int batch, int group,
                                     int query_begin, int key_begin,
                                     int key_limit) {
    auto q = cute::make_tensor(cute::make_smem_ptr(smem.query),
                               typename QkMma::LayoutA{});
    auto k = cute::make_tensor(cute::make_smem_ptr(smem.key),
                               typename QkMma::LayoutB{});
    int lane = int(threadIdx.x) & 31;
    int warp = int(threadIdx.x) >> 5;
    int group_width = (kQPerKV + 2) * kHeadDim;
    for (int local = warp; local < 16; local += 4) {
      int row = query_begin + local;
      int token = row / kQPerKV;
      int head = row % kQPerKV;
      bool valid = row < kQueryExtent;
      Element const* x = valid ? p.qkv +
          (batch * kTokens + token) * p.heads_kv * group_width +
          group * group_width + head * kHeadDim : nullptr;
      float square = 0.0f;
      if constexpr (kQkNorm)
        for (int d = lane; d < kHeadDim; d += 32)
          if (valid) square += float(x[d]) * float(x[d]);
      float inverse = kQkNorm ? rsqrtf(WarpSum(square) / kHeadDim + p.epsilon) : 1.0f;
      for (int d = lane; d < kHeadDim; d += 32)
        q(local, d) = valid ? NormalizeRotate(
            x, p.q_norm, p.cosine + (p.past + token) * kHeadDim,
            p.sine + (p.past + token) * kHeadDim, inverse, d) : Element(0.0f);
    }
    for (int local = warp; local < 64; local += 4) {
      int position = key_begin + local;
      bool valid = local < kKvTile && position < key_limit;
      if (!valid || position < p.past) {
        if (!valid)
          for (int d = lane; d < kHeadDim; d += 32) k(local, d) = Element(0.0f);
        continue;
      }
      int token = position - p.past;
      Element const* x = p.qkv +
          (batch * kTokens + token) * p.heads_kv * group_width +
          group * group_width + kQPerKV * kHeadDim;
      float square = 0.0f;
      if constexpr (kQkNorm)
        for (int d = lane; d < kHeadDim; d += 32)
          square += float(x[d]) * float(x[d]);
      float inverse = kQkNorm ? rsqrtf(WarpSum(square) / kHeadDim + p.epsilon) : 1.0f;
      bool owns_write = query_begin % kQRows == 0 &&
                        token * kQPerKV >= query_begin &&
                        token * kQPerKV < query_begin + kQRows;
      for (int d = lane; d < kHeadDim; d += 32) {
        Element output = NormalizeRotate(
            x, p.k_norm, p.cosine + position * kHeadDim,
            p.sine + position * kHeadDim, inverse, d);
        k(local, d) = output;
        if (owns_write)
          p.key_cache[((batch * p.heads_kv + group) * p.capacity + position) *
                      kHeadDim + d] = output;
      }
    }
    __syncthreads();
  }

  __device__ static void Run(ServingAttentionOperands const& p,
                             SharedStorage& smem,
                             int batch, int group, int query_block,
                             int cache_block) {
    int block_begin = cache_block * p.block_extent;
    int block_limit = min((cache_block + 1) * p.block_extent,
                          p.past + kTokens);
    if (block_begin >= block_limit) return;
    int q_begin = query_block * kQRows;
    int cmax = (p.capacity + p.block_extent - 1) / p.block_extent;
    for (int query_begin = q_begin;
         query_begin < min(q_begin + kQRows, kQueryExtent);
         query_begin += 16) {
      for (int i = int(threadIdx.x); i < 16 * kHeadDim; i += 128)
        smem.accumulated[i] = 0.0f;
      for (int i = int(threadIdx.x); i < 16; i += 128) {
        smem.row_max[i] = -INFINITY;
        smem.row_sum[i] = 0.0f;
      }
      __syncthreads();
      for (int key_begin = block_begin; key_begin < block_limit;
           key_begin += kKvTile) {
        PrepareRows(p, smem, batch, group, query_begin, key_begin,
                    block_limit);
        auto key = cute::make_tensor(cute::make_smem_ptr(smem.key),
                                     typename QkMma::LayoutB{});
        // Old cache rows are fetched with 16-byte cp.async.cg transactions.
        // Each source row is contiguous; the swizzled MMA layout is populated
        // after the asynchronous copy completes.
        for (int row = 0; row < kKvTile; ++row) {
          int position = key_begin + row;
          if (position >= min(block_limit, p.past)) break;
          CopyCached(p.key_cache +
              ((batch * p.heads_kv + group) * p.capacity + position) * kHeadDim,
              smem.raw, kHeadDim);
          for (int d = int(threadIdx.x); d < kHeadDim; d += 128)
            key(row, d) = smem.raw[d];
          __syncthreads();
        }
        QkMma::Run(smem.query, smem.key, smem.score, 64);
        __syncthreads();
        auto probability = cute::make_tensor(
            cute::make_smem_ptr(smem.probability),
            typename PvMma::LayoutA{});
        for (int row = int(threadIdx.x); row < 16; row += 128) {
          int global_row = query_begin + row;
          int token = global_row / kQPerKV;
          float m = smem.row_max[row];
          for (int col = 0; col < 64; ++col) {
            int position = key_begin + col;
            if (global_row < kQueryExtent && col < kKvTile &&
                position < block_limit && position <= p.past + token)
              m = fmaxf(m, smem.score[row * 64 + col] *
                                  (1.4426950408889634f / sqrtf(float(kHeadDim))));
          }
          float alpha = isfinite(smem.row_max[row]) ?
              exp2f(smem.row_max[row] - m) : 0.0f;
          float sum = smem.row_sum[row] * alpha;
          for (int col = 0; col < 64; ++col) {
            int position = key_begin + col;
            float value = global_row < kQueryExtent && col < kKvTile &&
                          position < block_limit && position <= p.past + token
                ? exp2f(smem.score[row * 64 + col] *
                    (1.4426950408889634f / sqrtf(float(kHeadDim))) - m)
                : 0.0f;
            probability(row, col) = Element(value);
            sum += value;
          }
          smem.row_max[row] = m;
          smem.row_sum[row] = sum;
          for (int d = 0; d < kHeadDim; ++d)
            smem.accumulated[row * kHeadDim + d] *= alpha;
        }
        __syncthreads();
        auto value = cute::make_tensor(cute::make_smem_ptr(smem.value),
                                       typename PvMma::LayoutB{});
        for (int row = 0; row < kKvTile; ++row) {
          int position = key_begin + row;
          if (position >= block_limit) break;
          if (position < p.past) {
            CopyCached(p.value_cache +
                ((batch * p.heads_kv + group) * p.capacity + position) * kHeadDim,
                smem.raw, kHeadDim);
          } else {
            int token = position - p.past;
            int group_width = (kQPerKV + 2) * kHeadDim;
            Element const* source = p.qkv +
                (batch * kTokens + token) * p.heads_kv * group_width +
                group * group_width + (kQPerKV + 1) * kHeadDim;
            for (int d = int(threadIdx.x); d < kHeadDim; d += 128) {
              smem.raw[d] = source[d];
              bool owns_write = query_begin == q_begin &&
                                token * kQPerKV >= q_begin &&
                                token * kQPerKV < q_begin + kQRows;
              if (owns_write)
                p.value_cache[((batch * p.heads_kv + group) * p.capacity +
                               position) * kHeadDim + d] = source[d];
            }
            __syncthreads();
          }
          for (int d = int(threadIdx.x); d < kHeadDim; d += 128)
            value(d, row) = smem.raw[d];
          __syncthreads();
        }
        for (int i = int(threadIdx.x); i < 64 * kHeadDim; i += 128)
          if (i / kHeadDim >= kKvTile || key_begin + i / kHeadDim >= block_limit)
            value(i % kHeadDim, i / kHeadDim) = Element(0.0f);
        __syncthreads();
        PvMma::Run(smem.probability, smem.value, smem.product, kHeadDim);
        __syncthreads();
        for (int i = int(threadIdx.x); i < 16 * kHeadDim; i += 128)
          smem.accumulated[i] += smem.product[i];
        __syncthreads();
      }
      for (int i = int(threadIdx.x); i < 16 * kHeadDim; i += 128) {
        int local_row = i / kHeadDim;
        int row = query_begin + local_row;
        if (row >= kQueryExtent) continue;
        int token = row / kQPerKV;
        int head = row % kQPerKV;
        float divided = smem.row_sum[local_row] > 0.0f ?
            smem.accumulated[i] / smem.row_sum[local_row] : 0.0f;
        if (cmax == 1 || kTokens > 1) {
          p.context[(batch * kTokens + token) * p.heads_kv * kQPerKV *
                    kHeadDim + (group * kQPerKV + head) * kHeadDim + i % kHeadDim]
              = Element(divided);
        } else {
          std::size_t base = (((std::size_t(batch) * p.heads_kv + group) *
              cmax + cache_block) * kQueryExtent + row);
          p.partial[base * kHeadDim + i % kHeadDim] = divided;
          if (i % kHeadDim == 0)
            p.lse[base] = smem.row_max[local_row] +
                          log2f(smem.row_sum[local_row]);
        }
      }
      __syncthreads();
    }
  }
};

}  // namespace tilemega::codegen
