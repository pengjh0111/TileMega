// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Backend/DmEpilogueValue.h>
#include <tilemega/Backend/DmTensorAddress.h>
#include <tilemega/Backend/DmMoeOperands.h>
#include <tilemega/Codegen/executor/ComputeGroup.cuh>
#include <cute/tensor.hpp>

namespace tilemega::backend {

struct DmEpilogueArguments {
  codegen::DmBufferView buffers{};
  codegen::DmEpilogueChain chain{};
  codegen::DmWriteMap write{};
  void* output = nullptr;
  int m = 0, n = 0, output_stride = 0;
  int norm_width = 0;
  float norm_eps = 0;
  // A gathered tile still uses its original token's normalization statistics.
  std::uint32_t const* statistic_rows = nullptr;
  int image_rows = 0;
  DmMoeRows moe{};
  bool partial_rows = false;
  unsigned routing_topk = 0;

  __device__ unsigned StatisticRow(unsigned row) const {
    return statistic_rows?statistic_rows[row]:moe.Token(row);
  }
  __device__ unsigned StorageRow(unsigned row) const {
    return partial_rows?row:moe.Storage(row);
  }
};

template <class Program> struct DmGateShape;
template <class... Steps>
struct DmGateShape<codegen::DmEpilogueProgram<Steps...>> {
  static constexpr unsigned kCount = (0u + ... +
      unsigned(Steps::kKind == codegen::DmEpilogueKind::kGatePair));
  static constexpr unsigned kUnit = (0u + ... +
      (Steps::kKind == codegen::DmEpilogueKind::kGatePair ? Steps::kUnit : 0u));
  template <unsigned Columns>
  static constexpr bool kFits = kCount <= 1 && (!kCount ||
      (kUnit && Columns % (2 * (kUnit ? kUnit : 1)) == 0));
};

template <class Spec> struct DmSideWalk;
template <class Program, codegen::DmWriteKind Write, unsigned Factor,
          codegen::DmRounding Store, class... Sides>
struct DmSideWalk<codegen::DmEpilogueSpec<Program, Write, Factor, Store, Sides...>> {
  template <unsigned Position, class Visitor>
  __device__ static void Visit(Visitor&) {}
  template <unsigned Position, class First, class... Rest, class Visitor>
  __device__ static void Visit(Visitor& visitor) {
    visitor.template SideOutput<Position, First>();
    Visit<Position + 1, Rest...>(visitor);
  }
  template <class Visitor>
  __device__ static void Run(Visitor& visitor) { Visit<0, Sides...>(visitor); }
};

template <class Arch, class Spec, int TileM, int TileN, bool Swizzled = true>
struct ServingDmEpilogue {
  using Program = typename Spec::Chain;
  using Gate = DmGateShape<Program>;
  static_assert(Gate::kCount <= 1, "one channel-pair contraction per tile");
  static_assert(Gate::template kFits<TileN>,
                "a channel tile must own complete gate/up pairs");
  static_assert(TileN >= 16 && (TileN & (TileN - 1)) == 0);
  static constexpr unsigned kColumns = Gate::kCount ? TileN / 2 : TileN;
  static constexpr int kSharedBytes = (TileM * TileN + 2 * TileM) * sizeof(float);

  __device__ static int Index(int row, int column) {
    if constexpr (Swizzled)
      return row * TileN + (column ^ (((row & 3) * 8) & (TileN - 1)));
    return row * TileN + column;
  }
  template <bool Gated>
  __device__ static int InputColumn(int column) {
    if constexpr (Gated && Gate::kCount)
      return 2 * Gate::kUnit * (column / Gate::kUnit) + column % Gate::kUnit;
    return column;
  }
  template <class T, unsigned Dtype>
  __device__ static T* Buffer(DmEpilogueArguments const& p, std::uint32_t id) {
    if (id >= p.buffers.count || !p.buffers.data || !p.buffers.dtypes ||
        p.buffers.dtypes[id] != Dtype || !p.buffers.data[id]) {
      asm volatile("trap;");
      return nullptr;
    }
    return reinterpret_cast<T*>(p.buffers.data[id]);
  }
  __device__ static codegen::DmBufferLayout Layout(
      DmEpilogueArguments const& p, std::uint32_t id) {
    if (id == codegen::kDmNoIndex) return {};
    if (id >= p.buffers.count || !p.buffers.layouts) {
      asm volatile("trap;");
      return {};
    }
    return p.buffers.layouts[id];
  }

  struct Visitor {
    float* tile;
    DmEpilogueArguments const& p;
    int tile_m, tile_n;

    __device__ void Statistics(std::uint32_t id, bool centered) {
      using namespace codegen::executor;
      float const* statistics = Buffer<float, 1>(p, id);
      auto layout = Layout(p, id);
      if (layout.rank != 3 || layout.logical[2] != 2 || p.norm_width <= 0) {
        asm volatile("trap;");
        return;
      }
      float* means = tile + TileM * TileN;
      float* inverses = means + TileM;
      int lane = ComputeThread() & 31;
      for (int row = ComputeThread() / 32; row < TileM; row += 4) {
        int global_row = tile_m * TileM + row;
        float sum = 0, square = 0;
        if (global_row < p.m) {
          auto source_row = p.StatisticRow(global_row);
          for (unsigned part = lane; part < layout.logical[1]; part += 32) {
            auto offset = source_row * layout.strides[0] + part * layout.strides[1];
            sum += statistics[offset];
            square += statistics[offset + layout.strides[2]];
          }
        }
        for (int shift = 16; shift; shift /= 2) {
          sum += __shfl_xor_sync(0xffffffff, sum, shift);
          square += __shfl_xor_sync(0xffffffff, square, shift);
        }
        if (lane == 0) {
          float mean = sum / p.norm_width;
          float variance = square / p.norm_width - (centered ? mean * mean : 0);
          means[row] = centered ? mean : 0;
          inverses[row] = rsqrtf(fmaxf(variance, 0) + p.norm_eps);
        }
      }
      ComputeSync();
    }

    template <unsigned Position, class Step, bool Gated>
    __device__ void Apply() {
      using K = codegen::DmEpilogueKind;
      using namespace codegen::executor;
      auto const& op = p.chain.operations[Position];
      if (op.kind != Step::kKind || op.activation != Step::kActivation ||
          op.gate != Step::kGate || op.unit != Step::kUnit ||
          op.input_rounding != Step::kInputRounding ||
          op.output_rounding != Step::kOutputRounding ||
          op.residual_map.kind != Step::kResidualMap ||
          op.residual_map.factor != Step::kResidualFactor) {
        asm volatile("trap;");
        return;
      }
      float const* first = nullptr;
      float const* second = nullptr;
      float const* third = nullptr;
      cutlass::bfloat16_t const* residual = nullptr;
      codegen::DmBufferLayout residual_layout{};
      std::uint32_t const* scatter = nullptr;
      if constexpr (Step::kKind == K::kBias || Step::kKind == K::kScale)
        first = Buffer<float, 1>(p, op.parameter[0]);
      if constexpr (Step::kKind == K::kResidual || Step::kKind == K::kResidualLN) {
        residual = Buffer<cutlass::bfloat16_t, 0>(p, op.parameter[0]);
        residual_layout = Layout(p, op.residual_map.layout);
        if constexpr (Step::kResidualMap == codegen::DmWriteKind::kRowScatter)
          scatter = Buffer<std::uint32_t, 2>(p, op.residual_map.rows);
        if constexpr (Step::kKind == K::kResidual) {
          if (op.parameter[1] != codegen::kDmNoIndex)
            first = Buffer<float, 1>(p, op.parameter[1]);
        } else {
          Statistics(op.parameter[1], true);
          second = Buffer<float, 1>(p, op.parameter[2]);
          third = Buffer<float, 1>(p, op.parameter[3]);
        }
      }
      if constexpr (Step::kKind == K::kDeferredLayerNorm) {
        Statistics(op.parameter[0], true);
        second = Buffer<float, 1>(p, op.parameter[1]);
        third = Buffer<float, 1>(p, op.parameter[2]);
      }
      if constexpr (Step::kKind == K::kDeferredRMSNorm)
        Statistics(op.parameter[0], false);
      constexpr bool contracting = Step::kKind == K::kGatePair;
      constexpr bool compact = Gated || contracting;
      constexpr int columns = compact ? kColumns : TileN;
      int global_columns = compact ? p.n / 2 : p.n;
      for (int i = ComputeThread(); i < TileM * columns; i += kComputeThreads) {
        int row = i / columns, column = i % columns;
        int global_row = tile_m * TileM + row;
        int global_column = tile_n * columns + column;
        if (global_row >= p.m || global_column >= global_columns) continue;
        int source_column = InputColumn<compact>(column);
        DmEpilogueInputs context;
        if constexpr (Step::kKind == K::kBias) context.bias = first[global_column];
        if constexpr (Step::kKind == K::kScale) context.scale = first[global_column];
        if constexpr (Step::kKind == K::kResidual || Step::kKind == K::kResidualLN) {
          auto offset = DmTensorAddress<Step::kResidualMap, Step::kResidualFactor>::Offset(
              residual_layout, global_row, global_column, p.output_stride, scatter);
          context.residual = float(residual[offset]);
          if constexpr (Step::kKind == K::kResidual)
            context.residual_scale = first ? first[global_column] : 1;
          else {
            context.gamma = second[global_column];
            context.beta = third[global_column];
          }
        }
        if constexpr (Step::kKind == K::kDeferredLayerNorm ||
                      Step::kKind == K::kDeferredRMSNorm || Step::kKind == K::kResidualLN) {
          context.mean = tile[TileM * TileN + row];
          context.rstd = tile[TileM * TileN + TileM + row];
        }
        if constexpr (Step::kKind == K::kDeferredLayerNorm) {
          context.u = second[global_column];
          context.v = third[global_column];
        }
        float partner = 0;
        if constexpr (contracting)
          partner = tile[Index(row, source_column + Step::kUnit)];
        float& value = tile[Index(row, source_column)];
        value = DmEpilogueValue<Arch, Step>::Apply(value, context, partner);
      }
      // Gate stores use their original gate slots; up slots remain read-only.
      // Compaction in place would let one lane overwrite another lane's input.
      ComputeSync();
    }
  };

  template <codegen::DmRounding Rounding = Spec::kStore>
  __device__ static float Stored(float value) { return DmRound<Rounding>(value); }

  __device__ static void Store(float* tile, DmEpilogueArguments const& p,
                               int tile_m, int tile_n) {
    using namespace codegen::executor;
    auto layout = Layout(p, p.write.layout);
    std::uint32_t const* scatter = nullptr;
    if constexpr (Spec::kWrite == codegen::DmWriteKind::kRowScatter)
      if(!p.moe.rows)scatter = Buffer<std::uint32_t, 2>(p, p.write.rows);
    for (int i = ComputeThread(); i < TileM * kColumns; i += kComputeThreads) {
      int row = i / kColumns, column = i % kColumns;
      int global_row = tile_m * TileM + row;
      int global_column = tile_n * kColumns + column;
      if (global_row >= p.m || global_column >= p.n / (Gate::kCount ? 2 : 1)) continue;
      std::uint64_t offset;
      if constexpr(Spec::kWrite==codegen::DmWriteKind::kRowScatter) {
        if(p.moe.rows) {
          if(!p.routing_topk || p.moe.rows[p.moe.Entry(global_row)].rank>=p.routing_topk) {
            asm volatile("trap;");return;
          }
          auto row=p.moe.Scatter(global_row,p.routing_topk);
          offset=DmTensorAddress<codegen::DmWriteKind::kDense>::Offset(
              layout,row,global_column,p.output_stride);
        }else offset=DmTensorAddress<Spec::kWrite,Spec::kFactor>::Offset(
            layout,global_row,global_column,p.output_stride,scatter);
      }else offset=DmTensorAddress<Spec::kWrite,Spec::kFactor>::Offset(
          layout,p.StorageRow(global_row),global_column,p.output_stride,scatter);
      float value = tile[Index(row, InputColumn<Gate::kCount != 0>(column))];
      if constexpr (Spec::kStore == codegen::DmRounding::kBF16)
        reinterpret_cast<cutlass::bfloat16_t*>(p.output)[offset] = cutlass::bfloat16_t(value);
      else reinterpret_cast<float*>(p.output)[offset] = value;
    }
    ComputeSync();
  }

  struct SideVisitor {
    float* tile;
    DmEpilogueArguments const& p;
    int tile_m, tile_n;

    __device__ float Value(int row, int column) const {
      return Stored(tile[Index(row, InputColumn<Gate::kCount != 0>(column))]);
    }
    template <unsigned Position, class Side>
    __device__ void SideOutput() {
      using namespace codegen::executor;
      using K = codegen::DmSideOutputKind;
      auto const& descriptor = p.chain.side[Position];
      if (descriptor.kind != Side::kKind || descriptor.count != Side::kCount) {
        asm volatile("trap;");
        return;
      }
      float* output = Buffer<float, 1>(p, descriptor.buffer);
      auto layout = Layout(p, descriptor.buffer);
      int columns = p.n / (Gate::kCount ? 2 : 1);
      if constexpr (Side::kKind == K::kRowStats) {
        if (layout.rank != 3 || layout.logical[2] != 2) {
          asm volatile("trap;");
          return;
        }
        int lane = ComputeThread() & 31;
        for (int row = ComputeThread() / 32; row < TileM; row += 4) {
          int global_row = tile_m * TileM + row;
          if (global_row >= p.m) continue;
          float sum = 0, square = 0;
          for (unsigned column = lane; column < kColumns; column += 32) {
            if (tile_n * kColumns + column >= columns) continue;
            float value = Value(row, column);
            sum += value;
            square += value * value;
          }
          for (int shift = 16; shift; shift /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, shift);
            square += __shfl_xor_sync(0xffffffff, square, shift);
          }
          if (lane == 0) {
            auto offset = p.StorageRow(global_row) * layout.strides[0] + tile_n * layout.strides[1];
            output[offset] = sum;
            output[offset + layout.strides[2]] = square;
          }
        }
      } else if constexpr (Side::kKind == K::kChannelPartialSums) {
        if (layout.rank != 3 || p.image_rows <= 0) {
          asm volatile("trap;");
          return;
        }
        int first = tile_m * TileM / p.image_rows;
        int end = min(p.m, (tile_m + 1) * TileM);
        int last = (end - 1) / p.image_rows;
        for (int image = first; image <= last; ++image) {
          int begin_row = max(0, image * p.image_rows - tile_m * TileM);
          int end_row = min(TileM, min(p.m, (image + 1) * p.image_rows) - tile_m * TileM);
          for (unsigned column = ComputeThread(); column < kColumns; column += kComputeThreads) {
            int global_column = tile_n * kColumns + column;
            if (global_column >= columns) continue;
            float sum = 0;
            for (int row = begin_row; row < end_row; ++row) sum += Value(row, column);
            output[image * layout.strides[0] + tile_m * layout.strides[1] +
                   global_column * layout.strides[2]] = sum;
          }
        }
      } else if constexpr (Side::kKind == K::kPartial) {
        if (layout.rank != 3) { asm volatile("trap;"); return; }
        for (int i = ComputeThread(); i < TileM * kColumns; i += kComputeThreads) {
          int row = i / kColumns, column = i % kColumns;
          int global_row = tile_m * TileM + row;
          int global_column = tile_n * kColumns + column;
          if (global_row < p.m && global_column < columns)
            output[global_row * layout.strides[0] + tile_n * layout.strides[1] +
                   column * layout.strides[2]] =
                tile[Index(row, InputColumn<Gate::kCount != 0>(column))];
        }
      } else {
        constexpr unsigned count = Side::kKind == K::kArgmaxPartial ? 1 : Side::kCount;
        static_assert(count > 0 && count <= kColumns);
        if (layout.rank != 3 || layout.logical[2] < count) {
          asm volatile("trap;");
          return;
        }
        auto indices = Buffer<std::int32_t, 2>(p, descriptor.auxiliary);
        auto index_layout = Layout(p, descriptor.auxiliary);
        // Only row leaders select from shared memory. The order is stable:
        // descending rounded logit, then ascending global expert index.
        for (int row = ComputeThread(); row < TileM; row += kComputeThreads) {
          int global_row = tile_m * TileM + row;
          if (global_row >= p.m) continue;
          float values[count];
          int selected[count];
          #pragma unroll
          for (unsigned rank = 0; rank < count; ++rank) {
            values[rank] = -INFINITY;
            selected[rank] = INT32_MAX;
          }
          for (unsigned column = 0; column < kColumns; ++column) {
            int global_column = tile_n * kColumns + column;
            if (global_column >= columns) continue;
            float value = Value(row, column);
            int index = global_column;
            #pragma unroll
            for (unsigned rank = 0; rank < count; ++rank)
              if (value > values[rank] || (value == values[rank] && index < selected[rank])) {
                float old_value = values[rank]; int old_index = selected[rank];
                values[rank] = value; selected[rank] = index;
                value = old_value; index = old_index;
              }
          }
          #pragma unroll
          for (unsigned rank = 0; rank < count; ++rank) {
            output[global_row * layout.strides[0] + tile_n * layout.strides[1] +
                   rank * layout.strides[2]] = values[rank];
            indices[global_row * index_layout.strides[0] + tile_n * index_layout.strides[1] +
                    rank * index_layout.strides[2]] = selected[rank];
          }
        }
      }
      ComputeSync();
    }
  };

  template <class Accumulator, class TiledMma>
  __device__ static void Run(Accumulator const& accum, TiledMma const& mma,
                             char* shared, DmEpilogueArguments const& p,
                             int tile_m, int tile_n) {
    using namespace codegen::executor;
    cute::cp_async_wait<0>();
    ComputeSync();
    auto coordinates = cute::make_identity_tensor(
        cute::Shape<cute::Int<TileM>, cute::Int<TileN>>{});
    auto owned = mma.get_thread_slice(ComputeThread()).partition_C(coordinates);
    CUTE_STATIC_ASSERT_V(cute::size(owned) == cute::size(accum));
    float* tile = reinterpret_cast<float*>(shared);
    for (int i = 0; i < cute::size(accum); ++i)
      tile[Index(cute::get<0>(owned(i)), cute::get<1>(owned(i)))] = accum(i);
    ComputeSync();
    RunFromTile(tile, p, tile_m, tile_n);
  }

  __device__ static void RunFromTile(float* tile, DmEpilogueArguments const& p,
                                     int tile_m, int tile_n) {
    if (p.chain.count != Program::kCount || p.chain.side_count != Spec::kSideCount ||
        p.write.kind != Spec::kWrite || p.write.factor != Spec::kFactor ||
        p.chain.store_rounding != Spec::kStore || !p.output) {
      asm volatile("trap;");
      return;
    }
    Visitor visitor{tile, p, tile_m, tile_n};
    DmEpilogueWalk<Program>::Run(visitor);
    Store(tile, p, tile_m, tile_n);
    SideVisitor sides{tile, p, tile_m, tile_n};
    DmSideWalk<Spec>::Run(sides);
    // A fast warp may start the next task's cp.async into this workspace.
    // Join all readers of the output tile before returning its ownership.
    codegen::executor::ComputeSync();
  }
};

}  // namespace tilemega::backend
