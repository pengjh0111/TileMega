// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/Mx/DecodeGemm.h>
#include <tilemega/Backend/Mx/RowTransformV2.h>
#include <cuda_runtime.h>

using namespace tilemega;
using namespace tilemega::backend::mx;
using tilemega::backend::LoadGlobal16Gvm;
using tilemega::backend::StoreGlobal16;

struct TransformArgs {
  Bf16 const *input, *weight, *cosine, *sine;
  Bf16* output;
  int rows;
  float epsilon;
};

template <int D, bool Norm>
__global__ void __launch_bounds__(576) TransformKernel(TransformArgs args) {
  extern __shared__ float exchange[];
  int const wave = threadIdx.x / 64, lane = threadIdx.x % 64;
  if (wave >= 8) return;
  int const row = int(blockIdx.x) * 128 + wave * 16 + (lane & 15), group = lane >> 4;
  uint4 x[D / 32], w[D / 32], cosine[D / 32], sine[D / 32];
#pragma unroll
  for (int dc = 0; dc < D / 32; ++dc) {
    int const at = dc * 32 + group * 8;
    x[dc] = LoadGlobal16Gvm(args.input + min(row, args.rows - 1) * D + at);
    w[dc] = LoadGlobal16Gvm(args.weight + at);
    cosine[dc] = LoadGlobal16Gvm(args.cosine + at);
    sine[dc] = LoadGlobal16Gvm(args.sine + at);
  }
  MxRowTransformV2<D, Norm>(x, w, cosine, sine, args.epsilon, exchange + wave * 64, lane);
  if (row < args.rows) {
#pragma unroll
    for (int dc = 0; dc < D / 32; ++dc)
      StoreGlobal16(args.output + row * D + dc * 32 + group * 8, x[dc]);
  }
}

template <int D, bool Norm>
__global__ void __launch_bounds__(512) TransformReference(TransformArgs args) {
  if (blockIdx.x) return;
  extern __shared__ __align__(16) char storage[];
  MxGemmSmem const layout(args.rows, D);
  float* const shared = reinterpret_cast<float*>(storage);
  float* const values = shared + layout.res;
  for (int i = threadIdx.x; i < args.rows * (D + 1); i += 512)
    values[i] = i % (D + 1) == D ? 0.f : float(args.input[i / (D + 1) * D + i % (D + 1)]);
  __syncthreads();
  MxGemmOperands operands{};
  operands.out = args.output; operands.q_norm = args.weight; operands.k_norm = args.weight;
  operands.cosine = args.cosine; operands.sine = args.sine;
  operands.capacity = 32; operands.past = 0; operands.epsilon = args.epsilon;
  using Reference = MxDecodeGemm<D, 1, MxDecodeOp::kQkvRope, false, MxPart::kTileRange, MxRope<D, 1, 1, Norm>>;
  Reference::RopeEpilogue(values, 0, operands, MxShape{3 * D, D, args.rows}, shared, layout);
}

template <int D, bool Norm>
int Launch(TransformArgs const& args, bool reference, cudaStream_t stream) {
  if (reference) {
    int const bytes = max(32784, MxGemmSmem(args.rows, D).bytes);
    if (cudaFuncSetAttribute(TransformReference<D, Norm>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess)
      return 1;
    TransformReference<D, Norm><<<104, 512, bytes, stream>>>(args);
  } else TransformKernel<D, Norm><<<104, 576, 32784, stream>>>(args);
  return int(cudaGetLastError());
}

extern "C" int mx5_transform(TransformArgs const* args, int dim, int norm, int reference, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (dim == 64 && !norm) return Launch<64, false>(*args, reference, s);
  if (dim == 128 && !norm) return Launch<128, false>(*args, reference, s);
  if (dim == 128 && norm) return Launch<128, true>(*args, reference, s);
  return 2;
}
