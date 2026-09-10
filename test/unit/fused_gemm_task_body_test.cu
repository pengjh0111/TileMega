// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/FusedGemmTaskBody.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::codegen;
using Arch = tilemega::arch::CurrentArch;
using Fused = FusedGemmTaskBody<Arch, 0, kGemmThreads>;
using Plain = Fused::Gemm;
using E = ModelElement;

static void Check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Buffer {
  E* pointer = nullptr;
  explicit Buffer(std::vector<E> const& values) {
    Check(cudaMalloc(&pointer, values.size() * sizeof(E)));
    Check(cudaMemcpy(pointer, values.data(), values.size() * sizeof(E), cudaMemcpyHostToDevice));
  }
  ~Buffer() { cudaFree(pointer); }
  Buffer(Buffer const&) = delete;
  std::vector<E> Read(int count) const {
    std::vector<E> out(count);
    Check(cudaMemcpy(out.data(), pointer, count * sizeof(E), cudaMemcpyDeviceToHost));
    return out;
  }
};

__global__ void plain_gemm(GemmInvocation invocation) {
  extern __shared__ char scratch[];
  Plain::RunTask<0>(invocation, int(blockIdx.x), scratch);
}
__global__ void plain_add(E const* input, E const* residual, E* output, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = E(float(input[index]) + float(residual[index]));
}
__global__ void plain_norm(E const* input, E const* weight, E* output, int width) {
  __shared__ float rms[kGemmThreads];
  int row = blockIdx.x;
  Fused::Norm::RunRow(input + row * width, weight, output + row * width, width, rms);
}
__global__ void fused_add(GemmInvocation invocation, E const* residual, E* output) {
  extern __shared__ char bytes[];
  Fused::Add(invocation, int(blockIdx.x), residual, output,
             *reinterpret_cast<Fused::SharedStorage*>(bytes));
}
__global__ void fused_norm(GemmInvocation invocation, E const* weight, E* output) {
  extern __shared__ char bytes[];
  Fused::RMSNorm(invocation, int(blockIdx.x), weight, output,
                 *reinterpret_cast<Fused::SharedStorage*>(bytes));
}

int main() try {
  int cases = 0;
  for (int m : {1, 5, 33}) for (int n : {112, 128}) for (float beta : {0.0f, 1.0f}) {
    constexpr int k = 64;
    std::vector<E> a(m*k), b(n*k), source(m*n), residual(m*n), weight(n);
    for (int i=0;i<m*k;++i) a[i]=E(float(i%31-15)/16);
    for (int i=0;i<n*k;++i) b[i]=E(float(i%23-11)/32);
    for (int i=0;i<m*n;++i) { source[i]=E(float(i%17-8)/16); residual[i]=E(float(i%19-9)/16); }
    for (int i=0;i<n;++i) weight[i]=E(1.0f+float(i%7)/16);
    std::vector<E> poison(m*n, E(-123.0f));
    Buffer da(a), db(b), dc(source), dr(residual), dw(weight);
    Buffer intermediate(poison), expected_add(poison), expected_norm(poison);
    Buffer result_add(poison), result_norm(poison), untouched(poison);
    GemmInvocation invocation{};
    invocation.problem = cute::make_shape(m,n,k,1);
    invocation.mainloop = {da.pointer, cute::make_stride(int64_t(k),cute::_1{},int64_t(m*k)),
                           db.pointer, cute::make_stride(int64_t(k),cute::_1{},int64_t(n*k))};
    invocation.epilogue.thread = {1.0f,beta};
    invocation.epilogue.ptr_C = dc.pointer;
    invocation.epilogue.dC = cute::make_stride(int64_t(n),cute::_1{},int64_t(m*n));
    invocation.epilogue.ptr_D = intermediate.pointer;
    invocation.epilogue.dD = invocation.epilogue.dC;
    invocation.tiles_m = (m + kGemmTileM - 1) / kGemmTileM;
    invocation.tiles_n = (n + kGemmTileN - 1) / kGemmTileN;
    invocation.tile_m=kGemmTileM; invocation.tile_n=kGemmTileN;
    int tiles = invocation.tiles_m * invocation.tiles_n;
    plain_gemm<<<tiles,kGemmThreads,sizeof(Plain::SharedStorage)>>>(invocation);
    Check(cudaGetLastError());
    plain_add<<<(m*n+kGemmThreads-1)/kGemmThreads,kGemmThreads>>>(intermediate.pointer,dr.pointer,expected_add.pointer,m*n);
    Check(cudaGetLastError());
    plain_norm<<<m,kGemmThreads>>>(intermediate.pointer,dw.pointer,expected_norm.pointer,n);
    Check(cudaGetLastError());
    invocation.epilogue.ptr_D=untouched.pointer;
    fused_add<<<tiles,kGemmThreads,sizeof(Fused::SharedStorage)>>>(invocation,dr.pointer,result_add.pointer);
    Check(cudaGetLastError());
    fused_norm<<<m,kGemmThreads,sizeof(Fused::SharedStorage)>>>(invocation,dw.pointer,result_norm.pointer);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    auto compare=[&](Buffer const& first, Buffer const& second) {
      auto x=first.Read(m*n), y=second.Read(m*n);
      if (std::memcmp(x.data(),y.data(),x.size()*sizeof(E)))
        throw std::runtime_error("shared fusion differs from separate GPU task bodies");
    };
    compare(expected_add,result_add); compare(expected_norm,result_norm);
    auto kept=untouched.Read(m*n);
    if (std::memcmp(kept.data(),poison.data(),poison.size()*sizeof(E)))
      throw std::runtime_error("fused producer wrote its global intermediate");
    cases+=2;
  }
  cudaFuncAttributes add{}, norm{};
  Check(cudaFuncGetAttributes(&add,fused_add)); Check(cudaFuncGetAttributes(&norm,fused_norm));
  int add_blocks=0,norm_blocks=0;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&add_blocks,fused_add,kGemmThreads,sizeof(Fused::SharedStorage)));
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&norm_blocks,fused_norm,kGemmThreads,sizeof(Fused::SharedStorage)));
  std::cout << "FUSED_GEMM_TASK_BODY cases=" << cases << " bit_equal=1 global_intermediate_untouched=1"
            << " threads=" << kGemmThreads << " shared=" << sizeof(Fused::SharedStorage)
            << " add_regs=" << add.numRegs << " norm_regs=" << norm.numRegs
            << " add_ctas=" << add_blocks << " norm_ctas=" << norm_blocks << '\n';
  return 0;
} catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
