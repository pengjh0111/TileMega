// SPDX-License-Identifier: BSD-3-Clause
// MX3-7 item 2: mcTlass on decode GEMM shapes, configured for decode.
//
// MX-2 timed `mctlass::gemm::device::Gemm` with its template defaults, which
// are OpClassSimt on Sm70: a SIMT GEMM with a 128x128 threadblock, so an M=16
// gate_up ran on a dozen threadblocks without tensor cores and read 6.3 MB at
// 0.5% of DRAM bandwidth. That measured the configuration, not the library.
// Here the tensor-op path, a small threadblock and serial split-K are chosen
// for M <= 16, on the same shapes as experiments/mx3/vendor_gemm.py.
//
// mcTlass's MetaX-native mainloops (gemm/threadblock/maca_mma_multistage)
// cover int8 and w4a16 dense GEMM and bf16 only for grouped/MoE 128x128 tiles,
// so the generic CUTLASS-2 path is the only bf16 dense route for decode.
//
// Usage: mctlass_decode <M> <N> <K> <split>
#include <mctlass/gemm/device/gemm.h>
#include <mctlass/numeric_types.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Element = mctlass::bfloat16_t;
// The generic tensor-op path (OpClassTensorOp on Sm75/Sm80) does not build
// for MACA: its shared-memory fragment loads are inline PTX `ldmatrix`
// (arch/memory_sm75.h:250), which the MACA assembler rejects as "invalid
// instruction" (MX-1 P-3). The SIMT path is therefore the only generic bf16
// route; it is configured here for M <= 16 -- a 32x64 threadblock instead of
// the default 128x128, so N=6144 gets 96 threadblocks before split-K -- with
// serial split-K over K.
using Gemm = mctlass::gemm::device::Gemm<
    Element, mctlass::layout::RowMajor,       // A: activations [M, K]
    Element, mctlass::layout::ColumnMajor,    // B: weights [N, K] row-major
    Element, mctlass::layout::RowMajor, float,
    mctlass::arch::OpClassSimt, mctlass::arch::Sm70,
    mctlass::gemm::GemmShape<32, 64, 8>, mctlass::gemm::GemmShape<32, 32, 8>,
    mctlass::gemm::GemmShape<1, 1, 1>,
    mctlass::epilogue::thread::LinearCombination<Element, 1, float, float>,
    mctlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1,
    /*SplitKSerial=*/true>;

int main(int argc, char** argv) {
  int M = argc > 1 ? std::atoi(argv[1]) : 1;
  int N = argc > 2 ? std::atoi(argv[2]) : 6144;
  int K = argc > 3 ? std::atoi(argv[3]) : 1024;
  int split = argc > 4 ? std::atoi(argv[4]) : 4;
  size_t wbytes = size_t(N) * K * 2;
  int copies = int(std::max<size_t>(2, (48ull << 20) / wbytes + 1));
  std::vector<Element> ha(size_t(M) * K), hb(size_t(N) * K);
  for (size_t i = 0; i < ha.size(); ++i) ha[i] = Element(float((i * 37 % 19) - 9) / 32.f);
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = Element(float((i * 53 % 23) - 11) / 64.f);
  Element *a, *c;
  std::vector<Element*> bs(copies);
  int* workspace;
  if (cudaMalloc(&a, ha.size() * 2) || cudaMalloc(&c, size_t(M) * N * 2) ||
      cudaMalloc(&workspace, size_t(N) * 1024 * 4)) return 2;
  for (auto& b : bs) {
    if (cudaMalloc(&b, wbytes)) return 2;
    cudaMemcpy(b, hb.data(), wbytes, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
  cudaMemset(workspace, 0, size_t(N) * 1024 * 4);
  auto args = [&](Element* b) {
    return typename Gemm::Arguments({M, N, K}, {a, K}, {b, K}, {c, N}, {c, N},
                                    {1.f, 0.f}, split);
  };
  Gemm gemm;
  if (gemm.can_implement(args(bs[0])) != mctlass::Status::kSuccess) {
    std::printf("cannot implement M=%d N=%d K=%d split=%d\n", M, N, K, split);
    return 3;
  }
  if (gemm(args(bs[0]), workspace) != mctlass::Status::kSuccess) {
    std::printf("launch failed\n");
    return 4;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("sync failed\n"); return 5; }
  // Numerics against an FP64 reference on a sample of columns.
  std::vector<Element> got(size_t(M) * N);
  cudaMemcpy(got.data(), c, got.size() * 2, cudaMemcpyDeviceToHost);
  double worst = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; n += 97) {
      double ref = 0;
      for (int k = 0; k < K; ++k) ref += double(float(ha[size_t(m) * K + k])) * float(hb[size_t(n) * K + k]);
      worst = std::max(worst, std::fabs(float(got[size_t(m) * N + n]) - ref) / std::max(1e-2, std::fabs(ref)));
    }
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int i = 0; i < 10; ++i) gemm(args(bs[i % copies]), workspace);
  cudaDeviceSynchronize();
  int const iters = 100;
  cudaEventRecord(s);
  for (int i = 0; i < iters; ++i) gemm(args(bs[i % copies]), workspace);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms = 0;
  cudaEventElapsedTime(&ms, s, e);
  double us = ms * 1000 / iters;
  std::printf("mctlass M=%d N=%d K=%d split=%d: %.2f us, %.1f GB/s, worst rel err %.3g\n",
              M, N, K, split, us, wbytes / (us * 1e-6) / 1e9, worst);
  return worst < 0.05 ? 0 : 6;
}
