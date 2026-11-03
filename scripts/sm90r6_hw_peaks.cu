// SPDX-License-Identifier: BSD-3-Clause
// H1 (prompt section 9.1): the measured peaks the lower bounds of section 9.3
// and N1 are taken against. Three items, each run alone on one claimed card:
//
//   hbm_read     8 GiB of DRAM, far past L2, read two ways: a cp.async.bulk
//                ring with 2/4/8 blocks in flight, and ld.global.nc.v4 with 8
//                per thread. 1 or 2 CTAs per SM. The maximum over configs.
//   bf16_wgmma   WGMMA m64n256k16 SS, bf16 in and fp32 accumulate, two
//                consumer warpgroups per CTA, operands resident in shared
//                memory so the loop measures the tensor cores alone.
//   bf16_mma_sync  the same, through mma.sync m16n8k16.
//
// Nothing here is a model: each number is a wall-clock rate over a loop long
// enough that launch and drain are below a percent.
#include <tilemega/Backend/Sm90Gmma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int kSms = 132;
constexpr int kBlockBytes = 32 * 1024;

void Check(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

// (b) ld.global.nc.v4 with `InFlight` independent loads per thread. The reads
// are non-temporal so they do not displace one another in L2.
template <int InFlight>
__global__ void HbmVector(float4 const* __restrict__ data, std::size_t vectors,
                          float4* __restrict__ sink) {
  std::size_t stride = std::size_t(gridDim.x) * blockDim.x * InFlight;
  float4 total = make_float4(0, 0, 0, 0);
  for (std::size_t base = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * InFlight;
       base + InFlight <= vectors; base += stride) {
    float4 value[InFlight];
#pragma unroll
    for (int i = 0; i < InFlight; ++i)
      asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                   : "=f"(value[i].x), "=f"(value[i].y), "=f"(value[i].z), "=f"(value[i].w)
                   : "l"(data + base + i) : "memory");
#pragma unroll
    for (int i = 0; i < InFlight; ++i) {
      total.x += value[i].x; total.y += value[i].y;
      total.z += value[i].z; total.w += value[i].w;
    }
  }
  // Keep the loads live without adding a store to the timed path.
  if (total.x == 1234.5678f) sink[0] = total;
}

// (a) a cp.async.bulk ring: `InFlight` 32 KiB blocks outstanding per CTA.
template <int InFlight>
__global__ void HbmBulk(char const* __restrict__ data, std::size_t blocks,
                        float* __restrict__ sink) {
  extern __shared__ char ring[];
  __shared__ std::uint64_t barrier[InFlight];
  if (threadIdx.x == 0)
#pragma unroll
    for (int i = 0; i < InFlight; ++i)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "l"(
          static_cast<std::uint64_t>(__cvta_generic_to_shared(&barrier[i]))));
  __syncthreads();
  std::size_t stride = std::size_t(gridDim.x) * InFlight;
  // An mbarrier's phase flips on every completion, so the parity a wait
  // compares against alternates per slot. A fixed parity succeeds once and
  // then hangs, which is what the first version of this probe did.
  std::uint32_t phase = 0;
  for (std::size_t base = std::size_t(blockIdx.x) * InFlight; base + InFlight <= blocks;
       base += stride, phase ^= 1u) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < InFlight; ++i) {
        auto slot = static_cast<std::uint64_t>(__cvta_generic_to_shared(ring + i * kBlockBytes));
        auto mbar = static_cast<std::uint64_t>(__cvta_generic_to_shared(&barrier[i]));
        // The barrier needs its arrival and its expected byte count before the
        // copy: complete_tx only credits the transaction, so without this the
        // wait never sees the arrival and spins forever.
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "l"(mbar), "n"(kBlockBytes) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], %2, [%3];"
                     :: "l"(slot), "l"(data + (base + i) * kBlockBytes),
                        "n"(kBlockBytes), "l"(mbar) : "memory");
      }
#pragma unroll
      for (int i = 0; i < InFlight; ++i) {
        auto mbar = static_cast<std::uint64_t>(__cvta_generic_to_shared(&barrier[i]));
        std::uint32_t done = 0;
        // The predicate form the repo's own Async.cuh uses; the bare
        // destination register form is not a legal operand list.
        while (!done)
          asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                       " selp.u32 %0, 1, 0, p; }"
                       : "=r"(done) : "l"(mbar), "r"(phase) : "memory");
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 1024) sink[0] = 1.0f;  // never taken; keeps `sink` live
}

using Arch = tilemega::arch::Sm90;
using Gmma = tilemega::backend::Sm90Gmma<Arch>;
using Element = cutlass::bfloat16_t;

// WGMMA m64n256k16 SS. Both operands stay in shared memory across the whole
// loop, so the rate is the tensor cores' and not the memory system's.
// One warpgroup per CTA and two CTAs per SM, so both warpgroup slots an SM
// offers are busy; the k16 atom's operands must be laid out at K = 64, the
// swizzle atom's tile, so each call issues four k16 steps.
__global__ __launch_bounds__(128) void Bf16Wgmma(int iterations, float* __restrict__ sink) {
  using namespace cute;
  extern __shared__ char storage[];
  auto* a = reinterpret_cast<Element*>(storage);
  auto* b = a + 64 * 64;
  for (int i = threadIdx.x; i < 64 * 64 + 256 * 64; i += blockDim.x)
    reinterpret_cast<Element*>(storage)[i] = Element(0.00390625f);
  __syncthreads();

  using Operation = typename Gmma::template SS<256>;
  auto tiled = make_tiled_mma(Operation{});
  auto thread_mma = tiled.get_thread_slice(threadIdx.x);
  auto sa = make_tensor(make_smem_ptr(a), tilemega::backend::Sm90KLayout<64, 64>{});
  auto sb = make_tensor(make_smem_ptr(b), tilemega::backend::Sm90KLayout<256, 64>{});
  auto fragment_a = thread_mma.partition_fragment_A(sa);
  auto fragment_b = thread_mma.partition_fragment_B(sb);
  auto accum = partition_fragment_C(tiled, Shape<_64, Int<256>>{});
  clear(accum);

  for (int i = 0; i < iterations; ++i) {
    Gmma::Fence(accum);
    Gmma::Arrive();
    gemm(tiled, fragment_a, fragment_b, accum);
    Gmma::Commit();
    Gmma::template Wait<0>();
    Gmma::Fence(accum);
  }
  if (threadIdx.x == 1024) sink[0] = float(accum(0));
}

// mma.sync m16n8k16, the same operands held in registers.
__global__ __launch_bounds__(256) void Bf16MmaSync(int iterations, float* __restrict__ sink) {
  std::uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
  std::uint32_t b[2] = {0x3c003c00u, 0x3c003c00u};
  float c[4] = {0, 0, 0, 0};
  for (int i = 0; i < iterations; ++i)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  if (threadIdx.x == 1024) sink[0] = c[0];
}

struct Sample {std::string item, config; double value = 0;};

double TimeKernel(void (*launch)(void*), void* context, int repeats) {
  cudaEvent_t start, stop;
  Check(cudaEventCreate(&start), "event");
  Check(cudaEventCreate(&stop), "event");
  launch(context);  // warm up
  Check(cudaDeviceSynchronize(), "warmup");
  Check(cudaEventRecord(start), "record");
  for (int i = 0; i < repeats; ++i) launch(context);
  Check(cudaEventRecord(stop), "record");
  Check(cudaDeviceSynchronize(), "sync");
  float ms = 0;
  Check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return double(ms) / repeats;
}
}  // namespace

int main(int argc, char** argv) {
  std::size_t gib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8;
  // argv[2] runs one item alone, which is how each is validated and how the
  // collection script keeps a process to a single measurement.
  std::string only = argc > 2 ? argv[2] : "";
  auto wanted = [&](char const* item) {return only.empty() || only == item;};
  std::size_t bytes = gib << 30;
  char* data = nullptr;
  Check(cudaMalloc(&data, bytes), "alloc");
  Check(cudaMemset(data, 1, bytes), "memset");
  float* sink = nullptr;
  Check(cudaMalloc(&sink, 256), "alloc sink");

  std::vector<Sample> samples;
  // (b) vector reads.
  for (int ctas : wanted("hbm_read") ? std::vector<int>{kSms, 2 * kSms} : std::vector<int>{}) {
    std::size_t vectors = bytes / sizeof(float4);
    struct Ctx {float4 const* d; std::size_t n; float4* s; int ctas;} ctx{
        reinterpret_cast<float4 const*>(data), vectors, reinterpret_cast<float4*>(sink), ctas};
    auto run = +[](void* raw) {
      auto* c = static_cast<Ctx*>(raw);
      HbmVector<8><<<c->ctas, 256>>>(c->d, c->n, c->s);
    };
    double ms = TimeKernel(run, &ctx, 5);
    samples.push_back({"hbm_read", "ld.global.nc.v4 x8, " + std::to_string(ctas / kSms) +
                                   " cta/sm", double(bytes) / (ms * 1e-3) / 1e9});
  }
  // (a) the bulk ring.
  for (int ctas : wanted("hbm_read_bulk") ? std::vector<int>{kSms, 2 * kSms} : std::vector<int>{}) {
    std::size_t blocks = bytes / kBlockBytes;
    for (int flight : {2, 4, 8}) {
      struct Ctx {char const* d; std::size_t n; float* s; int ctas, flight;} ctx{
          data, blocks, sink, ctas, flight};
      auto run = +[](void* raw) {
        auto* c = static_cast<Ctx*>(raw);
        std::size_t smem = std::size_t(c->flight) * kBlockBytes;
        if (c->flight == 2) {
          cudaFuncSetAttribute(HbmBulk<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
          HbmBulk<2><<<c->ctas, 128, smem>>>(c->d, c->n, c->s);
        } else if (c->flight == 4) {
          cudaFuncSetAttribute(HbmBulk<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
          HbmBulk<4><<<c->ctas, 128, smem>>>(c->d, c->n, c->s);
        } else {
          cudaFuncSetAttribute(HbmBulk<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
          HbmBulk<8><<<c->ctas, 128, smem>>>(c->d, c->n, c->s);
        }
      };
      double ms = TimeKernel(run, &ctx, 5);
      if (cudaGetLastError() != cudaSuccess) continue;
      samples.push_back({"hbm_read", "cp.async.bulk x" + std::to_string(flight) + ", " +
                                     std::to_string(ctas / kSms) + " cta/sm",
                         double(bytes) / (ms * 1e-3) / 1e9});
    }
  }
  // tensor cores
  if (wanted("bf16_wgmma")) {
    int iterations = 100000;
    struct Ctx {int n; float* s;} ctx{iterations, sink};
    auto run = +[](void* raw) {
      auto* c = static_cast<Ctx*>(raw);
      std::size_t smem = (64 * 64 + 256 * 64) * sizeof(Element);
      cudaFuncSetAttribute(Bf16Wgmma, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
      Bf16Wgmma<<<2 * kSms, 128, smem>>>(c->n, c->s);
    };
    double ms = TimeKernel(run, &ctx, 3);
    // 264 CTAs, one warpgroup each, m64n256k64 per iteration (four k16 steps).
    double flops = 2.0 * kSms * 64.0 * 256.0 * 64.0 * 2.0 * iterations;
    samples.push_back({"bf16_wgmma", "m64n256k16 SS, 1 wg/cta, 2 cta/sm",
                       flops / (ms * 1e-3) / 1e12});
  }
  if (wanted("bf16_mma_sync")) {
    int iterations = 200000;
    struct Ctx {int n; float* s;} ctx{iterations, sink};
    auto run = +[](void* raw) {
      auto* c = static_cast<Ctx*>(raw);
      Bf16MmaSync<<<kSms, 256, 0>>>(c->n, c->s);
    };
    double ms = TimeKernel(run, &ctx, 3);
    // Eight warps per CTA, one m16n8k16 each: 8*16*8*16*2 flops per iteration.
    double flops = double(kSms) * 8.0 * 16.0 * 8.0 * 16.0 * 2.0 * iterations;
    samples.push_back({"bf16_mma_sync", "m16n8k16, 8 warps/cta, 132 cta",
                       flops / (ms * 1e-3) / 1e12});
  }

  std::printf("{\"samples\":[");
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::printf("%s{\"item\":\"%s\",\"config\":\"%s\",\"value\":%.3f}", i ? "," : "",
                samples[i].item.c_str(), samples[i].config.c_str(), samples[i].value);
  std::printf("],\"units\":{\"hbm_read\":\"GB/s\",\"bf16_wgmma\":\"TFLOPS\","
              "\"bf16_mma_sync\":\"TFLOPS\"}}\n");
  cudaFree(data); cudaFree(sink);
  return 0;
}
