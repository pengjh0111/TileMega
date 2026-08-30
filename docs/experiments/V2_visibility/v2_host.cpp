// ============================================================================
// V2 host harness -- generalised from v1_host.cpp for the visibility study.
//
// Differences from v1_host.cpp, each of which was needed by a V2 question:
//   * The check mode is a command-line argument rather than a hard-coded
//     variant letter, so a new .mlir does not require a new host binary.
//   * Failing values are histogrammed. V1 reported "reads 768 or 512" from a
//     handful of eyeballed samples; the granularity of the loss (128 elements
//     = one STG.E share, vs 32 = one warp/cache line) is the single most
//     load-bearing fact in the SASS audit, so it is measured, not sampled.
//   * --reps runs the launch N times in one process. Process-per-run remains
//     the default because that is what V1 measured and comparability matters.
//   * Optional launch-timing perturbation (V2-a'), to separate "mechanically
//     safe" from "timing-lucky".
//
// usage: v2_host <cubin> <entry> <grid> <mode> [args]
//   modes:
//     scalar <expect> <firstConsumer>   out[b] == expect, b in [first, grid)
//     block1024 <expect>                out[b*1024+i] == expect (V1-a shape)
//     chunked <expect> <nprod>          out[b] == expect for consumers only
// ============================================================================
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define CU_CHECK(x)                                                            \
  do {                                                                         \
    CUresult _e = (x);                                                         \
    if (_e != CUDA_SUCCESS) {                                                  \
      const char *s = nullptr;                                                 \
      cuGetErrorString(_e, &s);                                                \
      std::fprintf(stderr, "%s:%d CUDA error %d: %s\n", __FILE__, __LINE__,    \
                   (int)_e, s ? s : "?");                                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const int kDataElems = 524288;  // 512 chunks x 1024 (V2-e needs grid+1 chunks)
static const int kOutElems = 524288;
static const int kFlagElems = 16384;  // 512 slots x 32 ints (128B padding)

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <cubin> <entry> <grid> <mode> [args]\n"
                 "  scalar <expect> <first> | block1024 <expect> | "
                 "chunked <expect> <nprod>\n",
                 argv[0]);
    return 2;
  }
  const char *cubin = argv[1];
  const char *entry = argv[2];
  int grid = std::atoi(argv[3]);
  std::string mode = argv[4];
  double expect = argc > 5 ? std::atof(argv[5]) : 1024.0;
  int aux = argc > 6 ? std::atoi(argv[6]) : 1;

  int reps = 1;
  if (const char *r = std::getenv("V2_REPS")) reps = std::atoi(r);
  // Perturbation knobs for V2-a': change when the consumer blocks actually get
  // to run relative to the producer, without changing the kernel itself.
  int spinBeforeLaunch = std::getenv("V2_HOST_DELAY_US")
                             ? std::atoi(std::getenv("V2_HOST_DELAY_US"))
                             : 0;
  bool useStream = std::getenv("V2_USE_STREAM") != nullptr;

  CU_CHECK(cuInit(0));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU_CHECK(cuCtxSetCurrent(ctx));

  CUmodule mod;
  CU_CHECK(cuModuleLoad(&mod, cubin));
  CUfunction fn;
  CU_CHECK(cuModuleGetFunction(&fn, mod, entry));

  CUstream stream = 0;
  if (useStream) CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  CUdeviceptr d_data, d_flag, d_out;
  CU_CHECK(cuMemAlloc(&d_data, (size_t)kDataElems * sizeof(float)));
  CU_CHECK(cuMemAlloc(&d_flag, (size_t)kFlagElems * sizeof(int)));
  CU_CHECK(cuMemAlloc(&d_out, (size_t)kOutElems * sizeof(float)));

  bool noSync = std::getenv("V2_NO_SYNC") != nullptr;
  std::vector<float> fill((size_t)kDataElems, noSync ? 1.0f : -1.0f);
  // V2_HOSTFILL_CHUNK: pre-fill data from this 1024-element chunk onwards with
  // the final value. A consumer reading there exercises the identical spin,
  // load latency and reduction, but its data was written by the HOST before the
  // launch, so cross-block visibility cannot be the explanation for a wrong
  // answer. This is the control that separates "visibility" from "reduction".
  if (mode == "v2e")
    for (size_t i = 0; i < 1024; ++i) fill[i] = 1.0f;   // seed chunk 0
  if (const char *hc = std::getenv("V2_HOSTFILL_CHUNK")) {
    for (size_t i = (size_t)std::atoi(hc) * 1024; i < fill.size(); ++i)
      fill[i] = 1.0f;
  }
  std::vector<float> out((size_t)kOutElems);

  std::map<double, int> badHist;  // observed wrong value -> count
  int runsFailed = 0, totalBadSlots = 0;

  // V2_ALT_FILL: give every rep a DIFFERENT value (1.0, 2.0, 1.0, ...).
  // Reps after the first normally pass, which the "stale shared-memory warp
  // partial" story explains as an accident: the SM's shared memory still holds
  // the identical partial from the previous launch, so reading it stale gives
  // the right number anyway. Alternating the value removes that accident -- if
  // the story is right, later reps must now fail too.
  bool altFill = std::getenv("V2_ALT_FILL") != nullptr;
  double baseExpect = expect;

  for (int r = 0; r < reps; ++r) {
    std::map<double, int> repHist;
    if (altFill) {
      float hi = std::getenv("V2_ALT_HI") ? (float)std::atof(std::getenv("V2_ALT_HI")) : 2.0f;
      float v = (r % 2) ? hi : 1.0f;
      std::fill(fill.begin(), fill.end(), v);
      // Scaling the expectation is only correct when the HOST is the source of
      // the reduced data (V2_HOSTFILL_CHUNK / v2_f_hostdata). In v1_min the
      // producer block overwrites chunk 0 with a constant 1.0 from the MLIR, so
      // the right answer does NOT scale -- an early run of this harness scaled
      // it anyway and mis-read the resulting mismatch as corruption.
      // V2_ALT_NOSCALE pins the expectation for that case.
      if (!std::getenv("V2_ALT_NOSCALE")) expect = baseExpect * v;
    }
    CU_CHECK(cuMemcpyHtoD(d_data, fill.data(), fill.size() * sizeof(float)));
    CU_CHECK(cuMemsetD32(d_flag, noSync ? 1 : 0, kFlagElems));
    if (mode == "v2e") {
      // V2-e is a chain: block bx produces chunk bx+1 and releases flag[(bx+1)*32],
      // then consumes chunk bx. Nobody produces chunk 0, so the host seeds it and
      // its flag; that is the only thing block 0 waits on.
      // V2_NOSEED_FLAG: for the shared-flag cells of the 2x2, block 0 releases
      // flag[0] itself, so seeding it would make every consumer's wait trivially
      // satisfied on the peeled first poll -- exactly the condition that hides
      // the defect. Leave it clear so the spin is real.
      if (!std::getenv("V2_NOSEED_FLAG")) {
        int one = 1;
        CU_CHECK(cuMemcpyHtoD(d_flag, &one, sizeof(int)));
      }
    }
    CU_CHECK(cuMemsetD32(d_out, 0x7fc00000u, kOutElems));  // NaN
    CU_CHECK(cuCtxSynchronize());

    if (spinBeforeLaunch) {
      // Busy-wait on the host between the memsets and the launch.
      volatile long long x = 0;
      for (long long i = 0; i < (long long)spinBeforeLaunch * 300; ++i) x += i;
    }

    void *args[] = {&d_data, &d_flag, &d_out};
    CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
    CUresult sync =
        useStream ? cuStreamSynchronize(stream) : cuCtxSynchronize();
    if (sync != CUDA_SUCCESS) {
      const char *s = nullptr;
      cuGetErrorString(sync, &s);
      std::printf("sync FAILED: %s\n", s ? s : "?");
      return 2;
    }

    CU_CHECK(cuMemcpyDtoH(out.data(), d_out, out.size() * sizeof(float)));

    int bad = 0;
    if (mode == "blockN") {
      // aux = elements per consumer block. Record WHICH element indices went
      // stale: the loss granularity (1 / 32 / 128 elements) is the whole point.
      int N = aux;
      for (int b = 1; b < grid; ++b) {
        int bbad = 0, lo = -1, hi = -1;
        for (int i = 0; i < N; ++i) {
          float v = out[(size_t)b * N + i];
          if (v != (float)expect) {
            ++bad; ++bbad; badHist[v]++;
            if (lo < 0) lo = i;
            hi = i;
          }
        }
        if (bbad && std::getenv("V2_VERBOSE"))
          std::printf("  rep %d block %d: %d/%d stale, idx %d..%d\n", r, b,
                      bbad, N, lo, hi);
      }
    } else {  // scalar / chunked: one slot per consumer block
      int first = (mode == "chunked") ? aux : ((mode == "v2e") ? 0 : aux);
      int last = grid;
      if (const char *vl = std::getenv("V2_VERIFY_LAST")) last = std::atoi(vl) + 1;
      if (last > grid) last = grid;
      for (int b = first; b < last; ++b) {
        float v = out[b];
        if (v != (float)expect) { ++bad; badHist[v]++; repHist[v]++; }
      }
    }
    if (bad) { ++runsFailed; totalBadSlots += bad;
      if (std::getenv("V2_VERBOSE")) {
        // Per-rep histogram: aggregating across reps is ambiguous when the fill
        // value alternates, because the same wrong value can be reached from
        // either rep's partials. Per-rep, each value has one decomposition.
        std::printf("  rep %d expect=%-6g %d bad slots :", r, expect, bad);
        for (auto &kv : repHist) std::printf("  %g x%d", kv.first, kv.second);
        std::printf("\n");
      } }
  }

  std::printf("grid=%-4d reps=%-4d mode=%s expect=%g : failedRuns=%d/%d "
              "badSlots=%d\n",
              grid, reps, mode.c_str(), expect, runsFailed, reps,
              totalBadSlots);
  if (!badHist.empty()) {
    std::printf("  bad-value histogram:");
    int shown = 0;
    for (auto &kv : badHist) {
      std::printf("  %g x%d", kv.first, kv.second);
      if (++shown >= 12) { std::printf("  ..."); break; }
    }
    std::printf("\n");
  }
  if (runsFailed) { std::printf("VERIFY FAILED\n"); return 3; }
  std::printf("all slots verified.\n");
  return 0;
}
