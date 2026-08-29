// Shared host harness for the V1 corrected-synchronisation experiments.
//
// Differences from the R1/R2 harness, all of which matter:
//   * Every output slot is verified, not just one value. R1/R2 checked a single
//     racily-written scalar (Bug C), so a block that produced garbage could go
//     unnoticed.
//   * The data buffer is pre-filled with a poison value (-1.0f) before launch.
//     If the producer's release is reordered ahead of its stores (Bug A), the
//     consumer reads poison and verification fails loudly instead of passing by
//     luck.
//   * The launch path matches R2/V0 (plain cuLaunchKernel with blockDim 1,1,1;
//     the driver expands to EIATTR_REQNTID) so hang rates stay comparable.
//
// usage: v1_host <cubin> <entry> <grid> <variant>
//   variant: ctrl | a | b | d
#include <cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static const int kDataElems = 262144;   // 256 chunks x 1024
static const int kOutElems  = 524288;   // 512 slots x 1024, covers any grid
static const int kFlagElems = 16384;    // 512 slots x 32 ints (128B padding)

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <cubin> <entry> <grid> <ctrl|a|b|d>\n",
                 argv[0]);
    return 2;
  }
  const char *cubin = argv[1];
  const char *entry = argv[2];
  int grid = std::atoi(argv[3]);
  const char *variant = argv[4];

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

  CUdeviceptr d_data, d_flag, d_out;
  CU_CHECK(cuMemAlloc(&d_data, (size_t)kDataElems * sizeof(float)));
  CU_CHECK(cuMemAlloc(&d_flag, (size_t)kFlagElems * sizeof(int)));
  CU_CHECK(cuMemAlloc(&d_out, (size_t)kOutElems * sizeof(float)));

  // Poison the data buffer: any value the consumer reads before the producer's
  // stores land is -1.0f and will fail verification.
  // V1_NO_SYNC isolates the compute path from the synchronisation path: the
  // host pre-fills the data with the final value and pre-sets the flag, so the
  // consumer's spin exits immediately and reads data that is already correct.
  // Any mismatch under V1_NO_SYNC is a bug in the compute (reduce), not in the
  // producer/consumer handshake.
  bool noSync = std::getenv("V1_NO_SYNC") != nullptr;
  std::vector<float> fill((size_t)kDataElems, noSync ? 1.0f : -1.0f);
  CU_CHECK(cuMemcpyHtoD(d_data, fill.data(), fill.size() * sizeof(float)));
  CU_CHECK(cuMemsetD32(d_flag, noSync ? 1 : 0, kFlagElems));
  // NaN-fill the output so an unwritten slot is distinguishable from zero.
  CU_CHECK(cuMemsetD32(d_out, 0x7fc00000u, kOutElems));
  CU_CHECK(cuCtxSynchronize());

  void *args[] = {&d_data, &d_flag, &d_out};
  std::printf("launching grid=%d variant=%s ...\n", grid, variant);
  std::fflush(stdout);
  CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, 1, 1, 1, 0, 0, args, nullptr));
  CUresult sync = cuCtxSynchronize();
  if (sync != CUDA_SUCCESS) {
    const char *s = nullptr;
    cuGetErrorString(sync, &s);
    std::printf("sync FAILED: %s\n", s ? s : "?");
    return 2;
  }

  std::vector<float> out((size_t)kOutElems);
  CU_CHECK(cuMemcpyDtoH(out.data(), d_out, out.size() * sizeof(float)));

  // Diagnostic: was the producer's data fully written by the time the kernel
  // ended? This separates "the producer never wrote it" from "the consumer
  // read it too early / saw a stale copy".
  if (std::getenv("V1_CHECK_DATA")) {
    std::vector<float> dat((size_t)kDataElems);
    CU_CHECK(cuMemcpyDtoH(dat.data(), d_data, dat.size() * sizeof(float)));
    long poisonLeft = 0;
    for (float v : dat)
      if (v != 1.0f)
        ++poisonLeft;
    std::printf("[diag] data elements still poisoned at kernel end: %ld / %d\n",
                poisonLeft, kDataElems);
  }

  int bad = 0;
  const int kMaxReport = 5;
  if (std::strcmp(variant, "ctrl") == 0) {
    // The unmodified kernel writes a single racily-shared scalar (Bug C) into
    // checksum_out[0]. There is no per-block output to verify; the point of
    // this variant is the hang rate, not correctness.
    std::printf("ctrl checksum=%g (single racy slot; not verified)\n", out[0]);
  } else if (std::strcmp(variant, "a") == 0) {
    // Consumer bx writes out[bx*1024 .. +1023] = 2.0f
    for (int b = 1; b < grid && bad < kMaxReport; ++b)
      for (int i = 0; i < 1024; ++i) {
        float v = out[(size_t)b * 1024 + i];
        if (v != 2.0f) {
          if (++bad <= kMaxReport)
            std::printf("MISMATCH out[%d*1024+%d] = %g, expected 2\n", b, i, v);
          break;
        }
      }
  } else {
    // b / c / d: consumer bx writes a single reduced value into out[bx].
    float expect = (std::strcmp(variant, "d") == 0) ? 65536.0f : 262144.0f;
    if (const char *e = std::getenv("V1_EXPECT"))
      expect = (float)std::atof(e);
    int firstConsumer = (std::strcmp(variant, "d") == 0) ? 4 : 1;
    for (int b = firstConsumer; b < grid; ++b) {
      float v = out[b];
      if (v != expect) {
        if (++bad <= kMaxReport)
          std::printf("MISMATCH out[%d] = %g, expected %g\n", b, v, expect);
      }
    }
  }

  if (bad) {
    std::printf("VERIFY FAILED (%d bad slots)\n", bad);
    return 3;
  }
  std::printf("kernel completed without hang; all slots verified.\n");
  return 0;
}
