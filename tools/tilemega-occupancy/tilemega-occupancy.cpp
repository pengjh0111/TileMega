// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// tilemega-occupancy -- query a cubin's occupancy and cooperative launch limit
// correctly.
//
// Why this exists (TILEMEGA_SKELETON.md section 6.7 / P0.3):
//   Every occupancy diagnostic in the R1 study called
//       cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, 1, 0)
//   passing a literal 1 as blockSize. But cubins produced by tileiras require a
//   specific blockDim through EIATTR_REQNTID. Querying a kernel that really
//   needs N threads with blockSize=1 underestimates resource usage by a factor
//   of N and yields a completely wrong grid limit (R1 concluded "8 blocks/SM,
//   max grid 1360" when the truth was 1 block/SM, max grid 170).
//
//   This tool reads blockSize from the kernel itself so that misdiagnosis
//   cannot recur.
//
// CAVEAT (found while writing this, see V0): the value read here is
// CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, which is the register-limited
// *upper bound*, not the blockDim that REQNTID demands. The two coincide for
// the kernels measured so far, but they are not the same quantity -- under
// tileiras 13.3 REQNTID is 128 while this attribute still reports 256. Reading
// EIATTR_REQNTID out of the ELF, as section 6.7 requires literally, is still
// outstanding.
//
// Usage: tilemega-occupancy <cubin> <entry-name>

#include <cuda.h>

#include <cstdio>
#include <cstdlib>

static void check(CUresult rc, const char *what) {
  if (rc == CUDA_SUCCESS)
    return;
  const char *msg = nullptr;
  cuGetErrorString(rc, &msg);
  std::fprintf(stderr, "error: %s failed: %s\n", what, msg ? msg : "?");
  std::exit(1);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <cubin> <entry-name>\n", argv[0]);
    return 2;
  }

  check(cuInit(0), "cuInit");
  CUdevice dev;
  check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  // Use the primary context rather than cuCtxCreate: the latter is the _v4
  // four-argument signature as of CUDA 13.3, so calling it directly fails to
  // compile depending on the toolkit version. The primary context API is
  // stable.
  CUcontext ctx;
  check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

  char name[256];
  check(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");

  int numSM = 0, regsPerSM = 0, smemPerSM = 0, major = 0, minor = 0;
  cuDeviceGetAttribute(&numSM, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
  cuDeviceGetAttribute(&regsPerSM,
                       CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                       dev);
  cuDeviceGetAttribute(
      &smemPerSM, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
      dev);
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       dev);

  CUmodule mod;
  check(cuModuleLoad(&mod, argv[1]), "cuModuleLoad");
  CUfunction fn;
  check(cuModuleGetFunction(&fn, mod, argv[2]), "cuModuleGetFunction");

  // The key step: read blockSize from the kernel itself. Do not guess, and do
  // not pass 1.
  int reqBlockSize = 0, numRegs = 0, sharedBytes = 0, localBytes = 0;
  check(cuFuncGetAttribute(&reqBlockSize,
                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn),
        "cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)");
  cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
  cuFuncGetAttribute(&sharedBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
  cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);

  int blocksPerSM = 0;
  check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, fn,
                                                    reqBlockSize, 0),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor");

  // Control: reproduce the mistake R1 made, so the gap is visible at a glance.
  int blocksPerSMWrong = 0;
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSMWrong, fn, 1, 0);

  int coopMaxGrid = blocksPerSM * numSM;

  std::printf("device           : %s (sm_%d%d, %d SMs)\n", name, major, minor,
              numSM);
  std::printf("cubin            : %s\n", argv[1]);
  std::printf("entry            : %s\n", argv[2]);
  std::printf("--\n");
  std::printf("required blockDim: %d   <-- read from the kernel, not guessed\n",
              reqBlockSize);
  std::printf("REG / thread     : %d\n", numRegs);
  std::printf("static SHM       : %d B\n", sharedBytes);
  std::printf("local / thread   : %d B\n", localBytes);
  std::printf("--\n");
  std::printf("blocks/SM        : %d\n", blocksPerSM);
  std::printf("cooperative limit: %d  (= blocks/SM x SM count)\n", coopMaxGrid);
  std::printf("--\n");
  std::printf("[control] passing blockSize=1 as R1 did: blocks/SM = %d "
              "(%.0fx overestimate)\n",
              blocksPerSMWrong,
              blocksPerSM ? (double)blocksPerSMWrong / blocksPerSM : 0.0);
  if (numRegs > 0 && reqBlockSize > 0) {
    std::printf("[by hand] %d / (%d x %d) = %.2f\n", regsPerSM, numRegs,
                reqBlockSize, (double)regsPerSM / (numRegs * reqBlockSize));
  }

  cuModuleUnload(mod);
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}
