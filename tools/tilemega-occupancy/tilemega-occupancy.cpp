// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// tilemega-occupancy —— 正确地查询一个 cubin 的 occupancy 与 cooperative launch 上限。
//
// 存在理由（TILEMEGA_SKELETON.md §6.7 / P0.3）：
//   R1 调研的全部 occupancy 诊断都调用
//       cuOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, 1, 0)
//   ——blockSize 传了字面量 1。而 tileiras 产出的 cubin 通过 EIATTR_REQNTID 硬性
//   要求一个特定的 blockDim。用 1 去查一个真实需要 N 线程的 kernel，会把资源占用
//   低估 N 倍，从而算出完全错误的 grid 上限（R1 因此误判 "8 blocks/SM, max grid
//   1360"，实际是 1 block/SM, max grid 170）。
//
//   本工具把 blockSize 从 kernel 自身读出来，杜绝这类误诊。读法用
//   CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK：对带 REQNTID 的 cubin，driver 报告的
//   上限就等于它被要求的 blockDim，比解析 ELF 更稳。
//
// 用法： tilemega-occupancy <cubin> <entry-name>

#include <cuda.h>

#include <cstdio>
#include <cstdlib>

static void check(CUresult rc, const char *what) {
  if (rc == CUDA_SUCCESS)
    return;
  const char *msg = nullptr;
  cuGetErrorString(rc, &msg);
  std::fprintf(stderr, "错误: %s 失败: %s\n", what, msg ? msg : "?");
  std::exit(1);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "用法: %s <cubin> <entry-name>\n", argv[0]);
    return 2;
  }

  check(cuInit(0), "cuInit");
  CUdevice dev;
  check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  // 用 primary context 而不是 cuCtxCreate：后者在 CUDA 13.3 里已是 _v4 四参数
  // 签名，直接调会随 toolkit 版本编不过。primary context 的签名是稳定的。
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

  // 关键一步：blockSize 从 kernel 自身读，不猜、不传 1。
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

  // 对照组：R1 犯的错误，保留下来让差距一眼可见。
  int blocksPerSMWrong = 0;
  cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSMWrong, fn, 1, 0);

  int coopMaxGrid = blocksPerSM * numSM;

  std::printf("设备            : %s (sm_%d%d, %d SM)\n", name, major, minor,
              numSM);
  std::printf("cubin           : %s\n", argv[1]);
  std::printf("entry           : %s\n", argv[2]);
  std::printf("--\n");
  std::printf("要求 blockDim   : %d   <-- 从 kernel 读出，不是猜的\n",
              reqBlockSize);
  std::printf("REG / thread    : %d\n", numRegs);
  std::printf("static SHM      : %d B\n", sharedBytes);
  std::printf("local / thread  : %d B\n", localBytes);
  std::printf("--\n");
  std::printf("blocks/SM       : %d\n", blocksPerSM);
  std::printf("cooperative 上限: %d  (= blocks/SM x SM 数)\n", coopMaxGrid);
  std::printf("--\n");
  std::printf("[对照] 若像 R1 那样错传 blockSize=1: blocks/SM = %d "
              "(高估 %.0fx)\n",
              blocksPerSMWrong,
              blocksPerSM ? (double)blocksPerSMWrong / blocksPerSM : 0.0);
  if (numRegs > 0 && reqBlockSize > 0) {
    std::printf("[手算] %d / (%d x %d) = %.2f\n", regsPerSM, numRegs,
                reqBlockSize, (double)regsPerSM / (numRegs * reqBlockSize));
  }

  cuModuleUnload(mod);
  cuDevicePrimaryCtxRelease(dev);
  return 0;
}
