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
//   An earlier version of this tool read CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
//   instead. That is the REGISTER-LIMITED UPPER BOUND, not the blockDim REQNTID
//   demands, and the two can differ wildly: for a kernel with REG=24 the driver
//   reports 1024 while REQNTID is 128. Section 6.7 says "read EIATTR_REQNTID"
//   literally, and it means it -- so this tool now parses the cubin ELF.
//
// Usage: tilemega-occupancy <cubin> <entry-name>

#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

/// Read EIATTR_REQNTID out of a cubin's `.nv.info.<entry>` section.
///
/// Encoding (verified empirically against `cuobjdump -elf`): the section is a
/// sequence of entries, each [format:u8][attribute:u8], and for EIFMT_SVAL
/// (format 0x04) followed by [length:u16][value bytes]. EIATTR_REQNTID is
/// attribute 0x10 with a 12-byte value holding three uint32 (x, y, z).
///
/// Returns 0 if the attribute is absent (the kernel then has no required
/// block dimension).
unsigned readRequiredBlockDim(const char *path, const char *entry) {
  FILE *f = std::fopen(path, "rb");
  if (!f)
    return 0;
  std::vector<unsigned char> d;
  {
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); return 0; }
    d.resize((size_t)n);
    if (std::fread(d.data(), 1, d.size(), f) != d.size()) {
      std::fclose(f); return 0;
    }
    std::fclose(f);
  }
  if (d.size() < 64 || std::memcmp(d.data(), "\x7f" "ELF", 4) != 0)
    return 0;

  auto u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, &d[o], 2); return v; };
  auto u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, &d[o], 4); return v; };
  auto u64 = [&](size_t o) { uint64_t v; std::memcpy(&v, &d[o], 8); return v; };

  uint64_t shoff = u64(0x28);
  uint16_t shentsize = u16(0x3a), shnum = u16(0x3c), shstrndx = u16(0x3e);
  if (!shoff || !shnum || shstrndx >= shnum)
    return 0;
  uint64_t strOff = u64(shoff + (uint64_t)shstrndx * shentsize + 0x18);

  std::string want = std::string(".nv.info.") + entry;
  for (uint16_t i = 0; i < shnum; ++i) {
    uint64_t sh = shoff + (uint64_t)i * shentsize;
    if (sh + shentsize > d.size())
      break;
    const char *nm = (const char *)&d[strOff + u32(sh)];
    if (want != nm)
      continue;
    uint64_t off = u64(sh + 0x18), size = u64(sh + 0x20);
    if (off + size > d.size())
      return 0;
    // Scan for [0x04][0x10][0x000c] followed by three uint32.
    for (uint64_t p = off; p + 4 + 12 <= off + size; ++p) {
      if (d[p] == 0x04 && d[p + 1] == 0x10 && u16(p + 2) == 12)
        return u32(p + 4);
    }
    return 0;
  }
  return 0;
}

} // namespace

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

  // The key step: read the REQUIRED block size out of the cubin's
  // EIATTR_REQNTID. Do not guess, do not pass 1, and do not substitute
  // MAX_THREADS_PER_BLOCK (that is the register-limited ceiling, a different
  // quantity entirely).
  int reqBlockSize = (int)readRequiredBlockDim(argv[1], argv[2]);
  int regLimitedMax = 0, numRegs = 0, sharedBytes = 0, localBytes = 0;
  check(cuFuncGetAttribute(&regLimitedMax,
                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn),
        "cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)");
  if (reqBlockSize == 0) {
    std::fprintf(stderr,
                 "warning: no EIATTR_REQNTID in %s; falling back to the "
                 "register-limited maximum (%d)\n",
                 argv[1], regLimitedMax);
    reqBlockSize = regLimitedMax;
  }
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
  std::printf("REQNTID blockDim : %d   <-- from EIATTR_REQNTID in the cubin\n",
              reqBlockSize);
  std::printf("reg-limited max  : %d   (NOT the required blockDim)\n",
              regLimitedMax);
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
