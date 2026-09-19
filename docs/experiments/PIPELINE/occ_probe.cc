// B1-b: ask the driver, on a real cubin, how many CTAs stay resident as a
// prefetch page is appended after the TaskSmem union.  This is the same query
// production makes (TargetSpec::ActiveBlocksPerSM); going through the driver
// API on a cubin lets the page size vary without editing the cell source.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ char const* s=nullptr; \
  cuGetErrorName(r,&s); std::fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#x,s?s:"?"); \
  std::exit(1);} } while(0)

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: probe <cubin> <mangled> <block> <page>...\n"); return 1; }
  char const* cubin = argv[1];
  char const* name = argv[2];
  int const block = std::atoi(argv[3]);
  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev));
  CUmodule mod; CK(cuModuleLoad(&mod, cubin));
  CUfunction fn; CK(cuModuleGetFunction(&fn, mod, name));
  int regs = 0, stat_smem = 0, cap = 0;
  CK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
  CK(cuFuncGetAttribute(&stat_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
  CK(cuDeviceGetAttribute(&cap, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
  std::printf("regs=%d static_smem=%d optin_cap=%d\n", regs, stat_smem, cap);
  for (int i = 4; i < argc; ++i) {
    int const dyn = std::atoi(argv[i]);
    if (dyn > cap) { std::printf("page=%d dyn=%d ctas=-1\n", dyn, dyn); continue; }
    // Production raises the opt-in limit the same way before launching.
    CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dyn));
    int ctas = 0;
    CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, fn, block, dyn));
    std::printf("page=%d dyn=%d ctas=%d\n", dyn - 16384, dyn, ctas);
  }
  return 0;
}
