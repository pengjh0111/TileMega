// SPDX-License-Identifier: BSD-3-Clause
// Load a cubin function and compute its hardware residency from real metadata.

#include <cuda.h>

#include <cstdlib>
#include <iostream>
#include <string>

static void check(CUresult result, char const* call) {
  if (result == CUDA_SUCCESS) return;
  char const* message = nullptr;
  cuGetErrorString(result, &message);
  std::cerr << call << ": " << (message ? message : "CUDA driver error") << "\n";
  std::exit(2);
}

#define CU_CHECK(call) check((call), #call)

int main(int argc, char** argv) {
  std::string cubin;
  std::string kernel;
  int dynamic_smem = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() -> char const* {
      if (++i >= argc) { std::cerr << "missing value for " << arg << "\n"; std::exit(2); }
      return argv[i];
    };
    if (arg == "--cubin") cubin = value();
    else if (arg == "--kernel") kernel = value();
    else if (arg == "--dynamic-smem") dynamic_smem = std::atoi(value());
    else { std::cerr << "unknown argument: " << arg << "\n"; return 2; }
  }
  if (cubin.empty() || kernel.empty()) {
    std::cerr << "usage: tilemega-occupancy --cubin FILE --kernel NAME [--dynamic-smem BYTES]\n";
    return 2;
  }

  CU_CHECK(cuInit(0));
  CUdevice device;
  CU_CHECK(cuDeviceGet(&device, 0));
  CUcontext context;
  CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CU_CHECK(cuCtxSetCurrent(context));
  CUmodule module;
  CU_CHECK(cuModuleLoad(&module, cubin.c_str()));
  CUfunction function;
  CU_CHECK(cuModuleGetFunction(&function, module, kernel.c_str()));

  int block_size = 0, regs = 0, static_smem = 0, max_dynamic_smem = 0, num_sms = 0;
  CU_CHECK(cuFuncGetAttribute(&block_size, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
  CU_CHECK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function));
  CU_CHECK(cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
  CU_CHECK(cuFuncGetAttribute(&max_dynamic_smem,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function));
  CU_CHECK(cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  if (dynamic_smem < 0 || dynamic_smem > max_dynamic_smem) {
    std::cerr << "dynamic shared memory exceeds function limit " << max_dynamic_smem << "\n";
    return 2;
  }

  int ctas_per_sm = 0;
  CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &ctas_per_sm, function, block_size, static_cast<size_t>(dynamic_smem)));
  long long resident_limit = static_cast<long long>(ctas_per_sm) * num_sms;
  std::cout << "OCCUPANCY_TOOL"
            << " block_size=" << block_size
            << " registers_per_thread=" << regs
            << " static_smem_bytes=" << static_smem
            << " dynamic_smem_bytes=" << dynamic_smem
            << " ctas_per_sm=" << ctas_per_sm
            << " num_sms=" << num_sms
            << " resident_limit=" << resident_limit << "\n";
  CU_CHECK(cuModuleUnload(module));
  CU_CHECK(cuDevicePrimaryCtxRelease(device));
  return 0;
}
