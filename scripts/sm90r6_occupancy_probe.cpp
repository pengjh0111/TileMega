// SPDX-License-Identifier: BSD-3-Clause
#include <cuda.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

void check(CUresult status) {
  if(status==CUDA_SUCCESS)return;
  char const* error=nullptr;cuGetErrorString(status,&error);
  throw std::runtime_error(error?error:"CUDA driver query failed");
}
int main(int argc,char** argv) {try {
  if(argc!=7)throw std::invalid_argument("cubin symbol dynamic_smem threads grid cluster");
  if(!std::getenv("TILEMEGA_GPU_LOCK_HELD") ||
      std::string(std::getenv("TILEMEGA_GPU_LOCK_HELD"))!="1")
    throw std::invalid_argument("occupancy query requires the physical-card guard");
  int shared=std::stoi(argv[3]),threads=std::stoi(argv[4]),grid=std::stoi(argv[5]),cluster=std::stoi(argv[6]);
  if(shared<0 || threads<1 || grid<1 || cluster<1 || grid%cluster)
    throw std::invalid_argument("invalid occupancy launch dimensions");
  check(cuInit(0));CUdevice device;check(cuDeviceGet(&device,0));
  CUcontext context;check(cuDevicePrimaryCtxRetain(&context,device));check(cuCtxSetCurrent(context));
  CUmodule module;check(cuModuleLoad(&module,argv[1]));
  CUfunction function;check(cuModuleGetFunction(&function,module,argv[2]));
  check(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
  int resident=0,registers=0,static_shared=0,sms=0;
  check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident,function,threads,shared));
  check(cuFuncGetAttribute(&registers,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
  check(cuFuncGetAttribute(&static_shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function));
  check(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,device));
  int clusters=0;
  if(cluster>1) {
    CUlaunchAttribute attribute{};attribute.id=CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attribute.value.clusterDim={unsigned(cluster),1,1};
    CUlaunchConfig config{unsigned(grid),1,1,unsigned(threads),1,1,unsigned(shared),nullptr,&attribute,1};
    check(cuOccupancyMaxActiveClusters(&clusters,function,&config));
  }
  int supported=cluster>1?std::min(resident,clusters*cluster/sms):resident;
  std::printf("{\"resident_blocks_per_sm\":%d,\"resident_clusters\":%d,"
      "\"supported_residency\":%d,\"registers\":%d,\"static_shared_bytes\":%d,"
      "\"dynamic_shared_bytes\":%d,\"sms\":%d,\"grid\":%d,\"cluster\":%d}\n",
      resident,clusters,supported,registers,static_shared,shared,sms,grid,cluster);
  check(cuModuleUnload(module));check(cuDevicePrimaryCtxRelease(device));return 0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}}
