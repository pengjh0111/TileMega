// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Support/Json.h>
#include <tilemega/Target/TargetSpec.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
namespace tilemega::commands::device {
int RunDevice(int argc,char** argv) {
  int device=0;
  if(argc==3 && std::string(argv[1])=="--device")device=std::stoi(argv[2]);
  else if(argc!=1)throw std::runtime_error("usage: tilemega probe device [--device N]");
  auto check=[](cudaError_t result) {if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));};
  auto attr=[&](cudaDeviceAttr key) {int value;check(cudaDeviceGetAttribute(&value,key,device));return value;};
  cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,device));
  int driver,runtime;check(cudaDriverGetVersion(&driver));check(cudaRuntimeGetVersion(&runtime));
  auto target=TargetSpec::Probe(device);auto record=json::Parse(target.ToJson());
  record.Set("name",prop.name);record.Set("device",device);
  record.Set("driver_version",driver);record.Set("runtime_version",runtime);
  record.Set("l2_bytes",attr(cudaDevAttrL2CacheSize));
  record.Set("theoretical_bandwidth_gbps",2.0*attr(cudaDevAttrMemoryClockRate)*1000.0*
      attr(cudaDevAttrGlobalMemoryBusWidth)/8.0/1e9);
  std::cout<<record.Dump()<<'\n';return 0;
}
}
