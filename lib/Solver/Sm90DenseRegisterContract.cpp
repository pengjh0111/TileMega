// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseRegisterContract.h>
#include <tilemega/Backend/Sm90DenseRoleContract.h>
#include <cuda_occupancy.h>
#include <algorithm>
#include <stdexcept>

namespace tilemega::solver {
Sm90DenseRegisterContract NativeDenseRegisterContract(TargetSpec const& target,int residency) {
  if(target.NvccArch()!="sm_90a" || target.sm_major!=9 || target.sm_minor!=0 ||
      target.res.warp_size!=32 || (residency!=1 && residency!=2) || target.res.regs_per_sm<=0)
    throw std::invalid_argument("native register contract requires an SM90a resource target");
  cudaOccDeviceProp properties{};
  properties.computeMajor=target.sm_major;properties.computeMinor=target.sm_minor;
  Sm90DenseRegisterContract result;
  result.threads=backend::Sm90DenseRoleAbi::kThreads;result.residency=residency;
  if(cudaOccRegAllocationGranularity(&result.granularity,&properties)!=CUDA_OCC_SUCCESS ||
      cudaOccSubPartitionsPerMultiprocessor(&result.subpartitions,&properties)!=CUDA_OCC_SUCCESS ||
      cudaOccRegAllocationMaxPerThread(&result.architecture_limit,&properties)!=CUDA_OCC_SUCCESS)
    throw std::invalid_argument("CUDA occupancy model lacks the native register topology");
  auto ceil=[](int a,int b){return (a+b-1)/b;};
  int warps=ceil(result.threads,target.res.warp_size);
  int required=ceil(residency*warps,result.subpartitions);
  int partition=target.res.regs_per_sm/result.subpartitions;
  // Invert cudaOccMaxBlocksPerSMRegsLimit (PARTITIONED_GC_OFF). Rounding the
  // total CTA budget alone misses the per-subpartition allocation constraint.
  int per_warp=(partition/(required*result.granularity))*result.granularity;
  result.callee_limit=std::min(result.architecture_limit,per_warp/target.res.warp_size);
  if(result.callee_limit<1)
    throw std::invalid_argument("native register contract cannot allocate a resident warp");
  return result;
}
}  // namespace tilemega::solver
