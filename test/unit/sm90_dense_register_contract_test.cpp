// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseRegisterContract.h>
#include <cuda_occupancy.h>
#include <cassert>
#include <stdexcept>

using namespace tilemega;
using namespace tilemega::solver;
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  auto target=TargetSpec::FromJson(argv[1]);
  auto one=NativeDenseRegisterContract(target,1);
  auto two=NativeDenseRegisterContract(target,2);
  assert(!one.Constrained() && two.Constrained() && two.callee_limit==168);
  cudaOccDeviceProp properties{};
  properties.computeMajor=target.sm_major;properties.computeMinor=target.sm_minor;
  properties.warpSize=target.res.warp_size;
  properties.regsPerMultiprocessor=target.res.regs_per_sm;
  // The fixture's SM90 per-block limit; production never infers it from this
  // test or treats the register-only bound as full resource feasibility.
  properties.regsPerBlock=65536;
  auto blocks=[&](int regs){
    cudaOccFuncAttributes attributes{};attributes.numRegs=regs;
    cudaOccResult result{};auto gc=PARTITIONED_GC_OFF;int count=0;
    assert(cudaOccMaxBlocksPerSMRegsLimit(&count,&gc,&result,&properties,
        &attributes,two.threads)==CUDA_OCC_SUCCESS);
    return count;
  };
  assert(blocks(two.callee_limit)>=2 && blocks(two.callee_limit+1)<2);
  assert(blocks(200)<2);
  auto smaller=target;smaller.res.regs_per_sm/=2;
  assert(NativeDenseRegisterContract(smaller,2).callee_limit==80);
  auto rejects=[&](TargetSpec const& t,int r){
    bool failed=false;try {NativeDenseRegisterContract(t,r);}
    catch(std::invalid_argument const&) {failed=true;}assert(failed);
  };
  rejects(target,3);smaller.res.regs_per_sm=0;rejects(smaller,2);
  smaller=target;smaller.nvcc_arch="sm_89";rejects(smaller,1);
  smaller=target;smaller.sm_major=6;rejects(smaller,2);
  smaller=target;smaller.res.warp_size=16;rejects(smaller,2);
}
