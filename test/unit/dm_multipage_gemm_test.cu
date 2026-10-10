// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Backend/ServingDmGemm.h>
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef DM_TEST_TM
#define DM_TEST_TM 16
#endif
#ifndef DM_TEST_TN
#define DM_TEST_TN 16
#endif
#ifndef DM_TEST_TK
#define DM_TEST_TK 16
#endif
#ifndef DM_TEST_STAGES
#define DM_TEST_STAGES 2
#endif
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<tilemega::arch::CurrentArch>,
    tilemega::arch::Sm89,tilemega::arch::CurrentArch>;
using Body=tilemega::codegen::ServingGemmTaskBody<Arch,
    DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES>;
using Packing=tilemega::backend::ServingDmGemm<Arch,DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES>;
constexpr int PageBytes=DM_TEST_TN*DM_TEST_TK*2>8192?16384:8192;
using Paged=tilemega::codegen::PagedGemmTaskBody<Arch,DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,PageBytes,(DM_TEST_TN*DM_TEST_TK*2/PageBytes)+1>;
using Ring=typename Paged::Ring;
constexpr int Pool=(1024+Paged::kDmWorkspaceBytes+1023)/1024*1024;
constexpr int PagedBytes=Pool+PageBytes*((DM_TEST_TN*DM_TEST_TK*2/PageBytes)+1);
using Spec=tilemega::codegen::DmEpilogueSpec<tilemega::codegen::DmEpilogueProgram<>>;
__global__ void Run(tilemega::codegen::ServingGemmOperands p,E* out,bool tiled) {
  extern __shared__ __align__(16) char shared[];
  p.output=out;
  Body::template RunDm<Spec>(p,blockIdx.x,blockIdx.y,shared);
}
__global__ void RunPaged(tilemega::codegen::ServingGemmOperands p,E* out) {
  extern __shared__ __align__(1024) char shared[];
  Ring ring{reinterpret_cast<typename Ring::Slot*>(shared),shared+Pool};
  ring.Initialize();std::uint64_t sequence=0;p.output=out;
  // Start the B group at a nonzero ring slot. Groups of two/four pages
  // deliberately wrap rings of three/five slots on subsequent iterations.
  if(tilemega::codegen::executor::IsCompute()) {
    ring.AwaitFull(sequence);ring.Release(sequence);
  } else {
    ring.AcquireEmpty(sequence);ring.PublishCopies(sequence);
  }
  ++sequence;
  for(int tm=0;tm<(p.m+DM_TEST_TM-1)/DM_TEST_TM;++tm)
    for(int tn=0;tn<(p.n+DM_TEST_TN-1)/DM_TEST_TN;++tn) {
      if(tilemega::codegen::executor::IsCompute())
        Paged::template RunDm<Spec>(p,tm,tn,ring,sequence,shared+1024);
      else Paged::Load(p,tn,ring,sequence);
    }
}
template<class T>T* Managed(std::size_t count) {
  T* result;assert(cudaMallocManaged(&result,count*sizeof(T))==cudaSuccess);return result;
}
int main() {
  unsigned cases=0;
  int device=0,shared_limit=0;
  assert(cudaGetDevice(&device)==cudaSuccess);
  assert(cudaDeviceGetAttribute(&shared_limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device)==cudaSuccess);
  bool dense_fits=Body::kSharedBytes<=shared_limit;
  if(!dense_fits)std::printf("tile=%dx%dx%d dense workspace=%d exceeds device limit=%d; paged FP32 oracle remains required\n",
      DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,Body::kSharedBytes,shared_limit);
  for(int rows:{1,17,35})for(int cols:{3,17,67,129,259})
    for(int reduction:{16,24,27,32,64,128,191,257})for(int begin:{0,DM_TEST_TK}) {
      int total=reduction+begin,pitch=(total+7)&~7;
      auto* a=Managed<E>(rows*pitch);auto* b=Managed<E>(cols*pitch);
      int kt=(total+DM_TEST_TK-1)/DM_TEST_TK,nt=(cols+DM_TEST_TN-1)/DM_TEST_TN;
      auto* packed=Managed<E>(nt*kt*DM_TEST_TN*DM_TEST_TK);
      auto* output=Managed<E>(rows*cols+32);
      for(int m=0;m<rows;++m)for(int k=0;k<pitch;++k)
        a[m*pitch+k]=E(k<total?((m*7+k*13)%29-14)*0.03125f:0);
      for(int n=0;n<cols;++n)for(int k=0;k<pitch;++k)
        b[n*pitch+k]=E(k<total?((n*11+k*5)%23-11)*0.015625f:0);
      for(int tn=0;tn<nt;++tn)for(int tk=0;tk<kt;++tk)
        for(int n=0;n<DM_TEST_TN;++n)for(int k=0;k<DM_TEST_TK;++k) {
          int gn=tn*DM_TEST_TN+n,gk=tk*DM_TEST_TK+k;
          packed[(tn*kt+tk)*DM_TEST_TN*DM_TEST_TK+typename Packing::LayoutB{}(n,k)]=
              gn<cols && gk<total?b[gn*pitch+gk]:E(0);
        }
      tilemega::codegen::ServingGemmOperands p{};
      p.a=a;p.b=b;p.m=rows;p.n=cols;p.k_total=total;p.k_begin=begin;p.k_count=reduction;
      p.a_row_stride=p.b_row_stride=pitch;p.output_stride=cols;p.weight_base=packed;p.k_total_full=total;
      std::vector<E> dense;
      for(bool paged:{false,true}) {
        if(!paged && !dense_fits)continue;
        bool tiled=TILEMEGA_NONPAGED_TILED;
        for(int i=0;i<rows*cols+32;++i)output[i]=E(-12345);
        if(paged) {
          assert(cudaFuncSetAttribute(RunPaged,cudaFuncAttributeMaxDynamicSharedMemorySize,PagedBytes)==cudaSuccess);
          RunPaged<<<1,160,PagedBytes>>>(p,output+16);
        }else {
          assert(cudaFuncSetAttribute(Run,cudaFuncAttributeMaxDynamicSharedMemorySize,Body::kSharedBytes)==cudaSuccess);
          Run<<<dim3((rows+DM_TEST_TM-1)/DM_TEST_TM,nt),128,Body::kSharedBytes>>>(p,output+16,tiled);
        }
        auto status=cudaGetLastError();if(status==cudaSuccess)status=cudaDeviceSynchronize();
        if(status!=cudaSuccess) {
          std::fprintf(stderr,"tile=%dx%dx%d rows=%d cols=%d k=%d begin=%d paged=%d cuda=%s\n",
              DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,rows,cols,reduction,begin,paged,cudaGetErrorString(status));return 2;
        }
        for(int i=0;i<16;++i)assert(output[i]==E(-12345) && output[rows*cols+16+i]==E(-12345));
        for(int m=0;m<rows;++m)for(int n=0;n<cols;++n) {
          float ref=0;for(int k=begin;k<total;++k)ref+=float(a[m*pitch+k])*float(b[n*pitch+k]);
          float actual=float(output[16+m*cols+n]);
          if(!std::isfinite(actual) || std::abs(actual-ref)>0.016f+0.016f*std::abs(ref)) {
            std::fprintf(stderr,"tile=%dx%dx%d rows=%d cols=%d k=%d begin=%d paged=%d m=%d n=%d actual=%g ref=%g\n",
                DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,rows,cols,reduction,begin,paged,m,n,actual,ref);return 3;
          }
        }
        dense.assign(output+16,output+16+rows*cols);
        ++cases;
      }
      cudaFree(a);cudaFree(b);cudaFree(packed);cudaFree(output);
    }
  std::printf("DM multipage GEMM %dx%dx%d stages=%d: %u task BF16-store FP32 oracles, tails, offsets and canaries passed\n",
      DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES,cases);
  return 0;
}
