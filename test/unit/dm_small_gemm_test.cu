// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ServingDmGemm.h>
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
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
using Body=tilemega::backend::ServingDmGemm<Arch,
    DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES>;
static_assert(cute::size(typename Body::TiledMma{})==128);
static_assert(cute::size(typename Body::ComputeMma{})==128/Body::kKSplits);
__global__ void Run(tilemega::codegen::ServingGemmOperands p,float* out,bool tiled) {
  extern __shared__ __align__(16) char shared[];
  auto* tile=Body::Dense(p,blockIdx.x,blockIdx.y,shared,tiled);
  for(int i=threadIdx.x;i<DM_TEST_TM*DM_TEST_TN;i+=128) {
    int row=blockIdx.x*DM_TEST_TM+i/DM_TEST_TN;
    int col=blockIdx.y*DM_TEST_TN+i%DM_TEST_TN;
    if(row<p.m && col<p.n)out[row*p.n+col]=tile[i];
  }
}
template<class T>T* Managed(std::size_t count) {
  T* result;assert(cudaMallocManaged(&result,count*sizeof(T))==cudaSuccess);return result;
}
int main() {
  unsigned cases=0;
  for(int rows:{1,17,35})for(int cols:{3,17,67})
    for(int reduction:{16,24,27,32,64,128,191,257})for(int begin:{0,DM_TEST_TK}) {
      int total=reduction+begin,pitch=(total+7)&~7;
      auto* a=Managed<E>(rows*pitch);auto* b=Managed<E>(cols*pitch);
      int kt=(total+DM_TEST_TK-1)/DM_TEST_TK,nt=(cols+DM_TEST_TN-1)/DM_TEST_TN;
      auto* packed=Managed<E>(nt*kt*DM_TEST_TN*DM_TEST_TK);
      auto* output=Managed<float>(rows*cols+32);
      for(int m=0;m<rows;++m)for(int k=0;k<pitch;++k)
        a[m*pitch+k]=E(k<total?((m*7+k*13)%29-14)*0.03125f:0);
      for(int n=0;n<cols;++n)for(int k=0;k<pitch;++k)
        b[n*pitch+k]=E(k<total?((n*11+k*5)%23-11)*0.015625f:0);
      for(int tn=0;tn<nt;++tn)for(int tk=0;tk<kt;++tk)
        for(int n=0;n<DM_TEST_TN;++n)for(int k=0;k<DM_TEST_TK;++k) {
          int gn=tn*DM_TEST_TN+n,gk=tk*DM_TEST_TK+k;
          packed[(tn*kt+tk)*DM_TEST_TN*DM_TEST_TK+typename Body::LayoutB{}(n,k)]=
              gn<cols && gk<total?b[gn*pitch+gk]:E(0);
        }
      tilemega::codegen::ServingGemmOperands p{};
      p.a=a;p.b=b;p.m=rows;p.n=cols;p.k_total=total;p.k_begin=begin;p.k_count=reduction;
      p.a_row_stride=p.b_row_stride=pitch;p.weight_base=packed;p.k_total_full=total;
      std::vector<float> dense;
      for(bool tiled:{false,true}) {
        for(int i=0;i<rows*cols+32;++i)output[i]=-12345;
        assert(cudaFuncSetAttribute(Run,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Body::kSharedBytes)==cudaSuccess);
        Run<<<dim3((rows+DM_TEST_TM-1)/DM_TEST_TM,nt),128,Body::kSharedBytes>>>(p,output+16,tiled);
        assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
        for(int i=0;i<16;++i)assert(output[i]==-12345 && output[rows*cols+16+i]==-12345);
        for(int m=0;m<rows;++m)for(int n=0;n<cols;++n) {
          float ref=0;for(int k=begin;k<total;++k)ref+=float(a[m*pitch+k])*float(b[n*pitch+k]);
          float actual=float(E(output[16+m*cols+n]));
          if(!std::isfinite(actual) || std::abs(actual-ref)>0.016f+0.016f*std::abs(ref)) {
            std::fprintf(stderr,"tile=%dx%dx%d rows=%d cols=%d k=%d begin=%d tiled=%d m=%d n=%d actual=%g ref=%g\n",
                DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,rows,cols,reduction,begin,tiled,m,n,actual,ref);return 3;
          }
        }
        if(!tiled)dense.assign(output+16,output+16+rows*cols);
        else assert(std::memcmp(dense.data(),output+16,rows*cols*sizeof(float))==0);
        ++cases;
      }
      cudaFree(a);cudaFree(b);cudaFree(packed);cudaFree(output);
    }
  std::printf("DM small GEMM %dx%dx%d stages=%d: %u row/tiled FP32 oracles, tails, offsets and canaries passed\n",
      DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES,cases);
}
