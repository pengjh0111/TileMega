// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/tasks/ServingGemmTaskBody.h>
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cassert>
#include <cmath>
#include <cstdio>
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
#define DM_TEST_STAGES 3
#endif
using namespace tilemega;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
using Body=codegen::ServingGemmTaskBody<Arch,DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES>;
using Packing=backend::ServingDmGemm<Arch,DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,DM_TEST_STAGES>;
constexpr int PageBytes=DM_TEST_TN*DM_TEST_TK*2>8192?16384:8192;
using Paged=codegen::PagedGemmTaskBody<Arch,DM_TEST_TM,DM_TEST_TN,DM_TEST_TK,PageBytes,2>;
using Ring=typename Paged::Ring;
constexpr int Pool=(1024+Paged::kDmWorkspaceBytes+1023)/1024*1024;
constexpr int PagedBytes=Pool+PageBytes*2;
using Spec=codegen::DmEpilogueSpec<codegen::DmEpilogueProgram<>,
    codegen::DmWriteKind::kDense,1,codegen::DmRounding::kFP32>;
__device__ codegen::ServingGemmOperands Chunk(codegen::ServingGemmOperands p,int split,int chunk,float* output) {
  int tiles=(p.k_total+DM_TEST_TK-1)/DM_TEST_TK;
  int begin=chunk*tiles/split,end=(chunk+1)*tiles/split;
  p.k_begin=begin*DM_TEST_TK;p.k_count=min(p.k_total,end*DM_TEST_TK)-p.k_begin;
  p.partial=output+std::size_t(chunk)*p.m*p.n;
  p.output=reinterpret_cast<E*>(p.partial);p.output_stride=p.partial_stride=p.n;
  p.chain.store_rounding=codegen::DmRounding::kFP32;
  p.epilogue=backend::ServingEpilogueOp::kPartial;return p;
}
__global__ void Run(codegen::ServingGemmOperands p,int split,float* output) {
  extern __shared__ __align__(16) char shared[];
  p=Chunk(p,split,blockIdx.z,output);
  if(split==1)Body::template RunDm<Spec>(p,blockIdx.x,blockIdx.y,shared);
  else Body::Run(p,blockIdx.x,blockIdx.y,shared);
}
__global__ void RunPaged(codegen::ServingGemmOperands p,int split,float* output) {
  extern __shared__ __align__(1024) char shared[];
  Ring ring{reinterpret_cast<typename Ring::Slot*>(shared),shared+Pool};
  ring.Initialize();std::uint64_t sequence=0;p=Chunk(p,split,blockIdx.x,output);
  for(int tm=0;tm<(p.m+DM_TEST_TM-1)/DM_TEST_TM;++tm)
    for(int tn=0;tn<(p.n+DM_TEST_TN-1)/DM_TEST_TN;++tn) {
      if(codegen::executor::IsCompute()) {
        if(split==1)Paged::template RunDm<Spec>(p,tm,tn,ring,sequence,shared+1024);
        else Paged::Run(p,tm,tn,ring,sequence,shared+1024);
      }else Paged::Load(p,tn,ring,sequence);
    }
}
template<class T>T* Managed(std::size_t count) {
  T* out=nullptr;assert(cudaMallocManaged(&out,count*sizeof(T))==cudaSuccess);return out;
}
int main() {
  assert(cudaFuncSetAttribute(Run,cudaFuncAttributeMaxDynamicSharedMemorySize,Body::kSharedBytes)==cudaSuccess);
  assert(cudaFuncSetAttribute(RunPaged,cudaFuncAttributeMaxDynamicSharedMemorySize,PagedBytes)==cudaSuccess);
  unsigned cases=0;
  for(unsigned batch:{1,3})for(unsigned area:{7,19})for(unsigned k:{24,27,72,129})
    for(unsigned n:{3,19})for(unsigned dtype:{0,1}) {
      unsigned m=batch*area,pitch=(k+7)/8*8,kt=(k+DM_TEST_TK-1)/DM_TEST_TK,nt=(n+DM_TEST_TN-1)/DM_TEST_TN;
      auto* a=Managed<E>(m*pitch);auto* b=Managed<E>(n*pitch);
      auto* packed=Managed<E>(nt*kt*DM_TEST_TN*DM_TEST_TK);
      auto* scale=Managed<float>(batch*k);auto* scale_bf16=Managed<E>(batch*k);
      auto* layouts=Managed<codegen::DmBufferLayout>(1);auto* dtypes=Managed<unsigned>(1);
      *layouts={};layouts->rank=2;layouts->logical[0]=layouts->physical[0]=batch;
      layouts->logical[1]=layouts->physical[1]=k;layouts->strides[0]=k;layouts->strides[1]=1;*dtypes=dtype;
      for(unsigned i=0;i<m*pitch;++i)a[i]=E(int(i*7%31)-15);
      for(unsigned i=0;i<n*pitch;++i)b[i]=E((int(i*11%29)-14)*.015625f);
      for(unsigned image=0;image<batch;++image)for(unsigned c=0;c<k;++c) {
        auto at=image*k+c;scale[at]=.91371f+.00231f*c+.03177f*image;scale_bf16[at]=E(scale[at]);
      }
      for(unsigned tn=0;tn<nt;++tn)for(unsigned t=0;t<kt;++t)
        for(unsigned col=0;col<DM_TEST_TN;++col)for(unsigned c=0;c<DM_TEST_TK;++c) {
          auto index=(std::size_t(tn)*kt+t)*DM_TEST_TN*DM_TEST_TK+typename Packing::LayoutB{}(col,c);
          packed[index]=tn*DM_TEST_TN+col<n && t*DM_TEST_TK+c<k?b[(tn*DM_TEST_TN+col)*pitch+t*DM_TEST_TK+c]:E(0);
        }
      codegen::ServingGemmOperands p;p.a=a;p.b=b;p.weight_base=packed;p.m=m;p.n=n;
      p.k_total=p.k_total_full=k;p.a_row_stride=p.b_row_stride=pitch;
      p.access.rows_per_batch=area;p.access.a_scale=0;
      p.a_scale=dtype?scale:reinterpret_cast<float const*>(scale_bf16);
      p.dm_buffers.layouts=layouts;p.dm_buffers.dtypes=dtypes;p.dm_buffers.count=1;
      std::vector<float> expected(m*n);
      for(unsigned row=0;row<m;++row)for(unsigned col=0;col<n;++col) {
        double sum=0;
        for(unsigned c=0;c<k;++c) {
          auto at=(row/area)*k+c;float s=dtype?scale[at]:float(scale_bf16[at]);
          sum+=double(float(E(float(a[row*pitch+c])*s)))*float(b[col*pitch+c]);
        }
        expected[row*n+col]=sum;
      }
      for(unsigned split:{1,2,4}) {
        if(split>kt)continue;
        auto* output=Managed<float>(std::size_t(split)*m*n);
        std::vector<float> first;
        for(unsigned epoch=0;epoch<3;++epoch)for(unsigned paged:{0,1}) {
          for(unsigned i=0;i<split*m*n;++i)output[i]=NAN;
          if(paged)RunPaged<<<split,160,PagedBytes>>>(p,split,output);
          else Run<<<dim3((m+DM_TEST_TM-1)/DM_TEST_TM,nt,split),128,Body::kSharedBytes>>>(p,split,output);
          assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);
          if(first.empty())first.assign(output,output+split*m*n);
          else for(unsigned i=0;i<split*m*n;++i)assert(output[i]==first[i]);
          for(unsigned i=0;i<m*n;++i) {
            float sum=0;for(unsigned chunk=0;chunk<split;++chunk)sum+=output[chunk*m*n+i];
            assert(std::isfinite(sum) && std::abs(sum-expected[i])<=.016f+.016f*std::abs(expected[i]));
          }
        }
        assert(cudaFree(output)==cudaSuccess);++cases;
      }
      for(void* pointer:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(packed),
          static_cast<void*>(scale),static_cast<void*>(scale_bf16),static_cast<void*>(layouts),static_cast<void*>(dtypes)})
        assert(cudaFree(pointer)==cudaSuccess);
    }
  std::printf("DM_SCA_DENSE cases=%u bf16_f32_scale_dense_pages_splitK PASS\n",cases);
}
