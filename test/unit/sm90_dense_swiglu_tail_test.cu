// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DN_VECTOR_SUMS 1
#include <tilemega/Backend/Sm90DenseEpilogue.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <stdexcept>
using BF=cutlass::bfloat16_t;
using namespace tilemega::backend;
void Check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__host__ __device__ float Input(int row,int col) {
  return col%8<4?(row%5+1)/4.f:(col%19-9)/8.f;
}
template<int TN,bool Native>
__global__ void Evaluate(BF* output,float const* norm,int batch) {
  extern __shared__ __align__(1024) char shared[];
  auto* tile=reinterpret_cast<float*>(shared+1024);int n=TN*gridDim.x;
  for(int i=threadIdx.x;i<16*TN;i+=160)tile[i]=Input(i/TN,blockIdx.x*TN+i%TN);
  __syncthreads();
  if(threadIdx.x<128) {
    if constexpr(Native)Sm90DenseEpilogue<ServingEpilogueOp::kSwiGLU,TN,4>::Run(
        tile,blockIdx.x,batch,n,n/2,output,nullptr,nullptr,norm,nullptr,128,1e-6f);
    else ServingEpilogue<ServingEpilogueOp::kSwiGLU,16,TN,4>::template RunFromTile<false>(
        tile,0,blockIdx.x,batch,n,n/2,n,output,nullptr,nullptr,nullptr,nullptr,norm,nullptr,128,1e-6f);
    tilemega::codegen::executor::ComputeSync();
  }
  __syncthreads();
}
template<int TN>
int Run(int& old_errors,int& cases) {
  BF* output;float* norm;Check(cudaMalloc(&output,16*5*TN));Check(cudaMalloc(&norm,16*128/8*4));
  std::vector<float> sums(16*128/8,.25f);
  Check(cudaMemcpy(norm,sums.data(),sums.size()*4,cudaMemcpyHostToDevice));
  int errors=0,before=0;
  for(int batch:{1,16})for(bool dn:{false,true})for(bool native:{false,true}) {
    Check(cudaMemset(output,0xff,16*5*TN));
    constexpr int shared=1024+(16*TN+32)*4;
    if(native)Evaluate<TN,true><<<5,160,shared>>>(output,dn?norm:nullptr,batch);
    else Evaluate<TN,false><<<5,160,shared>>>(output,dn?norm:nullptr,batch);
    Check(cudaGetLastError());Check(cudaDeviceSynchronize());
    std::vector<BF> host(batch*5*TN/2);
    Check(cudaMemcpy(host.data(),output,host.size()*2,cudaMemcpyDeviceToHost));
    for(int row=0;row<batch;++row)for(int tile=0;tile<5;++tile)for(int col=0;col<TN/2;++col) {
      int input=tile*TN+8*(col/4)+col%4;
      float scale=dn?1.f/std::sqrt(1.f/32+1e-6f):1.f;
      float g=float(BF(Input(row,input)*scale)),u=float(BF(Input(row,input+4)*scale));
      BF activated(g/(1.f+std::exp(-g))),expected(float(activated)*u);
      auto actual=host[row*(5*TN/2)+tile*(TN/2)+col];
      if(actual.raw()!=expected.raw()) {
        if(native) {
          if(errors<3)std::fprintf(stderr,"TN%d B%d DN%d row%d tile%d col%d got%g expected%g\n",
              TN,batch,dn,row,tile,col,float(actual),float(expected));
          ++errors;
        }else ++before;
      }
    }
    if(native)++cases;
  }
  if constexpr(TN>8)if(!before)throw std::runtime_error("missing pre-fix failure witness");
  old_errors+=before;Check(cudaFree(norm));Check(cudaFree(output));return errors;
}
int main() {try {
  Check(cudaSetDevice(0));
  int errors=0,old=0,cases=0;
  errors+=Run<8>(old,cases);errors+=Run<24>(old,cases);errors+=Run<40>(old,cases);
  errors+=Run<56>(old,cases);errors+=Run<72>(old,cases);errors+=Run<88>(old,cases);
  errors+=Run<104>(old,cases);errors+=Run<120>(old,cases);
  std::printf("{\"pass\":%s,\"errors\":%d,\"cases\":%d,\"old_mismatches\":%d}\n",
      errors?"false":"true",errors,cases,old);return errors?1:0;
}catch(std::exception const& e) {std::fprintf(stderr,"%s\n",e.what());return 2;}}
