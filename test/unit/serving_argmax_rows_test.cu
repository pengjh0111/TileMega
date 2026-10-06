// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ServingEpilogue.h>
#include <cstdio>
#include <cstdlib>
using namespace tilemega;
using Op=backend::ServingEpilogue<backend::ServingEpilogueOp::kArgmaxPartial,16,128>;
__host__ __device__ float Input(int row,int col,int kind){
  if(kind==0)return (col%13==0)?-1.f:-2.f; // negative ties across warps
  if(kind==1)return col==127?8.f:-float(col+row+1);
  return -float((col+row)%17)*.25f;
}
__global__ void Rows(float* values,int* indices,int m,int n,int kind){
  __shared__ float tile[backend::ServingEpilogueScratchBytes(16,128,true)/4];
  for(int i=threadIdx.x;i<16*128;i+=128)tile[Op::SharedIndex(i/128,i%128)]=Input(i/128,i%128,kind);
  __syncthreads();Op::RunFromTile<true>(tile,0,0,m,n,1,n,nullptr,nullptr,nullptr,values,indices);
}
int main(){
  float* values;int* indices;
  if(cudaMallocManaged(&values,16*sizeof(float)) || cudaMallocManaged(&indices,16*sizeof(int)))return 2;
  for(int m:{1,4,16})for(int n:{1,31,73,127,128})for(int kind=0;kind<3;++kind)for(int repeat=0;repeat<3;++repeat){
    Rows<<<1,128>>>(values,indices,m,n,kind);if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
    for(int row=0;row<m;++row){float best=-INFINITY;int index=INT32_MAX;
      for(int col=0;col<n;++col){float v=float(cutlass::bfloat16_t(Input(row,col,kind)));if(v>best){best=v;index=col;}}
      if(values[row]!=best || indices[row]!=index){std::fprintf(stderr,"argmax mismatch m%d n%d row%d\n",m,n,row);return 3;}}
  }
  std::puts("Argmax rows: 45 negative/tied/tail shapes x 3 runs PASS");cudaFree(values);cudaFree(indices);
}
