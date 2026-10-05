// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ServingGemm.h>
#include <tilemega/Backend/ServingMmaPipeline.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Arch=tilemega::arch::CurrentArch;
using TestElement=cutlass::bfloat16_t;
__host__ __device__ float AValue(int m,int k) {return ((m*13+k*7)%29-14)*0.015625f;}
__host__ __device__ float BValue(int n,int k) {return ((n*17+k*11)%31-15)*0.015625f;}

template<int TN,int TK,bool Pipe>
__global__ void Run(float* output,int M,int N,int K) {
  using namespace cute;
  using Config=tilemega::backend::ServingGemmConfig<Arch,16,TN,TK,2>;
  using LA=decltype(tile_to_shape(typename Config::SmemLayoutAtom{},Shape<_16,Int<TK>>{}));
  using LB=decltype(tile_to_shape(typename Config::SmemLayoutAtom{},Shape<Int<TN>,Int<TK>>{}));
  __shared__ __align__(16) TestElement a[16*TK],b[TN*TK];
  typename Config::TiledMma mma;
  auto thread=mma.get_slice(threadIdx.x);
  auto accum=partition_fragment_C(mma,Shape<_16,Int<TN>>{});clear(accum);
  auto sa=make_tensor(make_smem_ptr(a),LA{});auto sb=make_tensor(make_smem_ptr(b),LB{});
  auto ra=thread.partition_fragment_A(sa);auto rb=thread.partition_fragment_B(sb);
  auto ca=make_tiled_copy_A(typename Config::SmemCopyAtom{},mma);
  auto cb=make_tiled_copy_B(typename Config::SmemCopyAtomB{},mma);
  auto as=ca.get_slice(threadIdx.x).partition_S(sa);auto bs=cb.get_slice(threadIdx.x).partition_S(sb);
  auto ad=ca.get_slice(threadIdx.x).retile_D(ra);auto bd=cb.get_slice(threadIdx.x).retile_D(rb);
  for(int first=0;first<K;first+=TK) {
    for(int i=threadIdx.x;i<16*TK;i+=blockDim.x)
      a[LA{}(i/TK,i%TK)]=TestElement(i/TK<M && first+i%TK<K?AValue(i/TK,first+i%TK):0);
    for(int i=threadIdx.x;i<TN*TK;i+=blockDim.x)
      b[LB{}(i/TK,i%TK)]=TestElement(i/TK<N && first+i%TK<K?BValue(i/TK,first+i%TK):0);
    __syncthreads();
    if constexpr(Pipe)tilemega::backend::ServingMmaRegisterPipeline<Arch,Config>(mma,accum,ra,rb,as,bs,ad,bd);
    else {
      #pragma unroll
      for(int k=0;k<size<2>(ra);++k) {
        copy(typename Config::SmemCopyAtom{},as(_,_,k),ad(_,_,k));
        copy(typename Config::SmemCopyAtomB{},bs(_,_,k),bd(_,_,k));
        gemm(mma,ra(_,_,k),rb(_,_,k),accum);
      }
    }
    __syncthreads();
  }
  auto c=make_tensor(make_gmem_ptr(output),make_layout(Shape<_16,Int<TN>>{},LayoutRight{}));
  auto dest=thread.partition_C(c);copy(accum,dest);
}
void CheckCuda(cudaError_t code) {
  if(code!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(code));std::exit(2);}
}
template<int TN,int TK> int Check() {
  float* output=nullptr;CheckCuda(cudaMallocManaged(&output,2*16*TN*sizeof(float)));
  int cases=0;
  for(int m:{1,4,16})for(int n:{TN-7,TN})for(int k:{TK-7,TK,3*TK+19}) {
    Run<TN,TK,false><<<1,128>>>(output,m,n,k);
    Run<TN,TK,true><<<1,128>>>(output+16*TN,m,n,k);
    CheckCuda(cudaDeviceSynchronize());
    if(std::memcmp(output,output+16*TN,16*TN*sizeof(float))) {
      std::fprintf(stderr,"pipeline changes bits TN=%d TK=%d M=%d N=%d K=%d\n",TN,TK,m,n,k);std::exit(3);
    }
    for(int row=0;row<16;++row)for(int col=0;col<TN;++col) {
      float reference=0;
      if(row<m && col<n)for(int r=0;r<k;++r)reference+=float(TestElement(AValue(row,r)))*float(TestElement(BValue(col,r)));
      if(std::abs(output[row*TN+col]-reference)>0.0002f*(1+std::abs(reference))) {
        std::fprintf(stderr,"position-coded reference mismatch [%d,%d]\n",row,col);std::exit(4);
      }
    }
    ++cases;
  }
  CheckCuda(cudaFree(output));return cases;
}
int main() {
  int cases=Check<32,64>()+Check<32,128>()+Check<64,64>()+Check<64,128>()+Check<128,64>()+Check<128,128>();
  std::printf("MMA register pipeline: %d position-coded cases, bitwise equality PASS\n",cases);
}
