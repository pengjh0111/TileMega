// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/PagedGemmTaskBody.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Body=PagedGemmTaskBody<tilemega::arch::Sm80,16,64,64,8192,3>;
__global__ void Norm(E const* input,E const* weight,E* output,int rows,int width) {
  __shared__ float scratch[4];
  ServingRMSNormTaskBody::RunRow(input,weight,output,blockIdx.x,1,0,width,1e-6f,scratch);
}
template<bool Fused>
__global__ void Gemm(ServingGemmOperands p,E const* input,E const* weight) {
  extern __shared__ __align__(1024) char storage[];
  constexpr int work=1024,pool=work+Body::kScratchBytes;
  typename Body::Ring ring{reinterpret_cast<typename Body::Ring::Slot*>(storage),storage+pool};
  ring.Initialize();std::uint64_t sequence=0;
  for(int m=0;m<(p.m+15)/16;++m)for(int n=0;n<(p.n+63)/64;++n) {
    if(!executor::IsCompute())Body::Load(p,n,ring,sequence);
    else Body::Run(p,m,n,ring,sequence,storage+work,Fused?input:nullptr,Fused?weight:nullptr,1e-6f);
  }
}
void Check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
int main(){
  int cases=0;
  for(int rows:{1,3,16,17})for(int width:{64,2048,3072})for(int columns:{64,128}) {
    E *input,*weight,*normalized,*b,*out,*ref;
    Check(cudaMallocManaged(&input,rows*width*2));Check(cudaMallocManaged(&weight,width*2));Check(cudaMallocManaged(&normalized,rows*width*2));
    Check(cudaMallocManaged(&b,columns*width*2));Check(cudaMallocManaged(&out,rows*columns*2));Check(cudaMallocManaged(&ref,rows*columns*2));
    for(int i=0;i<rows*width;++i)input[i]=E(float(i*7%61-30)/32);
    for(int i=0;i<width;++i)weight[i]=E(1+float(i*13%31-15)/64);
    for(int i=0;i<columns*width;++i)b[i]=E(float(i*17%67-33)/64);
    Norm<<<rows,128>>>(input,weight,normalized,rows,width);
    ServingGemmOperands p;p.a=normalized;p.b=b;p.output=ref;p.m=rows;p.n=columns;p.k_total=p.k_count=width;p.output_stride=columns;
    constexpr int bytes=1024+Body::kScratchBytes+3*8192;
    Gemm<false><<<1,160,bytes>>>(p,input,weight);p.output=out;
    Gemm<true><<<1,160,bytes>>>(p,input,weight);Check(cudaDeviceSynchronize());
    if(std::memcmp(out,ref,rows*columns*2)){std::fprintf(stderr,"norm prologue mismatch M=%d K=%d N=%d\n",rows,width,columns);return 3;}
    ++cases;Check(cudaFree(input));Check(cudaFree(weight));Check(cudaFree(normalized));Check(cudaFree(b));Check(cudaFree(out));Check(cudaFree(ref));
  }
  std::printf("NormPrologueGemm: %d bit-identical standalone norm plus page GEMM cases\n",cases);
}
