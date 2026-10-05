// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/PagedAttentionTaskBody.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace tilemega;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm80,arch::CurrentArch>;
using E=cutlass::bfloat16_t;
void Check(cudaError_t status){if(status!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(2);}}
template<int D,int Q>
__global__ void PVTest(float* out,int* matched) {
  using namespace cute;
  using Body=codegen::PagedAttentionTaskBody<Arch,D,Q,false,8192,4>;
  using Swap=backend::ServingAttentionPVSwap<Arch,D,typename Body::ValueLayout>;
  using Score=typename Body::QK;
  __shared__ E values[D*16];int lane=threadIdx.x;
  for(int k=0;k<16;++k)for(int d=lane;d<D;d+=32)
    values[typename Body::ValueLayout{}(d,k)]=E((k-7)*.125f+(d%11)*.0625f);
  __syncthreads();
  auto coords=typename Score::Mma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<_16,_16>{}));
  auto prob=Score::Accumulator();
  for(int i=0;i<size(prob);++i) {
    int q=get<0>(coords(i)),k=get<1>(coords(i));
    prob(i)=q<Q?float(E((q+1)*.03125f+(k+1)*.015625f)):0;
  }
  auto bcoords=typename Swap::Mma{}.get_slice(lane).partition_B(make_identity_tensor(Shape<_8,_16>{}));
  int matches=0;
  for(int i=0;i<size(bcoords);++i) {
    int count=0;
    for(int j=0;j<size(coords);++j)
      count+=get<0>(bcoords(i))==get<0>(coords(j)) && get<1>(bcoords(i))==get<1>(coords(j));
    if(count==1)++matches;
  }
  matched[lane]=matches==size(bcoords);
  auto acc=Swap::Accumulator();Swap::PV(prob,coords,values,acc);
  auto output=typename Swap::Mma{}.get_slice(lane).partition_C(make_identity_tensor(Shape<Int<D>,_8>{}));
  for(int i=0;i<size(acc);++i)out[get<1>(output(i))*D+get<0>(output(i))]=acc(i);
}
template<int D,int Q>void Run() {
  float* out;int* matched;Check(cudaMallocManaged(&out,D*8*sizeof(float)));Check(cudaMallocManaged(&matched,32*sizeof(int)));
  PVTest<D,Q><<<1,32>>>(out,matched);Check(cudaDeviceSynchronize());
  for(int l=0;l<32;++l)if(!matched[l]){std::fprintf(stderr,"probability fragment mapping failed lane %d\n",l);std::exit(3);}
  for(int q=0;q<8;++q)for(int d=0;d<D;++d){
    float expected=0;for(int k=0;k<16;++k)expected+=(q<Q?float(E((q+1)*.03125f+(k+1)*.015625f)):0)*float(E((k-7)*.125f+(d%11)*.0625f));
    if(std::abs(out[q*D+d]-expected)>1e-5f*(1+std::abs(expected))){std::fprintf(stderr,"PV mismatch D%d Q%d q%d d%d %.8g %.8g\n",D,Q,q,d,out[q*D+d],expected);std::exit(3);}
  }
  Check(cudaFree(out));Check(cudaFree(matched));
}
int main(){Run<64,2>();Run<64,4>();Run<64,8>();Run<128,2>();Run<128,4>();Run<128,8>();std::puts("PV transposed MMA: six position-coded fragment/accumulator cases PASS");}
