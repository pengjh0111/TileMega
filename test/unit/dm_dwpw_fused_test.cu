// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/tasks/DwPwFusedTaskBody.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace tilemega;
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
template<DmEpilogueKind Kind,DmActivation Activation=DmActivation::kRelu>
using Step=DmEpilogueStep<Kind,Activation,DmGatePair::kSimpleGate,16,
    DmRounding::kFP32,DmRounding::kFP32,DmWriteKind::kDense,1>;
using Bias=Step<DmEpilogueKind::kBias>;
using Relu6=Step<DmEpilogueKind::kActivation,DmActivation::kRelu6>;
using Residual=Step<DmEpilogueKind::kResidual>;
using DwProgram=DmEpilogueProgram<Bias,Relu6>;
using PwSpec=DmEpilogueSpec<DmEpilogueProgram<Bias,Residual,Relu6>>;
template<int C,int M,int N,int K>
using Body=DwPwFusedTaskBody<Arch,C,M,N,K,2,DwProgram,PwSpec>;
template<class T>T* Managed(std::size_t count) {
  T* p=nullptr;assert(cudaMallocManaged(&p,count*sizeof(T))==cudaSuccess);return p;
}
DmBufferLayout Image(unsigned batch,unsigned h,unsigned w,unsigned c,unsigned halo) {
  DmBufferLayout l;l.kind=DmLayout::kNHWC;l.rank=4;
  l.logical[0]=l.physical[0]=batch;l.logical[1]=h;l.logical[2]=w;l.logical[3]=c;
  l.physical[1]=h+2*halo;l.physical[2]=w+2*halo;l.physical[3]=c;
  l.strides[3]=1;l.strides[2]=c;l.strides[1]=l.physical[2]*c;
  l.strides[0]=l.physical[1]*l.strides[1];
  l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=halo;return l;
}
std::uint64_t Offset(DmBufferLayout const& l,unsigned b,unsigned y,unsigned x,unsigned c) {
  return b*l.strides[0]+(y+l.halo_top)*l.strides[1]+(x+l.halo_left)*l.strides[2]+c;
}
template<int C,int M,int N,int K,bool Tiled>
__global__ void Run(DwPwFusedOperands p) {
  extern __shared__ __align__(16) char scratch[];
  Body<C,M,N,K>::template Run<Tiled>(p,blockIdx.x,blockIdx.y,scratch);
}
template<int C,int M,int N,int K>
__global__ void RunPaged(DwPwFusedOperands p) {
  using B=Body<C,M,N,K>;
  constexpr int PageBytes=8192,Pages=2;
  using Paged=PagedGemmTaskBody<Arch,M,N,K,PageBytes,Pages>;
  using Ring=typename Paged::Ring;
  constexpr int Pool=(1024+B::template PagedSharedBytes<PageBytes,Pages>()+1023)/1024*1024;
  extern __shared__ __align__(1024) char shared[];
  Ring ring{reinterpret_cast<typename Ring::Slot*>(shared),shared+Pool};
  ring.Initialize();std::uint64_t sequence=0;
  for(int tm=0;tm<(p.pointwise.m+M-1)/M;++tm)
    for(int tn=0;tn<(p.pointwise.n+N-1)/N;++tn) {
      if(executor::IsCompute()) {
        B::template RunPaged<PageBytes,Pages>(p,tm,tn,ring,sequence,shared+1024);
      }else {
        Paged::Load(p.pointwise,tn,ring,sequence);
      }
    }
}
template<int C,int M,int N,int K>
unsigned Check(unsigned stride,unsigned dilation,unsigned batch,unsigned columns,bool keep_dw) {
  using B=Body<C,M,N,K>;
  using Packing=backend::ServingDmGemm<Arch,M,N,K,2>;
  using Paged=PagedGemmTaskBody<Arch,M,N,K,8192,2>;
  constexpr int Pool=(1024+B::template PagedSharedBytes<8192,2>()+1023)/1024*1024;
  constexpr int PageShared=Pool+2*8192;
  ConvDesc c;c.n=batch;c.h=7;c.w=11;c.c=c.k=C;c.r=c.s=3;
  c.pad_h=c.pad_w=dilation;c.dilation_h=c.dilation_w=dilation;
  c.stride_h=c.stride_w=stride;c.p=(c.h+stride-1)/stride;c.q=(c.w+stride-1)/stride;
  auto in=Image(batch,c.h,c.w,C,dilation),dw_layout=Image(batch,c.p,c.q,C,1);
  unsigned m=batch*c.p*c.q,nt=(columns+N-1)/N,iterations=(C+K-1)/K;
  auto input_count=batch*in.strides[0],dw_count=batch*dw_layout.strides[0];
  auto* input=Managed<E>(input_count);auto* dw_weight=Managed<E>(C*9*8);
  auto* dw_output=Managed<E>(dw_count+32);auto* output=Managed<E>(m*columns+32);
  auto* weight=Managed<E>(columns*C);auto* packed=Managed<E>(nt*iterations*N*K);
  auto* bias=Managed<float>(C+columns);auto* residual=Managed<E>(m*columns);
  auto** data=Managed<void*>(3);auto* dtypes=Managed<unsigned>(3);
  data[0]=bias;data[1]=bias+C;data[2]=residual;dtypes[0]=dtypes[1]=1;dtypes[2]=0;
  for(std::size_t i=0;i<input_count;++i)input[i]=E(0);
  for(unsigned b=0;b<batch;++b)for(unsigned y=0;y<c.h;++y)for(unsigned x=0;x<c.w;++x)
    for(unsigned ch=0;ch<C;++ch)
      input[Offset(in,b,y,x,ch)]=E(float(int((b*31+y*17+x*13+ch*7)%83)-41)*.0625f);
  for(unsigned ch=0;ch<C;++ch) {
    bias[ch]=float(int(ch%17)-8)*.125f;
    for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s)for(unsigned k=0;k<8;++k)
      dw_weight[(ch*9+r*3+s)*8+k]=E(k?0:float(int((ch*11+r*5+s*3)%19)-9)*.03125f);
  }
  for(unsigned n=0;n<columns;++n) {
    bias[C+n]=float(int(n%13)-6)*.03125f;
    for(unsigned k=0;k<C;++k)weight[n*C+k]=E(float(int((n*7+k*13)%19)-9)*.015625f);
  }
  for(unsigned i=0;i<m*columns;++i)residual[i]=E(float(int(i%23)-11)*.03125f);
  for(unsigned tn=0;tn<nt;++tn)for(unsigned it=0;it<iterations;++it)
    for(unsigned n=0;n<N;++n)for(unsigned k=0;k<K;++k)
      packed[(std::size_t(tn)*iterations+it)*N*K+typename Packing::LayoutB{}(n,k)]=
          tn*N+n<columns && it*K+k<C?weight[(tn*N+n)*C+it*K+k]:E(0);
  DwPwFusedOperands p;
  p.depthwise={input,dw_weight,keep_dw?dw_output+16:nullptr,c,in,dw_layout,8,{data,nullptr,dtypes,3}};
  p.depthwise.chain.count=2;
  p.depthwise.chain.operations[0].kind=DmEpilogueKind::kBias;
  p.depthwise.chain.operations[0].parameter[0]=0;
  p.depthwise.chain.operations[1].kind=DmEpilogueKind::kActivation;
  p.depthwise.chain.operations[1].activation=DmActivation::kRelu6;
  for(auto& op:p.depthwise.chain.operations)op.gate=DmGatePair::kSimpleGate;
  auto& pw=p.pointwise;pw.b=weight;pw.weight_base=packed;pw.output=output+16;
  pw.m=m;pw.n=columns;pw.k_total=pw.k_total_full=pw.k_count=C;pw.output_stride=columns;
  pw.dm_buffers={data,nullptr,dtypes,3};pw.chain.count=3;
  pw.chain.operations[0].kind=DmEpilogueKind::kBias;pw.chain.operations[0].parameter[0]=1;
  pw.chain.operations[1].kind=DmEpilogueKind::kResidual;pw.chain.operations[1].parameter[0]=2;
  pw.chain.operations[2].kind=DmEpilogueKind::kActivation;pw.chain.operations[2].activation=DmActivation::kRelu6;
  for(auto& op:pw.chain.operations)op.gate=DmGatePair::kSimpleGate;
  std::vector<E> expected_dw(m*C),expected(m*columns);
  std::vector<bool> dw_written(dw_count);
  for(unsigned row=0;row<m;++row)for(unsigned ch=0;ch<C;++ch) {
    unsigned b=row/(c.p*c.q),y=(row/c.q)%c.p,x=row%c.q;
    double sum=0;
    for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s) {
      long iy=long(y)*stride+r*dilation-dilation,ix=long(x)*stride+s*dilation-dilation;
      if(iy>=0 && iy<long(c.h) && ix>=0 && ix<long(c.w))
        sum+=double(float(input[Offset(in,b,iy,ix,ch)]))*float(dw_weight[(ch*9+r*3+s)*8]);
    }
    expected_dw[row*C+ch]=E(std::max(0.,std::min(6.,sum+bias[ch])));
    dw_written[Offset(dw_layout,b,y,x,ch)]=keep_dw;
  }
  for(unsigned row=0;row<m;++row)for(unsigned n=0;n<columns;++n) {
    double sum=0;for(unsigned k=0;k<C;++k)sum+=double(float(expected_dw[row*C+k]))*float(weight[n*C+k]);
    sum+=bias[C+n];sum+=float(residual[row*columns+n]);
    expected[row*columns+n]=E(std::max(0.,std::min(6.,sum)));
  }
  assert(cudaFuncSetAttribute(Run<C,M,N,K,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,B::kSharedBytes)==cudaSuccess);
  assert(cudaFuncSetAttribute(Run<C,M,N,K,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,B::kSharedBytes)==cudaSuccess);
  assert(cudaFuncSetAttribute(RunPaged<C,M,N,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,PageShared)==cudaSuccess);
  std::vector<E> control;
  for(unsigned epoch=0;epoch<3;++epoch)for(unsigned mode=0;mode<3;++mode) {
    std::fill(output,output+m*columns+32,E(-12345));
    std::fill(dw_output,dw_output+dw_count+32,E(-12345));
    if(mode==0)Run<C,M,N,K,false><<<dim3((m+M-1)/M,nt),128,B::kSharedBytes>>>(p);
    else if(mode==1)Run<C,M,N,K,true><<<dim3((m+M-1)/M,nt),128,B::kSharedBytes>>>(p);
    else RunPaged<C,M,N,K><<<1,160,PageShared>>>(p);
    auto launch=cudaGetLastError(),run=cudaDeviceSynchronize();
    if(launch!=cudaSuccess || run!=cudaSuccess) {
      std::fprintf(stderr,"C=%d tile=%dx%dx%d mode=%u launch=%s run=%s\n",C,M,N,K,mode,cudaGetErrorString(launch),cudaGetErrorString(run));return 0;
    }
    for(unsigned i=0;i<m*columns;++i) {
      float actual=float(output[16+i]),ref=float(expected[i]);
      if(!std::isfinite(actual) || std::abs(actual-ref)>.016f+.016f*std::abs(ref)) {
        std::fprintf(stderr,"C=%d tile=%dx%dx%d stride=%u dilation=%u B=%u N=%u mode=%u index=%u actual=%g ref=%g\n",
            C,M,N,K,stride,dilation,batch,columns,mode,i,actual,ref);return 0;
      }
    }
    if(control.empty())control.assign(output+16,output+16+m*columns);
    else assert(std::memcmp(control.data(),output+16,m*columns*sizeof(E))==0);
    for(unsigned i=0;i<dw_count;++i)if(!dw_written[i])assert(dw_output[16+i]==E(-12345));
    if(keep_dw)for(unsigned row=0;row<m;++row)for(unsigned ch=0;ch<C;++ch) {
      unsigned b=row/(c.p*c.q),y=(row/c.q)%c.p,x=row%c.q;
      assert(dw_output[16+Offset(dw_layout,b,y,x,ch)]==expected_dw[row*C+ch]);
    }
    for(unsigned i=0;i<16;++i)assert(output[i]==E(-12345) && output[m*columns+16+i]==E(-12345) &&
        dw_output[i]==E(-12345) && dw_output[dw_count+16+i]==E(-12345));
  }
  cudaFree(input);cudaFree(dw_weight);cudaFree(dw_output);cudaFree(output);cudaFree(weight);
  cudaFree(packed);cudaFree(bias);cudaFree(residual);cudaFree(data);cudaFree(dtypes);
  return 1;
}
int main() {
  unsigned cases=0;
  for(unsigned stride:{1,2})for(unsigned dilation:{1,2})for(unsigned batch:{1,2})for(bool keep:{false,true}) {
    cases+=Check<24,16,16,16>(stride,dilation,batch,37,keep);
    cases+=Check<144,32,64,32>(stride,dilation,batch,80,keep);
    cases+=Check<1024,16,64,64>(stride,dilation,batch,128,keep);
  }
  assert(cases==48);
  std::printf("DWPW_FUSED cases=%u epochs=3 dense_tiled_paged_shared_A_bias_residual_relu6_optional_DW_halo_canaries PASS\n",cases);
}
