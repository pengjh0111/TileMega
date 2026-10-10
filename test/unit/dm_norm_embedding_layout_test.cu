// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/LayerNormTaskBody.h>
#include <tilemega/Codegen/tasks/EmbeddingSumTaskBody.h>
#include <tilemega/Codegen/tasks/LayoutConvertTaskBody.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace tilemega;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
template<class T>T* Managed(std::size_t count) {
  T* p=nullptr;assert(cudaMallocManaged(&p,count*sizeof(T))==cudaSuccess);return p;
}
template<int Width>
__global__ void Norm(codegen::LayerNormOperands p) {
  codegen::LayerNormTaskBody<Arch,Width,8>::Run(p,blockIdx.x);
}
template<int Width>
__global__ void Embedding(codegen::EmbeddingSumOperands p) {
  __shared__ float scratch[8];
  for(auto row=std::uint64_t(blockIdx.x);row<p.rows;row+=gridDim.x)
    codegen::EmbeddingSumTaskBody<Arch,Width>::Run(p,row,scratch);
}
__global__ void Convert(codegen::LayoutConvertOperands p) {
  codegen::LayoutConvertTaskBody<Arch,16>::Run(p,blockIdx.x);
}
void Complete() {assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);}
void Close(double actual,double expected,double absolute,double relative) {
  if(!std::isfinite(actual) || std::abs(actual-expected)>absolute+relative*std::abs(expected)) {
    std::fprintf(stderr,"actual=%.12g expected=%.12g absolute=%.12g relative=%.12g\n",
        actual,expected,absolute,relative);assert(false);
  }
}
template<int Width>unsigned CheckNormEmbedding() {
  unsigned cases=0;
  for(unsigned rows:{1,6,17,262})for(bool image:{false,true}) {
    codegen::DmBufferLayout l;
    unsigned size=rows*Width;
    if(image) {
      l.kind=codegen::DmLayout::kNHWC;l.rank=4;
      l.logical[0]=l.physical[0]=rows%2?1:2;l.logical[1]=1;
      l.logical[2]=rows/l.logical[0];l.logical[3]=Width;
      l.halo_top=l.halo_bottom=1;l.halo_left=l.halo_right=2;
      l.physical[1]=3;l.physical[2]=l.logical[2]+4;l.physical[3]=(Width+7)/8*8;
      l.strides[3]=1;l.strides[2]=l.physical[3];
      l.strides[1]=l.physical[2]*l.strides[2];l.strides[0]=3*l.strides[1];
      size=l.physical[0]*l.strides[0];
    }
    auto offset=[&](unsigned row,unsigned c) {
      if(!image)return std::uint64_t(row)*Width+c;
      auto n=row/l.logical[2],w=row%l.logical[2];
      return n*l.strides[0]+l.strides[1]+(w+2)*l.strides[2]+c;
    };
    auto* input=Managed<E>(size);auto* output=Managed<E>(size+32);
    auto* gamma=Managed<E>(Width);auto* beta=Managed<E>(Width);
    auto* stats=Managed<float>(2*rows+8);
    for(unsigned i=0;i<size;++i)input[i]=E(-12345);
    for(unsigned c=0;c<Width;++c) {
      gamma[c]=E(0.9f+(int(c%7)-3)*0.125f);beta[c]=E((int(c%13)-6)*0.03125f);
    }
    for(unsigned row=0;row<rows;++row)for(unsigned c=0;c<Width;++c)
      input[offset(row,c)]=E(row?(int((row*17+c*13)%83)-41)*0.0625f:1.25f);
    for(float eps:{1e-12f,1e-6f}) {
      for(unsigned i=0;i<size+32;++i)output[i]=E(-12345);
      for(unsigned i=0;i<2*rows+8;++i)stats[i]=-12345;
      codegen::LayerNormOperands p{input,gamma,beta,output+16,l,l,rows,eps,stats+4};
      Norm<Width><<<(rows+7)/8,128>>>(p);Complete();
      std::vector<bool> written(size);
      for(unsigned row=0;row<rows;++row) {
        double mean=0,variance=0;
        for(unsigned c=0;c<Width;++c)mean+=float(input[offset(row,c)]);
        mean/=Width;
        for(unsigned c=0;c<Width;++c) {auto x=float(input[offset(row,c)])-mean;variance+=x*x;}
        double sum=0,square=0;
        for(unsigned c=0;c<Width;++c) {
          double expected=(float(input[offset(row,c)])-mean)/std::sqrt(variance/Width+eps)*
              float(gamma[c])+float(beta[c]);
          auto value=float(output[16+offset(row,c)]);
          Close(value,expected,0.016,0.016);sum+=value;square+=double(value)*value;
          written[offset(row,c)]=true;
        }
        Close(stats[4+2*row],sum,5e-5,1e-5);Close(stats[5+2*row],square,5e-5,1e-5);
      }
      for(unsigned i=0;i<size;++i)if(!written[i])assert(output[16+i]==E(-12345));
      for(unsigned i=0;i<16;++i)assert(output[i]==E(-12345) && output[16+size+i]==E(-12345));
      for(unsigned i=0;i<4;++i)assert(stats[i]==-12345 && stats[4+2*rows+i]==-12345);
      ++cases;
    }
    cudaFree(input);cudaFree(output);cudaFree(gamma);cudaFree(beta);cudaFree(stats);
  }
  for(unsigned rows:{1,6,17,262})for(bool statistics:{false,true}) {
    auto* ids=Managed<std::int64_t>(rows);auto* types=Managed<std::int64_t>(rows);
    auto* word=Managed<E>(29*Width);auto* type=Managed<E>(2*Width);
    auto* pos=Managed<E>(32*Width);auto* out=Managed<E>(rows*Width+32);
    auto* stats=Managed<float>(rows*2+8);
    for(unsigned row=0;row<rows;++row) {ids[row]=(row*7)%29;types[row]=row%2;}
    for(unsigned i=0;i<29*Width;++i)word[i]=E((int((i*19)%47)-23)*0.09375f);
    for(unsigned i=0;i<2*Width;++i)type[i]=E((int((i*11)%37)-18)*0.03125f);
    for(unsigned i=0;i<32*Width;++i)pos[i]=E((int((i*13)%31)-15)*0.046875f);
    for(unsigned i=0;i<rows*Width+32;++i)out[i]=E(-12345);
    for(unsigned i=0;i<rows*2+8;++i)stats[i]=-12345;
    codegen::EmbeddingSumOperands p{ids,types,word,type,pos,out+16,statistics?stats+4:nullptr,
        rows,11,29,2,32};
    Embedding<Width><<<rows<3?rows:3,128>>>(p);Complete();
    for(unsigned row=0;row<rows;++row) {
      double sum=0,square=0;
      for(unsigned c=0;c<Width;++c) {
        E first(float(word[ids[row]*Width+c])+float(type[types[row]*Width+c]));
        E expected(float(first)+float(pos[(row%11)*Width+c]));
        assert(out[16+row*Width+c]==expected);sum+=float(expected);square+=double(float(expected))*float(expected);
      }
      if(statistics) {Close(stats[4+row*2],sum,5e-5,1e-5);Close(stats[5+row*2],square,5e-5,1e-5);}
    }
    for(unsigned i=0;i<16;++i)assert(out[i]==E(-12345) && out[16+rows*Width+i]==E(-12345));
    for(unsigned i=0;i<4;++i)assert(stats[i]==-12345 && stats[4+rows*2+i]==-12345);
    ++cases;cudaFree(ids);cudaFree(types);cudaFree(word);cudaFree(type);cudaFree(pos);cudaFree(out);cudaFree(stats);
  }
  return cases;
}
int main() {
  unsigned cases=CheckNormEmbedding<32>()+CheckNormEmbedding<33>()+CheckNormEmbedding<64>()+
      CheckNormEmbedding<128>()+CheckNormEmbedding<256>()+CheckNormEmbedding<512>()+
      CheckNormEmbedding<768>()+CheckNormEmbedding<1024>();
  for(unsigned batch:{1,2})for(unsigned c:{3,4,16,24})for(unsigned cp:{4,8,16,24}) {
    if(cp<c || (c<=4 && cp>8) || (c>4 && cp!=(c+7)/8*8))continue;
    codegen::DmBufferLayout l;l.kind=codegen::DmLayout::kNHWC;l.rank=4;
    l.logical[0]=l.physical[0]=batch;l.logical[1]=5;l.logical[2]=7;l.logical[3]=c;
    l.physical[1]=9;l.physical[2]=11;l.physical[3]=cp;
    l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=2;
    l.strides[3]=1;l.strides[2]=(cp+7)/8*8;l.strides[1]=11*l.strides[2];l.strides[0]=9*l.strides[1];
    auto count=batch*l.strides[0];auto* input=Managed<E>(batch*c*35);
    auto* out=Managed<E>(count+32);
    for(unsigned i=0;i<batch*c*35;++i)input[i]=E((int(i%67)-33)*0.03125f);
    for(auto fill:{codegen::DmFill::kZero,codegen::DmFill::kNegativeInfinity}) {
      E border(fill==codegen::DmFill::kZero?0.f:-INFINITY);l.fill=fill;
      for(unsigned i=0;i<count+32;++i)out[i]=i<16 || i>=count+16?E(-12345):border;
      Convert<<<(batch*35+15)/16,128>>>({input,out+16,l});Complete();
      for(unsigned n=0;n<batch;++n)for(unsigned h=0;h<9;++h)for(unsigned w=0;w<11;++w)
        for(unsigned k=0;k<l.strides[2];++k) {
          E expected=border;
          if(h>=2 && h<7 && w>=2 && w<9)
            expected=k<c?input[(n*c+k)*35+(h-2)*7+w-2]:E(0);
          assert(out[16+n*l.strides[0]+h*l.strides[1]+w*l.strides[2]+k]==expected);
        }
      for(unsigned i=0;i<16;++i)assert(out[i]==E(-12345) && out[count+16+i]==E(-12345));
      ++cases;
    }
    cudaFree(input);cudaFree(out);
  }
  std::printf("NORM_EMBEDDING_LAYOUT cases=%u FP32_LN_BF16_add_order_stats_halo_canaries PASS\n",cases);
}
