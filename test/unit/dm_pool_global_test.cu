// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/PoolTaskBody.h>
#include <tilemega/Codegen/tasks/GlobalPoolReduceTaskBody.h>
#include <tilemega/Codegen/executor/LastArriver.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace tilemega;
using E=cutlass::bfloat16_t;
using Arch=std::conditional_t<std::is_void_v<arch::CurrentArch>,arch::Sm89,arch::CurrentArch>;
template<class T>T* Managed(std::size_t n) {
  T* value=nullptr;assert(cudaMallocManaged(&value,n*sizeof(T))==cudaSuccess);return value;
}
void Complete() {assert(cudaGetLastError()==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess);}
codegen::DmBufferLayout Image(unsigned b,unsigned h,unsigned w,unsigned c,unsigned halo) {
  codegen::DmBufferLayout l;l.kind=codegen::DmLayout::kNHWC;l.rank=4;
  l.logical[0]=l.physical[0]=b;l.logical[1]=h;l.logical[2]=w;l.logical[3]=c;
  l.physical[1]=h+2*halo;l.physical[2]=w+2*halo;l.physical[3]=(c+7)/8*8;
  l.strides[3]=1;l.strides[2]=l.physical[3];l.strides[1]=l.physical[2]*l.strides[2];
  l.strides[0]=l.physical[1]*l.strides[1];
  l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=halo;return l;
}
std::uint64_t Offset(codegen::DmBufferLayout const& l,unsigned b,unsigned h,unsigned w,unsigned c) {
  return b*l.strides[0]+(h+l.halo_top)*l.strides[1]+(w+l.halo_left)*l.strides[2]+c;
}
__host__ __device__ float Value(unsigned image,unsigned row,unsigned channel) {
  return float(int((image*19+row*13+channel*7)%71)-35)*.03125f;
}
template<int Rows,int Channels>
__global__ void Pool(codegen::PoolOperands p) {
  codegen::PoolTaskBody<Arch,Rows,Channels>::Run(p,blockIdx.x);
}
template<int Rows,int Channels>unsigned CheckPool() {
  unsigned cases=0;
  for(unsigned batch:{1,2})for(unsigned channels:{19,64,144})
    for(unsigned stride:{1,2})for(unsigned dilation:{1,2}) {
      codegen::ConvDesc w;w.n=batch;w.h=7;w.w=11;w.c=w.k=channels;
      w.r=w.s=3;w.pad_h=w.pad_w=dilation;w.dilation_h=w.dilation_w=dilation;
      w.stride_h=w.stride_w=stride;w.p=(w.h+stride-1)/stride;w.q=(w.w+stride-1)/stride;
      auto in=Image(batch,w.h,w.w,channels,2),out=Image(batch,w.p,w.q,channels,1);
      auto input_size=batch*in.strides[0],output_size=batch*out.strides[0];
      auto* input=Managed<E>(input_size);auto* output=Managed<E>(output_size+32);
      for(unsigned i=0;i<input_size;++i)input[i]=E(900);
      for(unsigned i=0;i<output_size+32;++i)output[i]=E(-12345);
      for(unsigned n=0;n<batch;++n)for(unsigned h=0;h<w.h;++h)
        for(unsigned x=0;x<w.w;++x)for(unsigned c=0;c<channels;++c)
          input[Offset(in,n,h,x,c)]=E(-1.0f-std::abs(Value(n,h*w.w+x,c)));
      codegen::PoolOperands p{input,output+16,w,in,out};
      Pool<Rows,Channels><<<codegen::PoolTaskBody<Arch,Rows,Channels>::Count(p),128>>>(p);Complete();
      std::vector<bool> written(output_size);
      for(unsigned n=0;n<batch;++n)for(unsigned h=0;h<w.p;++h)
        for(unsigned x=0;x<w.q;++x)for(unsigned c=0;c<channels;++c) {
          float expected=-INFINITY;
          for(unsigned r=0;r<w.r;++r)for(unsigned s=0;s<w.s;++s) {
            int y=int(h*stride+r*dilation)-int(dilation);
            int xx=int(x*stride+s*dilation)-int(dilation);
            if(y>=0 && y<int(w.h) && xx>=0 && xx<int(w.w))
              expected=std::max(expected,float(input[Offset(in,n,y,xx,c)]));
          }
          auto at=Offset(out,n,h,x,c);assert(float(output[16+at])==expected);written[at]=true;
        }
      for(unsigned i=0;i<output_size;++i)if(!written[i])assert(output[16+i]==E(-12345));
      for(unsigned i=0;i<16;++i)assert(output[i]==E(-12345) && output[16+output_size+i]==E(-12345));
      cudaFree(input);cudaFree(output);++cases;
    }
  return cases;
}
template<int Channels>
__global__ void Reduce(codegen::GlobalPoolReduceOperands p) {
  codegen::GlobalPoolReduceTaskBody<Arch,Channels>::Run(p,blockIdx.x);
}
template<int Channels>
__global__ void ProduceAndArrive(codegen::GlobalPoolReduceOperands p,float* partials,unsigned* tickets) {
  using namespace codegen::executor;
  unsigned blocks=(p.channels+Channels-1)/Channels;
  unsigned tile=blockIdx.x/blocks,block=blockIdx.x%blocks;
  unsigned first_row=tile*p.producer_tile_rows;
  unsigned end_row=min((tile+1)*p.producer_tile_rows,p.images*p.image_rows);
  unsigned first_image=first_row/p.image_rows,last_image=(end_row-1)/p.image_rows;
  __shared__ unsigned last;
  for(unsigned image=first_image;image<=last_image;++image) {
    for(unsigned c=ComputeThread();c<Channels;c+=128)if(block*Channels+c<p.channels) {
      unsigned channel=block*Channels+c;
      float sum=0;
      for(unsigned row=max(first_row,image*p.image_rows);row<min(end_row,(image+1)*p.image_rows);++row)
        sum+=Value(image,row-image*p.image_rows,channel);
      auto const& l=p.partial_layout;
      partials[image*l.strides[0]+tile*l.strides[1]+channel*l.strides[2]]=sum;
    }
    unsigned first=image*p.image_rows/p.producer_tile_rows;
    unsigned end=((image+1)*p.image_rows+p.producer_tile_rows-1)/p.producer_tile_rows;
    auto id=image*blocks+block;
    LastArriver::Run(tickets+id,end-first,&last,[&] {
      codegen::GlobalPoolReduceTaskBody<Arch,Channels>::Run(p,id);
    });
  }
}
template<int Channels>unsigned CheckReduce() {
  unsigned cases=0;
  for(unsigned batch:{1,2,5})for(unsigned area:{1,4,49,77})
    for(unsigned tile:{16,64,128})for(unsigned channels:{19,64,144}) {
      unsigned tiles=(batch*area+tile-1)/tile;
      codegen::DmBufferLayout l;l.rank=3;
      l.logical[0]=l.physical[0]=batch;l.logical[1]=l.physical[1]=tiles;
      l.logical[2]=l.physical[2]=channels;l.strides[2]=1;
      l.strides[1]=channels+3;l.strides[0]=tiles*l.strides[1];
      unsigned output_stride=channels+5,size=batch*l.strides[0],out_size=batch*output_stride;
      auto* partial=Managed<float>(size+8);auto* output=Managed<float>(out_size+8);
      auto* reference=Managed<float>(out_size+8);
      unsigned tasks=batch*((channels+Channels-1)/Channels);
      auto* tickets=Managed<unsigned>(tasks);
      for(unsigned i=0;i<size+8;++i)partial[i]=NAN;
      for(unsigned image=0;image<batch;++image) {
        unsigned first=image*area/tile,end=((image+1)*area+tile-1)/tile;
        for(unsigned part=first;part<end;++part)for(unsigned c=0;c<channels;++c) {
          float sum=0;
          for(unsigned row=std::max(part*tile,image*area);row<std::min((part+1)*tile,(image+1)*area);++row)
            sum+=Value(image,row-image*area,c);
          partial[4+image*l.strides[0]+part*l.strides[1]+c]=sum;
        }
      }
      for(unsigned i=0;i<out_size+8;++i)reference[i]=output[i]=-12345;
      codegen::GlobalPoolReduceOperands p{partial+4,reference+4,l,batch,channels,area,tile,output_stride};
      Reduce<Channels><<<tasks,128>>>(p);Complete();
      for(unsigned epoch=0;epoch<3;++epoch) {
        for(unsigned i=0;i<tasks;++i)tickets[i]=0;
        for(unsigned i=0;i<size+8;++i)partial[i]=NAN;
        for(unsigned i=0;i<out_size+8;++i)output[i]=-12345;
        p.output=output+4;
        ProduceAndArrive<Channels><<<tiles*((channels+Channels-1)/Channels),128>>>(p,partial+4,tickets);
        Complete();
        for(unsigned image=0;image<batch;++image)for(unsigned c=0;c<channels;++c) {
          double sum=0;for(unsigned row=0;row<area;++row)sum+=Value(image,row,c);
          unsigned at=4+image*output_stride+c;
          assert(std::isfinite(output[at]) && std::abs(output[at]-sum/area)<1e-6);
          assert(output[at]==reference[at]);
        }
        for(unsigned i=0;i<tasks;++i)assert(tickets[i]==0);
        for(unsigned i=0;i<out_size+8;++i) {
          bool interior=i>=4 && i<4+out_size && (i-4)%output_stride<channels;
          if(!interior)assert(output[i]==-12345 && reference[i]==-12345);
        }
      }
      cudaFree(partial);cudaFree(output);cudaFree(reference);cudaFree(tickets);++cases;
    }
  return cases;
}
int main() {
  unsigned pool=CheckPool<4,32>()+CheckPool<17,256>();
  unsigned reduce=CheckReduce<32>()+CheckReduce<128>()+CheckReduce<256>();
  std::printf("POOL_GLOBAL window_cases=%u reduce_cases=%u epochs=3 halo_padding_LA_fixed_order PASS\n",pool,reduce);
}
