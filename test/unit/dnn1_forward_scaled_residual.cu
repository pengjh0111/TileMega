#include <tilemega/Backend/Sm90ForwardEpilogue.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace tilemega::backend;
using BF=cutlass::bfloat16_t;
using Op=Sm90ForwardEpilogueStep<Sm90ForwardOp::kResidual>;
using Chain=Sm90ForwardEpilogueChain<64,64>;
void Check(cudaError_t x){if(x!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(x));std::exit(2);}}
__global__ void Run(float* output,const float* input,const BF* skip,const float* scale,int rows,int cols,int channel,bool scaled){
 __shared__ float tile[64*64];
 for(int i=threadIdx.x;i<4096;i+=128)tile[i]=input[i];
 __syncthreads();
 Sm90ForwardEpilogueOperands p{};p.residual=skip;p.residual_stride=128;p.vectors[0]=scaled?scale:nullptr;
 int width=cols;Chain::Apply<Op>(tile,rows,channel,width,p);
 __syncthreads();
 for(int i=threadIdx.x;i<4096;i+=128)output[i]=tile[i];
}
int main(){
 if(!std::getenv("TILEMEGA_DEVICE_INDEX")||!std::getenv("CUDA_VISIBLE_DEVICES"))return 3;
 std::vector<float>a(4096),out(4096),scale(128);std::vector<BF>skip(64*128);
 for(int i=0;i<4096;++i)a[i]=float(i%41-20)/32;
 const float factors[]={0,.5f,-1,2};for(int i=0;i<128;++i)scale[i]=factors[i%4];
 for(int i=0;i<64*128;++i)skip[i]=BF(float(i%31-15)/16);
 float *d_a,*d_o,*d_s;BF*d_r;Check(cudaMalloc(&d_a,a.size()*4));Check(cudaMalloc(&d_o,out.size()*4));Check(cudaMalloc(&d_s,scale.size()*4));Check(cudaMalloc(&d_r,skip.size()*2));
 Check(cudaMemcpy(d_a,a.data(),a.size()*4,cudaMemcpyHostToDevice));Check(cudaMemcpy(d_s,scale.data(),scale.size()*4,cudaMemcpyHostToDevice));Check(cudaMemcpy(d_r,skip.data(),skip.size()*2,cudaMemcpyHostToDevice));
 bool all=true;for(int rows:{1,17,64})for(int cols:{19,64})for(int channel:{0,16})for(bool scaled:{false,true}){
 Run<<<1,128>>>(d_o,d_a,d_r,d_s,rows,cols,channel,scaled);Check(cudaGetLastError());Check(cudaDeviceSynchronize());Check(cudaMemcpy(out.data(),d_o,out.size()*4,cudaMemcpyDeviceToHost));bool pass=true;
 for(int r=0;r<64;++r)for(int c=0;c<64;++c){float expected=a[r*64+c];if(r<rows&&c<cols)expected=expected*(scaled?scale[channel+c]:1.f)+float(skip[r*128+channel+c]);pass&=std::memcmp(&expected,&out[r*64+c],4)==0;}
 std::printf("{\"rows\":%d,\"cols\":%d,\"channel\":%d,\"scaled\":%s,\"bitwise_equal\":%s}\n",rows,cols,channel,scaled?"true":"false",pass?"true":"false");all&=pass;
 }
 Check(cudaFree(d_a));Check(cudaFree(d_o));Check(cudaFree(d_s));Check(cudaFree(d_r));return all?0:1;
}
