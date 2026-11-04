#include <tilemega/Codegen/tasks/Sm90ForwardProgram.cuh>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
using namespace tilemega::backend;
using namespace tilemega::codegen::native_forward;
using BF=cutlass::bfloat16_t;
using Chain=Sm90ForwardEpilogueChain<1,64>;
using Body=Sm90ForwardDepthwise<1,1,64,1,2,Chain,1,1>;
struct Bind {
 static constexpr unsigned kMap=0,kWeight=1,kOutput=2;
 static constexpr unsigned kVector0=kAbsent,kVector1=kAbsent,kVector2=kAbsent,kVector3=kAbsent,kResidual=kAbsent,kStats=kAbsent,kRowStats=kAbsent,kRowPartials=kAbsent,kRowCounters=kAbsent,kChannelPartial=kAbsent;
};
struct Shape {
 static constexpr float kEpsilon=1e-5f;
 static constexpr int kResidualStride=0,kNormWidth=0,kWidth=2,kHeight=2,kChannels=64,kStrideH=1,kOutputWidth=2,kOutputHeight=2,kPaddedChannels=64,kOutputStride=64;
};
using Stage=ForwardDepthwiseStage<Body,Bind,Shape>;
using A=Placement<Stage,0>;using B=Placement<Stage,8192>;
using P=Program<1,1,3,1,8192,A,B,A,B,A,B,A,B>;
void Check(cudaError_t x){if(x!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(x));std::exit(2);}}
int main(){
 if(!std::getenv("TILEMEGA_DEVICE_INDEX")||!std::getenv("CUDA_VISIBLE_DEVICES"))return 3;
 std::vector<BF> input(4*64,BF(1)),weight(64*9,BF(1)),output(4*64);
 BF *di,*dw,*dout;Check(cudaMalloc(&di,input.size()*2));Check(cudaMalloc(&dw,weight.size()*2));Check(cudaMalloc(&dout,output.size()*2));
 Check(cudaMemcpy(di,input.data(),input.size()*2,cudaMemcpyHostToDevice));Check(cudaMemcpy(dw,weight.data(),weight.size()*2,cudaMemcpyHostToDevice));
 BF* host_buffers[]={di,dw,dout};BF**buffers;Check(cudaMalloc(&buffers,sizeof(host_buffers)));Check(cudaMemcpy(buffers,host_buffers,sizeof(host_buffers),cudaMemcpyHostToDevice));
 P::Binding binding{};binding.buffers=buffers;
 cuuint64_t dims[]={64,2,2,1},strides[]={128,256,512};unsigned box[]={64,3,3,1},element_strides[]={1,1,1,1};
 auto status=cuTensorMapEncodeTiled(&binding.maps[0],CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,4,di,dims,strides,box,element_strides,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(status!=CUDA_SUCCESS){std::fprintf(stderr,"tensor map %d\n",int(status));return 2;}
 ForwardCounter* events;Check(cudaMalloc(&events,P::kStages*sizeof(ForwardCounter)));Check(cudaFuncSetAttribute(sm90_forward_native_kernel<P>,cudaFuncAttributeMaxDynamicSharedMemorySize,P::kSharedBytes));
 for(int epoch=0;epoch<10;++epoch){
  Check(cudaMemset(events,0,P::kStages*sizeof(ForwardCounter)));Check(cudaMemset(dout,0,output.size()*2));
  sm90_forward_native_kernel<P><<<4,160,P::kSharedBytes>>>(binding,events);Check(cudaGetLastError());Check(cudaDeviceSynchronize());Check(cudaMemcpy(output.data(),dout,output.size()*2,cudaMemcpyDeviceToHost));
  for(auto x:output)if(float(x)!=4.f){std::fprintf(stderr,"bad output %f\n",float(x));return 1;}
 }
 std::puts("{\"passed\":true,\"stages\":8,\"ctas\":4,\"launches\":10,\"body\":\"real_depthwise\",\"expected_output\":4}");
 Check(cudaFree(events));Check(cudaFree(buffers));Check(cudaFree(di));Check(cudaFree(dw));Check(cudaFree(dout));return 0;
}
