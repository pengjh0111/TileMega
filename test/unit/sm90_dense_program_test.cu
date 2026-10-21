// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DN_VECTOR_SUMS 1
#define TILEMEGA_SWIGLU_U 8
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
namespace tilemega::codegen::native_dense {
using Op=backend::ServingEpilogueOp;
using Qkv=Gemm<1,4096,1024,32,256,3,4,1,Op::kStore,8,0,1,kAbsent,2,kAbsent,kAbsent,kAbsent>;
using Attn=Attention<1,128,2,true,16,64,3,0,1088,2,4,5,3,6,7,8,9,10,11,12>;
using Out=Gemm<1,1024,2048,8,256,3,1,4,Op::kStore,8,3,14,kAbsent,13,kAbsent,kAbsent,kAbsent>;
#if defined(TILEMEGA_NATIVE_TEST_PREISSUE)
using NativeProgram=Program<1,1088,1,15,Placement<Qkv>,
    Placement<Attn,0,16384,8192>,Placement<Out,0,4096,16384>>;
#else
using NativeProgram=Program<1,1088,1,15,Placement<Qkv>,Placement<Attn>,Placement<Out>>;
#endif
constexpr std::array<tm_buffer_info,15> kNativeBuffers={{{"input",1,0,1024,0,""},{"qkv_weight",1,0,4096*1024,0,""},{"qkv",0,0,4096,0,""},{"context",0,0,2048,0,""},{"key",1,0,8*1088*128,0,""},{"value",1,0,8*1088*128,0,""},{"cos",1,0,1088*128,0,""},{"sin",1,0,1088*128,0,""},{"qnorm",1,0,128,0,""},{"knorm",1,0,128,0,""},{"partial",0,1,8*16*2*128,0,""},{"lse",0,1,8*16*2,0,""},{"arrivals",0,2,16,0,""},{"output",0,0,1024,0,""},{"out_weight",1,0,1024*2048,0,""}}};
constexpr std::array<StageInfo,3> kNativeStages={{{"kGemm","layer.qkv.weight",2*4096*1024,128,0,0,0,0},{"kFusedAttention","",0,128,8,2,128,16},{"kGemm","layer.o_proj.weight",2*1024*2048,128,0,0,0,0}}};
constexpr float kNativeEpsilon=1e-6f;
}
#include <tilemega/Codegen/tasks/Sm90DenseRuntime.cuh>
#include <vector>
#include <algorithm>
using namespace tilemega::codegen::native_dense;
using BF=cutlass::bfloat16_t;
size_t packed(int n,int k,int N,int K,int TN,int TK){int r=n%TN,c=k%TK;size_t s=512*(r/8+(TN/8)*(c/64))+64*(r%8)+8*(((c%64)/8)^(r%8))+c%8;return (size_t(n/TN)*(K/TK)+k/TK)*TN*TK+s;}
BF* copy(std::vector<BF> const& host){BF* p=nullptr;Check(cudaMalloc(&p,host.size()*2));Check(cudaMemcpy(p,host.data(),host.size()*2,cudaMemcpyHostToDevice));return p;}
int main(){try{
 Check(cudaSetDevice(0));std::vector<BF> input(1024,BF(.25f)),qw(4096*1024,BF(0)),ow(1024*2048),ones(1088*128,BF(1)),zeros(8*1088*128,BF(0));
 for(int n=0;n<4096;++n)if(n%512>=384)for(int k=0;k<1024;++k)qw[packed(n,k,4096,1024,32,256)]=BF(float(n/512%4+1)/8/256);
 for(int n=0;n<1024;++n)for(int k=0;k<2048;++k)ow[packed(n,k,1024,2048,8,256)]=BF((n%7-3)/64.f);
 void* external[15]{};external[0]=copy(input);external[1]=copy(qw);external[4]=copy(zeros);external[5]=copy(zeros);external[6]=copy(ones);external[7]=copy(zeros);external[8]=copy(ones);external[9]=copy(ones);external[14]=copy(ow);
#ifndef TILEMEGA_NATIVE_TEST_PAST
#define TILEMEGA_NATIVE_TEST_PAST 64
#endif
 constexpr int initial=TILEMEGA_NATIVE_TEST_PAST;
 int past[]={initial,initial+1,initial+2};std::vector<BF> output[2];int errors=0;
 for(int loop=0;loop<2;++loop){
  Check(cudaMemset(external[4],0,zeros.size()*2));Check(cudaMemset(external[5],0,zeros.size()*2));
  auto* plan=tm_plan_create(1,external,0);if(!plan)throw std::runtime_error("native create rejected");
  if(tm_plan_set_steps(plan,past,3))throw std::runtime_error("native set steps rejected");
  if(loop){if(tm_plan_launch_steps(plan,0,3,1,0,nullptr))throw std::runtime_error("loop rejected");}
  else for(unsigned step=0;step<3;++step)if(tm_plan_launch(plan,step,1,step,nullptr))throw std::runtime_error("step rejected");
  Check(cudaDeviceSynchronize());output[loop].resize(1024);
  if(tm_plan_read_buffer(plan,13,output[loop].data(),2048))throw std::runtime_error("native read rejected");
  float sum=0;for(int bg=0;bg<8;++bg)sum+=256*float(BF(float(bg%4+1)/8*3/(initial+3)));
  for(int n=0;n<1024;++n){BF expected(sum*((n%7-3)/64.f));if(output[loop][n].raw()!=expected.raw()){if(errors<4)std::fprintf(stderr,"loop%d n%d got%f expected%f\n",loop,n,float(output[loop][n]),float(expected));++errors;}}
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP
  auto* directory=std::getenv("TILEMEGA_NATIVE_TEST_TRACE_DIR");
  if(directory && tm_plan_dump_serving_trace(plan,directory))throw std::runtime_error("trace dump rejected");
#endif
  tm_plan_destroy(plan);
 }
 for(int n=0;n<1024;++n)if(output[0][n].raw()!=output[1][n].raw())++errors;
 for(void* p:external)if(p)Check(cudaFree(p));
 std::printf("{\"errors\":%d,\"pass\":%s,\"stages\":3,\"steps\":3,\"step_loop_equal\":%s}\n",errors,errors?"false":"true",output[0]==output[1]?"true":"false");return errors?1:0;
}catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
