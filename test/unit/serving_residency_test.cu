// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_SERVING_PHASE 1
#define TILEMEGA_SERVING_SEQ 1
#define TILEMEGA_SOLVED_PAST 0
#define TILEMEGA_ARCH_ID 900
#define TILEMEGA_SERVING_RESIDENCY_REQUEST 2
#define TILEMEGA_SERVING_HEAD_DIM 64
#define TILEMEGA_SERVING_QPERKV 1
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 2
#if defined(TILEMEGA_PAGED) && TILEMEGA_PAGED
#define TILEMEGA_PAGE_BYTES 16384
#define TILEMEGA_PAGE_COUNT 6
#define TILEMEGA_PAGE_WORKSPACE_OFFSET 256
#define TILEMEGA_PAGE_POOL_OFFSET 9216
#define TILEMEGA_LOOKAHEAD_BYTES 0
#endif
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
namespace {
using namespace tilemega::codegen;
constexpr BufferDesc Buffer(char const* name,unsigned elements,bool readonly) {
  BufferDesc value{};value.name=name;value.constant=elements;
  value.source=BufferSource::kZero;value.role=1;value.external_name=name;
#if TILEMEGA_PREFETCH_RUNTIME
  value.no_producer=readonly;
#endif
  return value;
}
constexpr BufferDesc kBuffers[]={Buffer("input",64,false),Buffer("weight",4096,true),
    Buffer("output",64,false)};
constexpr GemmDesc kGemms[]={{64,64,0,1,2,2,0.0f}};
constexpr StageDesc kStages[]={{TaskKind::kGemm,0,0,0,1,{}}};
constexpr GemmRuntimeDesc kGeometry[]={{0,1,16,64,64,2}};
constexpr std::uint32_t kOffsets[]={0,0};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0}};
constexpr RuntimeVariantDesc kVariants[]={{kGeometry,nullptr,0,kOffsets,kSchedule,1,1,1,1,0}};
constexpr std::uint16_t kSeq[]={0,0};
constexpr ModelSpec kModel={{1,0,1,1,1},ScalarType::kBF16,kBuffers,3,kGemms,1,kStages,1,
    nullptr,0,kVariants,1,kSeq,2};
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
#include <cstdio>
#include <vector>
int main() {
  using E=cutlass::bfloat16_t;
  tm_plan_info info{};
  if(tm_plan_query(&info)!=0 || info.residency!=2)return 2;
  std::vector<E> input(64),weight(4096),output(64),expected(64);
  for(unsigned k=0;k<64;++k)input[k]=E((int(k%3)-1)*0.25f);
  for(unsigned n=0;n<64;++n) {
    float sum=0;
    for(unsigned k=0;k<64;++k) {
      weight[n*64+k]=E((int((n+k)%5)-2)*0.5f);
      sum+=float(input[k])*float(weight[n*64+k]);
    }
    expected[n]=E(sum);
  }
  void* external[3]{};
  for(unsigned i=0;i<3;++i)
    if(cudaMalloc(&external[i],(i==1?4096:64)*sizeof(E))!=cudaSuccess)return 3;
  cudaMemcpy(external[0],input.data(),128,cudaMemcpyHostToDevice);
  cudaMemcpy(external[1],weight.data(),8192,cudaMemcpyHostToDevice);
  void* plan=tm_plan_create(1,external,0);
  if(!plan) {
    int l1=0,l2=0,loop=2;
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&l1,tilemega_l1_kernel,kServingThreads,kServingSharedBytes)!=cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&l2,tilemega_l2_kernel,kServingThreads,kServingSharedBytes)!=cudaSuccess)return 4;
#if TILEMEGA_PAGED
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&loop,tilemega_loop_kernel,kServingThreads,kServingSharedBytes)!=cudaSuccess)return 4;
#endif
    for(auto* ptr:external)cudaFree(ptr);
    if(std::min({l1,l2,loop})<2) {
      std::printf("PASS serving runtime correctly rejects unsupported residency2: paged=%d l1=%d l2=%d loop=%d; numeric execution NOT RUN\n",
          TILEMEGA_PAGED,l1,l2,loop);return 0;
    }
    return 4;
  }
  std::int32_t past[4]={0,0,0,0};
  if(tm_plan_set_steps(plan,past,4)!=0)return 5;
  auto check=[&]() {
    if(cudaDeviceSynchronize()!=cudaSuccess)return false;
    cudaMemcpy(output.data(),external[2],128,cudaMemcpyDeviceToHost);
    for(unsigned n=0;n<64;++n)if(float(output[n])!=float(expected[n])) {
      std::fprintf(stderr,"residency2 column%u got%g expected%g\n",n,float(output[n]),float(expected[n]));return false;
    }
    return true;
  };
  for(unsigned mode:{TM_SERVING_L1,TM_SERVING_L2}) {
    for(unsigned step=0;step<4;++step) {
      cudaMemset(external[2],0,128);
      if(tm_plan_launch(plan,step,mode,step,nullptr)!=0 || !check())return 6;
    }
    if(tm_plan_loop_modes(plan)&mode) {
      cudaMemset(external[2],0,128);
      if(tm_plan_launch_steps(plan,0,4,mode,4,nullptr)!=0 || !check())return 7;
    }
  }
  tm_plan_destroy(plan);
  for(auto* ptr:external)cudaFree(ptr);
  std::printf("PASS real serving runtime residency=2 grid=%d paged=%d L1/L2/available-device-loop exact GEMM\n",info.grid,TILEMEGA_PAGED);
}
