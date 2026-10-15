// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_SERVING_PHASE 1
#define TILEMEGA_SERVING_SEQ 1
#define TILEMEGA_SOLVED_PAST 0
#define TILEMEGA_ARCH_ID 900
#define TILEMEGA_NONPAGED_LA 1
#define TILEMEGA_CLUSTER_COMBINE 1
#define TILEMEGA_SERVING_HEAD_DIM 64
#define TILEMEGA_SERVING_QPERKV 1
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 2
#define TILEMEGA_GEMM_IMPL_0 2
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
namespace {
using namespace tilemega::codegen;
constexpr unsigned N=321,K=256;
constexpr BufferDesc Buffer(char const* name,unsigned elements) {
  BufferDesc b{};b.name=name;b.constant=elements;b.source=BufferSource::kZero;
  b.role=1;b.external_name=name;return b;
}
constexpr BufferDesc kBuffers[]={Buffer("input",K),Buffer("weight",N*K),Buffer("output",N)};
constexpr GemmDesc kGemms[]={{N,K,0,1,2,2,0.0f}};
constexpr StageDesc Stage() {
  StageDesc s{};s.kind=TaskKind::kGemm;s.group=1;s.handoff_reduce_stage=kHandoffAutoCombine;return s;
}
constexpr StageDesc kStages[]={Stage()};
constexpr GemmRuntimeDesc kGeometry[]={{0,TILEMEGA_SERVING_CLUSTER_DIM,16,64,64,2}};
constexpr std::uint32_t kOffsets[]={0,0};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0}};
constexpr std::int64_t kPlanParams[]={std::int64_t(tilemega::dialect::PlacementTemplate::kClusterSplit)};
constexpr RuntimeVariantDesc Variant() {
  // Split-K producers and their elided reducer own the same output tile.
  // Cluster placement does not replace this last-arriver ownership contract.
  RuntimeVariantDesc v{kGeometry,nullptr,0,kOffsets,kSchedule,1,1,1,1,
      kCombinerTileOwnership};
  v.plan.mode=unsigned(tilemega::dialect::PlacementMode::kTemplate);
  v.plan.params=kPlanParams;v.plan.param_count=1;return v;
}
constexpr RuntimeVariantDesc kVariants[]={Variant()};
constexpr std::uint16_t kSeq[]={0,0};
constexpr ModelSpec kModel={{1,0,1,1,1},ScalarType::kBF16,kBuffers,3,kGemms,1,kStages,1,
    nullptr,0,kVariants,1,kSeq,2};
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
#include <cstdio>
#include <vector>
int main() {
  using E=cutlass::bfloat16_t;
  tm_plan_info info{};if(tm_plan_query(&info)!=0)return 2;
  std::vector<E> input(K),weight(N*K),output(N),expected(N);
  for(unsigned k=0;k<K;++k)input[k]=E((int(k%3)-1)*0.25f);
  for(unsigned n=0;n<N;++n) {
    float sum=0;
    for(unsigned k=0;k<K;++k) {
      weight[n*K+k]=E((int((n+k)%5)-2)*0.5f);
      sum+=float(input[k])*float(weight[n*K+k]);
    }
    expected[n]=E(sum);
  }
  void* external[3]{};unsigned const elements[]={K,N*K,N};
  for(unsigned i=0;i<3;++i)if(cudaMalloc(&external[i],elements[i]*sizeof(E))!=cudaSuccess)return 3;
  cudaMemcpy(external[0],input.data(),K*sizeof(E),cudaMemcpyHostToDevice);
  cudaMemcpy(external[1],weight.data(),N*K*sizeof(E),cudaMemcpyHostToDevice);
  void* plan=tm_plan_create(1,external,0);if(!plan)return 4;
  std::int32_t past[13]={};if(tm_plan_set_steps(plan,past,13)!=0)return 5;
  auto check=[&]() {
    if(cudaDeviceSynchronize()!=cudaSuccess)return false;
    cudaMemcpy(output.data(),external[2],N*sizeof(E),cudaMemcpyDeviceToHost);
    for(unsigned n=0;n<N;++n)if(float(output[n])!=float(expected[n])) {
      std::fprintf(stderr,"cluster combine column%u got%g expected%g\n",n,float(output[n]),float(expected[n]));return false;
    }
    return true;
  };
  for(unsigned mode:{TM_SERVING_L1,TM_SERVING_L2}) {
    for(unsigned step=0;step<13;++step) {
      cudaMemset(external[2],0,N*sizeof(E));
      if(tm_plan_launch(plan,step,mode,step,nullptr)!=0 || !check())return 6;
    }
    if(tm_plan_loop_modes(plan)&mode) {
      cudaMemset(external[2],0,N*sizeof(E));
      if(tm_plan_launch_steps(plan,0,13,mode,13,nullptr)!=0 || !check())return 7;
    }
  }
  tm_plan_destroy(plan);for(auto* pointer:external)cudaFree(pointer);
  std::printf("PASS production cluster combine c%d grid%d L1/L2/device-loop tail N321 splitK native WGMMA\n",
      TILEMEGA_SERVING_CLUSTER_DIM,info.grid);
}
