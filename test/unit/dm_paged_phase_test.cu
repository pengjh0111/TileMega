// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#ifndef DM_TEST_PHASE
#define DM_TEST_PHASE 2
#endif
#ifndef DM_TEST_ROWS
#define DM_TEST_ROWS 273
#endif
#ifndef DM_TEST_TILE_M
#define DM_TEST_TILE_M 128
#endif
#define TILEMEGA_SERVING_PHASE DM_TEST_PHASE
#if DM_TEST_PHASE == 2
#define TILEMEGA_SERVING_SEQ 1
#define DM_TEST_CAPACITY 0
#define DM_TEST_WORK_BYTES (8 * DM_TEST_TILE_M * 64)
#else
#define TILEMEGA_SERVING_SEQ 64
#define DM_TEST_CAPACITY 256
#define DM_TEST_WORK_BYTES ((8 * DM_TEST_TILE_M * 64 > tilemega::codegen::ServingAttentionSharedBytes(64,64)) ? 8 * DM_TEST_TILE_M * 64 : tilemega::codegen::ServingAttentionSharedBytes(64,64))
#endif
#define TILEMEGA_SERVING_BATCH_LO 1
#define TILEMEGA_SERVING_BATCH_HI 1
#define TILEMEGA_SERVING_PAST_LO 0
#define TILEMEGA_SERVING_PAST_HI 0
#define TILEMEGA_SERVING_QROWS 64
#define TILEMEGA_GEMM_TILE_M DM_TEST_TILE_M
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 2
#define TILEMEGA_PAGED 1
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#define TILEMEGA_PAGE_BYTES 8192
#define TILEMEGA_PAGE_COUNT 2
#define TILEMEGA_PAGE_WORKSPACE_OFFSET 128
#define TILEMEGA_PAGE_POOL_OFFSET (((128 + DM_TEST_WORK_BYTES + 1023) / 1024) * 1024)
#ifndef TILEMEGA_LOOKAHEAD_BYTES
#define TILEMEGA_LOOKAHEAD_BYTES 32768
#endif
#define TILEMEGA_DM_EPILOGUE_DISPATCH 1
#include <tilemega/Codegen/DmDescriptors.h>
namespace tilemega::codegen {
template<class Runner>
__device__ inline void DispatchDmEpilogue(std::uint32_t gemm,Runner const& runner) {
  if(gemm>1){asm volatile("trap;");return;}
  runner.template Run<DmEpilogueSpec<DmEpilogueProgram<>>>();
}
}
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
#include <cassert>
#include <cmath>

namespace {
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
#if DM_TEST_PHASE == 2
constexpr int kRows=DM_TEST_ROWS;
constexpr DmGemmAccess Access() { DmGemmAccess access; access.rows_per_batch=kRows; return access; }
#else
constexpr int kRows=64;
constexpr DmGemmAccess Access() { return {}; }
#endif
constexpr BufferDesc External(char const* name,unsigned count) {
  BufferDesc out{};out.name=name;out.constant=count;out.source=BufferSource::kZero;
  out.role=1;out.external_name=name;return out;
}
constexpr BufferDesc kBuffers[]={External("input",kRows*64), External("weight0",64*64),
  External("weight1",64*64),External("hidden",kRows*64),External("output",kRows*64)};
constexpr GemmDesc Gemm(unsigned a,unsigned b,unsigned d) {
  GemmDesc out{};out.n=out.k=64;out.a=a;out.b=b;out.c=out.d=d;out.access=Access();return out;
}
constexpr GemmDesc kGemms[]={Gemm(0,1,3),Gemm(3,2,4)};
constexpr StageDesc kStages[]={{TaskKind::kGemm,0,0,64,0,{}},{TaskKind::kGemm,1,0,64,0,{}}};
constexpr GemmRuntimeDesc kGeometry[]={{0,1,DM_TEST_TILE_M,64,64,2},{0,1,DM_TEST_TILE_M,64,64,2}};
constexpr StageDependency kDependencies[]={{0,1,StageDependency::Map::kIdentity,1,1,0,1}};
constexpr unsigned kOffsets[]={0,0,1};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0},{1,0,1}};
RuntimeVariantDesc kVariants[]={{kGeometry,kDependencies,1,kOffsets,kSchedule,2,1,
                               TILEMEGA_SERVING_SEQ,TILEMEGA_SERVING_SEQ,0}};
constexpr unsigned short kSeqVariant[TILEMEGA_SERVING_SEQ+1]={};
constexpr OutputDesc kOutputs[]={{4,nullptr}};
constexpr ModelSpec kModel={{0,0,0,1,DM_TEST_CAPACITY},ScalarType::kBF16,kBuffers,5,kGemms,2,
  kStages,2,kOutputs,1,kVariants,1,kSeqVariant,TILEMEGA_SERVING_SEQ+1};
float Weight(int n,int k,int layer) { return ((n*7+k*13+layer*3)%17-8)*0.015625f; }
std::vector<Element> Reference(std::vector<Element> const& input,int layer) {
  std::vector<Element> output(input.size());
  for(int m=0;m<kRows;++m)for(int n=0;n<64;++n) {
    float sum=0;for(int k=0;k<64;++k)sum+=float(input[m*64+k])*float(Element(Weight(n,k,layer)));
    output[m*64+n]=Element(sum);
  }
  return output;
}
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
int main() {
  using Layout=typename PagedGemmTaskBody<paged::PageArch,DM_TEST_TILE_M,64,64,8192,2>::LayoutB;
  std::vector<Element*> data(5);
  for(int i=0;i<5;++i)TILEMEGA_CUDA_CHECK(cudaMallocManaged(&data[i],kBuffers[i].constant*sizeof(Element)));
  for(int layer=0;layer<2;++layer)for(int n=0;n<64;++n)for(int k=0;k<64;++k)
    data[layer+1][Layout{}(n,k)]=Element(Weight(n,k,layer));
  auto target=tilemega::TargetSpec::Probe();
  kVariants[0].plan.eft_grid=std::min(16,target.res.num_sms);
  void* external[5];for(int i=0;i<5;++i)external[i]=data[i];
  auto* opaque=tm_plan_create(1,external,0);assert(opaque);
  int past=0;assert(tm_plan_set_steps(opaque,&past,1)==0);
  tm_plan_info info{};assert(tm_plan_query(&info)==0 && info.phase==DM_TEST_PHASE);
  for(unsigned epoch=0;epoch<16;++epoch) {
    std::vector<Element> input(kRows*64);
    for(int m=0;m<kRows;++m)for(int k=0;k<64;++k)
      data[0][m*64+k]=input[m*64+k]=Element(((m*11+k*5+int(epoch)*7)%23-11)*0.03125f);
    auto expected=Reference(Reference(input,0),1);
    std::vector<Element> first;
    for(int mode:{TM_SERVING_L1,TM_SERVING_L2}) {
      TILEMEGA_CUDA_CHECK(cudaMemset(data[3],0xFF,kRows*64*sizeof(Element)));
      TILEMEGA_CUDA_CHECK(cudaMemset(data[4],0xFF,kRows*64*sizeof(Element)));
      assert(tm_plan_launch(opaque,0,mode,epoch,nullptr)==0);
      TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
      for(int i=0;i<kRows*64;++i) {
        float actual=float(data[4][i]),ref=float(expected[i]);
        if(!std::isfinite(actual) || std::abs(actual-ref)>1.6e-2f+1.6e-2f*std::abs(ref)) {
          std::fprintf(stderr,"phase=%d tile=%d epoch=%u mode=%d element=%d actual=%g ref=%g\n",
            DM_TEST_PHASE,DM_TEST_TILE_M,epoch,mode,i,actual,ref);return 3;
        }
      }
      if(first.empty())first.assign(data[4],data[4]+kRows*64);
      else assert(std::memcmp(first.data(),data[4],first.size()*sizeof(Element))==0);
    }
  }
  tm_plan_destroy(opaque);for(auto ptr:data)TILEMEGA_CUDA_CHECK(cudaFree(ptr));
  std::puts("paged phase: produced activation, weight pages, poisoned intermediates and bitwise L1/L2 passed");
}
