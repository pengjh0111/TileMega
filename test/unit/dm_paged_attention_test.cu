// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_SERVING_SEQ 64
#define TILEMEGA_SERVING_BATCH_LO 1
#define TILEMEGA_SERVING_BATCH_HI 1
#define TILEMEGA_SERVING_PAST_LO 0
#define TILEMEGA_SERVING_PAST_HI 0
#define TILEMEGA_SERVING_HEAD_DIM 64
#define TILEMEGA_SERVING_QPERKV 2
#define TILEMEGA_SERVING_QROWS 64
#define TILEMEGA_SERVING_KV_TILE 64
#define TILEMEGA_GEMM_TILE_M 64
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 2
#define TILEMEGA_PAGED 1
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#define TILEMEGA_PAGE_BYTES 8192
#define TILEMEGA_PAGE_COUNT 2
#define TILEMEGA_PAGE_WORKSPACE_OFFSET 128
#define TILEMEGA_PAGE_POOL_OFFSET (((128 + tilemega::codegen::ServingAttentionSharedBytes(64,64) + 1023) / 1024) * 1024)
#define TILEMEGA_LOOKAHEAD_BYTES 32768
#define TILEMEGA_DM_EPILOGUE_DISPATCH 1
#include <tilemega/Codegen/DmDescriptors.h>
namespace tilemega::codegen {
template<class Runner>
__device__ inline void DispatchDmEpilogue(std::uint32_t id,Runner const& runner) {
  if(id>1){asm volatile("trap;");return;}
  runner.template Run<DmEpilogueSpec<DmEpilogueProgram<>>>();
}
}
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
#include <cassert>
#include <cmath>

namespace {
using namespace tilemega::codegen;
using Element=cutlass::bfloat16_t;
constexpr BufferDesc Buffer(char const* name,unsigned count,unsigned dtype=0) {
  BufferDesc out{};out.name=name;out.constant=count;out.source=BufferSource::kZero;
  out.dtype=dtype;out.role=1;out.external_name=name;return out;
}
constexpr BufferDesc kBuffers[]={Buffer("input",64*64),Buffer("weight",256*64),
  Buffer("qkv",64*256),Buffer("key",256*64),Buffer("value",256*64),
  Buffer("cos",256*64),Buffer("sin",256*64),Buffer("context",64*128),
  Buffer("partial",128*64,1),Buffer("lse",128,1),
  Buffer("out_weight",64*128),Buffer("output",64*64)};
constexpr GemmDesc kGemms[]={{256,64,0,1,2,2,0},{64,128,7,10,11,11,0}};
constexpr StageDesc Fused() {
  StageDesc out{};out.kind=TaskKind::kFusedAttention;out.extent=1;out.width=64;out.group=2;
  unsigned operands[]={2,3,4,5,6,kNoOperand,kNoOperand,7,8,9};
  for(unsigned i=0;i<10;++i)out.operand[i]=operands[i];
  out.attention_kv_block=256;out.attention_query_rows=64;
  return out;
}
constexpr StageDesc kStages[]={{TaskKind::kGemm,0,0,256,0,{}},Fused(),
  {TaskKind::kGemm,1,0,64,0,{}}};
constexpr GemmRuntimeDesc kGeometry[]={{0,1,64,64,64,2},{0,1,64,64,64,2}};
constexpr StageDependency kDependencies[]={{0,1,StageDependency::Map::kAll,1,0,0,4},
  {1,2,StageDependency::Map::kTable,1,0,0,1,0,0,1,1,0}};
constexpr RuntimeDependencyInterval kIntervals[]={{0,2}};
constexpr unsigned kOffsets[]={0,0,1,2};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0},{1,0,1},{2,1,1}};
RuntimeVariantDesc kVariants[]={{kGeometry,kDependencies,2,kOffsets,kSchedule,3,1,64,64,0,
  nullptr,true,false,nullptr,{},kIntervals,1}};
constexpr unsigned short kSeqVariant[65]={};
constexpr OutputDesc kOutputs[]={{11,nullptr}};
constexpr ModelSpec kModel={{0,0,0,1,256},ScalarType::kBF16,kBuffers,12,kGemms,2,kStages,3,
  kOutputs,1,kVariants,1,kSeqVariant,65};
float Weight(int n,int k) { return ((n*7+k*13)%17-8)*0.015625f; }
std::vector<Element> Reference(std::vector<Element> const& input) {
  std::vector<Element> qkv(64*256),context(64*128);
  for(int m=0;m<64;++m)for(int n=0;n<256;++n) {
    float sum=0;for(int k=0;k<64;++k)sum+=float(input[m*64+k])*float(Element(Weight(n,k)));
    qkv[m*256+n]=Element(sum);
  }
  for(int token=0;token<64;++token)for(int head=0;head<2;++head) {
    float logits[64],maximum=-INFINITY,total=0;
    for(int key=0;key<=token;++key) {
      float dot=0;for(int d=0;d<64;++d)dot+=float(qkv[token*256+head*64+d])*float(qkv[key*256+128+d]);
      logits[key]=dot*0.125f;maximum=std::max(maximum,logits[key]);
    }
    for(int key=0;key<=token;++key)total+=std::exp(logits[key]-maximum);
    for(int d=0;d<64;++d) {
      float sum=0;for(int key=0;key<=token;++key)
        sum+=std::exp(logits[key]-maximum)/total*float(qkv[key*256+192+d]);
      context[token*128+head*64+d]=Element(sum);
    }
  }
  return context;
}
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
int main() {
  using Layout=typename PagedGemmTaskBody<paged::PageArch,64,64,64,8192,2>::LayoutB;
  std::vector<void*> data(12);
  for(int i=0;i<12;++i)TILEMEGA_CUDA_CHECK(cudaMallocManaged(&data[i],kBuffers[i].constant*(kBuffers[i].dtype?4:2)));
  auto ptr=[&](int i){return static_cast<Element*>(data[i]);};
  for(int n=0;n<256;++n)for(int k=0;k<64;++k)ptr(1)[(n/64)*4096+Layout{}(n%64,k)]=Element(Weight(n,k));
  for(int n=0;n<64;++n)for(int k=0;k<128;++k)ptr(10)[(k/64)*4096+Layout{}(n,k%64)]=Element(Weight(n+3,k));
  for(int i=0;i<256*64;++i){ptr(5)[i]=Element(1);ptr(6)[i]=Element(0);}
  auto target=tilemega::TargetSpec::Probe();kVariants[0].plan.eft_grid=std::min(8,target.res.num_sms);
  auto* opaque=tm_plan_create(1,data.data(),0);assert(opaque);
  int past=0;assert(tm_plan_set_steps(opaque,&past,1)==0);
  for(unsigned epoch=0;epoch<16;++epoch) {
    std::vector<Element> input(64*64);
    for(int m=0;m<64;++m)for(int k=0;k<64;++k)
      ptr(0)[m*64+k]=input[m*64+k]=Element(((m*11+k*5+int(epoch)*7)%23-11)*0.03125f);
    auto expected=Reference(input);std::vector<Element> projected(64*64);
    for(int m=0;m<64;++m)for(int n=0;n<64;++n) {
      float sum=0;for(int k=0;k<128;++k)sum+=float(expected[m*128+k])*float(Element(Weight(n+3,k)));
      projected[m*64+n]=Element(sum);
    }
    std::vector<Element> first;
    for(int mode:{TM_SERVING_L1,TM_SERVING_L2}) {
      for(int i:{2,3,4,7,8,9,11})TILEMEGA_CUDA_CHECK(cudaMemset(data[i],0xFF,kBuffers[i].constant*(kBuffers[i].dtype?4:2)));
      assert(tm_plan_launch(opaque,0,mode,epoch,nullptr)==0);TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
      for(int i=0;i<64*128;++i) {
        float actual=float(ptr(7)[i]),ref=float(expected[i]);
        if(!std::isfinite(actual)||std::abs(actual-ref)>1.6e-2f+1.6e-2f*std::abs(ref)) {
          std::fprintf(stderr,"epoch=%u mode=%d context=%d actual=%g ref=%g\n",epoch,mode,i,actual,ref);return 3;
        }
      }
      for(int i=0;i<64*64;++i) {
        float actual=float(ptr(11)[i]),ref=float(projected[i]);
        if(!std::isfinite(actual)||std::abs(actual-ref)>1.6e-2f+1.6e-2f*std::abs(ref)) {
          std::fprintf(stderr,"epoch=%u mode=%d output=%d actual=%g ref=%g\n",epoch,mode,i,actual,ref);return 4;
        }
      }
      if(first.empty()) {
        first.assign(ptr(7),ptr(7)+64*128);
        first.insert(first.end(),ptr(11),ptr(11)+64*64);
      } else {
        assert(std::memcmp(first.data(),ptr(7),64*128*sizeof(Element))==0);
        assert(std::memcmp(first.data()+64*128,ptr(11),64*64*sizeof(Element))==0);
      }
    }
  }
  tm_plan_destroy(opaque);for(auto pointer:data)TILEMEGA_CUDA_CHECK(cudaFree(pointer));
  std::puts("paged prefill attention: QKV, two query blocks, exact consumer table, o_proj and L1/L2 passed");
}
