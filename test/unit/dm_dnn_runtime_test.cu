// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_SERVING_PHASE 2
#define TILEMEGA_SERVING_BATCH_LO 2
#define TILEMEGA_SERVING_BATCH_HI 2
#define TILEMEGA_SERVING_PAST_LO 0
#define TILEMEGA_SERVING_PAST_HI 0
#ifndef DM_TEST_EMBEDDING
#define DM_TEST_EMBEDDING 0
#endif
#ifndef DM_TEST_PAGED
#define DM_TEST_PAGED 0
#endif
#ifndef DM_TEST_SPLIT
#define DM_TEST_SPLIT 1
#endif
#if DM_TEST_EMBEDDING
#define TILEMEGA_SERVING_SEQ 128
#else
#define TILEMEGA_SERVING_SEQ 1
#endif
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 16
#define TILEMEGA_GEMM_TILE_K 16
#define TILEMEGA_GEMM_STAGES 3
#define TILEMEGA_PAGED DM_TEST_PAGED
#define TILEMEGA_WEIGHT_LAYOUT_TILED DM_TEST_PAGED
#define TILEMEGA_NONPAGED_TILED DM_TEST_PAGED
#define TILEMEGA_PAGE_BYTES 8192
#define TILEMEGA_PAGE_COUNT 2
#define TILEMEGA_PAGE_WORKSPACE_OFFSET 128
#define TILEMEGA_PAGE_POOL_OFFSET 3072
#define TILEMEGA_LOOKAHEAD_BYTES 32768
#define TILEMEGA_DM_EPILOGUE_DISPATCH 1
#define TILEMEGA_DM_STAGE_DISPATCH 1
#include <tilemega/Codegen/DmDescriptors.h>
#include <tilemega/Codegen/tasks/TaskBase.h>
namespace tilemega::codegen {
template<class Runner>
__device__ inline void DispatchDmEpilogue(std::uint32_t,Runner const& runner) {
  runner.template Run<DmEpilogueSpec<DmEpilogueProgram<>>>();
}
template<class Runner>
__device__ inline void DispatchDmStage(unsigned kind,unsigned width,unsigned rows,Runner const& runner) {
#if DM_TEST_EMBEDDING
  if(kind==unsigned(TaskKind::kEmbeddingSum) && width==768 && rows==1) {
    runner.template Run<TaskKind::kEmbeddingSum,768,1>();return;
  }
  if(kind==unsigned(TaskKind::kLayerNorm) && width==768 && rows==4) {
    runner.template Run<TaskKind::kLayerNorm,768,4>();return;
  }
#else
  if(kind==unsigned(TaskKind::kLayoutConvert) && width==3 && rows==17) {
    runner.template Run<TaskKind::kLayoutConvert,3,17>();return;
  }
  if(kind==unsigned(TaskKind::kLayerNorm) && width==19 && rows==4) {
    runner.template Run<TaskKind::kLayerNorm,19,4>();return;
  }
#endif
  asm volatile("trap;");
}
}
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
#include <cassert>
#include <cmath>

namespace {
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
constexpr BufferDesc External(char const* name,unsigned count,unsigned dtype=0) {
  BufferDesc out{};out.name=out.external_name=name;out.constant=count;
  out.source=BufferSource::kZero;out.role=1;out.dtype=dtype;return out;
}
constexpr DmBufferLayout Image(unsigned h,unsigned w,unsigned c,unsigned cp,unsigned pad) {
  DmBufferLayout l{};l.kind=DmLayout::kNHWC;l.rank=4;
  l.logical[0]=l.physical[0]=2;l.logical[1]=h;l.logical[2]=w;l.logical[3]=c;
  l.physical[1]=h+2*pad;l.physical[2]=w+2*pad;l.physical[3]=cp;
  l.strides[3]=1;l.strides[2]=cp;l.strides[1]=l.physical[2]*cp;
  l.strides[0]=l.physical[1]*l.strides[1];
  l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=pad;return l;
}
constexpr StageDesc Norm(unsigned input,unsigned gamma,unsigned beta,unsigned out,
                         unsigned stats,unsigned width,unsigned rows) {
  StageDesc s{};s.kind=TaskKind::kLayerNorm;s.width=width;s.group=4;
  s.rows_per_batch=rows;s.operand[0]=input;s.operand[1]=gamma;s.operand[2]=beta;
  s.operand[3]=out;s.operand[4]=stats;s.norm_epsilon=1.0e-6f;return s;
}
#if DM_TEST_EMBEDDING
constexpr int kRows=256,kWidth=768;
constexpr BufferDesc Table(char const* name,unsigned rows,unsigned width) {
  auto b=External(name,rows*width);b.layout.rank=2;b.layout.logical[0]=rows;
  b.layout.logical[1]=width;b.layout.physical[0]=rows;b.layout.physical[1]=width;
  b.layout.strides[0]=width;b.layout.strides[1]=1;return b;
}
constexpr BufferDesc kBuffers[]={External("ids",kRows,3),External("types",kRows,3),
  Table("word",257,kWidth),Table("type_table",2,kWidth),
  Table("position",128,kWidth),External("embedding",kRows*kWidth),
  External("gamma",kWidth),External("beta",kWidth),External("output",kRows*kWidth),
  External("row_stats",kRows*2,1)};
constexpr StageDesc Embedding() {
  StageDesc s{};s.kind=TaskKind::kEmbeddingSum;s.width=kWidth;s.group=1;s.extent=257;
  for(unsigned i=0;i<6;++i)s.operand[i]=i;
  s.operand[6]=kNoOperand;return s;
}
constexpr StageDesc kStages[]={Embedding(),Norm(5,6,7,8,9,kWidth,0)};
constexpr GemmDesc kGemms[]={};
constexpr ConvDesc kConvolutions[]={};
constexpr GemmRuntimeDesc kGeometry[]={};
constexpr StageDependency kDependencies[]={{0,1,StageDependency::Map::kWindow,1,4,0,4}};
constexpr unsigned kOffsets[]={0,0,1};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0},{1,0,1}};
constexpr OutputDesc kOutputs[]={{8,nullptr}};
constexpr unsigned kGemmCount=0,kConvCount=0,kStageCount=2,kBufferCount=10;
#else
constexpr int kRows=126,kWidth=19;
constexpr DmBufferLayout kInputLayout=Image(7,9,3,8,1),kConvLayout=Image(7,9,19,24,0);
constexpr BufferDesc WithLayout(BufferDesc b,DmBufferLayout l) {b.layout=l;return b;}
constexpr unsigned kWeightElements=DM_TEST_PAGED?2*5*16*16:19*3*3*8;
constexpr BufferDesc kBuffers[]={External("input",2*3*7*9),
  WithLayout(External("layout",2*kInputLayout.strides[0]),kInputLayout),
  External("weight",kWeightElements),
  WithLayout(External("convolution",2*kConvLayout.strides[0]),kConvLayout),
  External("gamma",19),External("beta",19),External("output",kRows*19),
  External("row_stats",2*kRows,1)};
constexpr StageDesc Convert() {
  StageDesc s{};s.kind=TaskKind::kLayoutConvert;s.width=3;s.group=17;
  s.rows_per_batch=63;s.operand[0]=0;s.operand[1]=1;return s;
}
constexpr StageDesc Conv() {
  StageDesc s{};s.kind=TaskKind::kGemm;s.gemm=0;return s;
}
constexpr GemmDesc Gemm() {
  GemmDesc g{};g.n=19;g.k=27;g.a=1;g.b=2;g.c=g.d=3;
  g.access.a=DmAAccess::kIm2Col;g.access.conv=0;g.access.rows_per_batch=63;
  g.access.write.layout=3;return g;
}
constexpr StageDesc kStages[]={Convert(),Conv(),Norm(3,4,5,6,7,19,63)};
constexpr GemmDesc kGemms[]={Gemm()};
constexpr ConvDesc kConvolutions[]={{2,7,9,3,19,3,3,1,1,1,1,1,1,7,9,1,3}};
constexpr GemmRuntimeDesc kGeometry[]={{0,DM_TEST_SPLIT,16,16,16,3}};
constexpr StageDependency kDependencies[]={{0,1,StageDependency::Map::kAll,1,0,0,1},
  {1,2,StageDependency::Map::kWindow,4,2,0,2}};
constexpr unsigned kOffsets[]={0,0,1,2};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0},{1,0,1},{2,1,1}};
constexpr OutputDesc kOutputs[]={{6,nullptr}};
constexpr unsigned kGemmCount=1,kConvCount=1,kStageCount=3,kBufferCount=8;
#endif
RuntimeVariantDesc kVariants[]={{kGeometry,kDependencies,kStageCount-1,kOffsets,
  kSchedule,kStageCount,1,TILEMEGA_SERVING_SEQ,TILEMEGA_SERVING_SEQ,kCombinerTileOwnership}};
constexpr unsigned short kSeqVariant[TILEMEGA_SERVING_SEQ+1]={};
constexpr ModelSpec kModel={{0,0,0,2,0},ScalarType::kBF16,kBuffers,kBufferCount,
  kGemms,kGemmCount,kStages,kStageCount,kOutputs,1,kVariants,1,kSeqVariant,
  TILEMEGA_SERVING_SEQ+1,TILEMEGA_NORM_EPSILON,kConvolutions,kConvCount};
float Value(unsigned a,unsigned b,unsigned epoch) {
  return (int((a*11+b*7+epoch*13)%31)-15)*0.03125f;
}
void Near(float actual,float expected) {
  if(!std::isfinite(actual) || std::abs(actual-expected)>1.6e-2f+1.6e-2f*std::abs(expected)) {
    std::fprintf(stderr,"actual=%g expected=%g\n",actual,expected);std::abort();
  }
}
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
int main() {
  std::vector<void*> data(kBufferCount);
  for(unsigned i=0;i<kBufferCount;++i) {
    auto bytes=kBuffers[i].dtype==3?8:kBuffers[i].dtype==1?4:2;
    TILEMEGA_CUDA_CHECK(cudaMallocManaged(&data[i],kBuffers[i].constant*bytes));
    TILEMEGA_CUDA_CHECK(cudaMemset(data[i],0,kBuffers[i].constant*bytes));
  }
  auto bf=[&](unsigned i){return static_cast<E*>(data[i]);};
#if DM_TEST_EMBEDDING
  for(unsigned t=2;t<5;++t)for(unsigned i=0;i<kBuffers[t].constant;++i)
    bf(t)[i]=E(Value(i/kWidth,i%kWidth,t));
  unsigned gamma=6,beta=7,out=8,stats=9;
#else
  unsigned gamma=4,beta=5,out=6,stats=7;
  auto geometry=tilemega::backend::ConvIterationGeometry::Build(kConvolutions[0],kInputLayout,16);
  using Config=tilemega::backend::ServingDmGemm<tilemega::arch::Sm89,16,16,16,3>;
  for(unsigned n=0;n<19;++n)for(unsigned r=0;r<3;++r)
    for(unsigned s=0;s<3;++s)for(unsigned c=0;c<8;++c) {
      E value(c<3?Value(n,r*9+s*3+c,1):0);
#if !DM_TEST_PAGED
      bf(2)[((n*3+r)*3+s)*8+c]=value;
#endif
    }
#if DM_TEST_PAGED
  for(unsigned tn=0;tn<2;++tn)for(unsigned it=0;it<5;++it)
    for(unsigned n=0;n<16;++n)for(unsigned k=0;k<16;++k) {
      auto point=geometry.At(it,k);unsigned column=tn*16+n;
      bf(2)[(tn*5+it)*256+typename Config::LayoutB{}(n,k)]=
          E(column<19 && point.valid && point.c<3?Value(column,point.r*9+point.s*3+point.c,1):0);
    }
#endif
#endif
  for(unsigned c=0;c<kWidth;++c){bf(gamma)[c]=E(0.75f+Value(c,1,0));bf(beta)[c]=E(Value(c,2,0));}
  auto target=tilemega::TargetSpec::Probe();kVariants[0].plan.eft_grid=std::min(7,target.res.num_sms);
  tm_plan_info info{};assert(tm_plan_query(&info)==0 && info.phase==2 && info.capacity==0);
#if !DM_TEST_EMBEDDING && DM_TEST_SPLIT > 1
  kVariants[0].ownership_flags=0;
  assert(tm_plan_create(2,data.data(),0)==nullptr);
  kVariants[0].ownership_flags=kCombinerTileOwnership;
#endif
  auto* plan=tm_plan_create(2,data.data(),0);assert(plan);
  int step=0;assert(tm_plan_set_steps(plan,&step,1)==0);
  for(unsigned epoch=0;epoch<8;++epoch) {
    std::vector<E> expected(kRows*kWidth);
#if DM_TEST_EMBEDDING
    for(unsigned row=0;row<kRows;++row) {
      auto word=(row*17+epoch*3)%257,type=(row+epoch)%2;
      static_cast<std::int64_t*>(data[0])[row]=word;
      static_cast<std::int64_t*>(data[1])[row]=type;
      for(unsigned c=0;c<kWidth;++c)
        expected[row*kWidth+c]=E(float(E(float(bf(2)[word*kWidth+c])+float(bf(3)[type*kWidth+c])))+
            float(bf(4)[(row%128)*kWidth+c]));
    }
#else
    for(unsigned n=0;n<2;++n)for(unsigned c=0;c<3;++c)for(unsigned pixel=0;pixel<63;++pixel)
      bf(0)[(n*3+c)*63+pixel]=E(Value(n*63+pixel,c,epoch));
    for(unsigned row=0;row<kRows;++row)for(unsigned column=0;column<kWidth;++column) {
      int pixel=row%63;float sum=0;
      for(int r=0;r<3;++r)for(int s=0;s<3;++s)for(unsigned c=0;c<3;++c) {
        int y=pixel/9+r-1,x=pixel%9+s-1;
        if(y>=0 && y<7 && x>=0 && x<9)
          sum+=float(bf(0)[((row/63)*3+c)*63+y*9+x])*float(E(Value(column,r*9+s*3+c,1)));
      }
      expected[row*kWidth+column]=E(sum);
    }
#endif
    auto before_norm=expected;
    for(unsigned row=0;row<kRows;++row) {
      double sum=0,var=0;for(unsigned c=0;c<kWidth;++c)sum+=float(expected[row*kWidth+c]);
      double mean=sum/kWidth;
      for(unsigned c=0;c<kWidth;++c){double d=float(expected[row*kWidth+c])-mean;var+=d*d;}
      for(unsigned c=0;c<kWidth;++c)expected[row*kWidth+c]=E(
          ((float(expected[row*kWidth+c])-mean)/std::sqrt(var/kWidth+1e-6))*float(bf(gamma)[c])+float(bf(beta)[c]));
    }
    std::vector<E> first;
    for(int mode:{TM_SERVING_L1,TM_SERVING_L2}) {
      TILEMEGA_CUDA_CHECK(cudaMemset(data[out],0xff,kBuffers[out].constant*2));
      assert(tm_plan_launch(plan,0,mode,epoch,nullptr)==0);TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
      for(unsigned i=0;i<expected.size();++i) {
#if DM_TEST_EMBEDDING
        auto intermediate=i;
        auto actual_before=float(bf(5)[intermediate]);
#else
        auto intermediate=(i/kWidth)*24+i%kWidth;
        auto actual_before=float(bf(3)[intermediate]);
#endif
        auto actual=float(bf(out)[i]),reference=float(expected[i]);
        if(std::abs(actual-reference)>1.6e-2f+1.6e-2f*std::abs(reference))
          std::fprintf(stderr,"epoch=%u mode=%d row=%u column=%u intermediate=%g reference_intermediate=%g\n",
              epoch,mode,i/kWidth,i%kWidth,actual_before,float(before_norm[i]));
        Near(actual_before,float(before_norm[i]));
        Near(actual,reference);
      }
      if(first.empty())first.assign(bf(out),bf(out)+expected.size());
      else assert(std::memcmp(first.data(),bf(out),expected.size()*2)==0);
      for(unsigned row=0;row<kRows;++row) {
        float sum=0,square=0;for(unsigned c=0;c<kWidth;++c) {
          float v=float(bf(out)[row*kWidth+c]);sum+=v;square+=v*v;
        }
        Near(static_cast<float*>(data[stats])[row*2],sum);
        Near(static_cast<float*>(data[stats])[row*2+1],square);
      }
#if !DM_TEST_EMBEDDING
      for(unsigned n=0;n<2;++n)for(unsigned y=0;y<9;++y)for(unsigned x=0;x<11;++x)
        for(unsigned c=0;c<8;++c)if(y==0 || y==8 || x==0 || x==10 || c>=3)
          assert(float(bf(1)[n*kInputLayout.strides[0]+y*88+x*8+c])==0);
#endif
    }
  }
  tm_plan_destroy(plan);for(auto p:data)TILEMEGA_CUDA_CHECK(cudaFree(p));
  std::puts("DNN runtime: native forward ABI, dependencies, L1/L2 bit equality and independent FP32 oracle passed");
}
