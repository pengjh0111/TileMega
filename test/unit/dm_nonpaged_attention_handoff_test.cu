// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_SERVING_PHASE 1
#define TILEMEGA_SERVING_SEQ 1
#define TILEMEGA_SERVING_BATCH_LO 3
#define TILEMEGA_SERVING_BATCH_HI 3
#define TILEMEGA_SERVING_PAST_LO 0
#define TILEMEGA_SERVING_PAST_HI 136
#define TILEMEGA_SERVING_HEAD_DIM 64
#define TILEMEGA_SERVING_QPERKV 4
#define TILEMEGA_SERVING_QROWS 4
#define TILEMEGA_SERVING_KV_TILE 64
#define TILEMEGA_NONPAGED_LA 1
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 2
#define TILEMEGA_DM_EPILOGUE_DISPATCH 1
#include <tilemega/Codegen/DmDescriptors.h>
namespace tilemega::codegen {
template<class Runner>
__device__ inline void DispatchDmEpilogue(std::uint32_t id,Runner const& runner) {
  if(id){asm volatile("trap;");return;}
  runner.template Run<DmEpilogueSpec<DmEpilogueProgram<>>>();
}
}
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
#include <cassert>
#include <cmath>

namespace {
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
constexpr int B=3,G=2,Q=4,D=64,C=137,Cb=3,H=G*Q*D;
constexpr BufferDesc External(char const* name,unsigned count,unsigned dtype=0) {
  BufferDesc out{};out.name=name;out.constant=count;out.source=BufferSource::kZero;
  out.dtype=dtype;out.role=1;out.external_name=name;return out;
}
constexpr BufferDesc kBuffers[]={External("qkv",B*G*(Q+2)*D),
  External("key",B*G*C*D),External("value",B*G*C*D),
  External("cos",C*D),External("sin",C*D),External("context",B*H),
  External("partial",B*G*Cb*Q*D,1),External("lse",B*G*Cb*Q,1),
  External("out_weight",64*H),External("output",B*64)};
constexpr GemmDesc kGemms[]={{64,H,5,8,9,9,0}};
constexpr StageDesc Attention() {
  StageDesc out{};out.kind=TaskKind::kFusedAttention;out.extent=G;out.width=D;out.group=Q;
  unsigned operands[]={0,1,2,3,4,kNoOperand,kNoOperand,5,6,7};
  for(unsigned i=0;i<10;++i)out.operand[i]=operands[i];
  out.attention_kv_block=64;out.attention_query_rows=4;return out;
}
constexpr StageDesc Merge() {
  StageDesc out{};out.kind=TaskKind::kAttentionMerge;out.extent=G;out.width=D;out.group=Q;
  out.operand[0]=6;out.operand[1]=7;out.operand[2]=5;
  out.attention_kv_block=64;return out;
}
StageDesc kStages[]={Attention(),Merge(),{TaskKind::kGemm,0,0,64,0,{}}};
constexpr GemmRuntimeDesc kGeometry[]={{0,1,16,64,64,2}};
constexpr StageDependency kDependencies[]={{0,1,StageDependency::Map::kWindow,1,Cb,0,Cb},
  {1,2,StageDependency::Map::kAll,1,0,0,B*G}};
constexpr unsigned kOffsets[]={0,0,1,2};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0},{1,0,1},{2,1,1}};
RuntimeVariantDesc kVariants[]={{kGeometry,kDependencies,2,kOffsets,kSchedule,3,1,1,1,0,
  nullptr,true}};
constexpr unsigned short kSeqVariant[2]={};
constexpr OutputDesc kOutputs[]={{9,nullptr}};
constexpr ModelSpec kModel={{1,0,1,B,C},ScalarType::kBF16,kBuffers,10,kGemms,1,
  kStages,3,kOutputs,1,kVariants,1,kSeqVariant,2};
constexpr std::uint8_t kServingFrontier[]={1,1,1,1,1,0,0,0,1,0};
float Input(int b,int g,int row,int d,int epoch,int salt) {
  return ((b*19+g*11+row*7+d*3+epoch*5+salt)%37-18)*0.015625f;
}
float Weight(int n,int k) {return ((n*7+k*13)%17-8)*0.015625f;}
std::vector<E> Reference(std::vector<void*> const& data,int past) {
  auto p=[&](int i){return static_cast<E const*>(data[i]);};
  std::vector<E> context(B*H);
  for(int b=0;b<B;++b)for(int g=0;g<G;++g)for(int q=0;q<Q;++q) {
    std::vector<float> score(past+1);float peak=-INFINITY,total=0;
    for(int position=0;position<=past;++position) {
      float dot=0;
      for(int d=0;d<D;++d) {
        auto key=position<past?p(1)[((b*G+g)*C+position)*D+d]:
            p(0)[(b*G+g)*(Q+2)*D+Q*D+d];
        dot+=float(p(0)[(b*G+g)*(Q+2)*D+q*D+d])*float(key);
      }
      score[position]=dot*0.125f;peak=std::max(peak,score[position]);
    }
    for(auto& s:score){s=std::exp(s-peak);total+=s;}
    for(int d=0;d<D;++d) {
      float sum=0;
      for(int position=0;position<=past;++position) {
        auto value=position<past?p(2)[((b*G+g)*C+position)*D+d]:
            p(0)[(b*G+g)*(Q+2)*D+(Q+1)*D+d];
        sum+=score[position]/total*float(value);
      }
      context[(b*G+g)*Q*D+q*D+d]=E(sum);
    }
  }
  return context;
}
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
int main() {
  std::vector<void*> data(10);
  for(int i=0;i<10;++i)TILEMEGA_CUDA_CHECK(cudaMallocManaged(&data[i],
      kBuffers[i].constant*(kBuffers[i].dtype?4:2)));
  auto p=[&](int i){return static_cast<E*>(data[i]);};
  for(int i=0;i<C*D;++i){p(3)[i]=E(1);p(4)[i]=E(0);}
  for(int n=0;n<64;++n)for(int k=0;k<H;++k)p(8)[n*H+k]=E(Weight(n,k));
  auto target=tilemega::TargetSpec::Probe();
  kVariants[0].plan.eft_grid=std::min(16,target.res.num_sms);
  std::vector<std::vector<E>> control;
  unsigned errors=0;bool printed=false;
  for(bool la:{false,true}) {
    kStages[0].handoff_reduce_stage=la?1:kNoOperand;kStages[1].handoff_elided=la;
    auto* opaque=tm_plan_create(B,data.data(),0);assert(opaque);
    auto* plan=static_cast<serving::Plan*>(opaque);
    assert(plan->model.stages.size()==3 && plan->model.stages[1].handoff_elided==la);
    if(la)for(auto const& task:plan->model.schedule)assert(task.stage!=1);
    constexpr int histories[]={0,1,63,64,65,127,128,136};
    assert(tm_plan_set_steps(opaque,histories,8)==0);
    for(unsigned epoch=0;epoch<16;++epoch) {
      int past=histories[epoch%8];
      for(int b=0;b<B;++b)for(int g=0;g<G;++g) {
        for(int row=0;row<Q+2;++row)for(int d=0;d<D;++d)
          p(0)[(b*G+g)*(Q+2)*D+row*D+d]=E(Input(b,g,row,d,epoch,1));
        for(int row=0;row<C;++row)for(int d=0;d<D;++d) {
          p(1)[((b*G+g)*C+row)*D+d]=E(Input(b,g,row,d,epoch,3));
          p(2)[((b*G+g)*C+row)*D+d]=E(Input(b,g,row,d,epoch,5));
        }
      }
      auto expected=Reference(data,past);std::vector<E> projected(B*64);
      for(int b=0;b<B;++b)for(int n=0;n<64;++n) {
        float sum=0;for(int k=0;k<H;++k)sum+=float(expected[b*H+k])*float(E(Weight(n,k)));
        projected[b*64+n]=E(sum);
      }
      std::vector<E> first;
      for(int mode:{TM_SERVING_L1,TM_SERVING_L2}) {
        for(int i:{5,6,7,9})TILEMEGA_CUDA_CHECK(cudaMemset(data[i],0xFF,
            kBuffers[i].constant*(kBuffers[i].dtype?4:2)));
        assert(tm_plan_launch(opaque,epoch%8,mode,epoch,nullptr)==0);
        TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
        for(int output:{5,9}) {
          auto const& ref=output==5?expected:projected;
          for(std::size_t i=0;i<ref.size();++i) {
            float actual=float(p(output)[i]),value=float(ref[i]);
            if(!std::isfinite(actual)||std::abs(actual-value)>1.6e-2f+1.6e-2f*std::abs(value)) {
              ++errors;
              if(!printed){std::fprintf(stderr,"la=%d epoch=%u past=%d mode=%d output=%d element=%zu actual=%g ref=%g\n",
                  la,epoch,past,mode,output,i,actual,value);printed=true;}
            }
          }
        }
        if(first.empty()) {
          first.assign(p(5),p(5)+B*H);first.insert(first.end(),p(9),p(9)+B*64);
        } else {
          assert(std::memcmp(first.data(),p(5),B*H*sizeof(E))==0);
          assert(std::memcmp(first.data()+B*H,p(9),B*64*sizeof(E))==0);
        }
        if(la) {
          auto stride=plan->model.params.serving_epoch_handoff_stride;
          std::vector<unsigned long long> tickets(stride);
          TILEMEGA_CUDA_CHECK(cudaMemcpy(tickets.data(),plan->model.params.serving_epoch_handoff_tickets+
              (mode==TM_SERVING_L2?3*stride:0),stride*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
          for(int i=0;i<B*G;++i)assert(tickets[i]==Cb*(epoch+1ull));
          EventCounter barrier;
          TILEMEGA_CUDA_CHECK(cudaMemcpy(&barrier,plan->model.events+1,sizeof(barrier),cudaMemcpyDeviceToHost));
          assert(barrier.arrivals==0 && barrier.epoch==0);
        }
      }
      if(!la)control.push_back(first);
      else assert(std::memcmp(first.data(),control[epoch].data(),first.size()*sizeof(E))==0);
    }
    tm_plan_destroy(opaque);
  }
  for(auto pointer:data)TILEMEGA_CUDA_CHECK(cudaFree(pointer));
  std::printf("native attention stage/LA and L1/L2 bitwise; independent numerical errors=%u\n",errors);
  return errors?3:0;
}
