// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#ifndef DM_TEST_PHASE
#define DM_TEST_PHASE 2
#endif
#define TILEMEGA_SERVING_PHASE DM_TEST_PHASE
#define TILEMEGA_SERVING_SEQ 1
#define TILEMEGA_SERVING_BATCH_LO 1
#define TILEMEGA_SERVING_BATCH_HI 1
#define TILEMEGA_SERVING_PAST_LO 0
#define TILEMEGA_SERVING_PAST_HI 0
#define TILEMEGA_NONPAGED_LA 1
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 16
#define TILEMEGA_GEMM_TILE_K 16
#define TILEMEGA_GEMM_STAGES 2
#define TILEMEGA_CG_SPLIT_TASK_ORDER 1
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
using E=cutlass::bfloat16_t;
constexpr int kRows=35,kInput=96,kHidden=32,kOutput=16,kChunks=3;
constexpr BufferDesc External(char const* name,unsigned count) {
  BufferDesc out{};out.name=name;out.constant=count;out.source=BufferSource::kZero;
  out.role=1;out.external_name=name;return out;
}
constexpr BufferDesc kBuffers[]={External("input",kRows*kInput),
  External("weight0",kHidden*kInput),External("weight1",kOutput*kHidden),
  External("hidden",kRows*kHidden),External("output",kRows*kOutput)};
constexpr GemmDesc Gemm(unsigned a,unsigned b,unsigned d,unsigned n,unsigned k) {
  GemmDesc out{};out.n=n;out.k=k;out.a=a;out.b=b;out.c=out.d=d;
  out.access.rows_per_batch=kRows;return out;
}
constexpr GemmDesc kGemms[]={Gemm(0,1,3,kHidden,kInput),Gemm(3,2,4,kOutput,kHidden)};
StageDesc kStages[]={{TaskKind::kGemm,0,0,kHidden,0,{}},
                    {TaskKind::kGemm,1,0,kOutput,0,{}}};
constexpr GemmRuntimeDesc kGeometry[]={{0,kChunks,16,16,16,2},{0,1,16,16,16,2}};
constexpr StageDependency kDependencies[]={{0,1,StageDependency::Map::kWindow,1,2,0,2}};
constexpr unsigned kOffsets[]={0,0,1};
constexpr ScheduleStageDesc kSchedule[]={{0,0,0},{1,0,1}};
RuntimeVariantDesc kVariants[]={{kGeometry,kDependencies,1,kOffsets,kSchedule,2,1,1,1,
  kCombinerTileOwnership,nullptr,true}};
constexpr unsigned short kSeqVariant[2]={};
constexpr OutputDesc kOutputs[]={{4,nullptr}};
constexpr ModelSpec kModel={{0,0,0,1,DM_TEST_PHASE==2?0:1},ScalarType::kBF16,kBuffers,5,kGemms,2,
  kStages,2,kOutputs,1,kVariants,1,kSeqVariant,2};
constexpr std::uint8_t kServingFrontier[]={1,1,1,0,0};
float Weight(int n,int k,int layer) {return ((n*7+k*13+layer*3)%17-8)*0.015625f;}
std::vector<E> Reference(std::vector<E> const& input,int n,int k,int layer) {
  std::vector<E> output(kRows*n);
  for(int row=0;row<kRows;++row)for(int col=0;col<n;++col) {
    float sum=0;for(int dim=0;dim<k;++dim)
      sum+=float(input[row*k+dim])*float(E(Weight(col,dim,layer)));
    output[row*n+col]=E(sum);
  }
  return output;
}
void RejectIncompatibleOwners() {
  for(unsigned bad=0;bad<11;++bad) {
    harness::DeviceModel model;
    model.params.ownership_flags=kCombinerTileOwnership;
    model.params.dims={1,0,1,1,1};
    model.stages.resize(3);
    model.stages[0].kind=TaskKind::kGemm;
    model.stages[0].handoff_reduce_stage=1;
    model.stages[1].kind=TaskKind::kGemmCombine;
    model.stages[1].handoff_elided=true;
    std::vector<GemmInvocation> gemms(1);
    gemms[0].chunks=3;gemms[0].tiles_m=2;gemms[0].tiles_n=2;
    if(bad==0)model.stages[0].handoff_reduce_stage=0;
    if(bad==1)model.stages[0].handoff_reduce_stage=3;
    if(bad==2)model.stages[1].handoff_elided=false;
    if(bad==3)model.params.ownership_flags=0;
    if(bad==4)model.stages[1].gemm=1;
    if(bad==5)gemms[0].chunks=1;
    if(bad==6)model.stages[1].kind=TaskKind::kArgmaxReduce;
    if(bad==7)model.stages[2].handoff_elided=true;
    if(bad==8) {
      model.stages[0].handoff_reduce_stage=2;
      model.stages[1]=model.stages[0];
      model.stages[2].kind=TaskKind::kGemmCombine;
      model.stages[2].handoff_elided=true;
    }
    if(bad>=9) {
      auto& source=model.stages[0];auto& target=model.stages[1];
      source.kind=TaskKind::kFusedAttention;target.kind=TaskKind::kAttentionMerge;
      source.extent=target.extent=1;source.width=target.width=64;
      source.attention_kv_block=target.attention_kv_block=64;
      source.operand[8]=target.operand[0]=10;
      source.operand[9]=target.operand[1]=11;
      source.operand[7]=target.operand[2]=12;
      if(bad==9)target.operand[0]=13;
      else model.params.dims.seq=64;
    }
    bool rejected=false;
    try {harness::PrepareEpochHandoffs(model,gemms);}
    catch(std::invalid_argument const&){rejected=true;}
    assert(rejected && !model.params.serving_epoch_handoff_tickets);
  }
}
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
int main() {
  RejectIncompatibleOwners();
  std::vector<E*> data(5);
  for(int i=0;i<5;++i)TILEMEGA_CUDA_CHECK(cudaMallocManaged(&data[i],kBuffers[i].constant*sizeof(E)));
  for(int layer=0;layer<2;++layer) {
    int n=layer?kOutput:kHidden,k=layer?kHidden:kInput;
    for(int col=0;col<n;++col)for(int dim=0;dim<k;++dim)
      data[layer+1][col*k+dim]=E(Weight(col,dim,layer));
  }
  auto target=tilemega::TargetSpec::Probe();
  kVariants[0].plan.eft_grid=std::min(16,target.res.num_sms);
  void* external[5];for(int i=0;i<5;++i)external[i]=data[i];
  std::vector<std::vector<E>> control;
  for(bool la:{false,true}) {
    kStages[0].handoff_reduce_stage=la?kHandoffAutoCombine:kNoOperand;
    auto* opaque=tm_plan_create(1,external,0);assert(opaque);
    auto* plan=static_cast<serving::Plan*>(opaque);
    assert(plan->model.stages.size()==3 && plan->model.stages[1].handoff_elided==la);
    if(la)for(auto const& task:plan->model.schedule)assert(task.stage!=1);
    int past[4]={};assert(tm_plan_set_steps(opaque,past,DM_TEST_PHASE==1?4:1)==0);
    GemmInvocation inv{};
    TILEMEGA_CUDA_CHECK(cudaMemcpy(&inv,plan->model.device_gemms,sizeof(inv),cudaMemcpyDeviceToHost));
    for(unsigned epoch=0;epoch<16;++epoch) {
      std::vector<E> input(kRows*kInput);
      for(int row=0;row<kRows;++row)for(int dim=0;dim<kInput;++dim)
        data[0][row*kInput+dim]=input[row*kInput+dim]=
            E(((row*11+dim*5+int(epoch)*7)%23-11)*0.03125f);
      auto expected=Reference(Reference(input,kHidden,kInput,0),kOutput,kHidden,1);
      std::vector<E> first;
      for(int mode:{TM_SERVING_L1,TM_SERVING_L2}) {
        for(int i:{3,4})TILEMEGA_CUDA_CHECK(cudaMemset(data[i],0xFF,kBuffers[i].constant*sizeof(E)));
        if(inv.serving_partial)TILEMEGA_CUDA_CHECK(cudaMemset(inv.serving_partial,0xFF,
            kChunks*kRows*kHidden*sizeof(float)));
        assert(tm_plan_launch(opaque,0,mode,epoch,nullptr)==0);
        TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
        for(int i=0;i<kRows*kOutput;++i) {
          float actual=float(data[4][i]),ref=float(expected[i]);
          if(!std::isfinite(actual) || std::abs(actual-ref)>1.6e-2f+1.6e-2f*std::abs(ref)) {
            std::fprintf(stderr,"la=%d epoch=%u mode=%d element=%d actual=%g ref=%g\n",
                la,epoch,mode,i,actual,ref);return 3;
          }
        }
        if(first.empty())first.assign(data[4],data[4]+kRows*kOutput);
        else assert(std::memcmp(first.data(),data[4],first.size()*sizeof(E))==0);
        if(la) {
          auto stride=plan->model.params.serving_epoch_handoff_stride;
          std::vector<unsigned long long> tickets(stride);
          TILEMEGA_CUDA_CHECK(cudaMemcpy(tickets.data(),
              plan->model.params.serving_epoch_handoff_tickets+
                  (mode==TM_SERVING_L2?plan->model.stages.size()*stride:0),
              stride*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
          for(int tile=0;tile<((kRows+15)/16)*2;++tile)
            assert(tickets[tile]==kChunks*(epoch+1ull));
          EventCounter barrier;
          TILEMEGA_CUDA_CHECK(cudaMemcpy(&barrier,plan->model.events+1,sizeof(barrier),cudaMemcpyDeviceToHost));
          assert(barrier.arrivals==0 && barrier.epoch==0);
        }
      }
      if(!la)control.push_back(first);
      else assert(std::memcmp(first.data(),control[epoch].data(),first.size()*sizeof(E))==0);
    }
#if DM_TEST_PHASE == 1
    // No token output changes the external input in this synthetic plan, so
    // every step has the same independent reference. This checks the loop's
    // event/ticket epochs without borrowing a forward-phase loop capability.
    TILEMEGA_CUDA_CHECK(cudaFree(plan->step_ns));plan->step_ns=nullptr;
    assert(tm_plan_loop_modes(opaque)&TM_SERVING_L1);
    assert(tm_plan_launch_steps(opaque,0,4,TM_SERVING_L1,16,nullptr)==0);
    TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
    assert(std::memcmp(data[4],control.back().data(),control.back().size()*sizeof(E))==0);
    if(la) {
      std::vector<unsigned long long> tickets(plan->model.params.serving_epoch_handoff_stride);
      TILEMEGA_CUDA_CHECK(cudaMemcpy(tickets.data(),plan->model.params.serving_epoch_handoff_tickets,
          tickets.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
      for(int tile=0;tile<((kRows+15)/16)*2;++tile)assert(tickets[tile]==kChunks*20ull);
      EventCounter barrier;
      TILEMEGA_CUDA_CHECK(cudaMemcpy(&barrier,plan->model.events+1,sizeof(barrier),cudaMemcpyDeviceToHost));
      assert(barrier.arrivals==0 && barrier.epoch==0);
    }
#endif
    tm_plan_destroy(opaque);
  }
  for(auto ptr:data)TILEMEGA_CUDA_CHECK(cudaFree(ptr));
  std::puts("small-tile native split-K: native L1/L2, elided barriers, epoch tickets, poison and bitwise stage/LA passed");
}
