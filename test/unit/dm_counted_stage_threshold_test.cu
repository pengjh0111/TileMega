// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#define TILEMEGA_SLOT_WINDOW 4
#define TILEMEGA_MODEL_BF16 1
#define TILEMEGA_SERVING_RUNTIME 1
#define TILEMEGA_SERVING_PHASE 2
#define TILEMEGA_SERVING_SEQ 1
#define TILEMEGA_SERVING_BATCH_LO 1
#define TILEMEGA_SERVING_BATCH_HI 1
#define TILEMEGA_SERVING_PAST_LO 0
#define TILEMEGA_SERVING_PAST_HI 0
#define TILEMEGA_GEMM_TILE_M 16
#define TILEMEGA_GEMM_TILE_N 64
#define TILEMEGA_GEMM_TILE_K 64
#define TILEMEGA_GEMM_STAGES 3
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
#include <cassert>

namespace {
using namespace tilemega::codegen;
constexpr int kProducers = 10, kConsumers = 7, kCountedConsumers=3;
constexpr std::uint32_t kThresholds[] = {91,8,8,2,73};
constexpr unsigned kCounterOffset=2, kBankSize=kCounterOffset+kCountedConsumers;
__host__ __device__ int Target(int producer,unsigned epoch) {
  int row=(producer+epoch)%kProducers;return row<4?0:row<8?1:2;
}
__host__ __device__ unsigned Weight(int producer,unsigned epoch) {
  return (producer+epoch)%kProducers<8?2u:1u;
}
constexpr BufferDesc kBuffers[] = {
    {"a", 64, 0, 0, 0, BufferSource::kZero, nullptr},
    {"w0", kProducers*64*64, 0, 0, 0, BufferSource::kZero, nullptr},
    {"w1", kConsumers*64*64, 0, 0, 0, BufferSource::kZero, nullptr},
    {"w2", kCountedConsumers*64*64, 0, 0, 0, BufferSource::kZero, nullptr},
    {"y0", kProducers*64, 0, 0, 0, BufferSource::kZero, nullptr},
    {"y1", kConsumers*64, 0, 0, 0, BufferSource::kZero, nullptr},
    {"y2", kCountedConsumers*64, 0, 0, 0, BufferSource::kZero, nullptr}};
constexpr GemmDesc kGemms[] = {{kProducers*64,64,0,1,4,4,0},
                              {kConsumers*64,64,0,2,5,5,0}, {kCountedConsumers*64,64,0,3,6,6,0}};
constexpr StageDesc kStages[] = {{TaskKind::kGemm,0,0,kProducers*64,0,{}},
    {TaskKind::kGemm,1,0,kConsumers*64,0,{}}, {TaskKind::kGemm,2,0,kCountedConsumers*64,0,{}}};
constexpr GemmRuntimeDesc kGeometry[] = {{0,1,16,64,64,3}, {0,1,16,64,64,3}, {0,1,16,64,64,3}};
constexpr unsigned kOffsets[] = {0,0,1,2};
constexpr ScheduleStageDesc kSchedule[] = {{0,0,0}, {1,0,1}, {2,1,1}};
StageDependency kDependencies[] = {{0,1,StageDependency::Map::kTable,1,0,0,1},
                                  {0,2,StageDependency::Map::kCounted,1,0,0,0}};
RuntimeVariantDesc kVariants[] = {{kGeometry,kDependencies,2,kOffsets,kSchedule,3,2,1,1,0}};
constexpr unsigned short kSeqVariant[] = {0,0};
constexpr OutputDesc kOutputs[] = {{5,nullptr}, {6,nullptr}};
constexpr ModelSpec kModel = {{0,0,0,1,0}, ScalarType::kBF16,kBuffers,7,kGemms,3,kStages,3,
                         kOutputs,2,kVariants,1,kSeqVariant,2};
__host__ __device__ bool Selected(int p, int c) { return c+1<kConsumers && (p+2*c)%3!=1; }
__host__ __device__ int Value(int p, int lane, int epoch) { return (p+1)*101+lane*3+epoch*17; }
__device__ void Body(Params const& params, unsigned stage, unsigned task,
                    int* values, int* output, unsigned epoch) {
  int lane = threadIdx.x;
  if (stage == 0) {
    values[task*128+lane] = Value(task,lane,epoch);
    executor::CountedDependency::Publish(params.counted_dependencies+kCounterOffset+Target(task,epoch), Weight(task,epoch));
  } else if (stage == 1) {
    int sum = 0;
    for (int p=0;p<kProducers;++p) if (Selected(p,task)) sum += values[p*128+lane];
    output[task*128+lane] = sum;
  } else {
    int sum = 0;
    for (int p=0;p<kProducers;++p) if (Target(p,epoch)==int(task)) sum += Value(p,lane,epoch)*Weight(p,epoch);
    // Read the released partials, rather than the oracle values above.
    int actual = 0;
    for (int p=0;p<kProducers;++p) if (Target(p,epoch)==int(task)) actual += values[p*128+lane]*Weight(p,epoch);
    if (actual != sum) asm volatile("trap;");
    output[(kConsumers+task)*128+lane] = actual;
  }
}
__global__ void SyntheticL1(Params const* params, EventCounter* events,
                            int* values, int* output, unsigned epoch) {
  for (unsigned stage=0;stage<params->stage_count;++stage) {
    int tasks = ActiveBlocks(*params,params->stages[stage]);
    for (int task=blockIdx.x;task<tasks;task+=gridDim.x) Body(*params,stage,task,values,output,epoch);
    GridBarrier(events,stage,epoch,params);
  }
}
__global__ void SyntheticL2(Params const* params, EventCounter* events,
                            int* values, int* output, unsigned epoch) {
  for (unsigned slot=params->schedule_offsets[blockIdx.x];slot<params->schedule_offsets[blockIdx.x+1];++slot) {
    auto task = params->schedule[slot];
    WaitTaskDependencies(*params,events,task,epoch);
    if(!ProbeTaskDependencies(*params,events,task,epoch))asm volatile("trap;");
    Body(*params,task.stage,task.logical_task,values,output,epoch);
    NotifyTask(*params,events,task.stage,task.logical_task,epoch);
  }
}
}
#include <tilemega/Codegen/tasks/ServingRuntime.cuh>
int main() {
  std::vector<std::vector<RuntimeDependencyInterval>> rows(kConsumers);
  unsigned stride=1;
  for (int c=0;c<kConsumers;++c) {
    for (int p=0;p<kProducers;) {
      if (!Selected(p,c)) { ++p; continue; }
      int begin=p; while(p<kProducers && Selected(p,c)) ++p;
      rows[c].push_back({unsigned(begin),unsigned(p-begin)});
    }
    stride=std::max(stride,unsigned(rows[c].size()));
  }
  std::vector<RuntimeDependencyInterval> table(kConsumers*stride);
  for (int c=0;c<kConsumers;++c) std::copy(rows[c].begin(),rows[c].end(),table.begin()+c*stride);
  kDependencies[0].table_rows=kConsumers; kDependencies[0].table_stride=stride;
  kDependencies[1].table_offset=table.size();
  kDependencies[1].table_rows=kCountedConsumers;kDependencies[1].table_stride=1;
  kDependencies[1].counted_offset=kCounterOffset;kDependencies[1].counted_threshold_offset=1;
  kVariants[0].counted_thresholds={kThresholds,5};
  for(int target=0;target<kCountedConsumers;++target)table.push_back({0,kProducers});
  kVariants[0].dependency_intervals=table.data(); kVariants[0].dependency_interval_count=table.size();
  auto target=tilemega::TargetSpec::Probe();
  kVariants[0].plan.eft_grid=std::min(16,target.res.num_sms);
  void* external[7] = {};
  auto* opaque=tm_plan_create(1,external,0); assert(opaque);
  auto* plan=static_cast<serving::Plan*>(opaque);
  int past=0; assert(tm_plan_set_steps(opaque,&past,1)==0);
  assert(plan->model.params.counted_dependency_count==kBankSize);
  assert(plan->model.params.counted_thresholds.size==5);
  assert(plan->model.params.counted_thresholds.values!=kThresholds);
  std::uint32_t uploaded[5];
  TILEMEGA_CUDA_CHECK(cudaMemcpy(uploaded,plan->model.params.counted_thresholds.values,sizeof(uploaded),cudaMemcpyDeviceToHost));
  assert(std::equal(uploaded,uploaded+5,kThresholds));
  assert(plan->model.schedule.size()==kProducers+kConsumers+kCountedConsumers);
  int *values,*output;
  TILEMEGA_CUDA_CHECK(cudaMallocManaged(&values,kProducers*128*sizeof(int)));
  TILEMEGA_CUDA_CHECK(cudaMallocManaged(&output,(kConsumers+kCountedConsumers)*128*sizeof(int)));
  for (unsigned epoch=0;epoch<16;++epoch) {
    std::vector<int> first;
    for (unsigned bank:{0u,1u}) {
      TILEMEGA_CUDA_CHECK(cudaMemset(values,0xA5,kProducers*128*sizeof(int)));
      TILEMEGA_CUDA_CHECK(cudaMemset(output,0xA5,(kConsumers+kCountedConsumers)*128*sizeof(int)));
      if (!bank) SyntheticL1<<<plan->grid,128>>>(plan->ring,plan->model.events,values,output,epoch);
      else SyntheticL2<<<plan->grid,128>>>(plan->ring+plan->steps,plan->model.events,values,output,epoch);
      TILEMEGA_CUDA_CHECK(cudaGetLastError()); TILEMEGA_CUDA_CHECK(cudaDeviceSynchronize());
      for (int c=0;c<kConsumers+kCountedConsumers;++c) for (int lane=0;lane<128;++lane) {
        int expected=0;
        for (int p=0;p<kProducers;++p) {
          if(c<kConsumers && Selected(p,c)) expected+=Value(p,lane,epoch);
          else if(c>=kConsumers && Target(p,epoch)==c-kConsumers) expected+=Value(p,lane,epoch)*Weight(p,epoch);
        }
        assert(output[c*128+lane]==expected);
      }
      if (!bank) first.assign(output,output+(kConsumers+kCountedConsumers)*128);
      else assert(std::equal(first.begin(),first.end(),output));
      unsigned long long counters[kBankSize];
      TILEMEGA_CUDA_CHECK(cudaMemcpy(counters,plan->model.params.counted_dependencies+bank*kBankSize,
                                    sizeof(counters),cudaMemcpyDeviceToHost));
      assert(counters[0]==0 && counters[1]==0);
      for(unsigned c=0;c<kCountedConsumers;++c)assert(counters[kCounterOffset+c]==kThresholds[1+c]*(epoch+1));
    }
  }
  TILEMEGA_CUDA_CHECK(cudaFree(values)); TILEMEGA_CUDA_CHECK(cudaFree(output));
  tm_plan_destroy(opaque);
  std::puts("counted stage thresholds: exact ordering tables, uploaded [8,8,2] tails, offset counters, lookahead probes, separate L1/L2 banks and bitwise outputs passed");
}
