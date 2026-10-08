// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
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
#define TILEMEGA_GEMM_STAGES 2
#define TILEMEGA_CG_SPLIT_TASK_ORDER 1
#define TILEMEGA_PAGED 1
#define TILEMEGA_WEIGHT_LAYOUT_TILED 1
#define TILEMEGA_PAGE_BYTES 8192
#define TILEMEGA_PAGE_COUNT 2
#define TILEMEGA_PAGE_WORKSPACE_OFFSET 128
#define TILEMEGA_PAGE_POOL_OFFSET 9216
#define TILEMEGA_LOOKAHEAD_BYTES 32768
#define TILEMEGA_DM_EPILOGUE_DISPATCH 1
#include <tilemega/Codegen/DmDescriptors.h>
namespace tilemega::codegen {
template<class Runner>
__device__ inline void DispatchDmEpilogue(std::uint32_t,Runner const& runner) {
  runner.template Run<DmEpilogueSpec<DmEpilogueProgram<>>>();
}
}
#include <tilemega/Codegen/tasks/ModelHarness.cuh>
#include <cassert>
#include <cmath>
using namespace tilemega::codegen;
using E=cutlass::bfloat16_t;
constexpr int kVirtual=16,kExperts=8,kRows=kVirtual*16,kWidth=128;
constexpr int kShared=TILEMEGA_PAGE_POOL_OFFSET+2*8192;
__host__ __device__ bool Empty(unsigned v,unsigned epoch) {return (v+epoch)%3==0;}
__host__ __device__ unsigned Rows(unsigned v,unsigned epoch) {return 1+(v+epoch)%16;}
float Weight(int expert,int n,int k) {return ((expert*3+n*7+k*11)%19-9)*0.015625f;}
__global__ __launch_bounds__(160,1) void StreamProbe(Params const* pp,
    EventCounter* events,unsigned epoch,bool l2,unsigned long long* sequences,
    unsigned* errors,unsigned* tickets) {
  auto const& p=*pp;
  extern __shared__ __align__(1024) char storage[];
  __shared__ unsigned attempted;
  if(threadIdx.x==0)attempted=0;
  __syncthreads();
  Watch watch{p.serving_watchdog,10'000'000'000ull};
  paged::Ring ring{reinterpret_cast<paged::Ring::Slot*>(storage),
      storage+TILEMEGA_PAGE_POOL_OFFSET,nullptr,&watch};ring.Initialize();
  std::uint64_t sequence=0;
  auto* work=storage+TILEMEGA_PAGE_WORKSPACE_OFFSET;
  paged::PageStream ahead(&p,1,l2,events,epoch);
  unsigned long long prefetched=0,loaded=0;
  paged::Lookahead lookahead{&ahead,&prefetched,&loaded};
  bool loader=!executor::IsCompute();
  if(loader) {
    // Before the dense pages reach compute, the real lookahead cursor must
    // park at the future unready dispatch and return instead of spinning.
    executor::PrefetchRange range;
    unsigned ranges=0;while(ahead.Next(&range))++ranges;
    auto parked_stage=ahead.stage;auto parked_task=ahead.task;auto parked_offset=ahead.offset;
    if(ranges!=4 || parked_stage!=2 || ahead.Next(&range) ||
       ahead.stage!=parked_stage || ahead.task!=parked_task || ahead.offset!=parked_offset)
      if(executor::LoaderLane()==0)atomicAdd(errors,1);
    if(executor::LoaderLane()==0)atomicExch(&attempted,1);
    __syncwarp();ahead=paged::PageStream(&p,1,l2,events,epoch);
  }else {
    if(executor::ComputeThread()==0)while(atomicAdd(&attempted,0)==0) {}
    executor::ComputeSync();
  }
  for(int task=blockIdx.x;task<2*kVirtual;task+=gridDim.x) {
    if(loader) {
      if(l2)paged::Task<true,true>(p,0,task,ring,sequence,work,events,epoch,nullptr,0,&lookahead);
      else paged::Task<true,false>(p,0,task,ring,sequence,work,events,epoch,nullptr,0,&lookahead);
    }else {
      if(l2)paged::Task<false,true>(p,0,task,ring,sequence,work,events,epoch);
      else paged::Task<false,false>(p,0,task,ring,sequence,work,events,epoch);
      if(l2)paged::Publish(p,events,0,task,epoch);
    }
  }
  if(!loader) {
    if(!l2)paged::StageBarrier(events,0,epoch,&watch,&p);
    auto const* inv=static_cast<GemmInvocation const*>(p.gemms);
    auto* binding=const_cast<MoeBindingRecord*>(static_cast<MoeBindingRecord const*>(inv[2].binding));
    if(executor::ComputeThread()==0) {
      unsigned v=blockIdx.x;
      if(Empty(v,epoch))binding[v].valid=0;
      else binding[v]={(v*3+epoch)%kExperts,v*16,Rows(v,epoch),1};
    }
    if(l2)paged::Publish(p,events,1,blockIdx.x,epoch);
    else {__threadfence();executor::ComputeSync();paged::StageBarrier(events,1,epoch,&watch,&p);}
  }
  int chunks=static_cast<GemmInvocation const*>(p.gemms)[2].chunks;
  for(int task=blockIdx.x;task<2*kVirtual*chunks;task+=gridDim.x) {
    if(loader) {
      if(l2)paged::Task<true,true>(p,2,task,ring,sequence,work,events,epoch,nullptr,0,&lookahead);
      else paged::Task<true,false>(p,2,task,ring,sequence,work,events,epoch,nullptr,0,&lookahead);
    }else {
      if(l2) {
        auto ref=p.schedule[p.schedule_offsets[blockIdx.x]+2+task/gridDim.x];
        paged::WaitDependencies(p,events,ref,epoch);
        paged::Task<false,true>(p,2,task,ring,sequence,work,events,epoch);
        paged::Publish(p,events,2,task,epoch);
      }else paged::Task<false,false>(p,2,task,ring,sequence,work,events,epoch);
    }
  }
  if(!loader && chunks>1 && !p.stages[3].handoff_elided) {
    if(!l2)paged::StageBarrier(events,2,epoch,&watch,&p);
    for(int task=blockIdx.x;task<2*kVirtual;task+=gridDim.x) {
      if(l2) {
        auto ref=p.schedule[p.schedule_offsets[blockIdx.x]+2+2*chunks+task/gridDim.x];
        paged::WaitDependencies(p,events,ref,epoch);
        paged::Task<false,true>(p,3,task,ring,sequence,work,events,epoch);
        paged::Publish(p,events,3,task,epoch);
      }else paged::Task<false,false>(p,3,task,ring,sequence,work,events,epoch);
    }
  }
  if(loader)sequences[2*blockIdx.x]=sequence;
  else {
    executor::ComputeSync();
    if(executor::ComputeThread()==0)sequences[2*blockIdx.x+1]=sequence;
  }
}
template<class T>T* Managed(std::size_t count) {
  T* result;TILEMEGA_CUDA_CHECK(cudaMallocManaged(&result,count*sizeof(T)));return result;
}
GemmInvocation Invocation(E const* a,E const* b,E* out,int n,int k_begin=0,int k_count=128) {
  GemmInvocation inv{};inv.problem={kRows,n,k_count,1};
  inv.mainloop={a+k_begin,cute::make_stride(std::int64_t(128),cute::_1{},std::int64_t(kRows*128)),
                b+k_begin,cute::make_stride(std::int64_t(128),cute::_1{},std::int64_t(n*128))};
  inv.epilogue.ptr_C=out;inv.epilogue.ptr_D=out;
  inv.tiles_m=kVirtual;inv.tiles_n=n/64;inv.tile_m=16;inv.tile_n=64;inv.k_total=128;
  inv.serving_k_begin=k_begin;inv.serving_weight_base=b;inv.serving_k_total_full=128;
  inv.serving_tile_k=64;inv.serving_output_stride=n;inv.serving_partial_stride=n;
  return inv;
}
int main() {
  auto target=tilemega::TargetSpec::Probe();int resident=0;
  TILEMEGA_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,StreamProbe,160,kShared));
  assert(kVirtual<=resident*target.res.num_sms);
  auto* input=Managed<E>(kRows*128);auto* hidden=Managed<E>(kRows*128);
  auto* output=Managed<E>(kRows*128);auto* dense=Managed<E>(128*128);
  auto* experts=Managed<E>(kExperts*128*128);auto* partials=Managed<float>(2*kRows*128);
  auto* bindings=Managed<MoeBindingRecord>(kVirtual);auto* rows=Managed<MoeBindingRow>(kRows);
  auto* inv=Managed<GemmInvocation>(4);auto* stages=Managed<StageDesc>(4);
  auto* deps=Managed<StageDependency>(4);auto* intervals=Managed<RuntimeDependencyInterval>(2*kVirtual*2);
  auto* offsets=Managed<unsigned>(5);auto* event_offsets=Managed<unsigned>(5);
  auto* flags=Managed<unsigned>(4);auto* waits=Managed<TaskWait>(512);
  auto* schedule=Managed<TaskRef>(8*kVirtual);auto* schedule_offsets=Managed<unsigned>(kVirtual+1);
  auto* buffers=Managed<ModelElement*>(2);buffers[0]=reinterpret_cast<E*>(partials);buffers[1]=output;
  auto* tickets=Managed<unsigned>(4*2*kVirtual);auto* sequences=Managed<unsigned long long>(2*kVirtual);
  auto* errors=Managed<unsigned>(1);auto* watchdog=Managed<WatchdogRecord>(1);auto* params=Managed<Params>(1);
  using Layout=PagedGemmTaskBody<paged::PageArch,16,64,64,8192,2>::LayoutB;
  for(int n=0;n<128;++n)for(int k=0;k<128;++k) {
    auto index=((n/64)*2+k/64)*64*64+Layout{}(n%64,k%64);
    dense[index]=E(n==k?0.5f:0.0f);
    for(int e=0;e<kExperts;++e)experts[e*128*128+index]=E(Weight(e,n,k));
  }
  auto* events=Managed<EventCounter>(4+4*(1+2*kVirtual*2));
  for(int split:{1,2})for(bool la:{false,true})for(bool table:{false,true}) {
    if(split==1 && la)continue;
    inv[0]=Invocation(input,dense,hidden,128);inv[1]=Invocation(input,dense,hidden,64);
    inv[1].serving_weight_base=nullptr;
    for(int chunk=0;chunk<split;++chunk) {
      inv[2+chunk]=Invocation(hidden,experts,output,128,chunk*128/split,128/split);
      inv[2+chunk].chunks=split;inv[2+chunk].serving_partial=partials+chunk*kRows*128;
      auto& a=inv[2+chunk].access;a.b=DmBAccess::kExpertIndirect;a.expert_stride=128*128;
      a.binding_blocks=kVirtual;a.binding_rows=kRows;a.experts=kExperts;a.block_rows=16;
      inv[2+chunk].binding=bindings;inv[2+chunk].rows=rows;
    }
    for(int i=0;i<4;++i) {stages[i]={};stages[i].kind=TaskKind::kGemm;stages[i].gemm=i==0?0:i==1?1:2;}
    stages[2].binding_producer=1;stages[3].kind=TaskKind::kGemmCombine;
    stages[3].operand[0]=0;stages[3].operand[1]=1;stages[3].width=128;stages[3].group=split;
    if(la) {stages[2].handoff_reduce_stage=3;stages[3].handoff_elided=true;}
    offsets[0]=offsets[1]=offsets[2]=0;offsets[3]=2;offsets[4]=split==1?2:3;
    deps[0]={0,2,StageDependency::Map::kWindow,unsigned(2*split),2,0,2};
    deps[1]={1,2,table?StageDependency::Map::kTable:StageDependency::Map::kWindow,
             unsigned(2*split),1,0,1};
    if(table) {
      deps[1].table_rows=2*kVirtual*split;deps[1].table_stride=1;
      for(unsigned task=0;task<unsigned(2*kVirtual*split);++task)
        intervals[task]={task/unsigned(2*split),1};
    }
    deps[2]={2,3,StageDependency::Map::kWindow,1,split,0,unsigned(split)};
    unsigned cursor=0;
    for(int i=0;i<4;++i) {
      flags[i]=kNeedsAggregateEvent|(TILEMEGA_EVENT_KAPPA>0?kNeedsFineEvents:0);event_offsets[i]=cursor;
      int count=i==0?2*kVirtual:i==1?kVirtual:i==2?2*kVirtual*split:2*kVirtual;
      cursor+=1+(TILEMEGA_EVENT_KAPPA>0?CeilDiv(count,std::max(1,TILEMEGA_EVENT_KAPPA)):0);
    }
    event_offsets[4]=cursor;
    unsigned q=0,w=0;
    for(unsigned worker=0;worker<kVirtual;++worker) {
      schedule_offsets[worker]=q;
      for(unsigned task=worker;task<2*kVirtual;task+=kVirtual)schedule[q++]={0,task,0,0,w,0,0};
      for(unsigned task=worker;task<unsigned(2*kVirtual*split);task+=kVirtual) {
        auto& ref=schedule[q++];ref={2,task,0,2,w,0,0};
        unsigned v=task/(2*split);
        if(TILEMEGA_EVENT_KAPPA==0) {
          waits[w++]={0,kWholeStageEventGroup};waits[w++]={1,kWholeStageEventGroup};
        }else {
        for(unsigned group=2*v/std::max(1,TILEMEGA_EVENT_KAPPA);group<CeilDiv(2*v+2,std::max(1,TILEMEGA_EVENT_KAPPA));++group)
          waits[w++]={0,group};
        waits[w++]={1,v/std::max(1,TILEMEGA_EVENT_KAPPA)};
        }
        ref.wait_count=w-ref.wait_begin;
      }
      if(split>1 && !la)for(unsigned task=worker;task<2*kVirtual;task+=kVirtual) {
        auto& ref=schedule[q++];ref={3,task,2,1,w,0,0};
        if(TILEMEGA_EVENT_KAPPA==0)waits[w++]={2,kWholeStageEventGroup};
        else for(unsigned group=task*split/std::max(1,TILEMEGA_EVENT_KAPPA);
            group<CeilDiv((task+1)*split,std::max(1,TILEMEGA_EVENT_KAPPA));++group)waits[w++]={2,group};
        ref.wait_count=w-ref.wait_begin;
      }
    }
    schedule_offsets[kVirtual]=q;
    *params={};params->dims={1,0,1,1,0};params->gemms=inv;params->stages=stages;params->stage_count=4;
    params->ownership_flags=kCombinerTileOwnership;
    params->buffers=buffers;params->dependencies=deps;params->dependency_count=offsets[4];
    params->dependency_offsets=offsets;params->dependency_intervals=intervals;
    params->event_offsets=event_offsets;params->event_flags=flags;
    params->schedule=schedule;params->schedule_offsets=schedule_offsets;params->task_waits=waits;
    params->serving_handoff_tickets=tickets;params->serving_handoff_ticket_stride=2*kVirtual;
    params->serving_watchdog=watchdog;params->serving_watchdog_ns=10'000'000'000ull;
    std::vector<std::vector<E>> l1(16);
    for(bool l2:{false,true}) {
      std::printf("case split=%d la=%d table=%d l2=%d\n",split,la,table,l2);std::fflush(stdout);
      TILEMEGA_CUDA_CHECK(cudaMemset(events,0,(4+cursor)*sizeof(EventCounter)));
      TILEMEGA_CUDA_CHECK(cudaMemset(tickets,0,4*2*kVirtual*sizeof(unsigned)));
      for(unsigned epoch=0;epoch<16;++epoch) {
        for(int row=0;row<kRows;++row)for(int k=0;k<128;++k)
          input[row*128+k]=E(((row*11+k*5+int(epoch)*7)%23-11)*0.03125f);
        TILEMEGA_CUDA_CHECK(cudaMemset(bindings,0xff,kVirtual*sizeof(MoeBindingRecord)));
        TILEMEGA_CUDA_CHECK(cudaMemset(hidden,0xff,kRows*128*sizeof(E)));
        TILEMEGA_CUDA_CHECK(cudaMemset(output,0xff,kRows*128*sizeof(E)));
        TILEMEGA_CUDA_CHECK(cudaMemset(partials,0xff,2*kRows*128*sizeof(float)));
        *errors=0;*watchdog={};
        StreamProbe<<<kVirtual,160,kShared>>>(params,events,epoch,l2,sequences,errors,tickets);
        TILEMEGA_CUDA_CHECK(cudaGetLastError());
        auto status=cudaDeviceSynchronize();
        if(status!=cudaSuccess) {
          std::fprintf(stderr,"split=%d la=%d table=%d l2=%d epoch=%u: %s\n",
              split,la,table,l2,epoch,cudaGetErrorString(status));return 4;
        }
        assert(!*errors && !watchdog->fired);
        unsigned expected_total=0;
        for(unsigned v=0;v<kVirtual;++v) {
          bool empty=Empty(v,epoch);
          for(unsigned row=0;row<16;++row)for(int n=0;n<128;++n) {
            int m=v*16+row;
            assert(hidden[m*128+n]==E(float(input[m*128+n])*0.5f));
            if(empty || row>=Rows(v,epoch)) {
              assert(std::isnan(float(output[m*128+n])));continue;
            }
            float ref=0;
            for(int k=0;k<128;++k)ref+=float(E(float(input[m*128+k])*0.5f))*float(E(Weight((v*3+epoch)%kExperts,n,k)));
            float actual=float(output[m*128+n]);
            if(!std::isfinite(actual) || std::abs(actual-ref)>0.016f+0.016f*std::abs(ref)) {
              std::fprintf(stderr,"split=%d la=%d table=%d l2=%d epoch=%u row=%d n=%d actual=%g ref=%g\n",split,la,table,l2,epoch,m,n,actual,ref);return 3;
            }
          }
        }
        // Sum page counts across workers: split task ownership may distribute
        // a virtual block's N/chunk tasks across several CTAs.
        for(unsigned v=0;v<kVirtual;++v)expected_total+=4+(Empty(v,epoch)?0:4);
        unsigned long long loader_total=0,compute_total=0;
        for(int worker=0;worker<kVirtual;++worker) {
          assert(sequences[2*worker]==sequences[2*worker+1]);
          loader_total+=sequences[2*worker];compute_total+=sequences[2*worker+1];
        }
        assert(loader_total==expected_total && compute_total==expected_total);
        if(!l2)l1[epoch].assign(output,output+kRows*128);
        else assert(std::memcmp(l1[epoch].data(),output,kRows*128*sizeof(E))==0);
        for(int i=0;i<4*2*kVirtual;++i)assert(tickets[i]==0);
        if(l2) {
          for(unsigned stage:{1u,2u}) {
            unsigned count=stage==1?kVirtual:2*kVirtual*split;
            assert(events[4+event_offsets[stage]].arrivals==std::uint64_t(count)*(epoch+1));
          }
          if(split>1)assert(events[4+event_offsets[3]].arrivals==std::uint64_t(2*kVirtual)*(epoch+1));
        }
      }
    }
  }
  std::puts("binding PageStream: parked lookahead, production page GEMMs, sparse binding waits, empty tasks, split-K and LA passed");
}
