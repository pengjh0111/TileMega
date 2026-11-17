// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/Sm90DenseExplicit.cuh>

namespace tilemega::codegen::native_dense {
struct StaticTaskEntry {
  int op,layer,coordinate,wait_begin,wait_count,publish_begin,publish_count;
};
struct StaticTaskRange {int begin,count;};

template<unsigned Buffers,unsigned Maps,unsigned Layers,unsigned Slots,class Ops,class Plan,bool Embed>
struct StaticTaskProgram {
  using Binding=Bindings<Buffers,Maps,Layers,Slots>;
  using RoleLaunch=backend::Sm90DenseRoleLaunch<160>;
  using Head=std::tuple_element_t<std::tuple_size_v<Ops>-1,Ops>;
  template<int>using GroupStage=std::tuple_element_t<1,Ops>;
  static constexpr int kBatch=1,kCapacity=1088,kResidency=1,kThreads=160,kCluster=1,kTp=1;
  static constexpr bool kEmbed=Embed,kPostHead=false;
  static constexpr unsigned kBuffers=Buffers,kMaps=Maps,kLayers=Layers;
  static constexpr unsigned kStages=Plan::kUnits,kCounters=Plan::kEvents+1;
  static constexpr int kSharedBytes=232448,kHeadMaxGrid=Head::kMaxGrid;

  template<unsigned I=0>
  __device__ static void Dispatch(int op,Binding const& b,int layer,int past,char* shared,float epsilon) {
    if constexpr(I<std::tuple_size_v<Ops>) {
      if(op==I) {
        using Stage=std::tuple_element_t<I,Ops>;
        auto& r=backend::Sm90DenseOperandHeader::Bank(shared,0);
        auto& ring=*reinterpret_cast<backend::Sm90DenseRingState*>(shared);
        if constexpr(Stage::kUsesOperandRecord) {
          if(threadIdx.x==0)Stage::PublishOperands(b,layer,past,epsilon,r);
          __syncthreads();
          Stage::template Run<0>(b,past,shared,epsilon,ring,backend::Sm90DenseNoTail{},r,false);
        }else Stage::template Run<0>(b,past,shared,epsilon,ring,backend::Sm90DenseNoTail{});
      }else Dispatch<I+1>(op,b,layer,past,shared,epsilon);
    }
  }
  template<bool Events>
  __device__ static void Execute(StaticTaskEntry const& e,Binding const& b,int past,char* shared,
      float epsilon,GridCounter* events,unsigned long long iteration,unsigned absolute_step) {
    if(threadIdx.x==0) {
      if constexpr(Events)for(int i=0;i<e.wait_count;++i) {
        int id=Plan::Wait(e.wait_begin+i);auto* address=&events[id].arrivals;
        unsigned long long expected=(iteration+1)*Plan::EventCount(id),seen;
        do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(address):"memory");}
        while(seen<expected);
      }
      auto& c=backend::Sm90DenseExplicitTasks::Context();
      if(e.coordinate<0) {
        auto r=Plan::HeadRange(blockIdx.x);c={Plan::HeadCoordinates()+r.begin,0,r.count};
      }else c={nullptr,e.coordinate,1};
    }
    __syncthreads();
    // Only the initial request token needs a prologue when head already
    // contains embedding. Its completion event remains one per iteration.
    if(!Embed || e.op!=0 || absolute_step==0)Dispatch(e.op,b,e.layer,past,shared,epsilon);
    if constexpr(Events) {
      if(threadIdx.x==0)for(int i=0;i<e.publish_count;++i) {
        int id=Plan::Publish(e.publish_begin+i);auto* address=&events[id].arrivals;
        asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(address),"l"(1ull):"memory");
      }
    }
    __syncthreads();
  }
  __device__ static void Step(Binding const& b,int past,char* shared,float epsilon,
      GridCounter* events,unsigned long long iteration,TraceState,unsigned absolute_step) {
    auto range=Plan::Track(blockIdx.x);
    for(int i=0;i<range.count;++i)
      Execute<true>(Plan::Entry(range.begin+i),b,past,shared,epsilon,events,iteration,absolute_step);
    // Head's final arriving CTA may still be writing the next embedding.
    // All tracks must acquire that write before the next iteration reuses x.
    if(threadIdx.x==0) {
      auto* address=&events[Plan::kEvents].arrivals;
      asm volatile("red.release.gpu.global.add.u64 [%0], %1;"::"l"(address),"l"(1ull):"memory");
      unsigned long long seen;
      do {asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(seen):"l"(address):"memory");}
      while(seen<(iteration+1)*gridDim.x);
    }
    __syncthreads();
  }
  __device__ static void Unit(unsigned unit,Binding const& b,int past,char* shared,float epsilon,
      unsigned absolute_step) {
    auto range=Plan::Unit(unit,blockIdx.x);
    for(int i=0;i<range.count;++i)
      Execute<false>(Plan::Entry(range.begin+i),b,past,shared,epsilon,nullptr,0,absolute_step);
  }
};
} // namespace tilemega::codegen::native_dense
