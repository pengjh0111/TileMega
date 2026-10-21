// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmPlanABI.h>
#include <tilemega/Target/Ascend/ProgressPrimitives.h>

namespace tilemega::ascend {
template<class T> __aicore__ inline T ReadLlmPod(GM_ADDR address,unsigned index=0) {
  static_assert(sizeof(T)%4==0);
  AscendC::GlobalTensor<uint32_t> source;source.SetGlobalBuffer((__gm__ uint32_t*)address);
  T value{};auto* bytes=(unsigned char*)&value;
  // Access the object representation through characters. Writing uint32_t*
  // into uint64_t/float fields violates C++ aliasing at optimized device builds.
  for(unsigned i=0;i<sizeof(T)/4;++i) {
    uint32_t word=source.GetValue(index*(sizeof(T)/4)+i);
    __builtin_memcpy(bytes+4*i,&word,4);
  }
  return value;
}
// Body owns a worker-lifetime pipeline. Scalar plan/control reads and DMA
// payloads keep the AS-1 visibility contract; no cache operations are removed.
template<bool Cube,class Body> __aicore__ inline void ExecuteLlm(Body& body,
    GM_ADDR arena,GM_ADDR metadata,GM_ADDR stages,GM_ADDR tasks,GM_ADDR waits,
    GM_ADDR offsets,GM_ADDR queue,GM_ADDR controls,GM_ADDR scratch,GM_ADDR trace,
    llm::Forward forward) {
  auto m=ReadLlmPod<llm::Metadata>(metadata);unsigned worker=GetWorkerId(m.groups);
  auto* cells=(__gm__ uint32_t*)controls;
  auto* abort=cells+(m.workers+2*m.groups)*128;
  if(m.version!=llm::kVersion || m.workers!=m.groups*3 || worker>=m.workers ||
      (m.seq!=1 && m.seq!=64) || uint64_t(forward.past)+m.seq>=m.capacity || !m.epoch_stride || !m.wait_ticks ||
      uint64_t(forward.epoch)*m.epoch_stride+m.epoch_stride-1>UINT32_MAX) {
    WriteControl(abort,2);return;
  }
  unsigned begin=ReadLlmPod<uint32_t>(offsets,worker),end=ReadLlmPod<uint32_t>(offsets,worker+1);
  if(begin>end || end>m.queue_refs){WriteControl(abort,3);return;}
  uint32_t base=forward.epoch*m.epoch_stride;
  if(begin==end)PublishSingleOwnerEvent(cells+worker*128,base);
  for(unsigned slot=begin;slot<end;++slot) {
    unsigned id=ReadLlmPod<uint32_t>(queue,slot);
    if(id>=m.tasks){WriteControl(abort,4);return;}
    auto task=ReadLlmPod<llm::Task>(tasks,id);
    if(task.stage>=m.stages || task.owner>=m.workers || !task.ordinal || task.ordinal>=m.epoch_stride ||
        task.wait_begin>m.waits || task.wait_count>m.waits-task.wait_begin || task.mix_aiv_lanes>1) {
      WriteControl(abort,5);return;
    }
    bool participant=task.mix_aiv_lanes && !Cube && worker==m.groups+2*task.owner;
    if((task.owner!=worker && !participant) || (task.mix_aiv_lanes && task.owner>=m.groups)) {
      WriteControl(abort,6);return;
    }
    for(unsigned i=0;i<task.wait_count;++i) {
      auto wait=ReadLlmPod<llm::Wait>(waits,task.wait_begin+i);
      if(wait.worker>=m.workers || !wait.prefix || wait.prefix>=m.epoch_stride){WriteControl(abort,7);return;}
      if(wait.worker!=worker && !WaitProgress(cells+wait.worker*128,base+wait.prefix,abort,m.wait_ticks))return;
    }
    if(ReadControl(abort))return;
    // The entire-cache acquire is the measured stale-cache-safe baseline.
    // It also invalidates scalar reads of recycled per-forward plan-adjacent data.
    AscendC::GlobalTensor<uint8_t> payload;payload.SetGlobalBuffer(arena);AcquireData(payload);
    auto stage=ReadLlmPod<llm::Stage>(stages,task.stage);
    const uint64_t start=AscendC::GetSystemCycle();bool ok=false;
    uint64_t body_start=0,body_end=0;
    unsigned owner_group=task.mix_aiv_lanes?task.owner:0;
    auto* ready=cells+(m.workers+owner_group)*128;
    auto* done=cells+(m.workers+m.groups+owner_group)*128;
    if constexpr(Cube) {
      if(forward.diagnostic==2)body_start=AscendC::GetSystemCycle();
      ok=body.Run(stage,task.tile,arena);
      if(forward.diagnostic==2)body_end=AscendC::GetSystemCycle();
      if(ok && task.mix_aiv_lanes) {
        PublishSingleOwnerEvent(ready,base+task.ordinal);
        if(!WaitProgress(done,base+task.ordinal,abort,m.wait_ticks))return;
      }
    }else {
      if(participant) {
        if(!WaitProgress(ready,base+task.ordinal,abort,m.wait_ticks))return;
        AcquireData(payload);
        if(forward.diagnostic==2)body_start=AscendC::GetSystemCycle();
        ok=body.Epilogue(stage,task.tile,arena);
        if(forward.diagnostic==2)body_end=AscendC::GetSystemCycle();
        if(ok) {
          PublishSingleOwnerEvent(done,base+task.ordinal);
          // Reverse acknowledgement prevents this lane recycling its buffers
          // or advertising a FIFO prefix before the logical owner completes.
          if(!WaitProgress(cells+task.owner*128,base+task.ordinal,abort,m.wait_ticks))return;
        }
      }else {
        if(forward.diagnostic==2)body_start=AscendC::GetSystemCycle();
        ok=body.Run(stage,task.tile,arena,scratch+uint64_t(worker)*m.worker_scratch_bytes,m,forward.past);
        if(forward.diagnostic==2)body_end=AscendC::GetSystemCycle();
      }
    }
    if(!ok){WriteControl(abort+1,worker);WriteControl(abort+2,id);WriteControl(abort+3,task.stage);WriteControl(abort+4,task.tile);WriteControl(abort,8);return;}
    if(forward.diagnostic==1 && task.owner==worker) {
      AscendC::GlobalTensor<uint64_t> record;record.SetGlobalBuffer((__gm__ uint64_t*)(trace+uint64_t(id)*64));
      record.SetValue(0,worker);record.SetValue(1,task.stage);record.SetValue(2,task.tile);
      record.SetValue(3,start);record.SetValue(4,AscendC::GetSystemCycle());record.SetValue(5,forward.epoch);
      record.SetValue(6,stage.operands[0]);record.SetValue(7,stage.operands[1]);
      AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(record);
    }
    PublishSingleOwnerEvent(cells+worker*128,base+slot-begin+1);
    if(forward.diagnostic==2) {
      // One cache line per queue reference prevents different workers sharing
      // dirty trace lines. Timers exclude dependency/ready/done waits, metadata
      // acquisition and these diagnostic writes; they are Body service only.
      AscendC::GlobalTensor<uint64_t> record;
      record.SetGlobalBuffer((__gm__ uint64_t*)(trace+uint64_t(slot)*512));
      record.SetValue(0,worker);record.SetValue(1,task.stage);record.SetValue(2,task.tile);
      record.SetValue(3,body_start);record.SetValue(4,body_end);record.SetValue(5,forward.epoch);
      record.SetValue(6,id);record.SetValue(7,participant?1:0);
      AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(record);
    }
  }
}
} // namespace tilemega::ascend
