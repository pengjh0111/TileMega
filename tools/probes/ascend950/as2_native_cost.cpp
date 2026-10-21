#include <tilemega/Target/Ascend/LlmExecutor.h>
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::ascend;
constexpr unsigned kGroups=8,kWorkers=24,kRepeats=32,kBlock=16384;
constexpr uint64_t kPerLane=16ULL<<20,kDeadline=2000000000ULL;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void native_cost(GM_ADDR table,GM_ADDR control,GM_ADDR source,GM_ADDR dest,GM_ADDR timing) {
  auto worker=GetWorkerId(kGroups);auto* cells=(__gm__ uint32_t*)control;
  auto* abort=cells+kWorkers*128;AscendC::GlobalTensor<uint8_t> metadata;
  metadata.SetGlobalBuffer(table);uint64_t dispatch=0,roundtrip=0;
  for(unsigned i=1;i<=kRepeats;++i) {
    auto begin=AscendC::GetSystemCycle();AcquireData(metadata);
    auto stage=ReadLlmPod<llm::Stage>(table);
    if(stage.width!=128){WriteControl(abort,4);return;}
    PublishSingleOwnerEvent(cells+worker*128,i);
    dispatch+=AscendC::GetSystemCycle()-begin;
  }
  // The round trip uses the same single-writer GM frontier protocol as the
  // executor. Half a round trip is an empirical symmetric estimate, not a
  // claim that opposite directions have identical one-way latency.
  auto* ping=cells+(kWorkers+1)*128;auto* ack=cells+(kWorkers+2)*128;
  if(worker==0) {
    for(unsigned i=1;i<=kRepeats;++i) {
      auto begin=AscendC::GetSystemCycle();PublishSingleOwnerEvent(ping,i);
      if(!WaitProgress(ack,i,abort,kDeadline))return;
      roundtrip+=AscendC::GetSystemCycle()-begin;
    }
  }else if(worker==kGroups+2) {
    for(unsigned i=1;i<=kRepeats;++i) {
      if(!WaitProgress(ping,i,abort,kDeadline))return;
      PublishSingleOwnerEvent(ack,i);
    }
  }
  PublishSingleOwnerEvent(cells+worker*128,kRepeats+1);
  for(unsigned w=0;w<kWorkers;++w)
    if(!WaitProgress(cells+w*128,kRepeats+1,abort,kDeadline))return;
  uint64_t copy_ticks=0;
  if ASCEND_IS_AIV {
    AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer,kBlock*4);auto local=buffer.Get<float>();
    AscendC::GlobalTensor<float> src,dst;
    auto lane=worker-kGroups;
    src.SetGlobalBuffer((__gm__ float*)(source+lane*kPerLane));
    dst.SetGlobalBuffer((__gm__ float*)(dest+lane*kPerLane));
    auto begin=AscendC::GetSystemCycle();
    for(unsigned i=0;i<kPerLane/4;i+=kBlock) {
      AscendC::DataCopy(local,src[i],kBlock);
      CompleteDma<AscendC::HardEvent::MTE2_MTE3>();
      AscendC::DataCopy(dst[i],local,kBlock);
      CompleteDma<AscendC::HardEvent::MTE3_MTE2>();
    }
    CompleteDma<AscendC::HardEvent::MTE3_S>();
    copy_ticks=AscendC::GetSystemCycle()-begin;
  }
  AscendC::GlobalTensor<uint64_t> times;
  times.SetGlobalBuffer((__gm__ uint64_t*)(timing+worker*512));
  times.SetValue(0,dispatch);times.SetValue(1,roundtrip);times.SetValue(2,copy_ticks);
  AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(times);
}
static void Check(aclError value,char const* call) {
  if(value)throw std::runtime_error(std::string(call)+": "+std::to_string(value));
}
int main() {
  try {
    Check(aclInit(nullptr),"init");uint32_t devices=0;Check(aclrtGetDeviceCount(&devices),"device count");
    if(devices!=1)throw std::runtime_error("Require one visible device");
    Check(aclrtSetDevice(0),"device");auto* soc=aclrtGetSocName();
    if(!soc || std::string(soc).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> memory;
    auto allocate=[&](size_t bytes,unsigned value=0) {
      void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");memory.push_back(p);
      Check(aclrtMemset(p,bytes,value,bytes),"initialize");return (uint8_t*)p;
    };
    llm::Stage stage{};stage.width=128;
    auto table=allocate(sizeof(stage)),control=allocate((kWorkers+3)*512),times=allocate(kWorkers*512);
    constexpr uint64_t bytes=2*kGroups*kPerLane;
    auto source=allocate(bytes,0x3f),dest=allocate(bytes,0xa5);
    Check(aclrtMemcpy(table,sizeof(stage),&stage,sizeof(stage),ACL_MEMCPY_HOST_TO_DEVICE),"upload table");
    native_cost<<<kGroups,nullptr,stream>>>(table,control,source,dest,times);
    Check(aclrtSynchronizeStream(stream),"synchronize");
    std::array<uint64_t,kWorkers*64> timing{};
    Check(aclrtMemcpy(timing.data(),sizeof(timing),times,sizeof(timing),ACL_MEMCPY_DEVICE_TO_HOST),"read timing");
    uint32_t abort=0;Check(aclrtMemcpy(&abort,4,control+kWorkers*512,4,ACL_MEMCPY_DEVICE_TO_HOST),"read abort");
    if(abort)throw std::runtime_error("Finite calibration barrier/roundtrip aborted");
    unsigned samples=0;
    for(uint64_t offset=0;offset<bytes;offset+=kBlock*4) {
      for(auto delta:{uint64_t(0),uint64_t(kBlock*4-4)}) {
        uint32_t actual=0;Check(aclrtMemcpy(&actual,4,dest+offset+delta,4,ACL_MEMCPY_DEVICE_TO_HOST),"sample payload");
        if(actual!=0x3f3f3f3f)throw std::runtime_error("Copy payload sample mismatch");++samples;
      }
    }
    std::cout<<"{\"schema\":\"tilemega.as2.native_cost_probe.v1\",\"groups\":8,\"workers\":24,\"repeats\":32,\"dispatcher_ticks_sum\":[";
    for(unsigned w=0;w<kWorkers;++w){if(w)std::cout<<',';std::cout<<timing[w*64];}
    std::cout<<"],\"aiv_copy_ticks\":[";
    for(unsigned w=kGroups;w<kWorkers;++w){if(w!=kGroups)std::cout<<',';std::cout<<timing[w*64+2];}
    std::cout<<"],\"roundtrip_ticks_sum\":"<<timing[1]<<",\"copy_logical_read_write_bytes\":"<<2*bytes
      <<",\"copy_working_set_bytes\":"<<2*bytes<<",\"payload_samples_passed\":"<<samples
      <<",\"host_kernel_launches\":1,\"device_abort\":0,\"status\":\"passed\"}"<<std::endl;
    for(auto p:memory)Check(aclrtFree(p),"free");Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");return 0;
  }catch(std::exception const& error){std::cerr<<error.what()<<std::endl;return 1;}
}
