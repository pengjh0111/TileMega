#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::ascend;
constexpr unsigned kCell=128, kElements=4096, kStride=kElements+kCell;
constexpr unsigned kEpochs=8, kPhases=12, kGlobalFields=6, kPhaseFields=8;
constexpr unsigned kFields=kGlobalFields+kPhases*kPhaseFields;
constexpr uint64_t kWaitTicks=2000000000ULL;
enum GlobalField { Barrier, Abort, CurrentPhase, CurrentEpoch, CurrentStep, WorkerId };
enum PhaseField { Ready, Ack, Armed, Finished, PreErrors, ScalarErrors, DmaErrors, OldChecksum };

struct Protocol {
  __gm__ uint32_t* state;
  unsigned groups,worker,workers;
  __aicore__ inline __gm__ uint32_t* Global(unsigned f,unsigned w) const {
    return state+(f*workers+w)*kCell;
  }
  __aicore__ inline __gm__ uint32_t* Cell(unsigned phase,unsigned f,unsigned w) const {
    return Global(kGlobalFields+phase*kPhaseFields+f,w);
  }
  __aicore__ inline bool Wait(__gm__ uint32_t* address,unsigned required,unsigned step) const {
    WriteControl(Global(CurrentStep,worker),step);
    const auto start=AscendC::GetSystemCycle();unsigned polls=0;
    for(;;) {
      // Each abort cell has exactly one writer, including timeout propagation.
      if((polls++&63u)==0) {
        for(unsigned w=0;w<workers;++w)if(ReadControl(Global(Abort,w))) {
          PublishSingleOwnerEvent(Global(Abort,worker),2);return false;
        }
        if(AscendC::GetSystemCycle()-start>=kWaitTicks) {
          PublishSingleOwnerEvent(Global(Abort,worker),1);return false;
        }
      }
      if(ReadControl(address)>=required) {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();return true;
      }
      AscendC::Nop<8>();
    }
  }
  __aicore__ inline bool Rendezvous(unsigned sequence) const {
    PublishSingleOwnerEvent(Global(Barrier,worker),sequence);
    for(unsigned w=0;w<workers;++w)if(!Wait(Global(Barrier,w),sequence,1))return false;
    return true;
  }
};

__aicore__ inline float Expected(unsigned worker,unsigned epoch,unsigned i) {
  return float(64*(worker+1)+8*epoch+i%7);
}
__aicore__ inline GM_ADDR PayloadSlot(GM_ADDR payload,unsigned phase,unsigned workers,unsigned worker) {
  return payload+uint64_t(phase*workers+worker)*kStride*sizeof(float);
}

struct CubeDma {
  GemmTile<float,float> tile;
  GM_ADDR identity;
  __aicore__ explicit CubeDma(GM_ADDR id):identity(id) {}
  __aicore__ inline void Copy(GM_ADDR input,GM_ADDR output,unsigned elements) {
    const unsigned rows=elements/32;
    for(unsigned t=0;t<(rows+31)/32;++t) {
      tile.RunTask(input,identity,output,rows,32,32,t,0);
      CompleteDma<AscendC::HardEvent::FIX_S>();
    }
  }
  __aicore__ inline unsigned Consume(GM_ADDR input,GM_ADDR output,unsigned elements,
                                     unsigned producer,unsigned epoch) {
    // The read goes through CATLASS MTE2/L1/MMAD, independently of Scalar reads.
    Copy(input,output,elements);
    AscendC::GlobalTensor<float> result;result.SetGlobalBuffer((__gm__ float*)output);
    AcquireData(result);unsigned errors=0;
    for(unsigned i=0;i<elements;++i)errors+=result.GetValue(i)!=Expected(producer,epoch,i);
    return errors;
  }
};

struct VectorDma {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
  __aicore__ VectorDma() { pipe.InitBuffer(buffer,kElements*sizeof(float)); }
  __aicore__ inline void Copy(GM_ADDR input,GM_ADDR output,unsigned elements) {
    AscendC::GlobalTensor<float> in,out;in.SetGlobalBuffer((__gm__ float*)input);
    out.SetGlobalBuffer((__gm__ float*)output);auto local=buffer.Get<float>();
    AscendC::DataCopy(local,in,elements);CompleteDma<AscendC::HardEvent::MTE2_S>();
    CompleteDma<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopy(out,local,elements);CompleteDma<AscendC::HardEvent::MTE3_S>();
  }
  __aicore__ inline unsigned Consume(GM_ADDR input,GM_ADDR output,unsigned elements,
                                     unsigned producer,unsigned epoch) {
    AscendC::GlobalTensor<float> in,out;in.SetGlobalBuffer((__gm__ float*)input);
    out.SetGlobalBuffer((__gm__ float*)output);auto local=buffer.Get<float>();
    AscendC::DataCopy(local,in,elements);CompleteDma<AscendC::HardEvent::MTE2_S>();
    unsigned errors=0;
    for(unsigned i=0;i<elements;++i)errors+=local.GetValue(i)!=Expected(producer,epoch,i);
    CompleteDma<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopy(out,local,elements);CompleteDma<AscendC::HardEvent::MTE3_S>();
    return errors;
  }
};

template<class Dma>
__aicore__ inline void RunProtocol(Dma& dma,GM_ADDR values,GM_ADDR payload,GM_ADDR consumed,
                                   Protocol const& p,unsigned abort_worker) {
  WriteControl(p.Global(WorkerId,p.worker),p.worker);
  const bool aic=p.worker<p.groups;
  const unsigned group=aic?p.worker:(p.worker-p.groups)/2;
  const unsigned lane=aic?0:(p.worker-p.groups)%2;
  const unsigned prev=(group+p.groups-1)%p.groups,next=(group+1)%p.groups;
  for(unsigned phase=0;phase<kPhases;++phase) {
    WriteControl(p.Global(CurrentPhase,p.worker),phase);
    WriteControl(p.Global(CurrentEpoch,p.worker),0);
    if(!p.Rendezvous(2*phase+1))return;
    if(phase==0 && p.worker==abort_worker) {
      PublishSingleOwnerEvent(p.Global(Abort,p.worker),4);return;
    }
    const unsigned direction=phase%4,size=phase/4;
    const unsigned elements=size==0?32:size==1?1056:kElements;
    unsigned inputs[2],outputs[2],ni=0,no=0;
    if(direction==0 && aic){inputs[ni++]=prev;outputs[no++]=next;}
    if(direction==1) {
      if(aic){outputs[no++]=p.groups+2*next;outputs[no++]=p.groups+2*next+1;}
      else inputs[ni++]=prev;
    }
    if(direction==2) {
      if(aic){inputs[ni++]=p.groups+2*prev;inputs[ni++]=p.groups+2*prev+1;}
      else outputs[no++]=next;
    }
    if(direction==3 && !aic) {
      inputs[ni++]=p.groups+2*prev+1-lane;
      outputs[no++]=p.groups+2*next+1-lane;
    }
    unsigned pre_errors=0,scalar_errors=0,dma_errors=0;
    for(unsigned epoch=1;epoch<=kEpochs;++epoch) {
      WriteControl(p.Global(CurrentEpoch,p.worker),epoch);
      unsigned checksum=0;
      for(unsigned j=0;j<ni;++j) {
        AscendC::GlobalTensor<float> data;data.SetGlobalBuffer((__gm__ float*)PayloadSlot(payload,phase,p.workers,inputs[j]));
        AcquireData(data);
        for(unsigned i=0;i<elements;++i) {
          const auto old=data.GetValue(i);checksum+=unsigned(old);
          pre_errors+=old!=(epoch==1?0.0f:Expected(inputs[j],epoch-1,i));
        }
      }
      WriteControl(p.Cell(phase,OldChecksum,p.worker),checksum);
      WriteControl(p.Cell(phase,PreErrors,p.worker),pre_errors);
      if(pre_errors){PublishSingleOwnerEvent(p.Global(Abort,p.worker),3);return;}
      if(ni)PublishSingleOwnerEvent(p.Cell(phase,Armed,p.worker),epoch);
      for(unsigned j=0;j<no;++j)
        if(!p.Wait(p.Cell(phase,Ack,outputs[j]),epoch-1,2) ||
           !p.Wait(p.Cell(phase,Armed,outputs[j]),epoch,3))return;
      if(no) {
        dma.Copy(values+uint64_t(p.worker*kEpochs+epoch-1)*kElements*sizeof(float),
                 PayloadSlot(payload,phase,p.workers,p.worker),elements);
        PublishSingleOwnerEvent(p.Cell(phase,Ready,p.worker),epoch);
      }
      for(unsigned j=0;j<ni;++j) {
        if(!p.Wait(p.Cell(phase,Ready,inputs[j]),epoch,4))return;
        AscendC::GlobalTensor<float> data;data.SetGlobalBuffer((__gm__ float*)PayloadSlot(payload,phase,p.workers,inputs[j]));
        AcquireData(data);
        for(unsigned i=0;i<elements;++i)scalar_errors+=data.GetValue(i)!=Expected(inputs[j],epoch,i);
        auto result=consumed+uint64_t((phase*p.workers+p.worker)*2+j)*kStride*sizeof(float);
        dma_errors+=dma.Consume(PayloadSlot(payload,phase,p.workers,inputs[j]),result,elements,inputs[j],epoch);
      }
      WriteControl(p.Cell(phase,ScalarErrors,p.worker),scalar_errors);
      WriteControl(p.Cell(phase,DmaErrors,p.worker),dma_errors);
      if(scalar_errors || dma_errors){PublishSingleOwnerEvent(p.Global(Abort,p.worker),3);return;}
      // Every fanout consumer acknowledges after Scalar and DMA reads complete.
      if(ni)PublishSingleOwnerEvent(p.Cell(phase,Ack,p.worker),epoch);
    }
    for(unsigned j=0;j<no;++j)if(!p.Wait(p.Cell(phase,Ack,outputs[j]),kEpochs,5))return;
    PublishSingleOwnerEvent(p.Cell(phase,Finished,p.worker),kEpochs);
    // Even empty roles participate, so phase-local resources cannot overlap.
    if(!p.Rendezvous(2*phase+2))return;
  }
}

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_message_passing(GM_ADDR identity,GM_ADDR values,GM_ADDR payload,GM_ADDR consumed,
                         GM_ADDR controls,unsigned groups,unsigned abort_worker) {
  Protocol p{(__gm__ uint32_t*)controls,groups,GetWorkerId(groups),3*groups};
  if ASCEND_IS_AIC {CubeDma dma(identity);RunProtocol(dma,values,payload,consumed,p,abort_worker);}
  else if ASCEND_IS_AIV {VectorDma dma;RunProtocol(dma,values,payload,consumed,p,abort_worker);}
}

static void Check(aclError error,char const* call) {
  if(error)throw std::runtime_error(std::string(call)+": "+std::to_string(error));
}
int main(int argc,char** argv) {
  try {
    if(argc!=3 && argc!=4)throw std::invalid_argument("usage: as2_message_passing <8|14> new_output [abort-worker]");
    const auto parsed=std::stoul(argv[1]);
    if(parsed!=8 && parsed!=14)throw std::invalid_argument("Bounded G8/G14 only");
    const unsigned groups=parsed,workers=3*groups;
    const auto fault=argc==4?std::stoul(argv[3]):workers;
    if(fault>workers || (argc==4 && fault==workers))throw std::invalid_argument("Abort worker must be a launched actor");
    namespace fs=std::filesystem;const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    const auto out=fs::weakly_canonical(argv[2]),relative=out.lexically_relative(root);
    if(out==root || relative.empty() || *relative.begin()==".." || fs::exists(out) || !fs::is_directory(out.parent_path()))
      throw std::invalid_argument("Require a new controlled output and an existing parent");
    std::vector<float> identity(1024,0),values(workers*kEpochs*kElements);
    for(unsigned i=0;i<32;++i)identity[i*32+i]=1;
    for(unsigned w=0;w<workers;++w)for(unsigned e=1;e<=kEpochs;++e)for(unsigned i=0;i<kElements;++i)
      values[(w*kEpochs+e-1)*kElements+i]=64*(w+1)+8*e+i%7;
    std::vector<uint32_t> states(kFields*workers*kCell+kCell,0);
    std::fill(states.end()-kCell,states.end(),0xa5a5a5a5);
    std::vector<uint32_t> data(kPhases*workers*kStride,0xa5a5a5a5);
    std::vector<uint32_t> readback(kPhases*workers*2*kStride,0xa5a5a5a5);
    for(unsigned phase=0;phase<kPhases;++phase)for(unsigned w=0;w<workers;++w) {
      unsigned elements=phase/4==0?32:phase/4==1?1056:kElements;
      std::fill_n(data.begin()+(phase*workers+w)*kStride,elements,0);
    }
    std::cerr<<"phase: aclInit"<<std::endl;Check(aclInit(nullptr),"init");
    uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::runtime_error("Use one visible device");
    Check(aclrtSetDevice(0),"device");const auto* soc=aclrtGetSocName();
    if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::runtime_error("Wrong SKU");
    int64_t cube=0,vector=0;
    Check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_CUBE_CORE_NUM,&cube),"cube count");
    Check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_VECTOR_CORE_NUM,&vector),"vector count");
    if(cube<groups || vector<2*groups)throw std::runtime_error("Requested actors exceed device counts");
    std::vector<void*> allocations;
    auto upload=[&](const void* source,size_t bytes){
      void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");
      allocations.push_back(p);Check(aclrtMemcpy(p,bytes,source,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
    };
    auto id=upload(identity.data(),identity.size()*4),v=upload(values.data(),values.size()*4);
    auto d=upload(data.data(),data.size()*4),r=upload(readback.data(),readback.size()*4),s=upload(states.data(),states.size()*4);
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");
    std::cerr<<"phase: kernelLaunch"<<std::endl;
    as2_message_passing<<<groups,nullptr,stream>>>(id,v,d,r,s,groups,fault);
    Check(aclrtSynchronizeStream(stream),"synchronize");std::cerr<<"phase: kernelComplete"<<std::endl;
    Check(aclrtMemcpy(states.data(),states.size()*4,s,states.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"download controls");
    Check(aclrtMemcpy(data.data(),data.size()*4,d,data.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"download payload");
    Check(aclrtMemcpy(readback.data(),readback.size()*4,r,readback.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"download consumers");
    for(auto p:allocations)Check(aclrtFree(p),"free");
    Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");
    auto field=[&](unsigned f,unsigned w){return states[(f*workers+w)*kCell];};
    unsigned aborts=0,deadlines=0,pre_errors=0,scalar_errors=0,dma_errors=0,guard_errors=0,host_errors=0,unfinished=0;
    for(unsigned w=0;w<workers;++w){aborts+=field(Abort,w)!=0;deadlines+=field(Abort,w)==1;
      host_errors+=field(WorkerId,w)!=w;
      if(fault==workers)unfinished+=field(Barrier,w)!=2*kPhases;
    }
    for(unsigned f=0;f<kFields;++f)for(unsigned w=0;w<workers;++w)for(unsigned i=1;i<kCell;++i)
      guard_errors+=states[(f*workers+w)*kCell+i]!=0;
    for(unsigned i=0;i<kCell;++i)guard_errors+=states[kFields*workers*kCell+i]!=0xa5a5a5a5;
    for(unsigned phase=0;phase<kPhases;++phase)for(unsigned w=0;w<workers;++w) {
      const unsigned elements=phase/4==0?32:phase/4==1?1056:kElements;
      pre_errors+=field(kGlobalFields+phase*kPhaseFields+PreErrors,w);
      scalar_errors+=field(kGlobalFields+phase*kPhaseFields+ScalarErrors,w);
      dma_errors+=field(kGlobalFields+phase*kPhaseFields+DmaErrors,w);
      if(fault==workers)unfinished+=field(kGlobalFields+phase*kPhaseFields+Finished,w)!=kEpochs;
      for(unsigned i=elements;i<kStride;++i){
        guard_errors+=data[(phase*workers+w)*kStride+i]!=0xa5a5a5a5;
        for(unsigned j=0;j<2;++j)guard_errors+=readback[((phase*workers+w)*2+j)*kStride+i]!=0xa5a5a5a5;
      }
    }
    const bool pass=!aborts && !pre_errors && !scalar_errors && !dma_errors && !guard_errors && !host_errors && !unfinished;
    const bool expected_abort=fault<workers && aborts==workers && !deadlines && !pre_errors && !scalar_errors && !dma_errors && !guard_errors && !host_errors;
    fs::create_directory(out);
    auto save=[&](const char* name,const std::vector<uint32_t>& bytes){
      std::ofstream file(out/name,std::ios::binary);file.write((const char*)bytes.data(),bytes.size()*4);
      file.close();if(!file)throw std::runtime_error("Cannot save evidence");
    };
    save("controls.bin",states);save("payload.bin",data);save("consumer.bin",readback);
    std::ofstream result(out/"result.json");
    result<<"{\"status\":\""<<(pass?"pass":expected_abort?"expected_abort":"fail")<<"\",\"groups\":"<<groups
      <<",\"workers\":"<<workers<<",\"phases\":"<<kPhases<<",\"epochs_per_phase\":"<<kEpochs
      <<",\"payload_elements\":[32,1056,4096],\"self_timeouts\":"<<deadlines<<",\"aborted_actors\":"<<aborts
      <<",\"injected_abort_worker\":"<<(fault<workers?std::to_string(fault):"null")
      <<",\"pre_read_errors\":"<<pre_errors<<",\"scalar_errors\":"<<scalar_errors<<",\"dma_errors\":"<<dma_errors
      <<",\"guard_errors\":"<<guard_errors<<",\"mapping_errors\":"<<host_errors<<",\"unfinished\":"<<unfinished
      <<",\"stale_scalar_preread\":true,\"consumer_DMA\":true,\"reverse_ack\":true,\"explicit_all_actor_phase_boundaries\":true,"
        "\"per_actor_single_writer_abort\":true,\"conservative_DSB_full_cache_acquire\":true,\"host_kernel_launches\":1,"
        "\"wait_deadline_ticks\":2000000000,\"directions\":[\"AIC_AIC\",\"AIC_AIV\",\"AIV_AIC\",\"AIV_AIV\"],"
        "\"cube_payload\":\"FP32 identity BlockMmad FIX\",\"aiv_payload\":\"MTE2_MTE3\",\"model_G14_enabled\":false}\n";
    if(!result)throw std::runtime_error("Cannot save result");return pass || expected_abort?0:2;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
