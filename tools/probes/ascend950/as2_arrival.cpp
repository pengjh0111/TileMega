#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

constexpr uint64_t kWaitTicks=2000000000ULL;
constexpr unsigned kFieldWords=128,kFields=10,kRecordWords=kFieldWords*kFields;
__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_arrival(GM_ADDR memory,unsigned groups) {
  using namespace tilemega::ascend;
  unsigned worker=GetWorkerId(groups),workers=3*groups;
  if(worker>=workers)return;
  auto* base=(__gm__ uint32_t*)memory;auto* own=base+worker*kRecordWords;
  const auto started=AscendC::GetSystemCycle();
  WriteControl(own+3*kFieldWords,worker);WriteControl(own+4*kFieldWords,AscendC::GetBlockIdx());
  WriteControl(own+5*kFieldWords,AscendC::GetSubBlockIdx());
  WriteControl(own+6*kFieldWords,uint32_t(started));WriteControl(own+7*kFieldWords,uint32_t(started>>32));
  PublishSingleOwnerEvent(own,1);
  for(;;) {
    bool all=true,failed=false;
    for(unsigned w=0;w<workers;++w) {
      failed|=ReadControl(base+w*kRecordWords+kFieldWords)!=0;
      all&=ReadControl(base+w*kRecordWords)==1;
    }
    // Abort is per actor: even the failure path has no multi-writer word.
    if(failed){PublishSingleOwnerEvent(own+kFieldWords,2);break;}
    if(AscendC::GetSystemCycle()-started>=kWaitTicks) {
      PublishSingleOwnerEvent(own+kFieldWords,1);break;
    }
    if(all){PublishSingleOwnerEvent(own+2*kFieldWords,1);break;}
    AscendC::Nop<8>();
  }
  const auto finished=AscendC::GetSystemCycle();
  WriteControl(own+8*kFieldWords,uint32_t(finished));WriteControl(own+9*kFieldWords,uint32_t(finished>>32));
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}
static void Check(aclError e,char const* what) {
  if(e)throw std::runtime_error(std::string(what)+": "+std::to_string(e));
}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::invalid_argument("usage: as2_arrival groups new_output");
    auto parsed=std::stoul(argv[1]);
    if(parsed!=8 && parsed!=14)throw std::invalid_argument("Bounded G8/G14 only");
    unsigned groups=parsed,workers=3*groups;
    namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto out=fs::weakly_canonical(argv[2]);auto relative=out.lexically_relative(root);
    if(out==root || relative.empty() || *relative.begin()==".." || fs::exists(out) || !fs::is_directory(out.parent_path()))
      throw std::invalid_argument("Require a new controlled output and an existing parent");
    std::vector<uint32_t> data(workers*kRecordWords+kFieldWords,0);
    std::fill(data.end()-128,data.end(),0xa5a5a5a5);
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::runtime_error("Use one visible device");
    Check(aclrtSetDevice(0),"device");auto* soc=aclrtGetSocName();
    if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::runtime_error("Wrong SKU");
    int64_t cube=0,vector=0;
    Check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_CUBE_CORE_NUM,&cube),"cube count");
    Check(aclrtGetDeviceInfo(0,ACL_DEV_ATTR_VECTOR_CORE_NUM,&vector),"vector count");
    if(cube<groups || vector<2*groups)throw std::runtime_error("Requested actors exceed device counts");
    void* memory=nullptr;Check(aclrtMalloc(&memory,data.size()*4,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");
    Check(aclrtMemcpy(memory,data.size()*4,data.data(),data.size()*4,ACL_MEMCPY_HOST_TO_DEVICE),"upload");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");
    as2_arrival<<<groups,nullptr,stream>>>((GM_ADDR)memory,groups);
    Check(aclrtSynchronizeStream(stream),"synchronize");
    Check(aclrtMemcpy(data.data(),data.size()*4,memory,data.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"download");
    Check(aclrtFree(memory),"free");Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");
    bool pass=true;unsigned timed_out=0,peer_abort=0;uint64_t maximum=0;
    for(unsigned w=0;w<workers;++w) {
      auto* d=data.data()+w*kRecordWords;
      auto field=[&](unsigned i){return d[i*kFieldWords];};
      const bool aic=w<groups;unsigned raw=aic?w:w-groups;
      if(field(0)!=1 || field(3)!=w || field(4)!=raw || field(5)!=(aic?0:raw%2))
        throw std::runtime_error("Actor coverage/mapping failed");
      if(field(1)>2 || field(2)>1)throw std::runtime_error("Invalid arrival state");
      timed_out+=field(1)==1;peer_abort+=field(1)==2;pass&=field(1)==0 && field(2)==1;
      uint64_t start=uint64_t(field(6))+(uint64_t(field(7))<<32),finish=uint64_t(field(8))+(uint64_t(field(9))<<32);
      if(finish<start)throw std::runtime_error("Nonmonotone actor timer");
      maximum=std::max(maximum,finish-start);
      for(unsigned f=0;f<kFields;++f)for(unsigned i=1;i<kFieldWords;++i)
        if(d[f*kFieldWords+i])throw std::runtime_error("Actor guard overwritten");
    }
    if(!std::all_of(data.end()-128,data.end(),[](auto v){return v==0xa5a5a5a5;}))
      throw std::runtime_error("Tail guard overwritten");
    fs::create_directory(out);std::ofstream binary(out/"actors.bin",std::ios::binary);
    binary.write((char*)data.data(),data.size()*4);binary.close();if(!binary)throw std::runtime_error("Cannot save actor records");
    std::ofstream result(out/"result.json");result<<"{\"status\":\""<<(pass?"pass":"bounded_abort")
      <<"\",\"groups\":"<<groups<<",\"workers\":"<<workers<<",\"deadline_ticks\":"<<kWaitTicks
      <<",\"self_timeouts\":"<<timed_out<<",\"peer_aborts\":"<<peer_abort
      <<",\"maximum_actor_elapsed_ticks\":"<<maximum
      <<",\"host_kernel_launches\":1,\"per_actor_single_writer_abort\":true,\"cube_or_DMA_payload\":false,"
        "\"scope\":\"bounded scalar-wait arrival liveness; not physical-core affinity or payload visibility\",\"model_G14_enabled\":false}\n";
    if(!result)throw std::runtime_error("Cannot save arrival result");return pass?0:2;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
