#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_topology(GM_ADDR output,unsigned groups,unsigned hardware_groups) {
  unsigned role=0,raw=AscendC::GetBlockIdx(),slot=raw;
  if ASCEND_IS_AIV {role=1;slot=hardware_groups+raw;}
  if(raw>=(role?2:1)*hardware_groups)return;
  AscendC::GlobalTensor<uint32_t> data;data.SetGlobalBuffer((__gm__ uint32_t*)output);
  data.SetValue(slot*128,0x514d4754);data.SetValue(slot*128+1,role);
  data.SetValue(slot*128+2,raw);data.SetValue(slot*128+3,AscendC::GetSubBlockIdx());
  data.SetValue(slot*128+4,tilemega::ascend::GetWorkerId(groups));
  AscendC::DataCacheCleanAndInvalid<uint32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(data);
}
static void Check(aclError e,char const* name) {
  if(e)throw std::runtime_error(std::string(name)+": "+std::to_string(e));
}
int main(int argc,char** argv) {
  try {
    if(argc!=4)throw std::runtime_error("usage: as2_topology groups verified_hardware_groups new_output");
    unsigned groups=std::stoul(argv[1]),hardware=std::stoul(argv[2]);
    if((groups!=8 && groups!=14) || hardware<groups || hardware>64)
      throw std::runtime_error("Require bounded G8/G14 and recorded hardware capacity");
    namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto out=fs::weakly_canonical(argv[3]);auto relative=out.lexically_relative(root);
    if(out==root || relative.empty() || *relative.begin()==".." || fs::exists(out) || !fs::is_directory(out.parent_path()))
      throw std::runtime_error("Require a new controlled output with an existing parent");
    std::vector<uint32_t> data((3*hardware+1)*128,~uint32_t(0));
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::runtime_error("Use one visible device");
    Check(aclrtSetDevice(0),"device");auto* soc=aclrtGetSocName();
    if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::runtime_error("Wrong SKU");
    void* memory=nullptr;Check(aclrtMalloc(&memory,data.size()*4,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");
    Check(aclrtMemcpy(memory,data.size()*4,data.data(),data.size()*4,ACL_MEMCPY_HOST_TO_DEVICE),"upload");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");
    as2_topology<<<groups,nullptr,stream>>>((GM_ADDR)memory,groups,hardware);
    Check(aclrtSynchronizeStream(stream),"synchronize");
    Check(aclrtMemcpy(data.data(),data.size()*4,memory,data.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"download");
    Check(aclrtFree(memory),"free");Check(aclrtDestroyStream(stream),"stream destroy");Check(aclFinalize(),"finalize");
    unsigned observed=0;bool parity=true;
    for(unsigned slot=0;slot<3*hardware;++slot) {
      unsigned role=slot>=hardware,raw=role?slot-hardware:slot;
      const bool expected=raw<(role?2:1)*groups;
      auto* cell=data.data()+slot*128;
      if(!expected) {
        for(unsigned i=0;i<128;++i)if(cell[i]!=~uint32_t(0))throw std::runtime_error("Inactive slot overwritten");
        continue;
      }
      if(cell[0]!=0x514d4754 || cell[1]!=role || cell[2]!=raw || cell[4]!=(role?groups+raw:raw))
        throw std::runtime_error("Missing actor or unexpected raw/worker mapping");
      if(role)parity&=cell[3]==raw%2;
      for(unsigned i=5;i<128;++i)if(cell[i]!=~uint32_t(0))throw std::runtime_error("Slot guard overwritten");
      ++observed;
    }
    for(unsigned i=3*hardware*128;i<data.size();++i)
      if(data[i]!=~uint32_t(0))throw std::runtime_error("Tail guard overwritten");
    fs::create_directory(out);std::ofstream binary(out/"actors.bin",std::ios::binary);
    binary.write((char*)data.data(),data.size()*4);binary.close();if(!binary)throw std::runtime_error("Cannot save actor records");
    std::ofstream result(out/"result.json");result<<"{\"status\":\"pass\",\"groups\":"<<groups
      <<",\"hardware_groups_from_inventory\":"<<hardware<<",\"observed_workers\":"<<observed
      <<",\"aiv_subblock_matches_raw_parity\":"<<(parity?"true":"false")
      <<",\"host_kernel_launches\":1,\"cross_core_waits\":false,\"cube_operations\":false,"
        "\"scope\":\"eventual ID coverage only; not simultaneous residency or payload visibility\",\"model_group_capability_enabled\":false}\n";
    if(!result)throw std::runtime_error("Cannot save topology result");return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
