#include <tilemega/Backend/Ascend/FaiAttentionTile.h>
#include <acl/acl.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::ascend;
constexpr unsigned kTasks=2,kEpochs=4,kCapacity=1088,kControlBytes=7*512;
template<unsigned Width> __global__ __aicore__ __mix__(1,2) __schedmode__(1)
void fai_tile_probe(GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR out,GM_ADDR control,unsigned rows,unsigned keys) {
  unsigned lane=2;
  if ASCEND_IS_AIV {lane=AscendC::GetSubBlockIdx();}
  FaiAttentionTile<Width> body;
  for(unsigned epoch=0;epoch<kEpochs;++epoch)for(unsigned task=0;task<kTasks;++task) {
    const unsigned tag=epoch*kTasks+task+1;
    auto* witness=(__gm__ uint32_t*)(control+(4+lane)*512);
    WriteControl(witness,tag);
    FaiTileTask context{q+uint64_t(task)*32*Width*2,k+uint64_t(task)*kCapacity*Width*2,
      v+uint64_t(task)*kCapacity*Width*2,out+uint64_t(tag-1)*32*Width*2,control,
      rows,Width,keys,kCapacity,tag,10000000000ULL};
    if(!body.RunTask(context,lane)){WriteControl(witness,0x80000000u|tag);return;}
  }
}
static void Check(aclError code,char const* action) {
  if(code)throw std::runtime_error(std::string(action)+": "+std::to_string(code));
}
static std::vector<uint8_t> Read(std::filesystem::path const& path,size_t bytes) {
  if(std::filesystem::file_size(path)!=bytes)throw std::invalid_argument("FAI input extent mismatch");
  std::vector<uint8_t> data(bytes);std::ifstream f(path,std::ios::binary);f.read((char*)data.data(),bytes);
  if(!f)throw std::runtime_error("FAI input read failed");return data;
}
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::invalid_argument("usage: as2_fai_tile rows width valid_keys inputs new_output");
    unsigned rows=std::stoul(argv[1]),d=std::stoul(argv[2]),keys=std::stoul(argv[3]);
    if(!fai_tile::Shape(rows,d,keys,kCapacity))throw std::invalid_argument("Unadmitted physical tile shape");
    namespace fs=std::filesystem;const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[4]),out=fs::weakly_canonical(argv[5]);
    for(auto const& p:{input,out}) {
      auto rel=p.lexically_relative(root);if(p==root || rel.empty() || *rel.begin()=="..")throw std::invalid_argument("Use controlled AS2 paths");
    }
    if(fs::exists(out))throw std::invalid_argument("Never overwrite FAI output");
    auto q=Read(input/"q.bin",kTasks*32*d*2),k=Read(input/"k.bin",kTasks*kCapacity*d*2),v=Read(input/"v.bin",kTasks*kCapacity*d*2);
    std::vector<uint8_t> result(uint64_t(kTasks)*kEpochs*32*d*2+512,0xa5),control(kControlBytes,0);
    std::cerr<<"phase: aclInit"<<std::endl;Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::invalid_argument("Require one visible device");Check(aclrtSetDevice(0),"device");
    auto* sku=aclrtGetSocName();if(!sku || std::string(sku)!="Ascend950PR_9579")throw std::invalid_argument("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> allocated;
    auto upload=[&](std::vector<uint8_t> const& data) {
      void* p=nullptr;Check(aclrtMalloc(&p,data.size(),ACL_MEM_MALLOC_HUGE_FIRST),"allocate");allocated.push_back(p);
      Check(aclrtMemcpy(p,data.size(),data.data(),data.size(),ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
    };
    auto dq=upload(q),dk=upload(k),dv=upload(v),dy=upload(result),dc=upload(control);
    std::cerr<<"phase: kernelLaunch"<<std::endl;
    if(d==64)fai_tile_probe<64><<<1,nullptr,stream>>>(dq,dk,dv,dy,dc,rows,keys);
    else fai_tile_probe<128><<<1,nullptr,stream>>>(dq,dk,dv,dy,dc,rows,keys);
    std::cerr<<"phase: kernelLaunchReturned"<<std::endl;Check(aclrtSynchronizeStream(stream),"FAI synchronize");
    std::cerr<<"phase: kernelComplete"<<std::endl;
    Check(aclrtMemcpy(result.data(),result.size(),dy,result.size(),ACL_MEMCPY_DEVICE_TO_HOST),"output");
    Check(aclrtMemcpy(control.data(),control.size(),dc,control.size(),ACL_MEMCPY_DEVICE_TO_HOST),"controls");
    fs::create_directories(out);
    for(auto const& item:{std::pair{"output.bin",&result},std::pair{"control.bin",&control}}) {
      std::ofstream f(out/item.first,std::ios::binary);f.write((char*)item.second->data(),item.second->size());if(!f)throw std::runtime_error("FAI evidence write failed");
    }
    for(unsigned i=0;i<512;++i)if(result[result.size()-512+i]!=0xa5)throw std::runtime_error("Output guard overwritten");
    uint32_t words[7];for(unsigned i=0;i<7;++i)std::memcpy(words+i,control.data()+i*512,4);
    for(unsigned i:{0u,1u,2u,4u,5u,6u})if(words[i]!=kTasks*kEpochs)throw std::runtime_error("Missing FAI owner/participant completion");
    if(words[3])throw std::runtime_error("Bounded FAI completion wait aborted");
    for(auto p:allocated)Check(aclrtFree(p),"free");Check(aclrtDestroyStream(stream),"destroy");Check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"executed_requires_numeric_compare\",\"host_kernel_launches\":1,\"groups\":1,\"tasks\":2,\"epochs\":4,\"physical_rows\":32,\"logical_rows\":"<<rows
      <<",\"width\":"<<d<<",\"valid_keys\":"<<keys<<",\"kv_block\":128,\"declared_score_or_probability_GM_bytes\":0,\"declared_FP32_KV_expansion_bytes\":0,\"whole_operator_scheduler\":false,\"model_admitted\":false}"<<std::endl;
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
