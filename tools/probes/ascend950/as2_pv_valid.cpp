#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tilemega::ascend;
constexpr unsigned kCapacity=1088;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void pv_valid(GM_ADDR p,GM_ADDR v,GM_ADDR poison,GM_ADDR full,GM_ADDR short_,
    GM_ADDR guarded,GM_ADDR timing,unsigned rows,unsigned width,unsigned keys) {
  if ASCEND_IS_AIC {
    using Body=GemmTile<float,float,32,128,64,false>;
    Body::SharedResource resource;
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    AscendC::GlobalTensor<uint64_t> times;times.SetGlobalBuffer((__gm__ uint64_t*)timing);
    const unsigned rounded=((keys+63)/64)*64;
    for(unsigned epoch=0;epoch<4;++epoch) {
      for(unsigned recipe=0;recipe<3;++recipe) {
        Body::TileMmad mm(resource);auto begin=AscendC::GetSystemCycle();
        for(unsigned tile=0;tile<(rows+31)/32;++tile) {
          Body::RunKSliceBorrowed(mm,p,recipe==2?poison:v,
              recipe==0?full:recipe==1?short_:guarded,rows,width,kCapacity,0,
              recipe==0?kCapacity:rounded,tile,0);
          CompleteDma<AscendC::HardEvent::FIX_S>();
        }
        times.SetValue(epoch*8+recipe,AscendC::GetSystemCycle()-begin);
      }
    }
    AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(times);
  }
}
static void Check(aclError code,char const* operation) {
  if(code)throw std::runtime_error(std::string(operation)+": "+std::to_string(code));
}
static std::vector<float> Read(std::filesystem::path const& path,size_t count) {
  if(std::filesystem::file_size(path)!=count*4)throw std::invalid_argument("Wrong FP32 input extent");
  std::vector<float> v(count);std::ifstream file(path,std::ios::binary);
  file.read(reinterpret_cast<char*>(v.data()),count*4);if(!file)throw std::runtime_error("Read failed");return v;
}
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::invalid_argument("usage: as2_pv_valid rows width valid_keys input_directory new_output");
    const unsigned rows=std::stoul(argv[1]),width=std::stoul(argv[2]),keys=std::stoul(argv[3]);
    if(!rows || rows>64 || (width!=64 && width!=128) || !keys || keys>=kCapacity)
      throw std::invalid_argument("Require bounded actual GQA/prefill FP32 PV shape");
    namespace fs=std::filesystem;const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[4]),output=fs::weakly_canonical(argv[5]);
    auto inside=[&](auto const& p){auto relative=p.lexically_relative(root);return !relative.empty() && *relative.begin()!=".." && p!=root;};
    if(!inside(input)||!inside(output)||fs::exists(output))throw std::invalid_argument("Use controlled existing inputs/new output");
    auto p=Read(input/"p.bin",size_t(rows)*kCapacity),v=Read(input/"v.bin",size_t(kCapacity)*width),poison=Read(input/"poison.bin",size_t(kCapacity)*width);
    std::cerr<<"phase: aclInit"<<std::endl;
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"device count");
    std::cerr<<"phase: aclrtSetDevice"<<std::endl;
    if(count!=1)throw std::runtime_error("Require one visible device");Check(aclrtSetDevice(0),"device");
    auto* sku=aclrtGetSocName();if(!sku || std::string(sku).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    std::cerr<<"phase: allocate"<<std::endl;
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> memory;
    auto alloc=[&](size_t bytes,void const* host) {
      void* d=nullptr;Check(aclrtMalloc(&d,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");memory.push_back(d);
      Check(aclrtMemcpy(d,bytes,host,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)d;
    };
    const size_t bytes=size_t(rows)*width*4;std::vector<uint8_t> initial(bytes+512,0xa5);
    std::vector<uint64_t> timing(64,0);
    auto dp=alloc(p.size()*4,p.data()),dv=alloc(v.size()*4,v.data()),dpoison=alloc(poison.size()*4,poison.data());
    auto df=alloc(initial.size(),initial.data()),ds=alloc(initial.size(),initial.data()),dg=alloc(initial.size(),initial.data()),dt=alloc(timing.size()*8,timing.data());
    std::cerr<<"phase: kernelLaunch"<<std::endl;
    pv_valid<<<1,nullptr,stream>>>(dp,dv,dpoison,df,ds,dg,dt,rows,width,keys);
    std::cerr<<"phase: kernelLaunchReturned"<<std::endl;
    Check(aclrtSynchronizeStream(stream),"synchronize");
    std::cerr<<"phase: kernelComplete"<<std::endl;
    std::vector<std::vector<uint8_t>> results(3,initial);
    unsigned index=0;for(auto d:{df,ds,dg}) {
      Check(aclrtMemcpy(results[index].data(),initial.size(),d,initial.size(),ACL_MEMCPY_DEVICE_TO_HOST),"download");
      if(!std::all_of(results[index].begin()+bytes,results[index].end(),[](auto b){return b==0xa5;}))throw std::runtime_error("Output guard changed");
      ++index;
    }
    Check(aclrtMemcpy(timing.data(),timing.size()*8,dt,timing.size()*8,ACL_MEMCPY_DEVICE_TO_HOST),"timing");
    if(results[0]!=results[1] || results[0]!=results[2])throw std::runtime_error("Full/valid/poisoned-tail FP32 PV differs bitwise");
    for(size_t offset=0;offset<bytes;offset+=4) {
      float value;std::memcpy(&value,results[0].data()+offset,4);if(!std::isfinite(value))throw std::runtime_error("Nonfinite PV output");
    }
    fs::create_directory(output);index=0;
    for(auto name:{"full.bin","valid.bin","poison.bin"}) {
      std::ofstream file(output/name,std::ios::binary);file.write((char*)results[index++].data(),bytes);
    }
    std::cout<<"{\"status\":\"pass\",\"rows\":"<<rows<<",\"width\":"<<width<<",\"valid_keys\":"<<keys<<",\"physical_stride\":1088,\"rounded_k\":"<<((keys+63)/64)*64<<",\"epochs\":4,\"groups\":1,\"host_kernel_launches\":1,\"bitwise_equal\":true,\"guard_pass\":true,\"ticks\":[";
    for(unsigned i=0;i<4;++i)for(unsigned j=0;j<3;++j){if(i||j)std::cout<<',';std::cout<<timing[i*8+j];}
    std::cout<<"]}"<<std::endl;
    for(auto d:memory)Check(aclrtFree(d),"free");Check(aclrtDestroyStream(stream),"stream destroy");Check(aclFinalize(),"finalize");return 0;
  }catch(std::exception const& error){std::cerr<<error.what()<<std::endl;return 1;}
}
