#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::ascend;
constexpr unsigned kCapacity=1088,kEpochs=4;
__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void qk_phase_drain(GM_ADDR q,GM_ADDR k,GM_ADDR control,GM_ADDR candidate,
    GM_ADDR timing,unsigned rows,unsigned width,unsigned keys) {
  if ASCEND_IS_AIC {
    using Body=GemmTile<float,float,32,64,128,true>;
    Body::SharedResource resource;
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    AscendC::GlobalTensor<uint64_t> times;times.SetGlobalBuffer((__gm__ uint64_t*)timing);
    for(unsigned epoch=0;epoch<kEpochs;++epoch)for(unsigned recipe=0;recipe<2;++recipe) {
      auto begin=AscendC::GetSystemCycle();
      {
        Body::TileMmad mm(resource);
        for(unsigned row=0;row<(rows+31)/32;++row)
          for(unsigned col=0;col<(keys+63)/64;++col) {
            Body::RunBorrowed(mm,q,k,recipe?candidate:control,rows,kCapacity,width,row,col);
            if(!recipe)CompleteDma<AscendC::HardEvent::FIX_S>();
          }
        // Disjoint score tiles remain inside this phase. Only the final
        // completed payload may be published to an external softmax consumer.
        if(recipe)CompleteDma<AscendC::HardEvent::FIX_S>();
      }
      times.SetValue(epoch*2+recipe,AscendC::GetSystemCycle()-begin);
    }
    AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(times);
  }
}
static void Check(aclError code,char const* operation) {
  if(code)throw std::runtime_error(std::string(operation)+": "+std::to_string(code));
}
static std::vector<uint8_t> Read(std::filesystem::path const& path,size_t bytes) {
  if(std::filesystem::file_size(path)!=bytes)throw std::invalid_argument("Wrong QK input extent");
  std::vector<uint8_t> data(bytes);std::ifstream f(path,std::ios::binary);
  f.read(reinterpret_cast<char*>(data.data()),bytes);
  if(!f)throw std::runtime_error("Cannot read QK input");return data;
}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::invalid_argument("usage: as2_qk_phase_drain inputs new_output");
    namespace fs=std::filesystem;const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[1]),output=fs::weakly_canonical(argv[2]);
    auto inside=[&](auto const& p){auto r=p.lexically_relative(root);return p!=root && !r.empty() && *r.begin()!="..";};
    if(!inside(input)||!inside(output)||fs::exists(output))throw std::invalid_argument("Use controlled inputs and a new output");
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"device count");
    if(count!=1)throw std::runtime_error("Require one visible device");Check(aclrtSetDevice(0),"device");
    auto* sku=aclrtGetSocName();if(!sku || std::string(sku).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");fs::create_directory(output);
    std::cout<<"{\"status\":\"pass\",\"scope\":\"standalone QK phase; no CTG/Plan/model or performance admission\",\"groups\":1,\"epochs\":4,\"cases\":[";
    for(unsigned c=0;c<2;++c) {
      const unsigned rows=c?4:2,width=c?64:128,keys=c?1087:129;
      auto q=Read(input/("q"+std::to_string(c)+".bin"),size_t(rows)*width*4);
      auto k=Read(input/("k"+std::to_string(c)+".bin"),size_t(kCapacity)*width*4);
      const size_t bytes=size_t(rows)*kCapacity*4;
      std::vector<uint8_t> initial(bytes+512,0xa5);std::vector<void*> memory;
      auto allocate=[&](size_t size,void const* data) {
        void* p=nullptr;Check(aclrtMalloc(&p,size,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");memory.push_back(p);
        Check(aclrtMemcpy(p,size,data,size,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
      };
      std::vector<uint64_t> zero(64,0);
      auto dq=allocate(q.size(),q.data()),dk=allocate(k.size(),k.data());
      auto a=allocate(initial.size(),initial.data()),b=allocate(initial.size(),initial.data());
      auto t=allocate(zero.size()*8,zero.data());
      qk_phase_drain<<<1,nullptr,stream>>>(dq,dk,a,b,t,rows,width,keys);
      Check(aclrtSynchronizeStream(stream),"QK phase synchronize");
      std::vector<uint8_t> x(initial.size()),y(initial.size());
      Check(aclrtMemcpy(x.data(),x.size(),a,x.size(),ACL_MEMCPY_DEVICE_TO_HOST),"control");
      Check(aclrtMemcpy(y.data(),y.size(),b,y.size(),ACL_MEMCPY_DEVICE_TO_HOST),"candidate");
      if(x!=y) {
        for(auto const& item:{std::pair{"control",&x},std::pair{"candidate",&y}}) {
          std::ofstream failed(output/(std::string(item.first)+std::to_string(c)+".bin"),std::ios::binary);
          failed.write(reinterpret_cast<char*>(item.second->data()),item.second->size());
        }
        throw std::runtime_error("Per-tile/phase drain differs bitwise");
      }
      if(!std::all_of(x.begin()+bytes,x.end(),[](auto z){return z==0xa5;}))throw std::runtime_error("QK guard changed");
      const unsigned rounded=(keys+63)/64*64;
      for(unsigned row=0;row<rows;++row)
        for(unsigned col=rounded*4;col<kCapacity*4;++col)
          if(x[row*kCapacity*4+col]!=0xa5)throw std::runtime_error("Unconsumed capacity suffix changed");
      std::ofstream f(output/("qk"+std::to_string(c)+".bin"),std::ios::binary);
      f.write(reinterpret_cast<char*>(x.data()),bytes);if(!f)throw std::runtime_error("QK evidence write failed");
      Check(aclrtMemcpy(zero.data(),zero.size()*8,t,zero.size()*8,ACL_MEMCPY_DEVICE_TO_HOST),"timers");
      if(c)std::cout<<',';
      std::cout<<"{\"rows\":"<<rows<<",\"width\":"<<width<<",\"valid_keys\":"<<keys<<",\"bitwise_equal\":true,\"guards\":true,\"ticks\":[";
      for(unsigned i=0;i<kEpochs*2;++i){if(i)std::cout<<',';std::cout<<zero[i];}
      std::cout<<"]}";
      for(auto p:memory)Check(aclrtFree(p),"free");
    }
    std::cout<<"],\"host_kernel_launches\":2}"<<std::endl;
    Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
