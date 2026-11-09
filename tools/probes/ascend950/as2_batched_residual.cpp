#define TILEMEGA_LLM_BATCHED_RESIDUAL 1
#include <tilemega/Backend/Ascend/LlmVectorBody.h>
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tilemega::ascend;
constexpr unsigned kWorkers=4,kEpochs=4;
struct Case {unsigned rows,columns,left_stride,right_stride,output_stride;};
const std::array<Case,4> kCases{{{1,128,1024,1024,1024},{16,128,1024,1024,1024},
    {3,97,144,160,176},{16,17,32,48,64}}};

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void residual_tile_probe(GM_ADDR left,GM_ADDR right,GM_ADDR control,GM_ADDR candidate,
    GM_ADDR timing,unsigned rows,unsigned columns,unsigned ls,unsigned rs,unsigned os) {
  if ASCEND_IS_AIV {
    const unsigned worker=AscendC::GetBlockIdx();
    auto lhs=left+uint64_t(worker)*rows*ls*2,rhs=right+uint64_t(worker)*rows*rs*2;
    auto a=control+uint64_t(worker)*rows*os*2,b=candidate+uint64_t(worker)*rows*os*2;
    LlmVectorBody body;bool ok=true;
    AscendC::GlobalTensor<uint64_t> times;times.SetGlobalBuffer((__gm__ uint64_t*)timing);
    for(unsigned epoch=0;epoch<kEpochs;++epoch)for(unsigned arm=0;arm<2;++arm) {
      const unsigned recipe=arm^(epoch%2);auto start=AscendC::GetSystemCycle();
      if(recipe)ok=ok && body.ResidualTile(lhs,rhs,b,rows,columns,ls,rs,os);
      else for(unsigned r=0;r<rows;++r)
        ok=ok && body.Pointwise(lhs+uint64_t(r)*ls*2,rhs+uint64_t(r)*rs*2,
            a+uint64_t(r)*os*2,columns,false);
      times.SetValue(worker*64+1+epoch*2+recipe,AscendC::GetSystemCycle()-start);
    }
    times.SetValue(worker*64,ok?0:1);
    AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(times);
  }
}
static void Check(aclError code,char const* action) {
  if(code)throw std::runtime_error(std::string(action)+": "+std::to_string(code));
}
static std::vector<uint16_t> Read(std::filesystem::path const& path,size_t n) {
  if(std::filesystem::file_size(path)!=n*2)throw std::invalid_argument("Residual input size mismatch");
  std::vector<uint16_t> data(n);std::ifstream f(path,std::ios::binary);
  f.read(reinterpret_cast<char*>(data.data()),n*2);if(!f)throw std::runtime_error("Residual input read failed");return data;
}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::invalid_argument("usage: as2_batched_residual inputs new_output");
    namespace fs=std::filesystem;const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[1]),output=fs::weakly_canonical(argv[2]);
    for(auto const& p:{input,output}) {
      auto relative=p.lexically_relative(root);
      if(p==root || relative.empty() || *relative.begin()=="..")throw std::invalid_argument("Use controlled residual paths");
    }
    if(fs::exists(output))throw std::invalid_argument("Require a new residual output");
    std::cerr<<"phase: aclInit"<<std::endl;
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::invalid_argument("Require one visible device");Check(aclrtSetDevice(0),"device");
    auto* soc=aclrtGetSocName();if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::invalid_argument("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");fs::create_directory(output);
    std::cerr<<"phase: streamReady"<<std::endl;
    std::cout<<"{\"status\":\"pass\",\"scope\":\"component only; no model or TPOT admission\",\"groups\":2,\"aiv_workers\":4,\"epochs\":4,\"cases\":[";
    for(unsigned i=0;i<kCases.size();++i) {
      auto c=kCases[i];auto prefix=std::to_string(i);
      auto lhs=Read(input/(prefix+"_left.bin"),kWorkers*c.rows*c.left_stride);
      auto rhs=Read(input/(prefix+"_right.bin"),kWorkers*c.rows*c.right_stride);
      auto expected=Read(input/(prefix+"_expected.bin"),kWorkers*c.rows*c.output_stride+256);
      std::vector<uint16_t> a(expected.size(),0x7fc1),b=a;
      std::vector<uint64_t> times(kWorkers*64,~uint64_t(0));std::vector<void*> memory;
      auto allocate=[&](void const* src,size_t bytes) {
        void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");memory.push_back(p);
        Check(aclrtMemcpy(p,bytes,src,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
      };
      auto x=allocate(lhs.data(),lhs.size()*2),y=allocate(rhs.data(),rhs.size()*2);
      auto da=allocate(a.data(),a.size()*2),db=allocate(b.data(),b.size()*2);
      auto dt=allocate(times.data(),times.size()*8);
      std::cerr<<"phase: kernelLaunch case="<<i<<std::endl;
      residual_tile_probe<<<2,nullptr,stream>>>(x,y,da,db,dt,c.rows,c.columns,c.left_stride,c.right_stride,c.output_stride);
      std::cerr<<"phase: kernelLaunchReturned case="<<i<<std::endl;
      Check(aclrtSynchronizeStream(stream),"residual synchronize");
      std::cerr<<"phase: kernelComplete case="<<i<<std::endl;
      Check(aclrtMemcpy(a.data(),a.size()*2,da,a.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"control download");
      Check(aclrtMemcpy(b.data(),b.size()*2,db,b.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"candidate download");
      Check(aclrtMemcpy(times.data(),times.size()*8,dt,times.size()*8,ACL_MEMCPY_DEVICE_TO_HOST),"timers");
      for(auto const& item:{std::pair{"control",&a},std::pair{"candidate",&b}}) {
        std::ofstream f(output/(prefix+"_"+item.first+".bin"),std::ios::binary);
        f.write(reinterpret_cast<char*>(item.second->data()),item.second->size()*2);
        if(!f)throw std::runtime_error("Cannot save residual payload");
      }
      if(a!=b || b!=expected)throw std::runtime_error("Residual control/candidate/independent BF16 reference mismatch or guard overwrite");
      if(i)std::cout<<',';
      std::cout<<"{\"rows\":"<<c.rows<<",\"columns\":"<<c.columns<<",\"bitwise_equal\":true,\"independent_reference\":true,\"guards\":true,\"worker_ticks\":[";
      for(unsigned w=0;w<kWorkers;++w) {
        if(times[w*64])throw std::runtime_error("Residual AIV returned failure");
        if(w)std::cout<<',';std::cout<<'[';
        for(unsigned t=0;t<kEpochs*2;++t) {
          auto ticks=times[w*64+1+t];if(!ticks || ticks==~uint64_t(0))throw std::runtime_error("Residual lane did not execute");
          if(t)std::cout<<',';std::cout<<ticks;
        }
        std::cout<<']';
      }
      std::cout<<"]}";for(auto p:memory)Check(aclrtFree(p),"free");
    }
    std::cout<<"],\"host_kernel_launches\":4}"<<std::endl;
    Check(aclrtDestroyStream(stream),"destroy");Check(aclFinalize(),"finalize");return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
