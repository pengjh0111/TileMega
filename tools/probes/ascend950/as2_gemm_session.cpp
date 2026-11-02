#include <tilemega/Backend/Ascend/GemmSession.h>
#include <acl/acl.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace tilemega::ascend;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_gemm_session(GM_ADDR a,GM_ADDR b,GM_ADDR af,GM_ADDR bf,GM_ADDR reference,
    GM_ADDR output,GM_ADDR temporary,unsigned m,unsigned n,unsigned k) {
  if ASCEND_IS_AIC {
    using Bf16=GemmTile<bfloat16_t,bfloat16_t,16,128,128,true>;
    using Float=GemmTile<float,float,32,64,128,true>;
    Bf16::SharedResource resource;const unsigned tiles=(n+127)/128;
    for(unsigned tile=0;tile<tiles;++tile) {
      Bf16::TileMmad mm(resource);
      Bf16::RunBorrowed(mm,a,b,reference,m,n,k,0,tile);
      CompleteDma<AscendC::HardEvent::FIX_S>();
    }
    GemmSession<Bf16> session(resource);
    for(unsigned epoch=0;epoch<4;++epoch) {
      for(unsigned tile=0;tile<tiles;++tile)session.SubmitTile(a,b,output,m,n,k,0,tile);
      // The FP32 recipe takes the same bank/flag namespace only after drain.
      session.FinishRecipe();
      {
        Float::TileMmad mm(resource);AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
        Float::RunBorrowed(mm,af,bf,temporary,m,n,k,0,0);
        CompleteDma<AscendC::HardEvent::FIX_S>();
      }
    }
  }
}
static void Check(aclError code,char const* action) {
  if(code)throw std::runtime_error(std::string(action)+": "+std::to_string(code));
}
static std::vector<uint16_t> Read(std::filesystem::path const& p,size_t count) {
  if(std::filesystem::file_size(p)!=count*2)throw std::invalid_argument("Input extent mismatch");
  std::vector<uint16_t> v(count);std::ifstream f(p,std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()),count*2);if(!f)throw std::runtime_error("Input read failed");return v;
}
static std::vector<float> Expand(std::vector<uint16_t> const& v) {
  std::vector<float> out(v.size());
  for(size_t i=0;i<v.size();++i){uint32_t bits=uint32_t(v[i])<<16;std::memcpy(&out[i],&bits,4);}return out;
}
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::invalid_argument("usage: as2_gemm_session M N K inputs new_output");
    unsigned m=std::stoul(argv[1]),n=std::stoul(argv[2]),k=std::stoul(argv[3]);
    if((m!=1 && m!=16) || n<256 || n>512 || n%32 || k<256 || k>14336 || k%16)
      throw std::invalid_argument("Bounded actual small-M/large-K family required");
    namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto controlled=[&](char const* raw){fs::path p(raw);if(fs::weakly_canonical(p)!=p || !p.is_absolute() || p==root || p.lexically_relative(root).empty() || *p.lexically_relative(root).begin()=="..")throw std::invalid_argument("Use canonical AS-2 paths");return p;};
    auto input=controlled(argv[4]),out=controlled(argv[5]);if(fs::exists(out))throw std::invalid_argument("Use new output");
    auto a=Read(input/"a.bin",uint64_t(m)*k),b=Read(input/"b_nk.bin",uint64_t(n)*k);
    auto af=Expand(a),bf=Expand(b);
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"device count");
    if(count!=1)throw std::runtime_error("Exactly one visible device required");
    Check(aclrtSetDevice(0),"select device");auto soc=aclrtGetSocName();
    if(!soc || std::string(soc).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> memory;
    auto allocate=[&](size_t bytes,void const* data=nullptr){void* p=nullptr;Check(aclrtMalloc(&p,bytes+512,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");memory.push_back(p);std::vector<uint8_t> init(bytes+512,0xa5);if(data)std::memcpy(init.data(),data,bytes);Check(aclrtMemcpy(p,init.size(),init.data(),init.size(),ACL_MEMCPY_HOST_TO_DEVICE),"upload");return static_cast<uint8_t*>(p);};
    auto da=allocate(a.size()*2,a.data()),db=allocate(b.size()*2,b.data());
    auto daf=allocate(af.size()*4,af.data()),dbf=allocate(bf.size()*4,bf.data());
    size_t bytes=uint64_t(m)*n*2;auto dr=allocate(bytes),dc=allocate(bytes),dt=allocate(2*bytes);
    as2_gemm_session<<<1,nullptr,stream>>>(da,db,daf,dbf,dr,dc,dt,m,n,k);
    Check(aclrtSynchronizeStream(stream),"session synchronize");
    std::vector<uint8_t> reference(bytes),output(bytes),guard(512);
    Check(aclrtMemcpy(reference.data(),bytes,dr,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"read reference");
    Check(aclrtMemcpy(output.data(),bytes,dc,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"read output");
    for(auto pair:{std::pair{dr,bytes},std::pair{dc,bytes},std::pair{dt,2*bytes}}) {
      Check(aclrtMemcpy(guard.data(),512,pair.first+pair.second,512,ACL_MEMCPY_DEVICE_TO_HOST),"read guard");
      if(!std::all_of(guard.begin(),guard.end(),[](auto v){return v==0xa5;}))throw std::runtime_error("Output guard overwrite");
    }
    if(reference!=output)throw std::runtime_error("Session changed BF16 outputs after dtype transitions");
    fs::create_directory(out);std::ofstream f(out/"c.bin",std::ios::binary);f.write(reinterpret_cast<char*>(output.data()),output.size());if(!f)throw std::runtime_error("Output save failed");
    Check(aclrtDestroyStream(stream),"destroy stream");for(auto p:memory)Check(aclrtFree(p),"free");
    Check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"passed\",\"groups\":1,\"host_kernel_launches\":1,\"epochs\":4,\"tiles_per_epoch\":"<<(n+127)/128<<",\"bf16_fp32_bf16_transitions\":4,\"bitwise_equal\":true,\"guards_passed\":true,\"tilemega_pipeline\":false}\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
