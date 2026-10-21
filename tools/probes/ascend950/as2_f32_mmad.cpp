#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Backend/Ascend/LlmVectorBody.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void fp32_mmad(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR error,unsigned m,unsigned mode) {
  if ASCEND_IS_AIC {
    if(mode>=2)return;
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    if(!mode) {
      tilemega::ascend::GemmTile<float,float,32,64,128,true> body;
      body.RunTask(a,b,c,m,64,128,0,0);
    }else {
      tilemega::ascend::GemmTile<float,float,32,128,64,false> body;
      body.RunTask(a,b,c,m,128,1088,0,0);
    }
    tilemega::ascend::CompleteDma<AscendC::HardEvent::FIX_S>();
  }else if ASCEND_IS_AIV {
    if(mode>=2 && AscendC::GetBlockIdx()==0) {
      tilemega::ascend::LlmVectorBody body;
      for(unsigned row=0;row<m;++row)
        if(!body.ServingSoftmax(a+row*64*4,c+row*1088*4,row+1,true,1088,1088,mode==3)) {
          tilemega::ascend::WriteControl((__gm__ uint32_t*)error,8);return;
        }
    }
  }
}
static void Check(aclError e,char const* name) {
  if(e)throw std::runtime_error(std::string(name)+": "+std::to_string(e));
}
int main(int argc,char** argv) {
  try {
    namespace fs=std::filesystem;
    if(argc!=3 && argc!=5)throw std::runtime_error("usage: as2_f32_mmad input_directory new_output_directory [m:1|32 mode:0=QK|1=PV|2=softmax|3=softmax-vector-div]");
    unsigned m=argc==5?std::stoul(argv[3]):32,mode=argc==5?std::stoul(argv[4]):0;
    if((m!=1 && m!=32) || mode>3)throw std::runtime_error("Unsupported bounded FP32 attention family");
    unsigned n=mode>=2?1088:mode?128:64,k=mode>=2?64:mode?1088:128;
    const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[1]),out=fs::weakly_canonical(argv[2]);
    auto controlled=[&](auto const& p){auto r=p.lexically_relative(root);return p!=root && !r.empty() && *r.begin()!="..";};
    if(!controlled(input)||!controlled(out)||fs::exists(out))throw std::runtime_error("Use private AS-2 paths/new output");
    auto read=[&](char const* name,unsigned n) {
      auto p=input/name;if(fs::file_size(p)!=n*4)throw std::runtime_error("FP32 input extent mismatch");
      std::vector<float> values(n);std::ifstream f(p,std::ios::binary);f.read((char*)values.data(),n*4);
      if(!f)throw std::runtime_error("Cannot read FP32 input");return values;
    };
    auto a=read("a.bin",m*k),b=mode>=2?std::vector<float>{0}:read(mode?"b_kn.bin":"b_nk.bin",n*k);
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::runtime_error("Require one visible device");
    Check(aclrtSetDevice(0),"set device");auto* soc=aclrtGetSocName();
    if(!soc || std::string(soc).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> owned;
    auto alloc=[&](size_t bytes,void const* host) {
      void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");owned.push_back(p);
      Check(aclrtMemcpy(p,bytes,host,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
    };
    std::vector<uint8_t> c(m*n*4+512,0xa5);
    if(mode>=2)std::fill(c.begin(),c.end()-512,0);
    auto da=alloc(a.size()*4,a.data()),db=alloc(b.size()*4,b.data()),dc=alloc(c.size(),c.data());
    uint32_t error=0;std::vector<uint32_t> error_cell(128,0);auto de=alloc(512,error_cell.data());
    fp32_mmad<<<1,nullptr,stream>>>(da,db,dc,de,m,mode);Check(aclrtSynchronizeStream(stream),"synchronize");
    Check(aclrtMemcpy(&error,4,de,4,ACL_MEMCPY_DEVICE_TO_HOST),"read body error");
    if(error)throw std::runtime_error("Softmax body rejected valid input");
    Check(aclrtMemcpy(c.data(),c.size(),dc,c.size(),ACL_MEMCPY_DEVICE_TO_HOST),"output");
    if(!std::all_of(c.end()-512,c.end(),[](auto x){return x==0xa5;}))throw std::runtime_error("Output guard overwritten");
    fs::create_directory(out);std::ofstream f(out/"c.bin",std::ios::binary);f.write((char*)c.data(),c.size()-512);
    if(!f)throw std::runtime_error("Cannot save FP32 output");
    for(auto p:owned)Check(aclrtFree(p),"free");Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"executed_requires_numeric_comparison\",\"input_dtype\":\"fp32\",\"output_dtype\":\"fp32\",\"hf32_enabled\":false,\"shape_mnk\":["<<m<<','<<n<<','<<k<<"],\"host_kernel_launches\":1,\"groups\":1}"<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
