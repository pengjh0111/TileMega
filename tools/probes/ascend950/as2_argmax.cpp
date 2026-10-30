// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_LLM_VECTOR_ARGMAX 1
#include <tilemega/Backend/Ascend/LlmVectorBody.h>
#include <acl/acl.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace tilemega::ascend;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_argmax_probe(GM_ADDR x,GM_ADDR ids,GM_ADDR values,GM_ADDR indices,GM_ADDR status,
                     unsigned rows,unsigned n,unsigned stride,unsigned first,unsigned vocab,bool merge) {
  if ASCEND_IS_AIV {
    const unsigned w=AscendC::GetBlockIdx();LlmVectorBody body;
    for(unsigned epoch=0;epoch<2;++epoch) {
      uint32_t chosen=UINT32_MAX;
      bool ok=merge?body.ArgmaxMerge(x,ids,n,vocab,chosen):
          body.ArgmaxRows(x,values+uint64_t(w)*rows*4*4,indices+uint64_t(w)*rows*4*4,
                          rows,n,stride,4,first,vocab);
      WriteControl((__gm__ uint32_t*)status+w*16+epoch*2,ok?1:0);
      WriteControl((__gm__ uint32_t*)status+w*16+epoch*2+1,chosen);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
  }
}
namespace {
void Check(aclError e,char const* operation) {
  if(e)throw std::runtime_error(std::string(operation)+": "+std::to_string(e));
}
uint16_t Bf16(float x) {
  uint32_t bits;std::memcpy(&bits,&x,4);
  if((bits&0x7f800000u)==0x7f800000u)return uint16_t(bits>>16);
  return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
float Float(uint16_t x) {uint32_t bits=uint32_t(x)<<16;float y;std::memcpy(&y,&bits,4);return y;}
struct Case {unsigned rows,n,stride,first,vocab;bool merge;unsigned exceptional;};
}
int main(int argc,char** argv) {
  try {
    namespace fs=std::filesystem;
    if(argc!=2)throw std::invalid_argument("as2_argmax new_runs_as2_output; use the single-card job wrapper");
    const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    const auto out=fs::weakly_canonical(argv[1]),relative=out.lexically_relative(root);
    if(out==root || relative.empty() || *relative.begin()==".." || fs::exists(out))
      throw std::invalid_argument("Require a new AS-2 output directory");
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"device count");
    if(count!=1)throw std::runtime_error("Require exactly one visible NPU");
    Check(aclrtSetDevice(0),"device");auto soc=aclrtGetSocName();
    if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::runtime_error("Unexpected SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");
    // Negative values detect zero padding winning; FP32 neighbors collapse to a
    // BF16 tie before selection. Shuffled merge IDs test stable global ties.
    const std::vector<Case> cases={{1,17,64,128,151936,false,0},{16,128,151936,128,151936,false,0},
      {16,255,304,4096,128256,false,0},{1,128,160,0,151936,false,4},
      {1,128,160,0,151936,false,5},{1,17,64,0,151936,false,1},
      {1,17,64,0,151936,false,2},{1,17,64,0,151936,false,3},
      {1,1002,0,0,128256,true,0},{1,1187,0,0,151936,true,0},
      {1,1187,0,0,151936,true,1},{1,1187,0,0,151936,true,6}};
    unsigned passed=0;
    for(auto c:cases) {
      const bool expected=c.exceptional==0 || c.exceptional==4 || c.exceptional==5;
      const unsigned guard=0xa5a5a5a5u;
      std::vector<uint16_t> raw(c.merge?1:c.rows*c.stride,0x7fc1);
      std::vector<float> x(c.n,-9.0f);std::vector<uint32_t> ids(c.n),wanted(c.rows),status(32,guard);
      std::vector<uint32_t> values(2*c.rows*4+16,guard),indices(values.size(),guard),maxima(c.rows);
      if(c.merge) {
        for(unsigned i=0;i<c.n;++i)ids[i]=(c.n-1-i)*64;
        x[3]=x[c.n-7]=2.0f;wanted[0]=std::min(ids[3],ids[c.n-7]);
        if(c.exceptional==1)x[0]=std::numeric_limits<float>::quiet_NaN();
        if(c.exceptional==6)ids[0]=c.vocab;
      }else for(unsigned r=0;r<c.rows;++r) {
        for(unsigned i=0;i<c.n;++i)raw[r*c.stride+i]=Bf16(-20.0f+float(i%5));
        raw[r*c.stride+3]=Bf16(-1.0001f);raw[r*c.stride+c.n-1]=Bf16(-1.0002f);
        if(c.exceptional==4)raw[r*c.stride+3]=0x8000,raw[r*c.stride+c.n-1]=0;
        if(c.exceptional==5)raw[r*c.stride+3]=Bf16(1.0001f),raw[r*c.stride+c.n-1]=Bf16(1.0002f);
        if(c.exceptional>=1 && c.exceptional<=3)raw[0]=c.exceptional==1?0x7fc1:c.exceptional==2?0x7f80:0xff80;
        unsigned best=0;
        for(unsigned i=1;i<c.n;++i)if(Float(raw[r*c.stride+i])>Float(raw[r*c.stride+best]))best=i;
        wanted[r]=c.first+best;float maximum=Float(raw[r*c.stride+best]);std::memcpy(&maxima[r],&maximum,4);
      }
      std::vector<void*> allocations;
      auto upload=[&](void const* data,size_t size) {
        void* p=nullptr;Check(aclrtMalloc(&p,size,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");allocations.push_back(p);
        Check(aclrtMemcpy(p,size,data,size,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
      };
      auto dx=c.merge?upload(x.data(),x.size()*4):upload(raw.data(),raw.size()*2);
      auto di=upload(ids.data(),ids.size()*4),dv=upload(values.data(),values.size()*4);
      auto dout=upload(indices.data(),indices.size()*4),ds=upload(status.data(),status.size()*4);
      as2_argmax_probe<<<1,nullptr,stream>>>(dx,di,dv,dout,ds,c.rows,c.n,c.stride,c.first,c.vocab,c.merge);
      Check(aclrtSynchronizeStream(stream),"synchronize");
      Check(aclrtMemcpy(status.data(),status.size()*4,ds,status.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"status");
      Check(aclrtMemcpy(values.data(),values.size()*4,dv,values.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"values");
      Check(aclrtMemcpy(indices.data(),indices.size()*4,dout,indices.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"indices");
      for(unsigned w=0;w<2;++w)for(unsigned e=0;e<2;++e) {
        if(status[w*16+e*2]!=unsigned(expected))throw std::runtime_error("Wrong finite/index-domain rejection");
        if(c.merge && expected && status[w*16+e*2+1]!=wanted[0])throw std::runtime_error("Unstable merge tie");
      }
      for(unsigned i=0;i<values.size();++i) {
        const bool payload=!c.merge && expected && i<2*c.rows*4 && i%4==0;
        if(payload) {
          const unsigned r=(i/4)%c.rows;
          // Signed zeros compare equal; token identity still uses the left tie.
          float actual,reference;std::memcpy(&actual,&values[i],4);std::memcpy(&reference,&maxima[r],4);
          if(actual!=reference || indices[i]!=wanted[r])throw std::runtime_error("Wrong BF16 row argmax");
        }else if(values[i]!=guard || indices[i]!=guard)throw std::runtime_error("Stride/tail guard overwritten");
      }
      for(auto p:allocations)Check(aclrtFree(p),"free");++passed;
    }
    Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");
    fs::create_directory(out);std::ofstream result(out/"result.json");
    result<<"{\"status\":\"passed\",\"cases\":"<<passed<<",\"aiv_lanes\":2,\"epochs\":2,\"single_forward_claim\":false}\n";
    if(!result)throw std::runtime_error("Cannot save result");std::cout<<"passed "<<passed<<" cases\n";return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
