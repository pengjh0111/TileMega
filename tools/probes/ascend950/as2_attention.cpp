#include <tilemega/Backend/Ascend/LlmVectorBody.h>
#include <acl/acl.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tilemega::ascend;
__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_attention(GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR partial,GM_ADDR stats,GM_ADDR out,
    GM_ADDR status,unsigned qheads,unsigned kvheads,unsigned keys,unsigned d,float scale) {
  if ASCEND_IS_AIV {
    unsigned worker=AscendC::GetBlockIdx(),chunks=(keys+31)/32;
    LlmVectorBody body;
    for(unsigned h=worker;h<qheads;h+=4) {
      unsigned kh=h/(qheads/kvheads);
      for(unsigned c=0;c<chunks;++c) {
        unsigned valid=keys-c*32<32?keys-c*32:32;
        if(!body.AttentionChunk(q+h*d*2,k+(kh*keys+c*32)*d*2,v+(kh*keys+c*32)*d*2,
             partial+(h*chunks+c)*d*4,stats+(h*chunks+c)*8,valid,d,scale)) {
          WriteControl((__gm__ uint32_t*)status+worker*128,1);return;
        }
      }
      AscendC::GlobalTensor<float> view;view.SetGlobalBuffer((__gm__ float*)stats);AcquireData(view);
      if(!body.AttentionMerge(partial+h*chunks*d*4,stats+h*chunks*8,out+h*d*2,chunks,d)) {
        WriteControl((__gm__ uint32_t*)status+worker*128,2);return;
      }
      WriteControl((__gm__ uint32_t*)status+worker*128+1,
          ReadControl((__gm__ uint32_t*)status+worker*128+1)+1);
    }
  }
}
static void Check(aclError code,char const* name) {
  if(code)throw std::runtime_error(std::string(name)+": "+std::to_string(code));
}
static std::vector<uint8_t> Read(std::filesystem::path path,size_t size) {
  if(std::filesystem::file_size(path)!=size)throw std::runtime_error("Input size mismatch");
  std::vector<uint8_t> data(size);std::ifstream in(path,std::ios::binary);in.read((char*)data.data(),size);
  if(!in)throw std::runtime_error("Input read failed");return data;
}
int main(int argc,char** argv) {
  try {
    if(argc!=7)throw std::runtime_error("usage: as2_attention Qheads KVheads keys head_dim input_directory output_directory");
    unsigned qh=std::stoul(argv[1]),kvh=std::stoul(argv[2]),keys=std::stoul(argv[3]),d=std::stoul(argv[4]);
    if(!kvh || !qh || qh>32 || qh%kvh || !keys || keys>1088 || (d!=64 && d!=128))
      throw std::runtime_error("Bounded GQA/head/context shape required");
    namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[5]),output=fs::weakly_canonical(argv[6]);
    auto inside=[&](auto const& p){auto r=p.lexically_relative(root);return p!=root && !r.empty() && *r.begin()!="..";};
    if(!inside(input)||!inside(output)||fs::exists(output))throw std::runtime_error("Use controlled AS-2 inputs and a new output directory");
    auto q=Read(input/"q.bin",qh*d*2),k=Read(input/"k.bin",kvh*keys*d*2),v=Read(input/"v.bin",kvh*keys*d*2);
    std::vector<uint8_t> y(qh*d*2+512,0xa5);std::vector<uint32_t> state(4*128,0);
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::runtime_error("Use one visible device");Check(aclrtSetDevice(0),"device");
    auto* soc=aclrtGetSocName();if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> allocated;
    auto allocate=[&](size_t bytes,void const* data=nullptr) {
      void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");allocated.push_back(p);
      if(data)Check(aclrtMemcpy(p,bytes,data,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");
      else Check(aclrtMemset(p,bytes,0,bytes),"zero");return (uint8_t*)p;
    };
    unsigned chunks=(keys+31)/32;
    auto dq=allocate(q.size(),q.data()),dk=allocate(k.size(),k.data()),dv=allocate(v.size(),v.data()),
      dp=allocate(qh*chunks*d*4),ds=allocate(qh*chunks*8),dy=allocate(y.size(),y.data()),st=allocate(state.size()*4,state.data());
    as2_attention<<<2,nullptr,stream>>>(dq,dk,dv,dp,ds,dy,st,qh,kvh,keys,d,1.0f/std::sqrt(float(d)));
    Check(aclrtSynchronizeStream(stream),"synchronize");
    Check(aclrtMemcpy(state.data(),state.size()*4,st,state.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"read status");
    Check(aclrtMemcpy(y.data(),y.size(),dy,y.size(),ACL_MEMCPY_DEVICE_TO_HOST),"read output");
    for(unsigned w=0;w<4;++w)if(state[w*128] || state[w*128+1]!=(qh+3-w)/4)
      throw std::runtime_error("Attention worker failed/missing head: "+std::to_string(w));
    for(unsigned i=qh*d*2;i<y.size();++i)if(y[i]!=0xa5)throw std::runtime_error("Output guard overwritten");
    fs::create_directories(output);std::ofstream out(output/"c.bin",std::ios::binary);out.write((char*)y.data(),qh*d*2);out.close();
    for(auto p:allocated)Check(aclrtFree(p),"free");Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"executed_requires_numeric_compare\",\"qheads\":"<<qh<<",\"kvheads\":"<<kvh
      <<",\"gqa_ratio\":"<<qh/kvh<<",\"keys\":"<<keys<<",\"head_dim\":"<<d
      <<",\"chunk_keys\":32,\"chunks\":"<<chunks<<",\"host_kernel_launches\":1,\"tilemega_pipeline\":false}"<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
