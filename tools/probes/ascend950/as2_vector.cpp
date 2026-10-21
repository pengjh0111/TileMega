#include <tilemega/Backend/Ascend/LlmVectorBody.h>
#include <acl/acl.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tilemega::ascend;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_vector(GM_ADDR input,GM_ADDR rhs,GM_ADDR third,GM_ADDR output,GM_ADDR status,GM_ADDR statistics,
                 unsigned kind,unsigned rows,unsigned width) {
  if ASCEND_IS_AIV {
    unsigned worker=AscendC::GetBlockIdx();LlmVectorBody body;
    for(unsigned row=worker;row<rows;row+=4) {
      auto x=input+uint64_t(row)*width*2, y=output+uint64_t(row)*width*2;
      const uint64_t start=AscendC::GetSystemCycle();
      bool ok=true;
      if(kind==0 || kind>=4)ok=body.RmsNorm(x,rhs,y,width,1e-6f,kind>=5?kind-3:kind==4?1:0,
                                         statistics+uint64_t(row)*64);
      else if(kind==3)ok=body.Rope(x,rhs+uint64_t(row)*width*2,third+uint64_t(row)*width*2,y,width);
      else for(unsigned col=0;col<width;col+=4096) {
        auto n=width-col<4096?width-col:4096;
        ok=ok && body.Pointwise(x+col*2,rhs+(uint64_t(row)*width+col)*2,y+col*2,n,kind==1);
      }
      if(!ok){WriteControl((__gm__ uint32_t*)status+worker*128,1);return;}
      const uint64_t ticks=uint64_t(AscendC::GetSystemCycle())-start;
      if(ticks>0xffffffffu){WriteControl((__gm__ uint32_t*)status+worker*128,2);return;}
      WriteControl((__gm__ uint32_t*)status+worker*128+2,
                   ReadControl((__gm__ uint32_t*)status+worker*128+2)+uint32_t(ticks));
      WriteControl((__gm__ uint32_t*)status+worker*128+1,
                   ReadControl((__gm__ uint32_t*)status+worker*128+1)+1);
    }
  }
}
static void Check(aclError e,const char* name) {
  if(e)throw std::runtime_error(std::string(name)+": "+std::to_string(e));
}
static std::vector<uint16_t> Read(std::filesystem::path const& path,size_t n) {
  if(std::filesystem::file_size(path)!=n*2)throw std::runtime_error("Input size mismatch");
  std::vector<uint16_t> v(n);std::ifstream in(path,std::ios::binary);
  in.read((char*)v.data(),n*2);if(!in)throw std::runtime_error("Input read failed");return v;
}
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::runtime_error("usage: as2_vector kind(0 norm/1 swiglu/2 add/3 rope/4 log-exp norm/5 CANN Newton norm/6 folded Newton norm) rows width input_directory output_directory");
    unsigned kind=std::stoul(argv[1]),rows=std::stoul(argv[2]),width=std::stoul(argv[3]);
    if(kind>6 || !rows || rows>64 || !width || width>14336 || ((kind==0 || kind>=4) && width>4096) || (kind>=5 && width!=128 && width!=1024) ||
       (kind==3 && width!=64 && width!=128))throw std::runtime_error("Unsupported bounded vector shape");
    namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[4]),output=fs::weakly_canonical(argv[5]);
    auto inside=[&](auto const& p){auto r=p.lexically_relative(root);return !r.empty() && *r.begin()!=".." && p!=root;};
    if(!inside(input)||!inside(output)||fs::exists(output))throw std::runtime_error("Use canonical AS-2 paths and new output directory");
    auto x=Read(input/"input.bin",rows*width),rhs=Read(input/"rhs.bin",(kind==0 || kind>=4)?width:rows*width);
    std::vector<uint16_t> third=kind==3?Read(input/"third.bin",rows*width):std::vector<uint16_t>(16,0);
    std::vector<uint16_t> y(rows*width+256,0x7fc1);std::vector<uint32_t> states(4*128,0);
    std::vector<float> statistics(rows*16,0);
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::runtime_error("Use single-device wrapper");
    Check(aclrtSetDevice(0),"device");auto* soc=aclrtGetSocName();
    if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> memory;
    auto alloc=[&](const void* data,size_t bytes) {
      void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");memory.push_back(p);
      Check(aclrtMemcpy(p,bytes,data,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
    };
    auto dx=alloc(x.data(),x.size()*2),dr=alloc(rhs.data(),rhs.size()*2),dt=alloc(third.data(),third.size()*2),
      dy=alloc(y.data(),y.size()*2),ds=alloc(states.data(),states.size()*4),
      dstat=alloc(statistics.data(),statistics.size()*4);
    as2_vector<<<2,nullptr,stream>>>(dx,dr,dt,dy,ds,dstat,kind,rows,width);
    Check(aclrtSynchronizeStream(stream),"synchronize");
    Check(aclrtMemcpy(states.data(),states.size()*4,ds,states.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"read status");
    Check(aclrtMemcpy(y.data(),y.size()*2,dy,y.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"read output");
    Check(aclrtMemcpy(statistics.data(),statistics.size()*4,dstat,statistics.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"read statistics");
    for(unsigned w=0;w<4;++w)
      if(states[w*128] || states[w*128+1]!=(rows+3-w)/4)throw std::runtime_error("Missing AIV computation or device shape rejection");
    for(size_t i=rows*width;i<y.size();++i)if(y[i]!=0x7fc1)throw std::runtime_error("Output guard overwritten");
    fs::create_directories(output);std::ofstream out(output/"c.bin",std::ios::binary);out.write((char*)y.data(),rows*width*2);out.close();
    if(kind==0 || kind>=4){std::ofstream stats(output/"norm_statistics.bin",std::ios::binary);
      stats.write((char*)statistics.data(),statistics.size()*4);}
    for(auto p:memory)Check(aclrtFree(p),"free");
    Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"executed_requires_numeric_compare\",\"kind\":"<<kind<<",\"rows\":"<<rows
      <<",\"width\":"<<width<<",\"groups\":2,\"aiv_rows\":[";
    for(unsigned w=0;w<4;++w){if(w)std::cout<<',';std::cout<<states[w*128+1];}
    std::cout<<"],\"aiv_service_ticks_sum\":[";
    for(unsigned w=0;w<4;++w){if(w)std::cout<<',';std::cout<<states[w*128+2];}
    std::cout<<"],\"ub_bytes\":"<<LlmVectorBody::kLocalBytes
      <<",\"host_kernel_launches\":1,\"tilemega_pipeline\":false}"<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
