#include <tilemega/Backend/Ascend/LlmVectorBody.h>
#include <tilemega/Backend/Ascend/SwiGluTile.h>
#include <acl/acl.h>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tilemega::ascend;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_swiglu_tile(GM_ADDR input,GM_ADDR output,GM_ADDR timing,unsigned rows,
    unsigned columns,unsigned input_stride,unsigned output_stride,unsigned implementation) {
  if ASCEND_IS_AIV {
    const unsigned worker=AscendC::GetBlockIdx();
    auto dst=output+uint64_t(worker)*rows*output_stride*2;
    uint64_t elapsed=0;bool ok=true;
    if(implementation==0) {
      LlmVectorBody body;
      for(unsigned repeat=0;repeat<4;++repeat) {
        const uint64_t start=AscendC::GetSystemCycle();
        for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<columns;c+=32) {
          auto src=input+(uint64_t(r)*input_stride+c)*2;
          ok=ok && body.Pointwise(src,src+32,dst+(uint64_t(r)*output_stride+c/2)*2,16,true);
        }
        if(repeat)elapsed+=uint64_t(AscendC::GetSystemCycle())-start;
      }
    }else {
      AscendC::TPipe pipe;AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
      pipe.InitBuffer(storage,36864);auto raw=storage.Get<uint8_t>();
      SwiGluTileBuffers b{raw.ReinterpretCast<bfloat16_t>(),raw[4096].ReinterpretCast<bfloat16_t>(),
        raw[8192].ReinterpretCast<bfloat16_t>(),raw[12288].ReinterpretCast<float>(),
        raw[20480].ReinterpretCast<float>(),raw[28672].ReinterpretCast<float>()};
      for(unsigned repeat=0;repeat<4;++repeat) {
        const uint64_t start=AscendC::GetSystemCycle();
        ok=ok && RunSwiGluTile(b,input,dst,rows,columns,input_stride,output_stride);
        if(repeat)elapsed+=uint64_t(AscendC::GetSystemCycle())-start;
      }
    }
    AscendC::GlobalTensor<uint64_t> status;status.SetGlobalBuffer((__gm__ uint64_t*)timing);
    status.SetValue(worker*64,ok?0:1);status.SetValue(worker*64+1,elapsed);
    AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(status);
  }
}

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_vector(GM_ADDR input,GM_ADDR rhs,GM_ADDR third,GM_ADDR output,GM_ADDR status,GM_ADDR statistics,
                 unsigned kind,unsigned rows,unsigned width,float epsilon) {
  if ASCEND_IS_AIV {
    unsigned worker=AscendC::GetBlockIdx();LlmVectorBody body;
    for(unsigned row=worker;row<rows;row+=4) {
      auto x=input+uint64_t(row)*width*2, y=output+uint64_t(row)*width*2;
      const uint64_t start=AscendC::GetSystemCycle();
      bool ok=true;
      if(kind==0 || kind>=4)ok=body.RmsNorm(x,rhs,y,width,epsilon,kind>=5?kind-3:kind==4?1:0,
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
static int RunSwiGluProbe(char** argv) {
  unsigned rows=std::stoul(argv[2]),columns=std::stoul(argv[3]);
  unsigned input_stride=std::stoul(argv[4]),output_stride=std::stoul(argv[5]),impl=std::stoul(argv[6]);
  if(!rows || rows>16 || !columns || columns>128 || columns%32 || input_stride<columns ||
      output_stride<columns/2 || input_stride>32768 || output_stride>16384 ||
      input_stride%16 || output_stride%16 || impl>1)throw std::runtime_error("Invalid bounded SwiGLU tile");
  namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
  auto input=fs::weakly_canonical(argv[7]),output=fs::weakly_canonical(argv[8]);
  for(auto const& p:{input,output}) {
    auto relative=p.lexically_relative(root);
    if(p==root || relative.empty() || *relative.begin()=="..")throw std::runtime_error("Require AS2 probe paths");
  }
  if(fs::exists(output))throw std::runtime_error("New output directory required");
  auto x=Read(input/"input.bin",rows*input_stride);
  std::vector<uint16_t> y(4*rows*output_stride+256,0x7fc1);std::vector<uint64_t> timing(4*64,~uint64_t(0));
  Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
  if(count!=1)throw std::runtime_error("Use single-device wrapper");
  Check(aclrtSetDevice(0),"device");auto* soc=aclrtGetSocName();
  if(!soc || std::string(soc)!="Ascend950PR_9579")throw std::runtime_error("Wrong SKU");
  aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> memory;
  auto allocate=[&](void const* src,size_t bytes) {
    void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");memory.push_back(p);
    Check(aclrtMemcpy(p,bytes,src,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
  };
  auto dx=allocate(x.data(),x.size()*2),dy=allocate(y.data(),y.size()*2),dt=allocate(timing.data(),timing.size()*8);
  as2_swiglu_tile<<<2,nullptr,stream>>>(dx,dy,dt,rows,columns,input_stride,output_stride,impl);
  Check(aclrtSynchronizeStream(stream),"synchronize");
  Check(aclrtMemcpy(y.data(),y.size()*2,dy,y.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"output");
  Check(aclrtMemcpy(timing.data(),timing.size()*8,dt,timing.size()*8,ACL_MEMCPY_DEVICE_TO_HOST),"timing");
  for(unsigned w=0;w<4;++w) {
    if(timing[w*64] || !timing[w*64+1])throw std::runtime_error("AIV tile failed or did not execute");
    for(unsigned r=0;r<rows;++r)for(unsigned c=columns/2;c<output_stride;++c)
      if(y[(w*rows+r)*output_stride+c]!=0x7fc1)throw std::runtime_error("Output stride guard overwritten");
  }
  for(size_t i=4*rows*output_stride;i<y.size();++i)
    if(y[i]!=0x7fc1)throw std::runtime_error("Output tail guard overwritten");
  fs::create_directory(output);std::ofstream file(output/"c.bin",std::ios::binary);
  file.write((char*)y.data(),(y.size()-256)*2);file.close();
  if(!file)throw std::runtime_error("Cannot save SwiGLU result");
  for(auto p:memory)Check(aclrtFree(p),"free");
  Check(aclrtDestroyStream(stream),"stream destroy");Check(aclFinalize(),"finalize");
  std::cout<<"{\"status\":\"executed_requires_numeric_compare\",\"implementation\":"<<impl
    <<",\"rows\":"<<rows<<",\"columns\":"<<columns<<",\"input_stride\":"<<input_stride
    <<",\"output_stride\":"<<output_stride<<",\"groups\":2,\"workers\":4,\"timed_repetitions\":3,\"service_ticks\":[";
  for(unsigned w=0;w<4;++w){if(w)std::cout<<',';std::cout<<timing[w*64+1];}
  std::cout<<"],\"ub_bytes\":"<<(impl?36864:LlmVectorBody::kLocalBytes)
    <<",\"host_kernel_launches\":1,\"tilemega_pipeline\":false}"<<std::endl;return 0;
}
int main(int argc,char** argv) {
  try {
    if(argc==9 && std::string(argv[1])=="swiglu-tile")return RunSwiGluProbe(argv);
    if(argc!=6 && argc!=7)throw std::runtime_error("usage: as2_vector kind(0 norm/1 swiglu/2 add/3 rope/4 log-exp norm/5 CANN Newton norm/6 folded Newton norm) rows width input_directory output_directory [epsilon]");
    unsigned kind=std::stoul(argv[1]),rows=std::stoul(argv[2]),width=std::stoul(argv[3]);
    float epsilon=argc==7?std::stof(argv[6]):1e-6f;
    if(!std::isfinite(epsilon) || epsilon<=0 || epsilon>.01f)throw std::runtime_error("Invalid normalization epsilon");
    if(kind>6 || !rows || rows>64 || !width || width>14336 || ((kind==0 || kind>=4) && width>4096) || (kind>=5 && width!=128 && width!=1024 && width!=2048 && width!=3072 && width!=4096) ||
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
    as2_vector<<<2,nullptr,stream>>>(dx,dr,dt,dy,ds,dstat,kind,rows,width,epsilon);
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
      <<",\"width\":"<<width<<",\"epsilon\":"<<epsilon<<",\"groups\":2,\"aiv_rows\":[";
    for(unsigned w=0;w<4;++w){if(w)std::cout<<',';std::cout<<states[w*128+1];}
    std::cout<<"],\"aiv_service_ticks_sum\":[";
    for(unsigned w=0;w<4;++w){if(w)std::cout<<',';std::cout<<states[w*128+2];}
    std::cout<<"],\"ub_bytes\":"<<LlmVectorBody::kLocalBytes
      <<",\"host_kernel_launches\":1,\"tilemega_pipeline\":false}"<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
