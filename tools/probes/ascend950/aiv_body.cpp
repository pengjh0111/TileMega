#include <tilemega/Backend/Ascend/AivBody.h>
#include <acl/acl.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as1_aiv_body(GM_ADDR x,GM_ADDR rhs,GM_ADDR y,GM_ADDR status,unsigned kind,unsigned rows,unsigned width) {
  if ASCEND_IS_AIV {
    if(AscendC::GetBlockIdx()!=0)return;
    tilemega::ascend::AivBody body;
    for(unsigned row=0;row<rows;++row) {
      auto second=rhs+(kind==1?0:row*width*2);
      if(!body.Run(kind,x+row*width*2,second,y+row*width*2,width,1e-5f)) {
        tilemega::ascend::WriteControl((__gm__ uint32_t*)status,1);return;
      }
    }
  }
}
static void Check(aclError e,char const* name) {
  if(e)throw std::runtime_error(std::string(name)+": "+std::to_string(e));
}
static std::vector<uint16_t> Read(std::filesystem::path const& path,size_t n) {
  if(std::filesystem::file_size(path)!=n*2)throw std::runtime_error("Input size mismatch");
  std::vector<uint16_t> v(n);std::ifstream f(path,std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()),n*2);if(!f)throw std::runtime_error("Input read failed");return v;
}
static uint8_t* Allocate(void const* data,size_t bytes) {
  uint8_t* p=nullptr;Check(aclrtMalloc((void**)&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");
  Check(aclrtMemcpy(p,bytes,data,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return p;
}
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::runtime_error("usage: aiv_body kind rows width input_directory output_file");
    unsigned kind=std::stoul(argv[1]),rows=std::stoul(argv[2]),width=std::stoul(argv[3]);
    if((kind!=1&&kind!=4&&kind!=10)||rows<1||rows>4||(width!=128&&width!=129))
      throw std::runtime_error("Use the bounded RMSNorm/SiLU/Add body shapes");
    auto input=std::filesystem::path(argv[4]);
    auto x=Read(input/"input.bin",rows*width),rhs=Read(input/"rhs.bin",kind==1?width:rows*width);
    std::vector<uint16_t> output(rows*width+256,0x7fc1);std::vector<uint32_t> status(128,0);
    Check(aclInit(nullptr),"init");uint32_t visible=0;Check(aclrtGetDeviceCount(&visible),"device count");
    if(visible!=1)throw std::runtime_error("Use the single-device wrapper");
    Check(aclrtSetDevice(0),"set device");const char* sku=aclrtGetSocName();
    if(!sku||std::string(sku)!="Ascend950PR_9579")throw std::runtime_error("Wrong runtime SKU");
    aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream),"stream");
    auto dx=Allocate(x.data(),x.size()*2),dr=Allocate(rhs.data(),rhs.size()*2),
      dy=Allocate(output.data(),output.size()*2),ds=Allocate(status.data(),512);
    as1_aiv_body<<<1,nullptr,stream>>>(dx,dr,dy,ds,kind,rows,width);
    Check(aclrtSynchronizeStream(stream),"synchronize");
    Check(aclrtMemcpy(status.data(),512,ds,512,ACL_MEMCPY_DEVICE_TO_HOST),"status");
    Check(aclrtMemcpy(output.data(),output.size()*2,dy,output.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"output");
    if(status[0])throw std::runtime_error("Unsupported device body invocation");
    for(size_t i=rows*width;i<output.size();++i)if(output[i]!=0x7fc1)throw std::runtime_error("Output guard overwritten");
    std::ofstream f(argv[5],std::ios::binary);f.write(reinterpret_cast<char*>(output.data()),rows*width*2);
    if(!f)throw std::runtime_error("Cannot write output");
    for(auto p:{dx,dr,dy,ds})Check(aclrtFree(p),"free");
    Check(aclrtDestroyStream(stream),"destroy");Check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"executed\",\"kind\":"<<kind<<",\"rows\":"<<rows<<",\"width\":"<<width
      <<",\"ub_bytes\":"<<tilemega::ascend::AivBody::kLocalBytes<<",\"host_kernel_launches\":1,\"tilemega_pipeline\":false}"<<std::endl;return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
