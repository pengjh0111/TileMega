#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as1_gemm_tile_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR trace,
                         unsigned m, unsigned n, unsigned physical_k, unsigned groups) {
  if ASCEND_IS_AIC {
    tilemega::ascend::GemmTile<bfloat16_t,bfloat16_t> body;
    const unsigned worker=AscendC::GetBlockIdx(), nt=(n+63)/64, tasks=((m+31)/32)*nt;
    for(unsigned task=worker;task<tasks;task+=groups) {
      body.RunTask(a,b,c,m,n,physical_k,task/nt,task%nt);
      tilemega::ascend::CompleteDma<AscendC::HardEvent::FIX_S>();
      auto* record=(__gm__ uint32_t*)trace+task*128;
      tilemega::ascend::WriteControl(record,task+1);
      tilemega::ascend::WriteControl(record+1,worker);
    }
  }
}
static void Check(aclError e,char const* name) {
  if(e)throw std::runtime_error(std::string(name)+": "+std::to_string(e));
}
static std::vector<uint16_t> Read(std::filesystem::path const& file,size_t elements) {
  if(std::filesystem::file_size(file)!=elements*2)throw std::runtime_error("Input size mismatch: "+file.string());
  std::vector<uint16_t> data(elements);std::ifstream in(file,std::ios::binary);
  in.read(reinterpret_cast<char*>(data.data()),elements*2);
  if(!in)throw std::runtime_error("Cannot read input");return data;
}
static uint8_t* Allocate(void const* data,size_t bytes) {
  uint8_t* p=nullptr;Check(aclrtMalloc((void**)&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");
  Check(aclrtMemcpy(p,bytes,data,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"upload");return p;
}
int main(int argc,char** argv) {
  try {
    if(argc!=8)throw std::runtime_error("usage: gemm_tile M N K groups input_directory output_file trace_file");
    unsigned m=std::stoul(argv[1]),n=std::stoul(argv[2]),k=std::stoul(argv[3]),g=std::stoul(argv[4]);
    if(!((m==64&&n==128&&k==256)||(m==33&&n==96&&k==129)) || (g!=1&&g!=2))
      throw std::runtime_error("Use the fixed bounded E-0 shapes and G=1/2");
    unsigned kp=(k+15)/16*16,tasks=((m+31)/32)*((n+63)/64);
    auto input=std::filesystem::path(argv[5]);
    auto a=Read(input/"input.bin",m*k),b=Read(input/"weight_kn.bin",k*n);
    std::vector<uint16_t> pa(m*kp+256,0),pb(kp*n+256,0),pc(m*n+256,0x7fc1);
    for(unsigned row=0;row<m;++row)std::copy_n(a.data()+row*k,k,pa.data()+row*kp);
    std::copy(b.begin(),b.end(),pb.begin());
    std::vector<uint32_t> trace(tasks*128,0);
    Check(aclInit(nullptr),"init");uint32_t visible=0;Check(aclrtGetDeviceCount(&visible),"device count");
    if(visible!=1)throw std::runtime_error("Use the single-device wrapper");
    Check(aclrtSetDevice(0),"set device");
    const char* sku=aclrtGetSocName();
    if(!sku || std::string(sku)!="Ascend950PR_9579")throw std::runtime_error("Wrong runtime SKU");
    aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream),"stream");
    auto da=Allocate(pa.data(),pa.size()*2),db=Allocate(pb.data(),pb.size()*2),dc=Allocate(pc.data(),pc.size()*2);
    auto dt=Allocate(trace.data(),trace.size()*4);
    as1_gemm_tile_kernel<<<g,nullptr,stream>>>(da,db,dc,dt,m,n,kp,g);
    Check(aclrtSynchronizeStream(stream),"synchronize");
    Check(aclrtMemcpy(pc.data(),pc.size()*2,dc,pc.size()*2,ACL_MEMCPY_DEVICE_TO_HOST),"download output");
    Check(aclrtMemcpy(trace.data(),trace.size()*4,dt,trace.size()*4,ACL_MEMCPY_DEVICE_TO_HOST),"download trace");
    for(unsigned t=0;t<tasks;++t)if(trace[t*128]!=t+1 || trace[t*128+1]!=t%g)
      throw std::runtime_error("Tile ownership/trace mismatch");
    for(size_t i=m*n;i<pc.size();++i)if(pc[i]!=0x7fc1)throw std::runtime_error("Output tail overwrote guard");
    std::ofstream out(argv[6],std::ios::binary);out.write(reinterpret_cast<char*>(pc.data()),m*n*2);
    if(!out)throw std::runtime_error("Cannot write output");
    std::ofstream tr(argv[7]);tr<<"[";
    for(unsigned t=0;t<tasks;++t){if(t)tr<<',';tr<<"{\"tile\":"<<t<<",\"worker\":"<<trace[t*128+1]<<"}";}
    tr<<"]\n";if(!tr)throw std::runtime_error("Cannot write trace");
    for(auto p:{da,db,dc,dt})Check(aclrtFree(p),"free");
    Check(aclrtDestroyStream(stream),"destroy");Check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"executed\",\"host_kernel_launches\":1,\"tasks\":"<<tasks
      <<",\"groups\":"<<g<<",\"logical_k\":"<<k<<",\"physical_k\":"<<kp
      <<",\"output_guard_pass\":true,\"tilemega_pipeline\":false}"<<std::endl;
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
