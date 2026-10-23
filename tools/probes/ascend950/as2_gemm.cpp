#include <tilemega/Backend/Ascend/GemmTile.h>
#include <tilemega/Target/Ascend/DevicePrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template<unsigned TM,unsigned TN,unsigned TK,unsigned L0K=32,bool WeightNK=true>
__aicore__ inline void Profile(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR timing,
                               unsigned m,unsigned n,unsigned k,unsigned groups) {
  if ASCEND_IS_AIC {
    const unsigned worker=AscendC::GetBlockIdx(),columns=(n+TN-1)/TN;
    const unsigned tiles=((m+TM-1)/TM)*columns;
    const uint64_t init=AscendC::GetSystemCycle();
    tilemega::ascend::GemmTile<bfloat16_t,bfloat16_t,TM,TN,TK,WeightNK,L0K> body;
    const uint64_t initialized=AscendC::GetSystemCycle();
    AscendC::GlobalTensor<uint64_t> times;
    times.SetGlobalBuffer((__gm__ uint64_t*)timing);
    times.SetValue(worker*64,initialized-init);
    for(unsigned tile=worker;tile<tiles;tile+=groups) {
      body.RunTask(a,b,c,m,n,k,tile/columns,tile%columns);
      tilemega::ascend::CompleteDma<AscendC::HardEvent::FIX_S>();
      const uint64_t start=AscendC::GetSystemCycle();
      for(unsigned r=0;r<3;++r) {
        body.RunTask(a,b,c,m,n,k,tile/columns,tile%columns);
        tilemega::ascend::CompleteDma<AscendC::HardEvent::FIX_S>();
      }
      const uint64_t end=AscendC::GetSystemCycle();
      times.SetValue((groups+tile)*64,end-start);
    }
    AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(times);
  }
}
__aicore__ inline void ProfileObservedPartial(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR timing,
    unsigned m,unsigned n,unsigned k,unsigned groups,unsigned parts) {
  if ASCEND_IS_AIC {
    using Body=tilemega::ascend::GemmTile<bfloat16_t,float,16,128,128,true,64>;
    typename Body::SharedResource resource;
    typename Body::TileMmad mm(resource);
    const unsigned columns=(n+127)/128,tiles=((m+15)/16)*columns;
    AscendC::GlobalTensor<uint64_t> times;times.SetGlobalBuffer((__gm__ uint64_t*)timing);
    for(unsigned tile=AscendC::GetBlockIdx();tile<tiles;tile+=groups) {
      const uint64_t start=AscendC::GetSystemCycle();
      const unsigned chunk=(k+parts-1)/parts;
      for(unsigned part=0;part<parts;++part) {
        const unsigned begin=part*chunk,size=k-begin<chunk?k-begin:chunk;
        Body::RunKSliceBorrowed(mm,a,b,c+uint64_t(part)*m*n*sizeof(float),m,n,k,
            begin,size,tile/columns,tile%columns);
        tilemega::ascend::CompleteDma<AscendC::HardEvent::FIX_S>();
      }
      times.SetValue((groups+tile)*64,AscendC::GetSystemCycle()-start);
    }
    AscendC::DataCacheCleanAndInvalid<uint64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(times);
  }
}
__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void as2_gemm(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR timing,
               unsigned m,unsigned n,unsigned k,unsigned groups,unsigned implementation) {
  if(implementation==0)Profile<32,64,64>(a,b,c,timing,m,n,k,groups);
  else if(implementation==1)Profile<16,128,128>(a,b,c,timing,m,n,k,groups);
  else if(implementation==2)Profile<16,128,128,64>(a,b,c,timing,m,n,k,groups);
  else if(implementation==3)Profile<16,128,128,32,false>(a,b,c,timing,m,n,k,groups);
  else ProfileObservedPartial(a,b,c,timing,m,n,k,groups,implementation==5?2:implementation==6?7:3);
}
static void Check(aclError code,const char* name) {
  if(code)throw std::runtime_error(std::string(name)+": "+std::to_string(code));
}
static std::vector<uint8_t> Read(std::filesystem::path const& path,size_t bytes) {
  if(std::filesystem::file_size(path)!=bytes)throw std::runtime_error("Input shape/file mismatch");
  std::vector<uint8_t> v(bytes);std::ifstream in(path,std::ios::binary);in.read((char*)v.data(),v.size());
  if(!in)throw std::runtime_error("Read failed");return v;
}
int main(int argc,char** argv) {
  try {
    if(argc!=7)throw std::runtime_error("usage: as2_gemm M N K implementation input_directory output_directory");
    unsigned m=std::stoul(argv[1]),n=std::stoul(argv[2]),k=std::stoul(argv[3]),impl=std::stoul(argv[4]);
    if(!m || m>64 || !n || n>151936 || n%16 || !k || k>16384 || k%16 || impl>6 || uint64_t(n)*k>(1ULL<<29))
      throw std::runtime_error("Bounded BF16 profile shape required");
    if(impl==4 && ((m!=1 && m!=16 && m!=64) || n!=2048 || (k!=6144 && k!=8192)))
      throw std::runtime_error("Partial diagnostic requires the observed stream-K shape");
    if(impl==5 && ((m!=1 && m!=64) || n!=3072 || k!=8192))
      throw std::runtime_error("Two-part diagnostic requires the observed H3072 stream-K shape");
    if(impl==6 && ((m!=1 && m!=64) || n!=1024 || k!=4096))
      throw std::runtime_error("Seven-part diagnostic requires the observed Llama8B KV shape");
    const bool partial=impl>=4;const unsigned parts=impl==5?2:impl==6?7:3;
    namespace fs=std::filesystem;
    const fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[5]),output=fs::weakly_canonical(argv[6]);
    auto inside=[&](auto const& p){auto r=p.lexically_relative(root);return !r.empty() && *r.begin()!=".." && p!=root;};
    if(!inside(input)||!inside(output)||fs::exists(output))throw std::runtime_error("Use existing AS-2 inputs and new output directory");
    auto a=Read(input/"a.bin",size_t(m)*k*2),b=Read(input/(impl==3?"b_kn.bin":"b_nk.bin"),size_t(n)*k*2);
    unsigned tm=impl?16:32,tn=impl?128:64,groups=2,tiles=((m+tm-1)/tm)*((n+tn-1)/tn);
    std::cerr<<"phase: aclInit"<<std::endl;
    Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"count");
    if(count!=1)throw std::runtime_error("Use one-device wrapper");
    std::cerr<<"phase: aclrtSetDevice"<<std::endl;
    Check(aclrtSetDevice(0),"device");auto* soc=aclrtGetSocName();
    if(!soc || std::string(soc).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    std::cerr<<"phase: allocate"<<std::endl;
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");
    std::vector<uint8_t> c(size_t(m)*n*(partial?parts*sizeof(float):2)+512,0xa5);
    std::vector<uint64_t> timing((groups+tiles)*64,0);
    std::vector<void*> allocated;
    auto alloc=[&](size_t bytes,const void* host) {
      void* p=nullptr;Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"allocate");allocated.push_back(p);
      if(host)Check(aclrtMemcpy(p,bytes,host,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"copy in");
      return (uint8_t*)p;
    };
    auto da=alloc(a.size(),a.data()),db=alloc(b.size(),b.data()),dc=alloc(c.size(),c.data()),dt=alloc(timing.size()*8,timing.data());
    std::cerr<<"phase: kernelLaunch"<<std::endl;
    as2_gemm<<<groups,nullptr,stream>>>(da,db,dc,dt,m,n,k,groups,impl);
    Check(aclrtSynchronizeStream(stream),"synchronize");
    std::cerr<<"phase: kernelComplete"<<std::endl;
    Check(aclrtMemcpy(c.data(),c.size(),dc,c.size(),ACL_MEMCPY_DEVICE_TO_HOST),"read output");
    Check(aclrtMemcpy(timing.data(),timing.size()*8,dt,timing.size()*8,ACL_MEMCPY_DEVICE_TO_HOST),"read timing");
    if(!std::all_of(c.end()-512,c.end(),[](auto v){return v==0xa5;}))throw std::runtime_error("Output guard overwritten");
    fs::create_directories(output);std::ofstream out(output/(partial?"partial_f32.bin":"c.bin"),std::ios::binary);out.write((char*)c.data(),c.size()-512);out.close();
    std::cout<<"{\"status\":\"executed_requires_numeric_compare\",\"shape_mnk\":["<<m<<','<<n<<','<<k
      <<"],\"implementation\":"<<impl<<",\"l0_k\":"<<((impl==2||partial)?64:32)<<",\"groups\":2,\"service_ticks\":[";
    for(unsigned tile=0;tile<tiles;++tile){if(tile)std::cout<<',';std::cout<<timing[(groups+tile)*64];}
    std::cout<<"],\"host_kernel_launches\":1,\"logical_weight_layout\":\"NK\",\"physical_weight_layout\":\""<<(impl==3?"KN":"NK")
      <<"\",\"timed_repetitions\":"<<(partial?1:3)<<",\"partial_count\":"<<(partial?parts:1)
      <<",\"device_combine\":false,\"tilemega_pipeline\":false}"<<std::endl;
    for(auto p:allocated)Check(aclrtFree(p),"free");
    Check(aclrtDestroyStream(stream),"destroy stream");Check(aclFinalize(),"finalize");return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
