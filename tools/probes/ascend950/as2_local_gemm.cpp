#include <tilemega/Backend/Ascend/LocalGemmTile.h>
#include <tilemega/Target/Ascend/ProgressPrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tilemega::ascend;
constexpr unsigned kSlot=131584,kTileBytes=16*128*2,kEpochs=4;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void local_gemm(GM_ADDR a,GM_ADDR b,GM_ADDR gm,GM_ADDR local,GM_ADDR control,
    unsigned rows,unsigned columns,unsigned k,uint64_t wait_ticks) {
  using Global=GemmTile<bfloat16_t,bfloat16_t,16,128,128,true>;
  using Local=LocalGemmTile<bfloat16_t,bfloat16_t>;
  if ASCEND_IS_AIV {if(AscendC::GetSubBlockIdx()!=0)return;}
  Global::SharedResource resource;
  auto slot=resource.ubBuf.template GetBufferByByte<bfloat16_t>(kSlot);
  auto guarded=resource.ubBuf.template GetBufferByByte<bfloat16_t>(kSlot-512);
  auto* ready=(__gm__ uint32_t*)control;auto* ack=ready+128;auto* abort=ready+256;
  const unsigned nt=(columns+127)/128;
  for(unsigned epoch=0;epoch<kEpochs;++epoch)for(unsigned tile=0;tile<nt;++tile) {
    const unsigned tag=1+epoch*nt+tile,rn=columns-tile*128<128?columns-tile*128:128;
    if ASCEND_IS_AIC {
      if(!WaitProgress(ack,2*tag-1,abort,wait_ticks))return;
      {
        Global::TileMmad mm(resource);
        Global::RunBorrowed(mm,a+uint64_t(epoch)*rows*k*2,b,gm+uint64_t(epoch)*rows*columns*2,rows,columns,k,0,tile);
        CompleteDma<AscendC::HardEvent::FIX_S>();
      }
      {
        Local::TileMmad mm(resource);
        if(!Local::RunBorrowed(mm,a+uint64_t(epoch)*rows*k*2,b,slot,rows,columns,k,0,tile)) {
          WriteControl(abort,2);return;
        }
        CompleteDma<AscendC::HardEvent::FIX_S>();
      }
      PublishSingleOwnerEvent(ready,tag);
      if(!WaitProgress(ack,2*tag,abort,wait_ticks))return;
    }else {
      AscendC::Duplicate(guarded,bfloat16_t(.25f),(kTileBytes+1024)/2);
      CompleteDma<AscendC::HardEvent::V_S>();
      if(slot.ReinterpretCast<uint16_t>().GetValue(0)!=0x3e80u){WriteControl(abort,3);return;}
      PublishSingleOwnerEvent(ack,2*tag-1);
      if(!WaitProgress(ready,tag,abort,wait_ticks))return;
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::GlobalTensor<bfloat16_t> output;
      output.SetGlobalBuffer((__gm__ bfloat16_t*)(local+uint64_t(epoch)*rows*columns*2));
      AscendC::DataCopyExtParams copy{1,rn*2,0,0,0};
      for(unsigned row=0;row<rows;++row)AscendC::DataCopyPad(output[row*columns+tile*128],slot[row*128],copy);
      CompleteDma<AscendC::HardEvent::MTE3_S>();
      auto guard_bits=guarded.ReinterpretCast<uint16_t>();
      for(unsigned i=0;i<256;++i)if(guard_bits.GetValue(i)!=0x3e80u || guard_bits.GetValue(256+kTileBytes/2+i)!=0x3e80u) {
        WriteControl(abort,4);return;
      }
      PublishSingleOwnerEvent(ack,2*tag);
    }
  }
}
static void Check(aclError code,char const* op) {
  if(code)throw std::runtime_error(std::string(op)+": "+std::to_string(code));
}
static std::vector<uint8_t> Read(std::filesystem::path const& p,size_t bytes) {
  if(std::filesystem::file_size(p)!=bytes)throw std::invalid_argument("Input extent mismatch");
  std::vector<uint8_t> data(bytes);std::ifstream f(p,std::ios::binary);f.read((char*)data.data(),bytes);
  if(!f)throw std::runtime_error("Input read failed");return data;
}
int main(int argc,char** argv) {
  try {
    if(argc!=6)throw std::invalid_argument("usage: as2_local_gemm M N K inputs new_output");
    unsigned m=std::stoul(argv[1]),n=std::stoul(argv[2]),k=std::stoul(argv[3]);
    if((m!=1 && m!=16) || (n!=128 && n!=129) || !k || k>16384 || k%16)throw std::invalid_argument("Unadmitted local tile probe shape");
    namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[4]),out=fs::weakly_canonical(argv[5]);
    auto inside=[&](auto const& p){auto rel=p.lexically_relative(root);return !rel.empty() && *rel.begin()!=".." && p!=root;};
    if(!inside(input) || !inside(out) || fs::exists(out))throw std::invalid_argument("Require controlled input and new output");
    auto a=Read(input/"a.bin",uint64_t(kEpochs)*m*k*2),b=Read(input/"b.bin",uint64_t(n)*k*2);
    std::cerr<<"phase: aclInit"<<std::endl;Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"device count");
    if(count!=1)throw std::runtime_error("Require one visible device");Check(aclrtSetDevice(0),"device");
    auto sku=aclrtGetSocName();if(!sku || std::string(sku).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> allocations;
    auto upload=[&](std::vector<uint8_t> const& host) {
      void* p=nullptr;Check(aclrtMalloc(&p,host.size(),ACL_MEM_MALLOC_HUGE_FIRST),"allocate");allocations.push_back(p);
      Check(aclrtMemcpy(p,host.size(),host.data(),host.size(),ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
    };
    const uint64_t bytes=uint64_t(kEpochs)*m*n*2;std::vector<uint8_t> initial(bytes+512,0xa5),controls(1536,0);
    auto da=upload(a),db=upload(b),dg=upload(initial),dl=upload(initial),dc=upload(controls);
    std::cerr<<"phase: kernelLaunch"<<std::endl;
    local_gemm<<<1,nullptr,stream>>>(da,db,dg,dl,dc,m,n,k,10000000000ULL);
    std::cerr<<"phase: kernelLaunchReturned"<<std::endl;Check(aclrtSynchronizeStream(stream),"synchronize");
    std::cerr<<"phase: kernelComplete"<<std::endl;
    Check(aclrtMemcpy(controls.data(),controls.size(),dc,controls.size(),ACL_MEMCPY_DEVICE_TO_HOST),"controls");
    auto global=initial,local=initial;
    Check(aclrtMemcpy(global.data(),global.size(),dg,global.size(),ACL_MEMCPY_DEVICE_TO_HOST),"global output");
    Check(aclrtMemcpy(local.data(),local.size(),dl,local.size(),ACL_MEMCPY_DEVICE_TO_HOST),"local output");
    fs::create_directory(out);
    for(auto const& item:{std::make_pair("global.bin",&global),std::make_pair("local.bin",&local)}) {
      std::ofstream f(out/item.first,std::ios::binary);f.write((char*)item.second->data(),bytes);
    }
    std::ofstream ctrl(out/"control.bin",std::ios::binary);ctrl.write((char*)controls.data(),controls.size());ctrl.close();
    if(std::any_of(controls.begin()+1024,controls.end(),[](auto x){return x!=0;}))throw std::runtime_error("Local protocol abort");
    if(global!=local || !std::all_of(global.begin()+bytes,global.end(),[](auto x){return x==0xa5;}))throw std::runtime_error("GM/local output or guard mismatch");
    std::cout<<"{\"status\":\"pass\",\"host_kernel_launches\":1,\"groups\":1,\"epochs\":4,\"aiv_participant\":0,\"GM_local_bitwise_equal\":true,\"default_enabled\":false}"<<std::endl;
    for(auto p:allocations)Check(aclrtFree(p),"free");Check(aclrtDestroyStream(stream),"destroy");Check(aclFinalize(),"finalize");return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
