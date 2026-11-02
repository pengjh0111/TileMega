#include <tilemega/Backend/Ascend/LocalGemmTile.h>
#include <tilemega/Target/Ascend/ProgressPrimitives.h>
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tilemega::ascend;
constexpr unsigned kSlot=131584,kTileBytes=16*128*2,kEpochs=4;

__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void local_gemm(GM_ADDR a,GM_ADDR b,GM_ADDR gm,GM_ADDR local,GM_ADDR control,GM_ADDR witness,
    unsigned rows,unsigned columns,unsigned k,uint64_t wait_ticks) {
  using Global=GemmTile<bfloat16_t,bfloat16_t,16,128,128,true>;
  using Local=LocalGemmTile<bfloat16_t,bfloat16_t>;
  bool monitor=false;
  if ASCEND_IS_AIV {monitor=AscendC::GetSubBlockIdx()!=0;}
  unsigned group=0,clock_offset=1536;
  if ASCEND_IS_AIC {group=AscendC::GetBlockIdx();}
  else {group=AscendC::GetBlockIdx()/2;clock_offset=2048;}
  const unsigned stride=(columns+15)/16*16;
  const uint64_t group_bytes=uint64_t(kEpochs)*rows*stride*2+512;
  a+=uint64_t(group)*kEpochs*rows*k*2;gm+=group*group_bytes;local+=group*group_bytes;control+=group*3072;
  auto* clock=(__gm__ uint32_t*)(control+clock_offset);
  if(!monitor) {auto entry=AscendC::GetSystemCycle();WriteControl(clock,uint32_t(entry));WriteControl(clock+1,uint32_t(entry>>32));}
  {
  Global::SharedResource resource;
  auto slot=resource.ubBuf.template GetBufferByByte<bfloat16_t>(kSlot);
  auto guarded=resource.ubBuf.template GetBufferByByte<bfloat16_t>(kSlot-512);
  auto* ready=(__gm__ uint32_t*)control;auto* ack=ready+128;auto* abort=ready+256;
  auto* monitor_ack=ready+640;
  const unsigned nt=(columns+127)/128;
  for(unsigned epoch=0;epoch<kEpochs;++epoch)for(unsigned tile=0;tile<nt;++tile) {
    const unsigned tag=1+epoch*nt+tile,rn=columns-tile*128<128?columns-tile*128:128;
    if ASCEND_IS_AIC {
      if(!WaitProgress(ack,2*tag-1,abort,wait_ticks))return;
      if(!WaitProgress(monitor_ack,2*tag-1,abort,wait_ticks))return;
      {
        Global::TileMmad mm(resource);
        Global::RunBorrowed(mm,a+uint64_t(epoch)*rows*k*2,b,gm+uint64_t(epoch)*rows*stride*2,rows,stride,k,0,tile);
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
      if(!WaitProgress(monitor_ack,2*tag,abort,wait_ticks))return;
    }else if(!monitor) {
      AscendC::Duplicate(guarded,bfloat16_t(.25f),(kTileBytes+1024)/2);
      CompleteDma<AscendC::HardEvent::V_S>();
      if(slot.ReinterpretCast<uint16_t>().GetValue(0)!=0x3e80u){WriteControl(abort,3);return;}
      PublishSingleOwnerEvent(ack,2*tag-1);
      if(!WaitProgress(ready,tag,abort,wait_ticks))return;
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::GlobalTensor<bfloat16_t> output;
      output.SetGlobalBuffer((__gm__ bfloat16_t*)(local+uint64_t(epoch)*rows*stride*2));
      AscendC::DataCopyExtParams copy{1,rn*2,0,0,0};
      for(unsigned row=0;row<rows;++row)AscendC::DataCopyPad(output[row*stride+tile*128],slot[row*128],copy);
      CompleteDma<AscendC::HardEvent::MTE3_S>();
      auto guard_bits=guarded.ReinterpretCast<uint16_t>();
      for(unsigned i=0;i<256;++i)if(guard_bits.GetValue(i)!=0x3e80u || guard_bits.GetValue(256+kTileBytes/2+i)!=0x3e80u) {
        WriteControl(abort,4);return;
      }
      PublishSingleOwnerEvent(ack,2*tag);
    }else {
      AscendC::Duplicate(guarded,bfloat16_t(.75f),(kTileBytes+1024)/2);
      CompleteDma<AscendC::HardEvent::V_S>();PublishSingleOwnerEvent(monitor_ack,2*tag-1);
      if(!WaitProgress(ready,tag,abort,wait_ticks))return;
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::GlobalTensor<bfloat16_t> output;
      output.SetGlobalBuffer((__gm__ bfloat16_t*)(witness+(uint64_t(group)*kEpochs*nt+epoch*nt+tile)*(kTileBytes+1024)));
      AscendC::DataCopyExtParams copy{1,kTileBytes+1024,0,0,0};AscendC::DataCopyPad(output,guarded,copy);
      CompleteDma<AscendC::HardEvent::MTE3_S>();PublishSingleOwnerEvent(monitor_ack,2*tag);
    }
  }
  }
  if(!monitor) {auto end=AscendC::GetSystemCycle();WriteControl(clock+2,uint32_t(end));WriteControl(clock+3,uint32_t(end>>32));}
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
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
    if(argc!=6 && argc!=7)throw std::invalid_argument("usage: as2_local_gemm M N K inputs new_output [groups]");
    unsigned m=std::stoul(argv[1]),n=std::stoul(argv[2]),k=std::stoul(argv[3]);
    const unsigned groups=argc==7?std::stoul(argv[6]):1;
    if(groups!=1 && groups!=2 && groups!=8)throw std::invalid_argument("Unadmitted probe group count");
    if((m!=1 && m!=16) || (n!=128 && n!=129) || !k || k>16384 || k%16)throw std::invalid_argument("Unadmitted local tile probe shape");
    namespace fs=std::filesystem;fs::path root="/home/d50064638/junhuip/TileMega/runs/as2";
    auto input=fs::weakly_canonical(argv[4]),out=fs::weakly_canonical(argv[5]);
    auto inside=[&](auto const& p){auto rel=p.lexically_relative(root);return !rel.empty() && *rel.begin()!=".." && p!=root;};
    if(!inside(input) || !inside(out) || fs::exists(out))throw std::invalid_argument("Require controlled input and new output");
    auto a=Read(input/"a.bin",uint64_t(groups)*kEpochs*m*k*2),b=Read(input/"b.bin",uint64_t(n)*k*2);
    // The GM reference uses a legal aligned physical stride. Its extra weight
    // rows are zero padding, never part of the logical numerical comparison.
    const unsigned stride=(n+15)/16*16;b.resize(uint64_t(stride)*k*2,0);
    std::cerr<<"phase: aclInit"<<std::endl;Check(aclInit(nullptr),"init");uint32_t count=0;Check(aclrtGetDeviceCount(&count),"device count");
    if(count!=1)throw std::runtime_error("Require one visible device");Check(aclrtSetDevice(0),"device");
    auto sku=aclrtGetSocName();if(!sku || std::string(sku).find("Ascend950PR")!=0)throw std::runtime_error("Wrong SKU");
    aclrtStream stream;Check(aclrtCreateStream(&stream),"stream");std::vector<void*> allocations;
    auto upload=[&](std::vector<uint8_t> const& host) {
      void* p=nullptr;Check(aclrtMalloc(&p,host.size(),ACL_MEM_MALLOC_HUGE_FIRST),"allocate");allocations.push_back(p);
      Check(aclrtMemcpy(p,host.size(),host.data(),host.size(),ACL_MEMCPY_HOST_TO_DEVICE),"upload");return (uint8_t*)p;
    };
    const uint64_t payload=uint64_t(kEpochs)*m*stride*2,group_bytes=payload+512;
    std::vector<uint8_t> initial(groups*group_bytes,0xa5),controls(groups*3072,0);
    std::vector<uint8_t> witness(uint64_t(groups)*kEpochs*((n+127)/128)*(kTileBytes+1024),0xa5);
    auto da=upload(a),db=upload(b),dg=upload(initial),dl=upload(initial),dc=upload(controls),dw=upload(witness);
    std::cerr<<"phase: kernelLaunch"<<std::endl;
    auto submitted=std::chrono::steady_clock::now();
    local_gemm<<<groups,nullptr,stream>>>(da,db,dg,dl,dc,dw,m,n,k,10000000000ULL);
    std::cerr<<"phase: kernelLaunchReturned"<<std::endl;Check(aclrtSynchronizeStream(stream),"synchronize");
    std::cerr<<"phase: kernelComplete"<<std::endl;
    auto completed=std::chrono::steady_clock::now();
    Check(aclrtMemcpy(controls.data(),controls.size(),dc,controls.size(),ACL_MEMCPY_DEVICE_TO_HOST),"controls");
    auto global=initial,local=initial;
    Check(aclrtMemcpy(global.data(),global.size(),dg,global.size(),ACL_MEMCPY_DEVICE_TO_HOST),"global output");
    Check(aclrtMemcpy(local.data(),local.size(),dl,local.size(),ACL_MEMCPY_DEVICE_TO_HOST),"local output");
    Check(aclrtMemcpy(witness.data(),witness.size(),dw,witness.size(),ACL_MEMCPY_DEVICE_TO_HOST),"inactive lane witness");
    fs::create_directory(out);
    for(auto const& item:{std::make_pair("global.bin",&global),std::make_pair("local.bin",&local)}) {
      std::ofstream f(out/item.first,std::ios::binary);
      for(unsigned g=0;g<groups;++g)for(unsigned row=0;row<kEpochs*m;++row)
        f.write((char*)item.second->data()+g*group_bytes+uint64_t(row)*stride*2,n*2);
      std::ofstream raw(out/(std::string(item.first)+".physical"),std::ios::binary);
      raw.write((char*)item.second->data(),item.second->size());
    }
    std::ofstream ctrl(out/"control.bin",std::ios::binary);ctrl.write((char*)controls.data(),controls.size());ctrl.close();
    std::ofstream monitor_file(out/"aiv1_witness.bin",std::ios::binary);monitor_file.write((char*)witness.data(),witness.size());monitor_file.close();
    for(size_t i=0;i<witness.size();i+=2)if(witness[i]!=0x40 || witness[i+1]!=0x3f)
      throw std::runtime_error("AIV1 private slot was modified by AIV0 output route");
    for(unsigned g=0;g<groups;++g) {
    if(std::any_of(controls.begin()+g*3072+1024,controls.begin()+g*3072+1536,[](auto x){return x!=0;}))throw std::runtime_error("Local protocol abort");
    for(unsigned row=0;row<kEpochs*m;++row) {
      auto offset=g*group_bytes+uint64_t(row)*stride*2;
      if(!std::equal(global.begin()+offset,global.begin()+offset+n*2,local.begin()+offset) ||
          !std::all_of(local.begin()+offset+n*2,local.begin()+offset+stride*2,[](auto x){return x==0xa5;}))
        throw std::runtime_error("GM/local logical output or row padding mismatch");
    }
    for(auto const* value:{&global,&local})if(!std::all_of(value->begin()+g*group_bytes+payload,value->begin()+(g+1)*group_bytes,[](auto x){return x==0xa5;}))
      throw std::runtime_error("Output end guard mismatch");
    }
    auto tick=[&](unsigned offset){uint32_t halves[2];std::memcpy(halves,controls.data()+offset,8);return uint64_t(halves[0])|(uint64_t(halves[1])<<32);};
    uint64_t cube_ticks=0,vector_ticks=0;
    for(unsigned g=0;g<groups;++g) {
      auto c0=tick(g*3072+1536),c1=tick(g*3072+1544),v0=tick(g*3072+2048),v1=tick(g*3072+2056);
      if(!c0 || c1<c0 || !v0 || v1<v0)throw std::runtime_error("Missing per-group entry/exit clock");
      cube_ticks=std::max(cube_ticks,c1-c0);vector_ticks=std::max(vector_ticks,v1-v0);
    }
    std::cout<<"{\"status\":\"pass\",\"host_kernel_launches\":1,\"groups\":"<<groups<<",\"epochs\":4,\"aiv_participant\":0,\"aiv1_private_slot_untouched\":true,\"GM_local_bitwise_equal\":true,\"default_enabled\":false"
      <<",\"host_launch_to_sync_ms\":"<<std::chrono::duration<double,std::milli>(completed-submitted).count()
      <<",\"logical_columns\":"<<n<<",\"physical_output_stride\":"<<stride
      <<",\"maximum_group_aic_body_ticks\":"<<cube_ticks<<",\"maximum_group_aiv_body_ticks\":"<<vector_ticks<<"}"<<std::endl;
    for(auto p:allocations)Check(aclrtFree(p),"free");Check(aclrtDestroyStream(stream),"destroy");Check(aclFinalize(),"finalize");return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
