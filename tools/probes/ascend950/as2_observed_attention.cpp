#define TILEMEGA_LLM_VARIANT 1
#include <tilemega/Backend/Ascend/LlmCubeBody.h>
#include <tilemega/Backend/Ascend/LlmServingBody.h>
#include <tilemega/Target/Ascend/LlmExecutor.h>
#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace tilemega::ascend;
__global__ __aicore__ __mix__(1,2) __schedmode__(1)
void observed_attention(GM_ADDR arena,GM_ADDR metadata,GM_ADDR stages,GM_ADDR tasks,GM_ADDR waits,
    GM_ADDR offsets,GM_ADDR queue,GM_ADDR controls,GM_ADDR scratch,GM_ADDR trace,unsigned past) {
  llm::Forward forward{0,past,1,0};
  if ASCEND_IS_AIC {
    LlmCubeBody<16,128,128,1> body;
    ExecuteLlm<true>(body,arena,metadata,stages,tasks,waits,offsets,queue,controls,scratch,trace,forward);
  }else {
    LlmServingBody<16,128,true> body;
    ExecuteLlm<false>(body,arena,metadata,stages,tasks,waits,offsets,queue,controls,scratch,trace,forward);
  }
}

static llm_host::fs::path Controlled(char const* raw) {
  namespace fs=llm_host::fs;
  fs::path p(raw),root="/home/d50064638/junhuip/TileMega/runs/as2";
  if(fs::weakly_canonical(p)!=p || p==root || p.lexically_relative(root).empty() ||
      *p.lexically_relative(root).begin()=="..")throw std::invalid_argument("Use canonical AS2 paths");
  return p;
}

int main(int argc,char** argv) {
  try {
    if(argc!=4)throw std::invalid_argument("usage: as2_observed_attention case new_output past");
    auto root=Controlled(argv[1]),out=Controlled(argv[2]);
    std::size_t end=0;auto parsed=std::stoull(argv[3],&end);
    if(argv[3][0]=='-' || argv[3][end] || parsed>1086 || llm_host::fs::exists(out))
      throw std::invalid_argument("Bounded past and new output are required");
    unsigned past=parsed;llm_host::Plan p(root/"plan");
    if(p.m.groups!=8 || p.m.tasks!=8 || p.m.stages!=1 || p.m.batch!=1 || p.m.seq!=1 ||
        p.m.arena_bytes>(32ULL<<20) || p.stages[0].kind!=13 || (p.stages[0].variant!=5 && p.stages[0].variant!=6) ||
        p.stages[0].extent!=8 || !((p.stages[0].width==128 && p.stages[0].group>=2 && p.stages[0].group<=4) ||
          (p.stages[0].width==64 && p.stages[0].group==4)))
      throw std::invalid_argument("Observed attention probe domain mismatch");
    llm_host::Device device;auto arena=device.Allocate(p.m.arena_bytes);
    auto scratch=device.Allocate(p.m.workers*p.m.worker_scratch_bytes);
    auto plan=device.Upload(p,1);
    std::vector<uint8_t> guard(512,0xa5);
    for(unsigned i=0;i<p.allocations.size();++i) {
      auto const& a=p.allocations[i];
      llm_host::Check(aclrtMemcpy(arena+a.offset+a.bytes,512,guard.data(),512,ACL_MEMCPY_HOST_TO_DEVICE),"probe guard");
      if(!a.required)continue;
      auto bytes=llm_host::Read<uint8_t>(root/"inputs"/(std::to_string(i)+".bin"));
      if(bytes.size()!=a.bytes)throw std::invalid_argument("Probe input extent mismatch");
      llm_host::Check(aclrtMemcpy(arena+a.offset,a.bytes,bytes.data(),a.bytes,ACL_MEMCPY_HOST_TO_DEVICE),"probe input upload");
    }
    observed_attention<<<8,nullptr,device.stream>>>(arena,plan.metadata,plan.stages,plan.tasks,plan.waits,
        plan.offsets,plan.queue,plan.control,scratch,plan.trace,past);
    llm_host::Check(aclrtSynchronizeStream(device.stream),"observed attention synchronize");
    llm_host::fs::create_directory(out);llm_host::Complete(p,plan,0,out);
    for(auto const& a:p.allocations) {
      llm_host::Check(aclrtMemcpy(guard.data(),512,arena+a.offset+a.bytes,512,ACL_MEMCPY_DEVICE_TO_HOST),"probe guard read");
      if(!std::all_of(guard.begin(),guard.end(),[](auto v){return v==0xa5;}))throw std::runtime_error("Probe guard overwrite");
    }
    llm_host::Dump(p,plan,arena,past,1,out);
    std::vector<uint8_t> workspace(8*llm::attention::kGroupBytes);
    llm_host::Check(aclrtMemcpy(workspace.data(),workspace.size(),arena+p.stages[0].workspace,
        workspace.size(),ACL_MEMCPY_DEVICE_TO_HOST),"probe FP32 phase workspace");
    llm_host::Save(out/"attention_workspace.bin",workspace.data(),workspace.size());
    std::ofstream receipt(out/"result.json");
    receipt<<"{\"status\":\"executed_requires_comparison\",\"tilemega_pipeline\":false,"
      "\"scope\":\"observed Body replay only; diagnostic queues, not model acceptance\","
      "\"host_kernel_launches\":1,\"groups\":8,\"softmax_statistics\":true,\"past\":"<<past<<"}\n";
    std::cout<<"observed attention workspace saved; one diagnostic launch\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}
