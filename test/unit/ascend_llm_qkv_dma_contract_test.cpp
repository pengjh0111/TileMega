#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
int main(int argc,char** argv) {
  using namespace tilemega::ascend;
  llm::Stage s{};s.kind=0;s.variant=11;s.rows=1;s.columns=4096;s.k=1024;s.tasks=32;
#if defined(TILEMEGA_LLM_QKV_DMA_ACQUIRE) && TILEMEGA_LLM_QKV_DMA_ACQUIRE
  llm_host::ValidateCompiledBodies({s});
  for(unsigned mutation=0;mutation<9;++mutation) {
    auto v=s;
    if(mutation==0)v.rows=16;
    if(mutation==1)v.columns=2048;
    if(mutation==2)v.k=4096;
    if(mutation==3)v.epilogue=1;
    if(mutation==4)v.extent=3;
    if(mutation==5)v.tasks=31;
    if(mutation==6)v.operands[0]=llm::kAbsent;
    if(mutation==7)v.operands[1]=llm::kAbsent;
    if(mutation==8)v.operands[3]=llm::kAbsent;
    try {llm_host::ValidateCompiledBodies({v});return 1;}
    catch(std::invalid_argument const&) {}
  }
  if(argc==2) {
    llm_host::Plan p(argv[1]);unsigned count=0;
    for(auto const& t:p.stages)count+=t.kind==0 && t.variant==11;
    if(count!=28 || p.m.groups!=8 || p.m.batch!=1 || p.m.seq!=1)return 1;
  }
#else
  try {llm_host::ValidateCompiledBodies({s});return 1;}
  catch(std::invalid_argument const&) {}
  if(argc==2) {
    try {llm_host::Plan p(argv[1]);return 1;}
    catch(std::invalid_argument const&) {}
  }
#endif
  std::cout<<"{\"status\":\"pass\",\"acl_initialized\":false}\n";
}
