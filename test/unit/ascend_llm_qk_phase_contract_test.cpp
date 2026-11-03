#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
int main(int argc,char** argv) {
  using namespace tilemega::ascend;
  llm::Stage s{};s.kind=13;s.variant=13;s.width=128;s.group=2;s.extent=8;
  s.query_rows=2;s.tasks=8;s.kv_block=1088;s.workspace=512;
  s.operands[10]=1024;s.operands[11]=2048;
#if defined(TILEMEGA_LLM_QK_PHASE_DRAIN) && TILEMEGA_LLM_QK_PHASE_DRAIN
  llm_host::ValidateCompiledBodies({s});
  for(unsigned mutation=0;mutation<8;++mutation) {
    auto v=s;
    if(mutation==0)v.width=64;
    if(mutation==1)v.group=3;
    if(mutation==2)v.extent=4;
    if(mutation==3)v.query_rows=4;
    if(mutation==4)v.tasks=16;
    if(mutation==5)v.kv_block=128;
    if(mutation==6)v.operands[10]=v.operands[11];
    if(mutation==7)v.workspace=llm::kAbsent;
    try {llm_host::ValidateCompiledBodies({v});return 1;}
    catch(std::invalid_argument const&) {}
  }
  if(argc==2) {
    llm_host::Plan p(argv[1]);unsigned count=0;
    for(auto const& t:p.stages)count+=t.kind==13 && t.variant==13;
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
