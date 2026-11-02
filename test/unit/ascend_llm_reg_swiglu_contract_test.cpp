#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
int main(int argc,char** argv) {
  using namespace tilemega::ascend;
  llm::Stage s{};s.kind=0;s.variant=8;s.epilogue=2;s.rows=1;
  s.columns=6144;s.k=1024;s.tasks=48;s.workspace=512;
#if defined(TILEMEGA_LLM_REG_SWIGLU) && TILEMEGA_LLM_REG_SWIGLU
  llm_host::ValidateCompiledBodies({s});
  for(unsigned mutation=0;mutation<5;++mutation) {
    auto v=s;
    if(mutation==0)v.extent=3;
    if(mutation==1)v.epilogue=1;
    if(mutation==2)v.tasks=47;
    if(mutation==3)v.columns=6145;
    if(mutation==4)v.variant=9;
    try {llm_host::ValidateCompiledBodies({v});return 1;}
    catch(std::invalid_argument const&) {}
  }
#else
  try {llm_host::ValidateCompiledBodies({s});return 1;}
  catch(std::invalid_argument const&) {}
#endif
  if(argc>2)return 1;
  if(argc==2) {
    llm_host::Plan plan(argv[1]);
    if(plan.m.tasks<=4096 || plan.m.groups!=8)return 1;
  }
  std::cout<<"{\"status\":\"pass\",\"acl_initialized\":false}\n";
}
