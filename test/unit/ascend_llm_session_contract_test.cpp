#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
int main(int argc,char** argv) {
  using namespace tilemega::ascend;
  llm::Stage s{};s.kind=0;s.variant=7;s.rows=1;s.columns=1024;s.k=1024;s.tasks=8;
#if defined(TILEMEGA_LLM_GEMM_SESSION) && TILEMEGA_LLM_GEMM_SESSION
  llm_host::ValidateCompiledBodies({s});
  for(unsigned mutation=0;mutation<4;++mutation) {
    auto v=s;
    if(mutation==0)v.extent=3;
    if(mutation==1)v.rows=16;
    if(mutation==2)v.epilogue=4;
    if(mutation==3)v.variant=8;
    try {llm_host::ValidateCompiledBodies({v});return 1;}
    catch(std::invalid_argument const&) {}
  }
  s.epilogue=2;s.columns=6144;s.k=1024;s.tasks=48;
  llm_host::ValidateCompiledBodies({s});
  s.epilogue=3;s.columns=151936;s.tasks=1187;s.workspace=512;
  llm_host::ValidateCompiledBodies({s});
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
