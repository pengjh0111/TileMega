#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
int main(int argc,char** argv) {
  using namespace tilemega::ascend;
  llm::Stage s{};s.kind=0;s.variant=9;s.rows=1;s.columns=1024;s.k=2048;
  s.tasks=8;s.epilogue=1;s.workspace=512;
  static_assert(local_residual::kUbBytes==136192);
#if defined(TILEMEGA_LLM_LOCAL_RESIDUAL) && TILEMEGA_LLM_LOCAL_RESIDUAL
  llm_host::ValidateCompiledBodies({s});s.k=3072;llm_host::ValidateCompiledBodies({s});
  for(unsigned mutation=0;mutation<9;++mutation) {
    auto v=s;
    if(mutation==0)v.rows=16;
    if(mutation==1)v.columns=2048;
    if(mutation==2)v.k=4096;
    if(mutation==3)v.epilogue=2;
    if(mutation==4)v.extent=3;
    if(mutation==5)v.tasks=7;
    if(mutation==6)v.workspace=llm::kAbsent;
    if(mutation==7)v.operands[2]=llm::kAbsent;
    if(mutation==8)v.variant=10;
    try {llm_host::ValidateCompiledBodies({v});return 1;}
    catch(std::invalid_argument const&) {}
  }
  if(argc==2) {
    llm_host::Plan plan(argv[1]);unsigned local=0;
    for(auto const& t:plan.stages)local+=t.kind==0 && t.variant==9;
    if(local!=56 || plan.m.groups!=8 || plan.m.seq!=1 || plan.m.batch!=1)return 1;
  }
#else
  try {llm_host::ValidateCompiledBodies({s});return 1;}
  catch(std::invalid_argument const&) {}
  if(argc==2) {
    try {llm_host::Plan plan(argv[1]);return 1;}
    catch(std::invalid_argument const&) {}
  }
#endif
  std::cout<<"{\"status\":\"pass\",\"acl_initialized\":false}\n";
}
