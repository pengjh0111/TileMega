#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>

int main(int argc,char** argv) {
  using namespace tilemega::ascend;
  if(!batched_residual::Tile(16,128,1024,1024,1024) ||
      !batched_residual::Tile(3,97,144,160,176) ||
      batched_residual::Tile(17,128,1024,1024,1024) ||
      batched_residual::Tile(16,129,1024,1024,1024) ||
      batched_residual::Tile(1,17,16,32,32) ||
      batched_residual::Tile(1,17,33,32,32) ||
      batched_residual::Tile(16,128,1024,1024,UINT32_MAX))return 1;
  llm::Stage s{};s.kind=0;s.variant=12;s.rows=16;s.columns=1024;s.k=2048;
  s.tasks=8;s.epilogue=1;s.workspace=512;
#if defined(TILEMEGA_LLM_BATCHED_RESIDUAL) && TILEMEGA_LLM_BATCHED_RESIDUAL
  for(unsigned k:{2048u,3072u}) {s.k=k;llm_host::ValidateCompiledBodies({s});}
  for(unsigned mutation=0;mutation<12;++mutation) {
    auto v=s;
    if(mutation==0)v.rows=1;
    if(mutation==1)v.columns=2048;
    if(mutation==2)v.k=4096;
    if(mutation==3)v.epilogue=2;
    if(mutation==4)v.extent=3;
    if(mutation==5)v.tasks=7;
    if(mutation==6)v.workspace=llm::kAbsent;
    if(mutation>=7)v.operands[mutation-7]=llm::kAbsent;
    // Operand4 is not consumed by a residual epilogue.
    if(mutation==11) {llm_host::ValidateCompiledBodies({v});continue;}
    try {llm_host::ValidateCompiledBodies({v});return 1;}
    catch(std::invalid_argument const&) {}
  }
  if(argc==2) {
    llm_host::Plan p(argv[1]);unsigned count=0;
    for(auto const& t:p.stages)count+=t.kind==0 && t.variant==12;
    if(count!=56 || p.m.groups!=8 || p.m.batch!=16 || p.m.seq!=1)return 1;
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
