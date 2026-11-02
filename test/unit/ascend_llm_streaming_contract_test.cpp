#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>

int main() {
  using namespace tilemega::ascend;
  llm::Stage s{};s.kind=13;s.variant=10;s.extent=8;s.width=128;
  s.group=3;s.kv_block=1088;s.query_rows=48;
#if defined(TILEMEGA_LLM_STREAMING_ATTENTION) && TILEMEGA_LLM_STREAMING_ATTENTION
  s.variant=TILEMEGA_LLM_PRECISION_CONTRACT+8;
  llm_host::ValidateCompiledBodies({s});
  for(auto shape:{std::pair{128u,2u},std::pair{128u,4u},std::pair{64u,4u}}) {
    auto v=s;v.width=shape.first;v.group=shape.second;v.query_rows=v.group;
    llm_host::ValidateCompiledBodies({v});
  }
  for(unsigned mutation=0;mutation<6;++mutation) {
    auto v=s;
    if(mutation==0)v.group=0;
    if(mutation==1)v.width=32;
    if(mutation==2)v.kv_block=128;
    if(mutation==3)v.query_rows=47;
    if(mutation==4)v.variant=8;
    if(mutation==5)v.variant=s.variant==10?11:10;
    try {llm_host::ValidateCompiledBodies({v});return 1;}
    catch(std::invalid_argument const&) {}
  }
#else
  try {llm_host::ValidateCompiledBodies({s});return 1;}
  catch(std::invalid_argument const&) {}
#endif
  std::cout<<"{\"status\":\"pass\",\"acl_initialized\":false}\n";
}
