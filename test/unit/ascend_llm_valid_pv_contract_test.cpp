#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
int main(int argc,char** argv) {
  using namespace tilemega::ascend;
  llm::Stage s{};s.kind=13;s.variant=12;s.width=128;s.kv_block=1088;
  s.extent=8;s.group=2;s.query_rows=2;s.rows=1;s.tasks=8;
  static_assert(llm::IsCubeAttention(12) && llm::IsCachedAttention(12));
  static_assert(!llm::IsStreamingAttention(12) && llm::IsStreamingAttention(11));
#if defined(TILEMEGA_LLM_VALID_PV) && TILEMEGA_LLM_VALID_PV
  llm_host::ValidateCompiledBodies({s});
  s.width=64;
  try {llm_host::ValidateCompiledBodies({s});return 1;}
  catch(std::invalid_argument const&) {}
  if(argc==2) {
    llm_host::Plan plan(argv[1]);
    if(!std::any_of(plan.stages.begin(),plan.stages.end(),[](auto const& t){return t.kind==13 && t.variant==12;}))return 1;
  }
#else
  try {llm_host::ValidateCompiledBodies({s});return 1;}
  catch(std::invalid_argument const&) {}
#endif
  s.variant=13;
  try {llm_host::ValidateCompiledBodies({s});return 1;}
  catch(std::invalid_argument const&) {}
  std::cout<<"{\"status\":\"pass\",\"acl_initialized\":false}"<<std::endl;
}
