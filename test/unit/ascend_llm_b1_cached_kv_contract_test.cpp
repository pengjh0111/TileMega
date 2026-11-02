#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
int main() {
  using namespace tilemega::ascend;
  auto state=[](unsigned batch,bool broken) {
    llm::Metadata m{};m.batch=batch;m.capacity=1088;m.seq=1;
    llm::Stage s{};s.kind=13;s.variant=9;s.extent=8;s.group=2;s.width=128;
    s.query_rows=2;s.kv_block=1088;
    uint64_t bytes=uint64_t(batch)*8*1088*128*2,offset=5ULL<<30;
    std::vector<llm::Allocation> allocations;
    for(unsigned i:{1u,2u,10u,11u}) {
      auto n=bytes*(i<10?1:2);s.operands[i]=offset;
      allocations.push_back({offset,n,n+512,0,0});offset+=n+512;
    }
    std::set<uint32_t> resets{0,1,2,3};if(broken)resets.erase(3);
    auto decode=llm_host::ValidateCachedKvState(m,{s},allocations,resets);
    m.seq=64;s.query_rows=32;
    auto prefill=llm_host::ValidateCachedKvState(m,{s},allocations,resets);
    llm_host::ValidateCachedKvPair(prefill,decode);
  };
  state(16,false);
#if defined(TILEMEGA_LLM_FP32_KV_CACHE_B1) && TILEMEGA_LLM_FP32_KV_CACHE_B1
  state(1,false);
#else
  try {state(1,false);return 1;}
  catch(std::invalid_argument const&) {}
#endif
  for(unsigned batch:{1u,4u,16u}) {
    try {state(batch,true);return 1;}
    catch(std::invalid_argument const&) {}
  }
  std::cout<<"{\"status\":\"pass\",\"acl_initialized\":false}\n";
}
