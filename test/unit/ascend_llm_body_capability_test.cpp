#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>
#include <utility>

int main() {
  using namespace tilemega::ascend;
  static_assert(llm::SupportsGroups(8) && !llm::SupportsGroups(28));
  static_assert(llm::SupportsGroups(14)==bool(TILEMEGA_LLM_G14));
  llm::Stage local{};local.kind=0;local.variant=2;local.rows=1024;
  local.columns=2048;local.k=8192;local.epilogue=1;local.operands[5]=0;
#if defined(TILEMEGA_LLM_LOCAL_TAIL_K7) && TILEMEGA_LLM_LOCAL_TAIL_K7
  llm_host::ValidateCompiledBodies({local});
  local.k=6144;llm_host::ValidateCompiledBodies({local});local.k=8192;
  for(unsigned rows:{16u,64u}) {
    auto invalid=local;invalid.rows=rows;
    try {llm_host::ValidateCompiledBodies({invalid});return 1;}
    catch(std::invalid_argument const&) {}
  }
  local.operands[5]=llm::kAbsent;
#endif
  try {llm_host::ValidateCompiledBodies({local});return 1;}
  catch(std::invalid_argument const&) {}
  llm::Stage qtail{};qtail.kind=0;qtail.variant=3;qtail.rows=1024;
  qtail.columns=4096;qtail.k=4096;qtail.epilogue=4;qtail.workspace=qtail.operands[3]=512;
#if defined(TILEMEGA_LLM_LOCAL_Q_TAIL_K3) && TILEMEGA_LLM_LOCAL_Q_TAIL_K3
  llm_host::ValidateCompiledBodies({qtail});
  for(unsigned k:{4096u,14336u}) {
    auto residual=qtail;residual.k=k;residual.epilogue=1;
    llm_host::ValidateCompiledBodies({residual});
    residual.workspace=llm::kAbsent;
    try {llm_host::ValidateCompiledBodies({residual});return 1;}
    catch(std::invalid_argument const&) {}
  }
  for(unsigned mutation=0;mutation<4;++mutation) {
    auto invalid=qtail;
    if(mutation==0)invalid.rows=16;
    if(mutation==1)invalid.epilogue=0;
    if(mutation==2)invalid.workspace=1024;
    if(mutation==3)invalid.operands[5]=llm::kAbsent;
    try {llm_host::ValidateCompiledBodies({invalid});return 1;}
    catch(std::invalid_argument const&) {}
  }
#else
  try {llm_host::ValidateCompiledBodies({qtail});return 1;}
  catch(std::invalid_argument const&) {}
#endif
  llm::Stage attention{};attention.kind=13;attention.variant=4;
  llm_host::ValidateCompiledBodies({attention});
  llm::Stage swiglu{};swiglu.kind=0;swiglu.variant=4;swiglu.epilogue=2;
  swiglu.rows=16;swiglu.columns=12288;swiglu.k=2048;swiglu.tasks=96;
#if defined(TILEMEGA_LLM_SWIGLU_TILE) && TILEMEGA_LLM_SWIGLU_TILE
  llm_host::ValidateCompiledBodies({swiglu});
  for(auto family:{std::pair{6144u,1024u},std::pair{16384u,2048u},
                  std::pair{16384u,3072u},std::pair{28672u,4096u}}) {
    auto s=swiglu;s.columns=family.first;s.k=family.second;s.tasks=s.columns/128;
    llm_host::ValidateCompiledBodies({s});
    ++s.tasks;
    try {llm_host::ValidateCompiledBodies({s});return 1;}
    catch(std::invalid_argument const&) {}
  }
  for(unsigned mutation=0;mutation<5;++mutation) {
    auto s=swiglu;
    if(mutation==0)s.rows=1;
    if(mutation==1)s.columns=6144;
    if(mutation==2)s.epilogue=1;
    if(mutation==3)s.extent=3;
    if(mutation==4)s.workspace=llm::kAbsent;
    try {llm_host::ValidateCompiledBodies({s});return 1;}
    catch(std::invalid_argument const&) {}
  }
#else
  try {llm_host::ValidateCompiledBodies({swiglu});return 1;}
  catch(std::invalid_argument const&) {}
#endif
  swiglu.variant=5;
#if defined(TILEMEGA_LLM_SWIGLU_L1_K256) && TILEMEGA_LLM_SWIGLU_L1_K256
  llm_host::ValidateCompiledBodies({swiglu});
  auto invalid_k256=swiglu;invalid_k256.rows=1;
  try {llm_host::ValidateCompiledBodies({invalid_k256});return 1;}
  catch(std::invalid_argument const&) {}
#else
  try {llm_host::ValidateCompiledBodies({swiglu});return 1;}
  catch(std::invalid_argument const&) {}
#endif
  unsigned rejected=0;
  for(unsigned variant:{5u,6u,7u,8u,9u,99u}) {
    attention.variant=variant;
    try {llm_host::ValidateCompiledBodies({attention});}
    catch(std::invalid_argument const&) {++rejected;}
  }
  llm::Stage norm{};norm.kind=1;norm.width=1024;
  for(unsigned variant:{2u,3u}) {
    norm.variant=variant;
    try {llm_host::ValidateCompiledBodies({norm});}
    catch(std::invalid_argument const&) {++rejected;}
  }
  llm::Stage pack{};pack.kind=25;pack.extent=8;pack.width=768;
  pack.group=1;pack.query_rows=4;pack.kv_block=128;
  llm_host::ValidateCompiledBodies({pack});
  pack.query_rows=3;
  try {llm_host::ValidateCompiledBodies({pack});}
  catch(std::invalid_argument const&) {++rejected;}
  llm::Stage partial{};partial.kind=0;partial.extent=3;
  llm::Stage combine{};combine.kind=6;
#if defined(TILEMEGA_LLM_SPLIT_K) && TILEMEGA_LLM_SPLIT_K
  for(unsigned kind:{0u,6u}) {
    llm::Stage split{};split.kind=kind;split.extent=3;split.columns=2048;
    split.k=8192;split.rows=16;split.width=2731;split.group=1;
    split.variant=1;split.epilogue=kind==6;
    llm_host::ValidateCompiledBodies({split});
    split.k=6144;split.width=2048;llm_host::ValidateCompiledBodies({split});
    for(unsigned k:{4096u,7168u}) {
      auto invalid=split;invalid.k=k;invalid.width=(k+2)/3;
      try {llm_host::ValidateCompiledBodies({invalid});return 1;}
      catch(std::invalid_argument const&) {}
    }
    split.rows=1024;
    try {llm_host::ValidateCompiledBodies({split});return 1;}
    catch(std::invalid_argument const&) {}
  }
  for(unsigned extent:{2u,7u})for(unsigned kind:{0u,6u}) {
    llm::Stage split{};split.kind=kind;split.extent=extent;split.rows=16;
    split.columns=extent==2?3072:1024;split.k=extent==2?8192:4096;
    split.width=(split.k+extent-1)/extent;split.group=1;split.variant=1;
    split.epilogue=kind==6 && extent==2;
    llm_host::ValidateCompiledBodies({split});
    for(unsigned group:{0u,2u}) {
      auto invalid=split;invalid.group=group;
      try {llm_host::ValidateCompiledBodies({invalid});return 1;}
      catch(std::invalid_argument const&) {}
    }
    split.rows=1024;
    try {llm_host::ValidateCompiledBodies({split});return 1;}
    catch(std::invalid_argument const&) {}
  }
#endif
  for(auto const& stage:{partial,combine}) {
    try {llm_host::ValidateCompiledBodies({stage});}
    catch(std::invalid_argument const&) {++rejected;}
  }
  if(rejected!=11)return 1;
  llm::Metadata m{};m.batch=16;m.capacity=1088;m.seq=64;
  llm::Stage cached{};cached.kind=13;cached.variant=9;cached.extent=8;cached.group=2;
  cached.width=128;cached.query_rows=32;cached.kv_block=1088;
  const uint64_t bytes=16ULL*8*1088*128*2;
  cached.operands[1]=512;cached.operands[2]=bytes+1024;
  cached.operands[10]=(5ULL<<30)+512;cached.operands[11]=cached.operands[10]+2*bytes+512;
  std::vector<llm::Allocation> allocations;
  for(unsigned i:{1u,2u,10u,11u}) {
    auto size=bytes*(i<10?1:2);
    allocations.push_back({cached.operands[i],size,size+512,0,0});
  }
  std::set<uint32_t> resets{0,1,2,3};
  auto pre=llm_host::ValidateCachedKvState(m,{cached},allocations,resets);
  m.seq=1;cached.query_rows=2;
  auto dec=llm_host::ValidateCachedKvState(m,{cached},allocations,resets);
  llm_host::ValidateCachedKvPair(pre,dec);
  for(auto shape:{std::pair<unsigned,unsigned>{128,3},{128,4},{64,4}}) {
    auto s=cached;s.width=shape.first;s.group=shape.second;s.query_rows=s.group;
    auto a=allocations;for(auto& p:a){p.bytes=p.bytes*s.width/128;p.allocated=p.bytes+512;}
    auto d=llm_host::ValidateCachedKvState(m,{s},a,resets);
    auto pre_m=m;pre_m.seq=64;s.query_rows=16*s.group;
    llm_host::ValidateCachedKvPair(llm_host::ValidateCachedKvState(pre_m,{s},a,resets),d);
    s.group=5;s.query_rows=80;
    try {llm_host::ValidateCachedKvState(pre_m,{s},a,resets);return 1;}
    catch(std::invalid_argument const&) {}
  }
  try {llm_host::ValidateCachedKvPair(pre,{});return 1;}
  catch(std::invalid_argument const&) {}
  for(unsigned mutation=0;mutation<4;++mutation) {
    auto s=cached;auto a=allocations;auto r=resets;
    if(mutation==0)r.erase(2);
    if(mutation==1)s.operands[11]=s.operands[10];
    if(mutation==2)a[2].bytes-=4;
    if(mutation==3)a[3].required=1;
    try {llm_host::ValidateCachedKvState(m,{s},a,r);return 1;}
    catch(std::invalid_argument const&) {}
  }
  std::cout<<"{\"status\":\"pass\",\"missing_device_primitive_rejected\":true,"
    "\"unknown_implementation_rejected\":true,\"missing_split_body_rejected\":true,\"acl_initialized\":false}\n";
}
