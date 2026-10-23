#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>

int main() {
  using namespace tilemega::ascend;
  llm::Stage attention{};attention.kind=13;attention.variant=4;
  llm_host::ValidateCompiledBodies({attention});
  unsigned rejected=0;
  for(unsigned variant:{5u,6u,99u}) {
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
    for(unsigned k:{6144u,4096u}) {
      auto invalid=split;invalid.k=k;invalid.width=(k+2)/3;
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
  if(rejected!=8)return 1;
  std::cout<<"{\"status\":\"pass\",\"missing_device_primitive_rejected\":true,"
    "\"unknown_implementation_rejected\":true,\"missing_split_body_rejected\":true,\"acl_initialized\":false}\n";
}
