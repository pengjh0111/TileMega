#include <tilemega/Target/Ascend/LlmRequestHost.h>
#include <iostream>

int main() {
  using namespace tilemega::ascend;
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
  unsigned rejected=0;
  for(unsigned variant:{5u,6u,7u,99u}) {
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
  if(rejected!=9)return 1;
  std::cout<<"{\"status\":\"pass\",\"missing_device_primitive_rejected\":true,"
    "\"unknown_implementation_rejected\":true,\"missing_split_body_rejected\":true,\"acl_initialized\":false}\n";
}
