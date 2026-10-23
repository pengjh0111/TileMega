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
  for(auto const& stage:{partial,combine}) {
    try {llm_host::ValidateCompiledBodies({stage});}
    catch(std::invalid_argument const&) {++rejected;}
  }
  if(rejected!=8)return 1;
  std::cout<<"{\"status\":\"pass\",\"missing_device_primitive_rejected\":true,"
    "\"unknown_implementation_rejected\":true,\"missing_split_body_rejected\":true,\"acl_initialized\":false}\n";
}
