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
  if(rejected!=5)return 1;
  std::cout<<"{\"status\":\"pass\",\"missing_device_primitive_rejected\":true,"
    "\"unknown_implementation_rejected\":true,\"acl_initialized\":false}\n";
}
