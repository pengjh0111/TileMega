// SPDX-License-Identifier: BSD-3-Clause
#define TILEMEGA_DM_SUPPORT 1
#include <tilemega/Codegen/MoeRuntimeOperands.h>
#include <cassert>

using namespace tilemega::codegen;
int main() {
  BufferDesc buffer{};buffer.constant=16;
  StageDesc stage{};stage.operand[1]=stage.operand[3]=0;
  ModelSpec spec{};spec.buffers=&buffer;spec.buffer_count=1;
  ModelDims dims{};dims.seq=2;dims.batch=1;
  auto rejected=[](auto action) {
    bool failed=false;
    try{action();}catch(std::invalid_argument const&){failed=true;}
    assert(failed);
  };
  for(unsigned slot:{1u,3u})for(unsigned dtype:{0u,1u}) {
    buffer.dtype=dtype;
    ValidateMoeRoutingWeights(spec,stage,dims,slot,16);
    rejected([&]{ValidateMoeRoutingWeights(spec,stage,dims,slot,17);});
    rejected([&]{ValidateMoeRuntimeOperand(spec,stage,dims,slot,2,16);});
  }
  for(unsigned dtype:{2u,3u,4u}) {
    buffer.dtype=dtype;
    rejected([&]{ValidateMoeRoutingWeights(spec,stage,dims,1,16);});
  }
  stage.operand[1]=kNoOperand;
  ValidateMoeRuntimeOperand(spec,stage,dims,1,0,16,true);
  rejected([&]{ValidateMoeRoutingWeights(spec,stage,dims,1,16);});
  rejected([&]{ValidateMoeRuntimeOperand(spec,stage,dims,1,0,16);});
  rejected([&]{ValidateMoeRoutingWeights(spec,stage,dims,16,16);});
  rejected([&]{ValidateMoeRuntimeOperand(spec,stage,dims,16,0,16);});
}
