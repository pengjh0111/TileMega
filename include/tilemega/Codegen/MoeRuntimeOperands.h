// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/tasks/ModelRuntime.h>
#include <stdexcept>

namespace tilemega::codegen {
inline void ValidateMoeRuntimeOperand(ModelSpec const& spec, StageDesc const& stage,
    ModelDims const& dims, unsigned slot, unsigned dtype, std::uint64_t elements,
    bool optional=false) {
  if(slot>=16)throw std::invalid_argument("MoE operand slot outside descriptor");
  auto id=stage.operand[slot];
  if(optional && id==kNoOperand)return;
  if(id>=spec.buffer_count || spec.buffers[id].dtype!=dtype ||
      spec.buffers[id].Elements(dims)<elements)
    throw std::invalid_argument("invalid MoE runtime operand type or extent");
}
inline void ValidateMoeRoutingWeights(ModelSpec const& spec, StageDesc const& stage,
    ModelDims const& dims, unsigned slot, std::uint64_t elements) {
  if(slot>=16 || stage.operand[slot]>=spec.buffer_count)
    throw std::invalid_argument("missing MoE routing weight operand");
  auto dtype=spec.buffers[stage.operand[slot]].dtype;
  // Modern HF preserves routing probabilities in FP32 through the weighted
  // expert sum; the older BF16 boundary remains a separate valid contract.
  if(dtype!=0 && dtype!=1)
    throw std::invalid_argument("MoE routing weights must be BF16 or FP32");
  ValidateMoeRuntimeOperand(spec,stage,dims,slot,dtype,elements);
}
} // namespace tilemega::codegen
