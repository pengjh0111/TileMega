// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Frontend/ModelPlan.h>
namespace tilemega::frontend {
struct DnnPlanOptions {
  unsigned batch=1;
  unsigned small_input_channels=8;
  unsigned depthwise_rows=1,depthwise_channels=64;
  std::uint64_t workspace_budget_bytes=0;
  std::string memory_reuse="none";
  std::uint64_t memory_l2_budget_bytes=0;
  bool deferred_layernorm=false;
};
ModelPlan BuildDnnModelPlan(std::vector<FxNodeRecord> const&,
    std::vector<SignatureInput> const&,std::vector<std::string> const&,
    DnnPlanOptions const& = {});
} // namespace tilemega::frontend
