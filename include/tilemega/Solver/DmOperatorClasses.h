// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Analysis/CouplingCache.h>
#include <tilemega/Solver/DmSemanticSignature.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <map>
#include <sstream>
#include <stdexcept>

namespace tilemega::solver {

inline std::string GemmSemanticSignature(analysis::SemanticOp const& op,
    frontend::PlanGemm const& gemm, frontend::ModelPlan const& plan) {
  using namespace codegen;
  auto base=plan.dm && op.exact_task_access ? DmSemanticSignature(op)
      : analysis::SemanticSignature(op);
  auto const& a=gemm.access;
  bool extended=a.a!=DmAAccess::kDense || a.b!=DmBAccess::kDense ||
      a.rows_per_batch || a.a_row_stride>1 || a.a_row_offset ||
      a.conv!=kDmNoIndex || a.rows!=kDmNoIndex || a.binding!=kDmNoIndex ||
      a.a_scale!=kDmNoIndex || a.expert_stride || a.binding_blocks ||
      a.binding_rows || a.experts || a.block_rows ||
      a.write.kind!=DmWriteKind::kDense || a.write.factor!=1 ||
      a.write.layout!=kDmNoIndex || a.write.rows!=kDmNoIndex ||
      gemm.chain.count || gemm.chain.side_count;
  for(auto id:{gemm.a,gemm.b,gemm.d})
    extended|=plan.buffers.at(id).layout.rank!=0;
  // Default descriptors preserve the legacy class keys, including DM dense plans.
  if(!extended)return base;
  if(!plan.dm)throw std::invalid_argument("extended GEMM class requires DM descriptors");
  if(gemm.chain.count>8 || gemm.chain.side_count>5)
    throw std::invalid_argument("GEMM class epilogue exceeds descriptor capacity");
  std::ostringstream out;
  out<<base<<"\nDM-GEMM-v1;";
  auto put=[&](auto value){out<<static_cast<std::uint64_t>(value)<<',';};
  std::map<std::uint32_t,std::size_t> names;
  auto buffer=[&](std::uint32_t id) {
    if(id==kDmNoIndex) {out<<"none;";return;}
    auto const& b=plan.buffers.at(id);
    auto [it,added]=names.emplace(id,names.size());
    out<<"buffer"<<it->second<<':'<<b.dtype.size()<<':'<<b.dtype<<';';
    auto const& l=b.layout;
    put(l.kind);put(l.rank);
    for(auto x:l.logical)put(x);
    for(auto x:l.physical)put(x);
    for(auto x:l.strides)put(x);
    put(l.halo_top);put(l.halo_bottom);put(l.halo_left);put(l.halo_right);put(l.fill);
    out<<';';
  };
  auto write=[&](DmWriteMap const& w) {
    put(w.kind);put(w.factor);buffer(w.layout);buffer(w.rows);
  };
  // Canonical buffer ordinals retain alias relationships without FQNs or plan IDs.
  buffer(gemm.a);buffer(gemm.b);buffer(gemm.d);
  put(a.a);put(a.b);put(a.rows_per_batch);
  put(a.a_row_stride ? a.a_row_stride:1);put(a.a_row_offset);
  buffer(a.rows);buffer(a.binding);buffer(a.a_scale);
  put(a.expert_stride);put(a.binding_blocks);put(a.binding_rows);
  put(a.experts);put(a.block_rows);write(a.write);
  if(a.conv==kDmNoIndex)out<<"no-conv;";
  else {
    auto const& c=plan.convolutions.at(a.conv);
    for(auto x:{c.n,c.h,c.w,c.c,c.k,c.r,c.s,c.stride_h,c.stride_w,
                c.pad_h,c.pad_w,c.dilation_h,c.dilation_w,c.p,c.q})put(x);
    buffer(c.input_layout);buffer(c.output_layout);
  }
  auto const& chain=gemm.chain;
  put(chain.count);put(chain.side_count);put(chain.store_rounding);
  for(unsigned i=0;i<chain.count;++i) {
    auto const& e=chain.operations[i];put(e.kind);
    for(auto id:e.parameter)buffer(id);
    write(e.residual_map);put(e.activation);put(e.gate);put(e.unit);
    put(e.input_rounding);put(e.output_rounding);
  }
  for(unsigned i=0;i<chain.side_count;++i) {
    auto const& s=chain.side[i];put(s.kind);put(s.count);
    buffer(s.buffer);buffer(s.auxiliary);
  }
  for(auto const& stage:plan.stages)if(stage.kind==frontend::PlanTaskKind::kDwPwFused) {
    auto const& owner=plan.gemms.at(stage.gemm);
    if(&owner!=&gemm && (owner.a!=gemm.a || owner.b!=gemm.b || owner.d!=gemm.d))continue;
    out<<"dwpw;";auto const& c=plan.convolutions.at(stage.conv);
    for(auto x:{c.n,c.h,c.w,c.c,c.k,c.r,c.s,c.stride_h,c.stride_w,
                c.pad_h,c.pad_w,c.dilation_h,c.dilation_w,c.p,c.q})put(x);
    buffer(c.input_layout);buffer(stage.operands[1]);
    put(stage.chain.count);put(stage.chain.store_rounding);
    for(unsigned i=0;i<stage.chain.count;++i) {
      auto const& e=stage.chain.operations[i];put(e.kind);
      for(auto id:e.parameter)buffer(id);
      put(e.activation);put(e.input_rounding);put(e.output_rounding);
    }
  }
  return out.str();
}

}  // namespace tilemega::solver
