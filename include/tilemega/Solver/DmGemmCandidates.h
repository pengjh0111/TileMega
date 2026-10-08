// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/DmGemmTraits.h>
#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace tilemega::solver {

struct DmWeightLayoutConstraint {
  // A shared packed allocation has one physical N/K tile. TM is independent.
  int tile_n=0,tile_k=0;
};

inline std::uint64_t DmGemmIterations(frontend::PlanGemm const& gemm,
    frontend::ModelPlan const& plan,int tile_k) {
  if(tile_k<=0)throw std::invalid_argument("DM GEMM needs a positive K tile");
  if(gemm.access.a!=codegen::DmAAccess::kIm2Col)
    return (std::uint64_t(gemm.k)+tile_k-1)/tile_k;
  auto const& conv=plan.convolutions.at(gemm.access.conv);
  auto const& layout=plan.buffers.at(conv.input_layout).layout;
  if(layout.kind!=codegen::DmLayout::kNHWC || layout.rank!=4 ||
     !layout.physical[3] || !conv.r || !conv.s)
    throw std::invalid_argument("im2col GEMM lacks an NHWC channel extent");
  auto channels=std::uint64_t(layout.physical[3]);
  auto positions=std::uint64_t(conv.r)*conv.s;
  // Large channels finish each filter position with a predicated C block;
  // small channels concatenate complete filter positions into one K tile.
  return channels>=std::uint64_t(tile_k)
      ?positions*((channels+tile_k-1)/tile_k)
      :(positions*channels+tile_k-1)/tile_k;
}

inline std::string_view DmGemmCandidateRejection(frontend::PlanGemm const& gemm,
    frontend::ModelPlan const& plan,GemmConfig const& g,TargetSpec const& target,
    DmWeightLayoutConstraint shared={}) {
  using namespace codegen;
  if((shared.tile_n==0)!=(shared.tile_k==0))
    throw std::invalid_argument("shared DM weight layout needs both tile dimensions");
  if(!plan.dm || !target.caps.cp_async)return "DM/cp.async capability";
  if(!gemm.n || !gemm.k || !DmServingBF16ShapeLegal(g.tile_m,g.tile_n,g.tile_k,g.stages))
    return "shape";
  if(DmServingBF16SmemBytes(g.tile_m,g.tile_n,g.tile_k,g.stages)>
      target.res.max_dynamic_smem_per_cta)return "shared memory";
  if(shared.tile_n && (g.tile_n!=shared.tile_n || g.tile_k!=shared.tile_k))
    return "shared weight layout";
  if(gemm.chain.count>8 || gemm.chain.side_count>5)
    throw std::invalid_argument("DM candidate epilogue exceeds descriptor capacity");
  bool swiglu=gemm.epilogue==frontend::PlanGemm::Epilogue::kSwiGLU;
  unsigned gates=0;
  for(unsigned i=0;i<gemm.chain.count;++i) {
    auto const& e=gemm.chain.operations[i];
    if(e.kind==DmEpilogueKind::kGatePair) {
      ++gates;
      if(!e.unit || e.unit>unsigned(g.tile_n)/2 || unsigned(g.tile_n)%(2*e.unit))
        return "gate pairing";
      swiglu|=e.gate==DmGatePair::kSwiGLU;
    }
  }
  if(gates>1)throw std::invalid_argument("DM chain has multiple contracting gates");
  if(gemm.epilogue==frontend::PlanGemm::Epilogue::kSwiGLU &&
     (!gemm.interleave_u || gemm.interleave_u>unsigned(g.tile_n)/2 ||
      unsigned(g.tile_n)%(2*gemm.interleave_u)))
    return "legacy gate pairing";
  if(gemm.access.a==DmAAccess::kIm2Col) {
    auto const& c=plan.convolutions.at(gemm.access.conv);
    auto channels=plan.buffers.at(c.input_layout).layout.physical[3];
    if(channels<unsigned(g.tile_k) && (!channels || unsigned(g.tile_k)%channels))
      return "small-channel packing";
  }
  if(gemm.access.b==DmBAccess::kExpertIndirect) {
    if(!gemm.access.experts || !gemm.access.expert_stride ||
       gemm.access.binding==kDmNoIndex || gemm.access.a==DmAAccess::kIm2Col)
      throw std::invalid_argument("expert GEMM lacks a valid binding descriptor");
    if(swiglu && (gemm.n%2 || gemm.n%g.tile_n))return "expert gate/up width";
    auto intermediate=swiglu?gemm.n/2:gemm.k;
    if(intermediate%g.tile_k)return "expert intermediate K";
  }
  auto iterations=DmGemmIterations(gemm,plan,g.tile_k);
  if(g.split_k<1 || iterations<std::uint64_t(g.split_k) || iterations%g.split_k)
    return "split K";
  if(gemm.epilogue==frontend::PlanGemm::Epilogue::kArgmaxPartial && g.split_k!=1)
    return "argmax split";
  return {};
}

inline std::vector<GemmConfig> DmGemmCandidates(frontend::PlanGemm const& gemm,
    frontend::ModelPlan const& plan,TargetSpec const& target,
    DmWeightLayoutConstraint shared={}) {
  if(target.res.max_dynamic_smem_per_cta<=0)
    throw std::invalid_argument("DM candidate target has no shared-memory budget");
  std::vector<GemmConfig> out;
  for(int m:{16,32,64,128})for(int n:{16,32,64,128,256})for(int k:{16,32,64,128}) {
    if(m*n>16384)continue;
    // cp.async exposes eight outstanding groups; deeper allocated pipelines
    // would keep the same live slots and only consume residency resources.
    int stage_limit=std::min(8,target.res.max_dynamic_smem_per_cta/(2*k*(m+n)));
    for(int stages=2;stages<=stage_limit;++stages)for(int split:{1,2,4,8,16,32}) {
      GemmConfig g{m,n,k,stages,split};
      if(DmGemmCandidateRejection(gemm,plan,g,target,shared).empty())out.push_back(g);
    }
  }
  return out;
}

} // namespace tilemega::solver
