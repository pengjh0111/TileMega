// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <set>

namespace tilemega::solver {
namespace {
void Structure(Sm90DenseCoordinates& c) {
  c.dn_vector_sums=c.gemms[1].tile_n%32 || c.gemms[3].tile_n%32;
  c.swiglu_u=16;
  while(c.gemms[2].tile_n%(2*c.swiglu_u) && c.swiglu_u>4)c.swiglu_u/=2;
}
}
bool Sm90DenseSearchSpace::Legal(Sm90DenseCoordinates const& c) const {
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" || target.res.num_sms<=0 ||
     (head_dim!=64 && head_dim!=128) || queries<2 || queries>4)return false;
  try {c.Validate(shapes,batch);}catch(std::invalid_argument const&) {return false;}
  auto budget=std::min(target.res.max_dynamic_smem_per_cta,target.res.max_smem_per_sm/c.residency);
  if(budget<=0 || c.attention.SharedBytes(batch,head_dim,queries,1088)>budget ||
     (batch>=8 && c.attention.block_keys==16))return false;
  for(auto const& g:c.gemms)
    if(g.SharedBytes()>budget || g.mma!=Sm90DenseMma::kMmaSync ||
       g.preissue_smem_bytes || g.prefetch_l2_bytes)return false;
  return !c.attention.preissue_smem_bytes && !c.attention.prefetch_l2_bytes;
}
std::vector<Sm90DenseCoordinates> Sm90DenseSearchSpace::Seeds() const {
  std::vector<Sm90DenseCoordinates> result;
  for(int residency:{1,2}) {
    auto budget=std::min(target.res.max_dynamic_smem_per_cta,target.res.max_smem_per_sm/residency);
    if(budget<=0)continue;
    try {
      auto exact=Sm90DenseCoordinates::Exact(shapes,batch,budget);exact.residency=residency;
      for(;exact.attention.chunks<=64;exact.attention.chunks*=2)
        if(Legal(exact)) {result.push_back(exact);break;}
    }catch(std::invalid_argument const&) {}
  }
  return result;
}
std::vector<Sm90DenseCoordinates> Sm90DenseSearchSpace::Neighbours(Sm90DenseCoordinates const& center) const {
  if(!Legal(center))throw std::invalid_argument("native search center is illegal");
  std::vector<Sm90DenseCoordinates> result;std::set<std::string> seen{center.ToJson().Dump(0)};
  auto add=[&](Sm90DenseCoordinates candidate) {
    Structure(candidate);
    if(Legal(candidate) && seen.insert(candidate.ToJson().Dump(0)).second)
      result.push_back(std::move(candidate));
  };
  for(unsigned family=0;family<5;++family) {
    auto const& g=center.gemms[family];
    for(int tn=8;tn<=512 && tn<=shapes[family][0];tn+=8) {
      if(shapes[family][0]%tn)continue;
      for(int wn:{1,2,4}) {
        if(tn%(8*wn))continue;
        auto candidate=center;auto& next=candidate.gemms[family];
        next.tile_n=tn;next.warps_n=wn;next.warps_k=4/wn;
        next.slot_bytes=2*tn*g.TileK();add(std::move(candidate));
      }
    }
    for(int tk:{64,128,256,512,1024,2048})for(int slots:{2,3,4,6,8}) {
      auto candidate=center;auto& next=candidate.gemms[family];
      next.slot_bytes=2*next.tile_n*tk;next.slots=slots;add(std::move(candidate));
    }
  }
  for(int chunks:{1,2,4,8,16,32,64})for(int row_work:{0,8,16}) {
    auto candidate=center;candidate.attention.chunks=chunks;
    candidate.attention.new_row_work=row_work;add(std::move(candidate));
  }
  if(batch>=8)for(int keys:{32,64,128})for(int slots:{2,3,4,6,8}) {
    auto candidate=center;candidate.attention.block_keys=keys;
    candidate.attention.slots=slots;add(std::move(candidate));
  }
  for(int residency:{1,2}) {
    auto candidate=center;candidate.residency=residency;add(std::move(candidate));
  }
  return result;
}
}  // namespace tilemega::solver
