// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90DenseResources.h>

namespace tilemega::solver {

void Sm90DenseSearchSpace::Structure(Sm90DenseCoordinates& c) {
  if(c.attention.chunks==1)c.attention.new_row_work=0;
  c.swiglu_u=16;
  while(c.gemms[2].tile_n%(2*c.swiglu_u) && c.swiglu_u>4)c.swiglu_u/=2;
}

bool Sm90DenseSearchSpace::Legal(Sm90DenseCoordinates const& c) const {
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" || target.res.num_sms<=0 ||
     (head_dim!=64 && head_dim!=128) || (queries!=2 && queries!=3 && queries!=4 && queries!=8))
    return false;
  try {c.Validate(shapes,batch);}catch(std::invalid_argument const&) {return false;}
  auto budget=Sm90DenseSharedBudget(target,c.residency);
  if(budget<=0)return false;
  try {c.ValidateSchedule(shapes,batch,head_dim,queries,1088,budget);}
  catch(std::invalid_argument const&) {return false;}
  for(auto const& g:c.gemms)if(g.mma!=Sm90DenseMma::kMmaSync)return false;
  return true;
}

std::vector<Sm90DenseGemm> Sm90DenseSearchSpace::FamilyCandidates(unsigned family,int residency) const {
  std::vector<Sm90DenseGemm> result;
  int grid=Grid(residency),n=shapes[family][0],k=shapes[family][1];
  auto budget=Sm90DenseSharedBudget(target,residency);
  int arena=Sm90DenseArenaBytes(budget);
  bool residual=Sm90DenseCoordinates::ResidualFamily(family);
  std::vector<int> tiles,depths;
  if(family==4) {
    for(int tn:{128,256})if(n%tn==0)tiles.push_back(tn);
    depths={64,128,256};
  }else {
    for(int tn=8;tn<=n;tn+=8)
      if(n%tn==0 && n/tn*4>=grid && n/tn<=2*grid)tiles.push_back(tn);
    for(int tk=64;tk<=2048;tk*=2)depths.push_back(tk);
  }
  for(int tn:tiles)for(int wn:{1,2,4}) {
    if(tn%(8*wn) || tn/wn>256)continue;
    for(int tk:depths) {
      if(k%tk)continue;
      Sm90DenseGemm g;g.tile_n=tn;g.warps_n=wn;g.warps_k=4/wn;g.slot_bytes=2*tn*tk;
      double w_cta=double((n/tn+grid-1)/grid)*tn*k*2;
      auto fits=[&] {return g.SharedBytes(batch,residual)-1024<=arena;};
      // D5 ring rule: the whole task in flight when it fits the arena in at
      // most eight slots; otherwise the fewest slots covering min(W_cta, 48 KB).
      bool found=false;
      if(k/tk<=8) {g.slots=std::max(2,k/tk);found=fits();}
      for(int slots=2;slots<=8 && !found;++slots) {
        g.slots=slots;
        if(double(slots)*g.slot_bytes>=std::min(w_cta,48.0*1024)) {
          if(!fits())break;
          found=true;
        }
      }
      if(!found || g.register_stages>tk/(16*g.warps_k))continue;
      try {g.Validate(n,k);}catch(std::invalid_argument const&) {continue;}
      result.push_back(g);
    }
  }
  return result;
}

std::vector<Sm90DenseAttention> Sm90DenseSearchSpace::AttentionCandidates(int residency) const {
  std::vector<Sm90DenseAttention> result;
  int s0=1;
  while(2*s0*batch*hkv<=Grid(residency))s0*=2;
  auto budget=Sm90DenseSharedBudget(target,residency);
  std::vector<int> keys=head_dim==128?std::vector<int>{64}:std::vector<int>{64,128};
  for(int s:{s0,s0/2,s0/4}) {
    if(s<1 || s>64)continue;
    for(int bk:keys) {
      // D2 slot rule: the most slots (2..4) that fit, capped by the blocks of
      // the longest chunk.
      // Within one arena when two slots fit there; otherwise the stage takes
      // the whole budget (D5 full-width stage).
      Sm90DenseAttention a;a.chunks=s;a.block_keys=bk;
      int blocks=((1088+s-1)/s+bk-1)/bk;
      a.slots=2;
      int limit=a.SharedBytes(head_dim,queries)-1024<=Sm90DenseArenaBytes(budget)?
          Sm90DenseArenaBytes(budget)+1024:budget;
      for(a.slots=std::max(2,std::min(4,blocks));a.slots>2 && a.SharedBytes(head_dim,queries)>limit;)
        --a.slots;
      if(a.SharedBytes(head_dim,queries)<=limit)result.push_back(a);
    }
  }
  return result;
}

}  // namespace tilemega::solver
