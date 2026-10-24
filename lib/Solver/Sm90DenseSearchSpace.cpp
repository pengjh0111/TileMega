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
     (head_dim!=64 && head_dim!=128) || queries<2 || queries>4)return false;
  try {c.Validate(shapes,batch);}catch(std::invalid_argument const&) {return false;}
  auto budget=Sm90DenseSharedBudget(target,c.residency);
  if(budget<=0 || (batch>=8 && c.attention.block_keys==16))return false;
  try {c.ValidateSchedule(shapes,batch,head_dim,queries,1088,budget);}
  catch(std::invalid_argument const&) {return false;}
  for(auto const& g:c.gemms)if(g.mma!=Sm90DenseMma::kMmaSync)return false;
  return true;
}

std::vector<Sm90DenseGemm> Sm90DenseSearchSpace::FamilyCandidates(unsigned family,int residency,
                                                                 bool vector_sums) const {
  std::vector<Sm90DenseGemm> result;
  int grid=Grid(residency),n=shapes[family][0],k=shapes[family][1];
  auto budget=Sm90DenseSharedBudget(target,residency);
  std::vector<int> tiles,depths;
  if(family==4) {
    for(int tn:{128,256})if(n%tn==0)tiles.push_back(tn);
    depths={64,128,256};
  }else {
    for(int tn=8;tn<=n;tn+=8)
      if(n%tn==0 && n/tn*4>=grid && n/tn<=2*grid &&
         (vector_sums || (family!=1 && family!=3) || tn%32==0))tiles.push_back(tn);
    for(int tk=64;tk<=2048;tk*=2)depths.push_back(tk);
  }
  for(int tn:tiles)for(int wn:{1,2,4}) {
    if(tn%(8*wn) || tn/wn>256)continue;
    for(int tk:depths) {
      if(k%tk)continue;
      Sm90DenseGemm g;g.tile_n=tn;g.warps_n=wn;g.warps_k=4/wn;g.slot_bytes=2*tn*tk;
      double w_cta=double((n/tn+grid-1)/grid)*tn*k*2;
      // L0 ring rule: the fewest slots whose bytes in flight cover the CTA's
      // weights or 48 KB, whichever is smaller.
      bool found=false;
      for(int slots=2;slots<=8 && !found;++slots) {
        g.slots=slots;
        if(double(slots)*g.slot_bytes>=std::min(w_cta,48.0*1024))found=true;
      }
      if(!found || g.SharedBytes(batch)>budget || g.register_stages>tk/(16*g.warps_k))continue;
      try {g.Validate(n,k);}catch(std::invalid_argument const&) {continue;}
      result.push_back(g);
    }
  }
  return result;
}

std::vector<Sm90DenseAttention> Sm90DenseSearchSpace::AttentionCandidates(int residency) const {
  std::vector<Sm90DenseAttention> result;
  int s0=1;
  while(2*s0*batch*8<=Grid(residency))s0*=2;
  std::vector<int> keys=head_dim==128 || batch<8?std::vector<int>{64}:std::vector<int>{64,128};
  for(int s:{s0,s0/2,s0/4}) {
    if(s<1 || s>64)continue;
    for(int bk:keys) {
      Sm90DenseAttention a;a.chunks=s;a.block_keys=bk;a.slots=3;
      result.push_back(a);
    }
  }
  return result;
}

}  // namespace tilemega::solver
