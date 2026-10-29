// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90DenseResources.h>

namespace tilemega::solver {

void Sm90DenseSearchSpace::Structure(Sm90DenseCoordinates& c) {
  if(c.attention.chunks==1)c.attention.new_row_work=0;
  c.swiglu_u=16;
  if(!c.moe)while(c.gemms[2].tile_n%(2*c.swiglu_u) && c.swiglu_u>4)c.swiglu_u/=2;
}

bool Sm90DenseSearchSpace::Legal(Sm90DenseCoordinates const& c) const {
  // Q = 1 is OLMoE's MHA (A3); the attention body's constraint is Q in [1, 8].
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" || target.res.num_sms<=0 ||
     (head_dim!=64 && head_dim!=128) ||
     (queries!=1 && queries!=2 && queries!=3 && queries!=4 && queries!=8))
    return false;
  try {c.Validate(shapes,batch);}catch(std::invalid_argument const&) {return false;}
  auto budget=Sm90DenseSharedBudget(target,c.residency);
  if(budget<=0)return false;
  try {c.ValidateSchedule(shapes,batch,head_dim,queries,1088,budget);}
  catch(std::invalid_argument const&) {return false;}
  for(auto const& g:c.gemms)if(g.mma!=Sm90DenseMma::kMmaSync)return false;
  // D8: every qkv tile lies in one KV group of (Q + 2) D columns.
  if(c.group_wait && ((queries+2)*head_dim)%c.gemms[0].tile_n)return false;
  int grid=Grid(c.residency);
  if(c.cluster>1 && grid%c.cluster)return false;
  // D9: every CTA publishes one argmax partial into the head's [B][tasks]
  // outputs, so the head needs at least one task per CTA.
  if(shapes[4][0]/c.gemms[4].tile_n<grid)return false;
  // D10c: both CTAs of a pair run the same task count.
  for(unsigned f:c.Active())
    if(c.gemms[f].multicast && ((shapes[f][0]/c.gemms[f].tile_n)%grid)%2)return false;
  if(c.attention.dsmem) {
    int arena=Sm90DenseArena(budget);
    // D10b: one task wave; a group-waiting attention may start while a peer
    // still runs qkv, so its DSMEM inbox must lie outside the qkv arena.
    if(batch*hkv*c.attention.chunks>grid ||
       (c.group_wait && c.attention.FullWidth(head_dim,queries,arena)))
      return false;
  }
  return true;
}

std::vector<Sm90DenseGemm> Sm90DenseSearchSpace::FamilyCandidates(unsigned family,int residency) const {
  std::vector<Sm90DenseGemm> result;
  int grid=Grid(residency),n=shapes[family][0],k=shapes[family][1];
  auto budget=Sm90DenseSharedBudget(target,residency);
  int arena=Sm90DenseArenaBytes(budget);
  bool residual=Sm90DenseCoordinates::ResidualFamily(family);
  std::vector<int> tiles,depths;
  // Tasks of the busiest CTA: experts run n_active x N/TN tasks (§8).
  auto tasks=[&](int tn) {
    double count=double(n/tn)*(family>=6?ExpectedActive():1);
    return std::ceil(count/grid);
  };
  if(family==4) {
    for(int tn:{128,256})if(n%tn==0 && n/tn>=grid)tiles.push_back(tn);
    depths={64,128,256};
  }else if(family>=5) {
    // M4: TN8, 16 tasks. The expert tiles belong to the geometry: R3 fitted
    // {96, 192} / {128, 256} to Qwen3's 128 experts of width 768, and A3 gives
    // OLMoE's 64 experts of width 1024 TN in {64, 128, 256}. Validate below
    // drops any tile the shape does not divide.
    if(family==5)tiles={8};
    else if(experts==64)tiles={64,128,256};
    else if(family==6)tiles=batch==1?std::vector<int>{96}:std::vector<int>{96,192};
    else tiles={128,256};
    for(int tk=64;tk<=2048;tk*=2)depths.push_back(tk);
  }else {
    for(int tn=8;tn<=n;tn+=8)
      if(n%tn==0 && n/tn*4>=grid && n/tn<=2*grid)tiles.push_back(tn);
    for(int tk=64;tk<=2048;tk*=2)depths.push_back(tk);
  }
  for(int tn:tiles)for(int wn:{1,2,4}) {
    if(tn%(8*wn) || tn/wn>256)continue;
    // gate_up tiles an expert stack interleaved at U = 16 (pinned for native
    // MoE), so its TN spans whole 2U pairs.
    if(family==6 && tn%32)continue;
    for(int tk:depths) {
      if(k%tk)continue;
      Sm90DenseGemm g;g.tile_n=tn;g.warps_n=wn;g.warps_k=4/wn;g.slot_bytes=2*tn*tk;
      double w_cta=tasks(tn)*tn*k*2;
      auto fits=[&] {return g.SharedBytes(batch,residual)-1024<=arena;};
      // D5 ring rule: the whole task in flight when it fits the arena in at
      // most eight slots; otherwise the fewest slots covering min(W_cta, 48 KB).
      // MR2: an expert stream (384 KB per task at Qwen3) keeps the most slots
      // the arena holds; 48 KB in flight left it latency-bound (R3 MC2).
      bool found=false;
      if(family>=6)
        for(int slots=8;slots>=2 && !found;--slots) {g.slots=slots;found=fits();}
      if(!found && k/tk<=8) {g.slots=std::max(2,k/tk);found=fits();}
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
  // MR5: every power-of-two chunk count down to one (no merge).
  for(int s=s0;s>=1;s/=2) {
    if(s>64)continue;
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
