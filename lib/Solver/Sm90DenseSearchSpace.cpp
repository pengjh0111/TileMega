// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <tilemega/Solver/Sm90NativeConstants.h>

namespace tilemega::solver {

int Sm90DenseSearchSpace::DefaultGateInterleave(int tile_n) {
  int u=16;
  while(u>4 && tile_n%(2*u))u/=2;
  return u;
}

void Sm90DenseSearchSpace::Structure(Sm90DenseCoordinates& c,bool preserve_gate_interleave) {
  if(c.attention.chunks==1)c.attention.new_row_work=0;
  bool valid=c.swiglu_u==4 || c.swiglu_u==8 || c.swiglu_u==16;
  if(c.moe)c.swiglu_u=16;
  else if(!preserve_gate_interleave || (valid && c.gemms[2].tile_n%(2*c.swiglu_u)))
    c.swiglu_u=DefaultGateInterleave(c.gemms[2].tile_n);
}

bool Sm90DenseSearchSpace::Legal(Sm90DenseCoordinates const& c) const {
  // Q = 1 is OLMoE's MHA (A3); the attention body's constraint is Q in [1, 8].
  bool ampere=target.arch_tag=="sm_80" && target.NvccArch()=="sm_80" &&
      target.caps.cp_async && target.caps.mbarrier && !target.caps.tma && !target.caps.cluster;
  if((!ampere && (target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a")) || target.res.num_sms<=0 ||
     (head_dim!=64 && head_dim!=128) ||
     (queries!=1 && queries!=2 && queries!=3 && queries!=4 && queries!=8))
    return false;
  if(c.live_epilogue_rows && !ampere)return false;
  if(ampere && (c.cluster!=1 || c.attention.dsmem || c.tp!=1 ||
      std::any_of(c.gemms.begin(),c.gemms.end(),[](auto const& g){return g.multicast;})))
    return false;
  try {c.Validate(shapes,batch);}catch(std::invalid_argument const&) {return false;}
  auto budget=Sm90DenseSharedBudget(target,c.residency);
  if(budget<=0)return false;
  int grid=Grid(c.residency);
  try {c.ValidateSchedule(shapes,batch,head_dim,queries,1088,budget,grid);}
  catch(std::invalid_argument const&) {return false;}
  for(auto const& g:c.gemms)if(g.mma!=Sm90DenseMma::kMmaSync)return false;
  // D8: every qkv tile lies in one KV group of (Q + 2) D columns.
  if(c.group_wait && ((queries+2)*head_dim)%c.gemms[0].tile_n)return false;
  // A1 (RC1): a whole-row q/k norm reads every KV group of its token, but D8
  // releases only the groups this CTA's own qkv tiles wrote, so the attention
  // would read groups another CTA has not finished writing.
  if(c.group_wait && qk_norm_row)return false;
  // Unfused experts read a schedule published by global routing.
  if(c.moe && !c.expert_fuse && !c.GlobalRoute())return false;
  // The declared degree and the coordinate's must agree, and a whole-row norm
  // needs every head of the token, which no single rank holds.
  if(c.tp!=tp)return false;
  if(tp>1 && qk_norm_row)return false;
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
    // A sharded vocabulary is not a multiple of 128: Qwen3-1.7B gives 75968 at
    // TP2 and 37984 at TP4, which admit 16 and 32 but not 128 or 256. D9 still
    // requires at least one lm_head task per CTA.
    if(tp>1)for(int tn:{16,32,64,128,256}) {
      if(n%tn==0 && n/tn>=grid)tiles.push_back(tn);
    }else for(int tn:{128,256})if(n%tn==0 && n/tn>=grid)tiles.push_back(tn);
    depths={64,128,256};
  }else if(family>=5) {
    // M4: TN8, 16 tasks. The expert tiles belong to the geometry: R3 fitted
    // {96, 192} / {128, 256} to Qwen3's 128 experts of width 768, and A3 gives
    // OLMoE's 64 experts of width 1024 TN in {64, 128, 256}. Validate below
    // drops any tile the shape does not divide.
    if(target.arch_tag=="sm_80")
      for(int tn:{8,16,32,64,96,128,192,256}) {
        if(n%tn==0 && (family!=5 || n/tn<grid))tiles.push_back(tn);
      }
    else if(family==5)tiles={8};
    else if(experts==64)tiles={64,128,256};
    else if(family==6)tiles=batch==1?std::vector<int>{96}:std::vector<int>{96,192};
    else tiles={128,256};
    for(int tk=64;tk<=2048;tk*=2)depths.push_back(tk);
  }else {
    // Dense gates can issue a third task with the same ring. Unlike the
    // residual families, there is no per-task side buffer to enlarge.
    int waves=target.arch_tag=="sm_80" && (residual || family==2)?3:2;
    for(int tn=8;tn<=n;tn+=8)
      if(n%tn==0 && n/tn*4>=grid && n/tn<=waves*grid)
        tiles.push_back(tn);
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
      auto fits=[&] {return g.SharedBytes(batch,residual,n,grid)-1024<=arena;};
      if(target.arch_tag=="sm_80") {
        for(int slots=2;slots<=8;++slots) {
          g.slots=slots;
          if(!fits())break;
          for(int stages:{1,2,4}) {
            if(stages>tk/(16*g.warps_k))break;
            g.register_stages=stages;
            try {g.Validate(n,k);}catch(std::invalid_argument const&) {continue;}
            result.push_back(g);
          }
        }
        continue;
      }
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

std::vector<Sm90DenseAttention> Sm90DenseSearchSpace::AttentionCandidates(
    int residency,Sm90NativeConstants const* constants,int norm) const {
  std::vector<Sm90DenseAttention> result;
  int s0=1;
  int fill=std::min(64,Grid(residency)/(batch*hkv));
  while(2*s0<=fill)s0*=2;
  std::vector<int> chunks;
  for(int s=s0;s>=1;s/=2)chunks.push_back(s);
  // On Ampere, also price the chunk count that fills the last KV group in one wave.
  if(target.arch_tag=="sm_80" && constants && fill>s0 &&
      constants->HasAttentionMerge(head_dim,queries,norm,fill))chunks.push_back(fill);
  auto budget=Sm90DenseSharedBudget(target,residency);
  std::vector<int> keys=head_dim==128?std::vector<int>{64}:std::vector<int>{64,128};
  for(int s:chunks) {
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
      if(a.SharedBytes(head_dim,queries)>limit)continue;
      result.push_back(a);
      // Preserve the control ring depth for a matched comparison at the new grid fill.
      if(s>s0) {
        auto control=std::find_if(result.begin(),result.end(),[&](auto const& other) {
          return other.chunks==s0 && other.block_keys==bk;
        });
        if(control!=result.end() && control->slots!=a.slots) {
          a.slots=control->slots;
          if(a.SharedBytes(head_dim,queries)<=limit)result.push_back(a);
        }
      }
    }
  }
  return result;
}

}  // namespace tilemega::solver
