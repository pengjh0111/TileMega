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
  if(budget<=0 || (batch>=8 && c.attention.block_keys==16))return false;
  try {c.ValidateSchedule(shapes,batch,head_dim,queries,1088,budget);}
  catch(std::invalid_argument const&) {return false;}
  for(auto const& g:c.gemms)
    if(g.mma!=Sm90DenseMma::kMmaSync)return false;
  return true;
}
std::vector<Sm90DenseCoordinates> Sm90DenseSearchSpace::Seeds() const {
  std::vector<Sm90DenseCoordinates> result;
  for(int residency:{1,2}) {
    auto budget=std::min(target.res.max_dynamic_smem_per_cta,target.res.max_smem_per_sm/residency);
    if(budget<=0)continue;
    try {
      auto exact=Sm90DenseCoordinates::Exact(shapes,batch,budget);exact.residency=residency;
      for(;exact.attention.chunks<=64;exact.attention.chunks*=2)
        if(Legal(exact)) {
          result.push_back(exact);
          result.push_back(MaxPreissue(exact,false));
          result.push_back(MaxPreissue(exact,true));
          break;
        }
    }catch(std::invalid_argument const&) {}
  }
  return result;
}
Sm90DenseCoordinates Sm90DenseSearchSpace::MaxPreissue(
    Sm90DenseCoordinates const& base,bool l2) const {
  if(!Legal(base))throw std::invalid_argument("illegal native preissue seed");
  auto c=base;
  for(auto& g:c.gemms) {g.workspace_offset=0;g.preissue_smem_bytes=0;g.prefetch_l2_bytes=0;}
  c.attention.workspace_offset=0;c.attention.preissue_smem_bytes=0;c.attention.prefetch_l2_bytes=0;
  auto budget=std::min(target.res.max_dynamic_smem_per_cta,target.res.max_smem_per_sm/c.residency);
  auto& down=c.gemms[3];int offset=int(budget/2/1024)*1024;
  if(offset+down.SharedBytes()<=budget)down.workspace_offset=offset;
  auto maximize=[&](unsigned family,int maximum,int quantum) {
    int lower=0,upper=maximum/quantum;
    while(lower<upper) {
      int middle=(lower+upper+1)/2;auto candidate=c;
      if(family<5)candidate.gemms[family].preissue_smem_bytes=middle*quantum;
      else candidate.attention.preissue_smem_bytes=middle*quantum;
      if(Legal(candidate))lower=middle;else upper=middle-1;
    }
    if(family<5)c.gemms[family].preissue_smem_bytes=lower*quantum;
    else c.attention.preissue_smem_bytes=lower*quantum;
  };
  for(unsigned i=0;i<5;++i) {
    auto const& g=c.gemms[i];
    maximize(i,std::min(g.slots,shapes[i][1]/g.TileK())*g.slot_bytes,16);
  }
  int rows=batch<=4?backend::Sm90DenseLayout::AttentionKeys(
      c.attention.chunks,c.attention.new_row_work,1088):c.attention.slots*c.attention.block_keys;
  maximize(5,4*head_dim*rows,4*head_dim);
  if(l2) {
    int bytes=int(std::min<std::int64_t>(512*1024,
        target.res.l2_bytes/(2*target.res.num_sms*c.residency)))/32*32;
    for(auto& g:c.gemms)g.prefetch_l2_bytes=bytes;
    c.attention.prefetch_l2_bytes=bytes;
  }
  if(!Legal(c))throw std::invalid_argument("native preissue seed violates liveness");
  return c;
}
std::vector<Sm90DenseCoordinates> Sm90DenseSearchSpace::Neighbours(Sm90DenseCoordinates const& center) const {
  if(!Legal(center))throw std::invalid_argument("native search center is illegal");
  std::vector<Sm90DenseCoordinates> result;std::set<std::string> seen{center.ToJson().Dump(0)};
  auto add=[&](Sm90DenseCoordinates candidate) {
    Structure(candidate);
    if(Legal(candidate) && seen.insert(candidate.ToJson().Dump(0)).second)
      result.push_back(std::move(candidate));
  };
  auto preissue=[&](unsigned family,int bytes) {
    using Layout=backend::Sm90DenseLayout;
    auto candidate=center;
    auto geometry=[&](unsigned i) {
      if(i==5) {
        auto const& a=center.attention;
        return Layout::Attention(batch,head_dim,queries,a.chunks,a.block_keys,
            a.slots,a.new_row_work,1088);
      }
      auto const& g=center.gemms[i];
      return Layout::Gemm(g.tile_n,g.TileK(),g.slots,g.warps_k);
    };
    unsigned previous=std::array<unsigned,6>{3,5,1,2,3,0}[family];
    int previous_offset=previous==5?center.attention.workspace_offset:
        center.gemms[previous].workspace_offset;
    auto live=geometry(previous).Live(previous_offset);
    auto scratch=geometry(previous).Scratch(previous_offset);
    backend::Sm90DensePayload payload;
    if(family<5) {
      auto& g=candidate.gemms[family];g.preissue_smem_bytes=bytes;
      payload=Layout::GemmPayload(g.tile_n,g.TileK(),g.slots,shapes[family][1],bytes);
    }else {
      auto& a=candidate.attention;a.preissue_smem_bytes=bytes;
      payload=Layout::AttentionPayload(batch,head_dim,queries,a.chunks,a.block_keys,
          a.slots,a.new_row_work,1088,bytes);
    }
    add(candidate);
    if(!bytes)return;
    // Prefix and placement must move together: a whole first slot can overlap
    // the previous epilogue at offset zero, making either isolated move useless.
    std::set<int> offsets{0};
    for(int i=0;i<payload.count;++i) {
      auto range=payload.ranges[i];
      for(auto boundary:{live,scratch}) {
        int before=boundary.begin-range.end,after=boundary.end-range.begin;
        if(before>=0)offsets.insert(before/1024*1024);
        if(after>=0)offsets.insert((after+1023)/1024*1024);
      }
    }
    bool selected[2]={false,false};
    for(int offset:offsets) {
      if(family<5)candidate.gemms[family].workspace_offset=offset;
      else candidate.attention.workspace_offset=offset;
      bool overlap=payload.Shift(offset).Overlaps(live);
      if(!selected[overlap] && Legal(candidate)) {
        add(candidate);selected[overlap]=true;
      }
    }
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
    for(int depth:{1,2,4}) {
      auto candidate=center;candidate.gemms[family].register_stages=depth;add(std::move(candidate));
    }
    auto budget=std::min(target.res.max_dynamic_smem_per_cta,
                         target.res.max_smem_per_sm/center.residency);
    for(int offset:{0,int(budget/2/1024)*1024,int((budget-g.SharedBytes()+g.workspace_offset)/1024)*1024}) {
      auto candidate=center;candidate.gemms[family].workspace_offset=offset;add(std::move(candidate));
    }
    int maximum=std::min(g.slots,shapes[family][1]/g.TileK())*g.slot_bytes;
    for(int bytes:{0,4096,g.slot_bytes,2*g.slot_bytes,16384,32768,65536,maximum})
      preissue(family,bytes);
    for(int bytes:{0,64*1024,128*1024,194*1024}) {
      auto candidate=center;candidate.gemms[family].prefetch_l2_bytes=bytes;add(std::move(candidate));
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
  int kv_block=4*head_dim*(batch<=4?backend::Sm90DenseLayout::AttentionKeys(
      center.attention.chunks,center.attention.new_row_work,1088):center.attention.block_keys);
  for(int bytes:{0,4096,16384,32768,65536,kv_block,2*kv_block})preissue(5,bytes);
  for(int bytes:{0,64*1024,128*1024,194*1024}) {
    auto candidate=center;candidate.attention.prefetch_l2_bytes=bytes;add(std::move(candidate));
  }
  auto budget=std::min(target.res.max_dynamic_smem_per_cta,
                       target.res.max_smem_per_sm/center.residency);
  auto const& a=center.attention;
  for(int offset:{0,int(budget/2/1024)*1024,
      int((budget-a.SharedBytes(batch,head_dim,queries,1088)+a.workspace_offset)/1024)*1024}) {
    auto candidate=center;candidate.attention.workspace_offset=offset;add(std::move(candidate));
  }
  add(MaxPreissue(center,false));add(MaxPreissue(center,true));
  return result;
}
}  // namespace tilemega::solver
