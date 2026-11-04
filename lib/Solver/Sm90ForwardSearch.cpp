// SPDX-License-Identifier: BSD-3-Clause
// DNN1 S2/S4 CPU-only draft, not enabled in the tracked native solver.
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <chrono>
#include <numeric>

namespace tilemega::solver {
namespace {
std::string S(long long x){return std::to_string(x);}
Sm90ForwardSelector Tile(int m,int n) {
  return {{"tm",S(m)},{"tn",S(n)},{"elements",S(1LL*m*n)}};
}
void Term(json::Array& terms,std::string const& name,double value,std::string source) {
  if(!std::isfinite(value) || value<0)throw std::invalid_argument("invalid prediction term: "+name);
  terms.emplace_back(json::Object{{"name",name},{"ns",value},{"source",std::move(source)}});
}
double Measured(Sm90ForwardConstants const& constants,std::string const& metric,
    Sm90ForwardSelector const& selector,std::string const& unit,double scale,json::Array& terms) {
  if(!std::isfinite(scale) || !(scale>0))throw std::invalid_argument("nonpositive measured-term multiplicity");
  auto found=constants.Get(metric,selector,unit);double value=found.ns*scale;
  terms.emplace_back(json::Object{{"name",metric},{"ns",value},{"coefficient",found.ToJson()},
      {"multiplicity",scale}});return value;
}
bool Matrix(Sm90ForwardBody body) {
  return body==Sm90ForwardBody::kGemm || body==Sm90ForwardBody::kConv;
}
bool Fused(Sm90ForwardBody body) {
  return body==Sm90ForwardBody::kDwPw || body==Sm90ForwardBody::kInvertedResidual;
}
std::string NativeKey(Sm90ForwardConstants const& c,std::string const& key) {
  return c.File()+"#sm90_native_constants."+key;
}
json::Value ResourceReceipt(Sm90ForwardResourceUsage const& r) {
  return json::Object{{"shared_bytes",r.shared_bytes},{"slot_bytes",r.slot_bytes},
      {"store_columns",r.store_columns},{"arena_bytes",r.arena_bytes},{"budget_bytes",r.budget_bytes},
      {"task_count",double(r.task_count)},{"exclusive",r.exclusive},{"feasible",r.feasible},
      {"reason",r.reason},{"source","ForwardGeometry.h::QuerySm90ForwardResources; Backend/Sm90ForwardResources.h"}};
}
json::Array const& Steps(Sm90ForwardShape const& s,char const* key) {
  auto const* steps=s.dnn1_geometry.Find(key);
  if(!steps)throw Sm90ForwardMissing(std::string("plan metadata must explicitly provide ")+key+", including an empty array");
  return steps->AsArray(key);
}
} // namespace

Sm90ForwardCostModel::TaskWork Sm90ForwardCostModel::TaskBytes(
    Sm90ForwardShape const& s,Sm90ForwardStage const& c) {
  auto const& g=s.dnn1_geometry;
  long long rows=0,channels=0,bytes=0;
  switch(s.body) {
    case Sm90ForwardBody::kDepthwise: {
      long long halo_w=(c.tile_w-1)*s.stride+3,halo_h=(c.tile_h-1)*s.stride+3;
      channels=std::min<long long>(s.channels,c.channel_block);rows=c.tile_w*c.tile_h;
      bytes=2*(halo_w*halo_h*c.channel_block+channels*9*ForwardField(g,"depthwise_pack"));
      return {bytes,bytes+2*rows*channels,rows*channels};
    }
    case Sm90ForwardBody::kDwPw:case Sm90ForwardBody::kInvertedResidual: {
      long long halo_w=(c.tile_w-1)*s.stride+3,halo_h=(c.tile_h-1)*s.stride+3;
      long long blocks=ForwardCeil(ForwardField(g,"padded_channels"),c.channel_block);
      rows=c.tile_w*c.tile_h;
      // Actual DWPW stream: halo and TN*CB pointwise pages per C block,
      // packed filters only for live channels, then one output tile.
      bytes=2*(blocks*(halo_w*halo_h*c.channel_block+c.tile_n*c.channel_block)+
          s.channels*9*ForwardField(g,"depthwise_pack"));
      return {bytes,bytes+2*rows*std::min<long long>(s.out_channels,c.tile_n),rows*s.channels};
    }
    case Sm90ForwardBody::kEncoderAttention: {
      auto seq=ForwardField(g,"seq"),dim=ForwardField(g,"head_dim");
      bytes=2*(2*seq*dim+c.tile_m*dim);
      return {bytes,bytes+2*c.tile_m*dim,c.tile_m*dim};
    }
    case Sm90ForwardBody::kLayout:
      rows=std::min<long long>(128,ForwardField(g,"width")*ForwardField(g,"height"));
      channels=ForwardField(g,"input_channels");return {0,4*rows*channels,rows*channels};
    case Sm90ForwardBody::kPool:
      channels=s.channels;return {0,2*channels*(s.kernel*s.kernel+1),channels};
    case Sm90ForwardBody::kGlobalPool:
      channels=std::min<long long>(128,s.channels);rows=ForwardField(g,"parts");
      return {0,channels*(8*rows+2),channels*rows};
    case Sm90ForwardBody::kGemv:
      channels=std::min<long long>(c.tile_n,s.n);bytes=2*(1LL*c.tile_n*s.k+s.k);
      return {bytes,bytes+6*channels,channels*s.k};
    case Sm90ForwardBody::kLayerNorm:
      rows=std::min<long long>(4,ForwardField(g,"rows"));
      return {0,10*rows*s.channels,rows*s.channels};
    case Sm90ForwardBody::kEmbedding:
      rows=std::min<long long>(4,ForwardField(g,"rows"));
      return {0,8*rows*s.channels+24*rows,rows*s.channels};
    default:throw std::invalid_argument("TaskBytes requested for a matrix body");
  }
}

double Sm90ForwardCostModel::Epilogue(Sm90ForwardShape const& s,Sm90ForwardStage const& c,
    Sm90ForwardResourceUsage const& resource,json::Array& terms) const {
  auto const& g=s.dnn1_geometry;double result=0;
  int ep_m=s.body==Sm90ForwardBody::kDepthwise?c.tile_w*c.tile_h:(Fused(s.body)?64:c.tile_m);
  int ep_n=s.body==Sm90ForwardBody::kDepthwise?c.channel_block:c.tile_n;
  // §7.2: every step uses the exact [op][TM*TN] entry from the new measured
  // forward section. Sources include the file and complete measured key.
  for(auto const& entry:Steps(s,"epilogue_steps")) {
    auto selector=Tile(ep_m,ep_n);selector["op"]=entry.AsString("epilogue op");
    result+=Measured(forward_,"epilogue_step_ns",selector,"ns",1,terms);
  }
  if(resource.store_columns<=0)
    throw Sm90ForwardMissing("resource helper did not provide actual epilogue store_columns");
  // Store geometry can be smaller than TN (gate halves, final N=3, tails).
  // Missing widths are errors until the required-keys probe manifest is run.
  auto store=Tile(ep_m,int(resource.store_columns));
  store["op"]=ForwardString(g,"store_kind");
  result+=Measured(forward_,"epilogue_store_ns",store,"ns",1,terms);
  if(c.split_k>1) {
    auto selector=Tile(c.tile_m,c.tile_n);selector["slices"]=S(c.split_k);
    result+=Measured(forward_,"splitk_partial_ns",selector,"ns",1,terms);
  }else Term(terms,"splitk_partial_ns",0,"structural: split_k=1 has no partial or reduction");
  auto const* row=g.Find("has_row_stats");
  if(!row)throw Sm90ForwardMissing("missing plan-derived has_row_stats flag");
  if(row->AsBool("has_row_stats")) {
    if((ForwardCeil(s.n,ep_n)*ep_n)<=0 || (ForwardCeil(s.n,ep_n)*ep_n)%ep_n)
      throw std::invalid_argument("row-stat padded N must contain whole TN tiles");
    result+=Measured(forward_,"rowstat_ns",{{"rows",S(ep_m)},
        {"tiles",S(ForwardCeil(s.n,ep_n))}},"ns",1,terms);
  }else Term(terms,"rowstat_ns",0,"structural: stage has no row-statistics side output");
  return result;
}

double Sm90ForwardCostModel::BodyCost(Sm90ForwardShape const& s,Sm90ForwardStage const& c,
    Sm90ForwardResourceUsage const& resource,json::Array& terms) const {
  auto const& g=s.dnn1_geometry;double result=0;
  if(s.body==Sm90ForwardBody::kDepthwise || Fused(s.body)) {
    // Source: forward.dw_tap_ns[tw,th,live_channels,ct,stride,pack], measured
    // production Depthwise compute segment, normalized per live element/tap.
    auto depthwise=[&](int live,long long blocks) {
      if(!blocks)return;
      Sm90ForwardSelector selector{{"tw",S(c.tile_w)},{"th",S(c.tile_h)},
        {"channels",S(live)},{"ct",S(c.channel_block)},{"stride",S(s.stride)},
        {"pack",S(ForwardField(g,"depthwise_pack"))}};
      result+=Measured(forward_,"dw_tap_ns",selector,"ns/element/tap",
          double(c.tile_w)*c.tile_h*live*s.kernel*s.kernel*blocks,terms);
    };
    long long blocks=Fused(s.body)?ForwardCeil(ForwardField(g,"padded_channels"),c.channel_block):1;
    if(Fused(s.body)) {
      depthwise(c.channel_block,s.channels/c.channel_block);
      if(s.channels%c.channel_block)depthwise(int(s.channels%c.channel_block),1);
    }else depthwise(int(std::min<long long>(s.channels,c.channel_block)),1);
    if(Fused(s.body)) {
      // Fused-body interpretation (pending A2 confirmation): actual DW tap
      // work + its own channel epilogue; matrix async is priced by Stage.
      // No bare WGMMA instruction timing is mislabeled a fused body cost.
      for(auto const& entry:Steps(s,"depthwise_epilogue_steps")) {
        auto ep=Tile(c.tile_w*c.tile_h,c.channel_block);ep["op"]=entry.AsString("DW epilogue op");
        result+=Measured(forward_,"epilogue_step_ns",ep,"ns",blocks,terms);
      }
    }
    return result;
  }
  if(s.body==Sm90ForwardBody::kEncoderAttention) {
    // Head/batch replication changes task count; one task still reads one
    // head's full Seq K/V and produces Rows queries. The required table axes
    // are exactly seq/rows/PV implementation/segment, per §5.
    for(auto name:{"qk","softmax","pv"})
      result+=Measured(forward_,"attention_ns",{{"seq",S(ForwardField(g,"seq"))},
          {"rows",S(c.tile_m)},{"pv",ForwardString(g,"pv_kind")},{"segment",name}},"ns",1,terms);
    return result;
  }
  auto work=TaskBytes(s,c);
  if(work.elements<=0)
    throw Sm90ForwardMissing("resource helper did not provide exact small-body task_elements denominator");
  Sm90ForwardSelector selector;
  switch(s.body) {
    case Sm90ForwardBody::kLayout:
      selector={{"kind","layout"},{"channels",S(ForwardField(g,"input_channels"))},
          {"padded",S(ForwardField(g,"padded_channels"))}};break;
    case Sm90ForwardBody::kPool:
      selector={{"kind","pool"},{"window",S(s.kernel)},{"stride",S(s.stride)},
          {"channels",S(s.channels)}};break;
    case Sm90ForwardBody::kGlobalPool:
      selector={{"kind","global_pool"},{"parts",S(ForwardField(g,"parts"))},
          {"channels",S(s.channels)}};break;
    case Sm90ForwardBody::kGemv:
      selector={{"kind","gemv"},{"block",S(c.tile_n)},{"k",S(s.k)},{"n",S(s.n)}};break;
    case Sm90ForwardBody::kLayerNorm:
      selector={{"kind","layernorm"},{"channels",S(s.channels)},
          {"stride",S(ForwardField(g,"output_pitch"))}};break;
    case Sm90ForwardBody::kEmbedding:
      selector={{"kind","embedding"},{"channels",S(s.channels)}};break;
    default:throw std::invalid_argument("body has no measured constant mapping");
  }
  // Source: forward.small_body_ns[kind/exact geometry]. The denominator is
  // explicit in collector METHOD (layout live input, pool output, GAP input
  // partials, GEMV weights, LN/embedding outputs); never guess from s.pixels.
  return Measured(forward_,"small_body_ns",selector,"ns/element",work.elements,terms);
}

double Sm90ForwardCostModel::Stage(Sm90ForwardShape const& s,Sm90ForwardStage const& c,int residency,
    json::Value* terms) const {
  c.Validate();
  if(c.body!=s.body)throw std::invalid_argument("coordinate and shape body differ");
  auto resource=QuerySm90ForwardResources(s,c,target_,residency);
  if(!resource.feasible)throw std::invalid_argument("cost called for resource-infeasible candidate: "+resource.reason);
  long long tasks=Tasks(s,c);int grid=Grid(residency);
  if(resource.task_count!=tasks)throw std::invalid_argument("shared resource/task mapping disagrees with cost geometry");
  double waves=double(ForwardCeil(tasks,grid)),active=double(std::min<long long>(tasks,grid));
  // File+keys: target sm90_native_constants.c_bar_ns[G], dram_latency_ns,
  // sm_gbps, cta_dram_gbps, dram_gbps. No cross-stage prefetch this round:
  // L_first is ALWAYS measured dram_latency_ns, never L_l2.
  double barrier=native_.Barrier(grid),first=native_.dram_latency;
  double async_ns=0,memory_ns=0,data_ns=0,ep_ns=0,body_ns=0,rate=0;
  json::Array components;
  long long weight_task=0,activation_task=0,k_iter=0;
  if(Matrix(s.body)) {
    if(s.k%64 || s.k%c.split_k || (ForwardCeil(s.n,c.tile_n)*c.tile_n)%c.tile_n ||
        (ForwardCeil(s.n,c.tile_n)*c.tile_n)!=ForwardCeil(s.n,c.tile_n)*c.tile_n)
      throw std::invalid_argument("GEMM pad/split metadata inconsistent with generated body");
    k_iter=s.k/(c.split_k*64);
    if(k_iter<=0)throw std::invalid_argument("GEMM has zero K iterations");
    weight_task=2LL*c.tile_n*(s.k/c.split_k);
    activation_task=2LL*c.tile_m*(s.k/c.split_k);
    if(resource.slot_bytes!=2LL*64*(c.tile_m+c.tile_n))
      throw std::invalid_argument("shared helper slot_bytes differs from production GEMM ring");
    async_ns=double(weight_task+activation_task)/native_.sm_gbps;
    rate=std::min(native_.CtaDramGbps(),double(c.slots)*resource.slot_bytes/native_.dram_latency);
    memory_ns=double(weight_task)/rate;
    data_ns=std::max(async_ns,memory_ns)+std::min(async_ns,memory_ns)/k_iter;
    ep_ns=Epilogue(s,c,resource,components);
  }else {
    auto work=TaskBytes(s,c);
    // Non-GEMM interpretation pending original A2: max(CTA stream, DRAM
    // stream) plus the measured body primitive. Small bodies have no async
    // ring, so their measured CTA DRAM cap is used without a fake slot term.
    // Small::Run contains its own global accesses; this conservative extra
    // transfer term is visible for attribution, not hidden in a fitted rate.
    async_ns=double(work.async_bytes)/native_.sm_gbps;
    rate=native_.CtaDramGbps();
    // FX5 GEMV has one held A row and ONE rotating weight page. Coordinate
    // slots=2 describes those two roles, not two weight requests in flight.
    int streaming_slots=s.body==Sm90ForwardBody::kGemv?1:c.slots;
    if(resource.slot_bytes>0)
      rate=std::min(rate,double(streaming_slots)*resource.slot_bytes/native_.dram_latency);
    memory_ns=double(work.dram_bytes)/rate;
    data_ns=std::max(async_ns,memory_ns);
    body_ns=BodyCost(s,c,resource,components);
    if(s.body==Sm90ForwardBody::kDepthwise || Fused(s.body))ep_ns=Epilogue(s,c,resource,components);
  }
  // Source: true stage total weight bytes from the plan, native.dram_gbps.
  // This is the §7.2 total-weight HBM floor, not all activation bytes/DRAM.
  double floor=double(s.weight_bytes)/native_.dram_gbps;
  double raw=barrier+first+waves*(data_ns+ep_ns+body_ns);
  double total=std::max(raw,floor);
  if(terms)*terms=json::Object{{"tasks",double(tasks)},{"active_ctas",active},{"grid",grid},
      {"waves",waves},{"k_iter",double(k_iter)},{"weight_task_bytes",double(weight_task)},
      {"activation_task_bytes",double(activation_task)},{"rate_bytes_per_ns",rate},
      {"barrier_ns",barrier},{"barrier_source",NativeKey(forward_,"c_bar_ns["+S(grid)+"]")},
      {"first_byte_ns",first},{"first_byte_source",NativeKey(forward_,"dram_latency_ns")},
      {"cross_stage_prefetch",false},{"cta_async_ns",async_ns},
      {"cta_async_source",NativeKey(forward_,"sm_gbps")},{"dram_task_ns",memory_ns},
      {"rate_source",NativeKey(forward_,"cta_dram_gbps")+"; "+NativeKey(forward_,"dram_latency_ns")},
      {"data_task_ns",data_ns},{"epilogue_task_ns",ep_ns},{"body_task_ns",body_ns},
      {"hbm_weight_floor_ns",floor},{"hbm_floor_source",NativeKey(forward_,"dram_gbps")},
      {"unclamped_ns",raw},{"stage_ns",total},{"measured_components",components},
      {"resources",ResourceReceipt(resource)}};
  return total;
}

double Sm90ForwardCostModel::LegacyStage(Sm90ForwardShape const& s,Sm90ForwardStage const& c,
    int residency,json::Value* terms) const {
  long long tasks=Tasks(s,c);int grid=Grid(residency);
  bool wgmma=Matrix(s.body)||Fused(s.body);
  auto rate=forward_.HistoricalWgmma(wgmma?c.tile_n:256);
  double flops=2.0*s.m*s.n*s.k;
  if(s.body==Sm90ForwardBody::kDepthwise)flops=2.0*s.pixels*s.channels*s.kernel*s.kernel;
  double compute=flops/(rate.ns*std::min<long long>(tasks,grid));
  double bytes=double(s.weight_bytes+s.activation_bytes+s.output_bytes);
  double memory=bytes/native_.dram_gbps;
  double total=native_.Barrier(grid)+native_.l2_latency+std::max(compute,memory)+
      (c.split_k>1?native_.Barrier(grid):0);
  if(terms)*terms=json::Object{{"legacy_only",true},{"compute_ns",compute},{"dram_ns",memory},
      {"eta_times_sm_flops_per_ns",rate.ToJson()},
      {"barrier_source",NativeKey(forward_,"c_bar_ns["+S(grid)+"]")},
      {"first_byte_source",NativeKey(forward_,"l2_latency_ns")},
      {"memory_source",NativeKey(forward_,"dram_gbps")},{"stage_ns",total}};
  return total;
}

std::vector<Sm90ForwardStage> Sm90ForwardSearchSpace::Candidates(std::size_t index,int residency)const {
  if(index>=shapes.size() || (residency!=1 && residency!=2))throw std::invalid_argument("bad forward candidate request");
  auto const& s=shapes[index];std::vector<Sm90ForwardStage> result;
  auto push=[&](Sm90ForwardStage c){c.body=s.body;try{c.Validate();}catch(std::invalid_argument const&){return;}result.push_back(c);};
  if(Matrix(s.body)) {
    for(int m:{64,128})for(int n:{32,64,128,256}) {
      if(n>ForwardCeil(s.n,32)*32)continue;
      for(int split:{1,2,4,8,16}) {
        if(split>1 && (s.whole_row || s.k%(1LL*split*64) || s.k/split<128))continue;
        Sm90ForwardStage c;c.tile_m=m;c.tile_n=n;c.split_k=split;
        c.slots=s.k/64>=4?4:2;push(c);
      }
    }
  }else if(s.body==Sm90ForwardBody::kDepthwise || Fused(s.body)) {
    for(auto tile:{std::pair{8,8},std::pair{8,16},std::pair{16,16}}) {
      if(Fused(s.body) && (!s.fusable || tile.first*tile.second>64))continue;
      for(int n:(Fused(s.body)?std::vector<int>{64,128,256}:std::vector<int>{64})) {
        if(Fused(s.body) && n>s.out_channels && n!=64)continue;
        Sm90ForwardStage c;c.tile_w=tile.first;c.tile_h=tile.second;c.tile_n=n;
        c.fused=Fused(s.body);c.channel_block=64;c.slots=2;push(c);
      }
    }
  }else if(s.body==Sm90ForwardBody::kEncoderAttention) {
    auto seq=ForwardField(s.dnn1_geometry,"seq");
    for(int rows:{16,32,64,128})if(seq%rows==0) {Sm90ForwardStage c;c.tile_m=rows;push(c);}
  }else if(s.body==Sm90ForwardBody::kGemv) {
    for(int block:{64,128,256}) {Sm90ForwardStage c;c.tile_n=block;push(c);}
  }else {Sm90ForwardStage c;push(c);}
  // No [G/2,2G] window and no collapse-to-one fallback. Resource filtering
  // below is legality, never measured-constant-availability pruning.
  return result;
}

Sm90ForwardSearchResult SearchSm90Forward(Sm90ForwardSearchSpace const& space,
    Sm90ForwardCostModel const& cost,int top) {
  if(top!=4)throw std::invalid_argument("DNN1 compiles predicted top4; pass measure_top=4");
  if(space.shapes.empty())throw std::invalid_argument("forward search needs at least one stage");
  auto start=std::chrono::steady_clock::now();Sm90ForwardSearchResult result;
  std::vector<Sm90ForwardCandidate> pool;json::Array residency_receipts;
  for(int residency:{1,2}) {
    std::vector<std::vector<Sm90ForwardStage>> options(space.shapes.size());
    std::vector<std::vector<double>> prices(space.shapes.size());json::Array stage_receipts;
    bool complete=true;
    for(std::size_t i=0;i<space.shapes.size();++i) {
      auto raw=space.Candidates(i,residency);int filtered=0;json::Array rejected;
      for(auto const& c:raw) {
        auto resources=QuerySm90ForwardResources(space.shapes[i],c,space.target,residency);
        if(!resources.feasible) {++filtered;rejected.emplace_back(json::Object{{"coordinate",c.ToJson()},
            {"resources",ResourceReceipt(resources)}});continue;}
        options[i].push_back(c);
        // Missing constants abort the search; do not quietly make an
        // unmeasured candidate disappear and bias the selected coordinate.
        prices[i].push_back(cost.Stage(space.shapes[i],c,residency));++result.evaluated;
      }
      if(options[i].empty()) {complete=false;stage_receipts.emplace_back(json::Object{{"stage",int(i)},
          {"legal_candidate_count",int(raw.size())},{"shared_filtered_count",filtered},
          {"remaining_candidate_count",0},{"rejected",rejected}});break;}
      std::vector<std::size_t> order(options[i].size());std::iota(order.begin(),order.end(),0);
      std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return prices[i][a]<prices[i][b];});
      std::vector<Sm90ForwardStage> sorted;std::vector<double> sorted_prices;
      for(auto j:order){sorted.push_back(options[i][j]);sorted_prices.push_back(prices[i][j]);}
      options[i]=std::move(sorted);prices[i]=std::move(sorted_prices);
      json::Value chosen_terms;cost.Stage(space.shapes[i],options[i][0],residency,&chosen_terms);
      stage_receipts.emplace_back(json::Object{{"stage",int(i)},{"legal_candidate_count",int(raw.size())},
          {"shared_filtered_count",filtered},{"remaining_candidate_count",int(options[i].size())},
          {"selected",options[i][0].ToJson()},{"selected_terms",chosen_terms},{"rejected",rejected}});
    }
    if(!complete){residency_receipts.emplace_back(json::Object{{"residency",residency},
        {"feasible",false},{"stages",stage_receipts}});continue;}
    // "At most nine" applies independently inside the required residency
    // outer loop: <=9 per residency, <=18 combined, global predicted top4.
    std::vector<std::size_t> contributions(space.shapes.size());
    std::iota(contributions.begin(),contributions.end(),0);
    std::stable_sort(contributions.begin(),contributions.end(),[&](auto a,auto b){return prices[a][0]>prices[b][0];});
    if(contributions.size()>8)contributions.resize(8);
    auto build=[&](std::size_t changed,std::string origin) {
      Sm90ForwardCoordinates c;c.residency=residency;
      for(std::size_t i=0;i<options.size();++i)c.stages.push_back(options[i][i==changed?1:0]);
      auto prediction=cost.Predict(space.shapes,c);
      auto chosen_receipts=stage_receipts;
      for(std::size_t i=0;i<c.stages.size();++i) {
        chosen_receipts[i].Set("selected",c.stages[i].ToJson());
        chosen_receipts[i].Set("selected_terms",prediction.stages[i].terms);
      }
      json::Value receipt=json::Object{{"residency",residency},{"stages",chosen_receipts},
          {"changed_stage",changed<options.size()?int(changed):-1}};
      // The unchanged command serializes prediction.ToJson(). Keep the
      // complete new search receipt there so no out-of-scope tool edit is
      // needed to preserve candidate counts, coordinates and terms.
      prediction.search_receipt=receipt;
      pool.push_back({std::move(c),std::move(prediction),std::move(origin),std::move(receipt)});
    };
    auto prefix="r"+S(residency);build(options.size(),prefix+"_optimum");int generated=1;
    for(auto stage:contributions)if(options[stage].size()>1){build(stage,prefix+"_stage"+S(stage)+"_alt1");++generated;}
    residency_receipts.emplace_back(json::Object{{"residency",residency},{"feasible",true},
        {"program_candidate_count",generated},{"stages",stage_receipts}});
  }
  if(pool.empty())throw std::invalid_argument("forward search has no resource-legal program");
  auto programs=pool.size();
  std::stable_sort(pool.begin(),pool.end(),[](auto const& a,auto const& b){return a.prediction.total_ns<b.prediction.total_ns;});
  if(pool.size()>4)pool.resize(4);
  result.ranked=std::move(pool);
  result.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  result.receipt=json::Object{{"residencies",residency_receipts},{"program_candidate_count",int(programs)},
      {"compiled_top_count",int(result.ranked.size())},{"evaluated_stage_candidates",double(result.evaluated)},
      {"bound_interpretation","<=9 per residency; <=18 total; global top4"},
      {"elapsed_ms",result.elapsed_ms},{"constants",cost.ForwardConstants().Receipt()}};
  for(auto& selected:result.ranked) {
    selected.search_receipt.Set("program_candidate_count",int(programs));
    selected.search_receipt.Set("compiled_top_count",int(result.ranked.size()));
    selected.search_receipt.Set("bound_interpretation","<=9 per residency; <=18 total; global top4");
    selected.prediction.search_receipt=selected.search_receipt;
  }
  return result;
}
} // namespace tilemega::solver
