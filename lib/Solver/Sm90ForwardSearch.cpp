// SPDX-License-Identifier: BSD-3-Clause
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
      channels=std::min<long long>(128,ForwardField(g,"batch")*ForwardField(g,"output_width")*
          ForwardField(g,"output_height")*s.channels);
      return {0,2*channels*(s.kernel*s.kernel+1),channels};
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
  bool row_stats=g.At("has_row_stats").AsBool("has_row_stats");
  bool pool=g.At("has_channel_pool").AsBool("has_channel_pool");
  if(row_stats && pool)throw std::invalid_argument("Finish probes do not combine row and channel side outputs");
  if(row_stats) {
    if(!Matrix(s.body) || s.m%ep_m || s.n%ep_n || g.At("gated").AsBool("gated") ||
        ForwardString(g,"store_kind")!="vector_16byte")
      throw Sm90ForwardMissing("row-stat Finish needs an exact new geometry probe");
    result+=Measured(forward_,"rowstat_finish_store_ns",{{"tm",S(ep_m)},{"tn",S(ep_n)},
        {"live_rows",S(ep_m)},{"tiles",S(s.n/ep_n)},
        {"store_columns",S(resource.store_columns)},{"store","dense_vector16"},
        {"output_pitch",S(ForwardField(g,"output_pitch"))}},"ns",1,terms);
  }else if(pool) {
    // Source: forward.pool_finish_store_ns, empty-Apply production Finish.
    // Use the exact task-class histogram, never an unweighted key average.
    // The stage model remains waves times mean task service; class counts and
    // every measured coefficient are exposed for checking this approximation.
    auto rows=ForwardField(g,"pool_rows"),channels=ForwardField(g,"pool_channels");
    Sm90ForwardSelector base{{"pool_rows",S(rows)},{"pool_channels",S(channels)},
        {"parts",S(Sm90ForwardPoolParts(s,c))},{"tm",S(ep_m)},{"tn",S(ep_n)},
        {"width",S(resource.store_columns)},{"stride",S(ForwardField(g,"output_pitch"))}};
    std::map<Sm90ForwardSelector,long long> classes;long long count=0;
    if(Matrix(s.body)) {
      base["tw"]=base["th"]=base["image_width"]=base["live_tile_width"]="0";
      for(long long row=0;row<s.m;row+=ep_m) {
        auto key=base;key["live_rows"]=S(std::min<long long>(ep_m,s.m-row));
        key["row_base_mod_image"]=S(row%rows);++classes[key];++count;
      }
    }else {
      auto width=ForwardField(g,"output_width"),height=ForwardField(g,"output_height");
      if((width>c.tile_w && width%c.tile_w) || (height>c.tile_h && height%c.tile_h))
        throw Sm90ForwardMissing("mixed spatial edge pool Finish needs separate measured classes");
      base["tw"]=S(c.tile_w);base["th"]=S(c.tile_h);base["image_width"]=S(width);
      base["live_tile_width"]=S(std::min<long long>(width,c.tile_w));
      base["live_rows"]=S(std::min<long long>(width,c.tile_w)*std::min<long long>(height,c.tile_h));
      base["row_base_mod_image"]="0";
      count=ForwardCeil(width,c.tile_w)*ForwardCeil(height,c.tile_h)*ForwardField(g,"batch");
      classes[base]=count;
    }
    if(!count)throw std::invalid_argument("pool Finish task class set is empty");
    for(auto const& [key,multiplicity]:classes) {
      result+=Measured(forward_,"pool_finish_store_ns",key,"ns",double(multiplicity)/count,terms);
      terms.back().Set("task_class_count",double(multiplicity));
      terms.back().Set("row_task_count",double(count));
    }
  }else {
    auto store=Tile(ep_m,int(resource.store_columns));
    store["op"]=ForwardString(g,"store_kind");
    result+=Measured(forward_,"epilogue_store_ns",store,"ns",1,terms);
  }
  if(c.split_k>1) {
    auto selector=Tile(c.tile_m,c.tile_n);selector["slices"]=S(c.split_k);
    result+=Measured(forward_,"splitk_partial_ns",selector,"ns",1,terms);
  }else Term(terms,"splitk_partial_ns",0,"structural: split_k=1 has no partial or reduction");
  // Row-stat Finish already includes local sums, partials, last-arriver and
  // output store. Adding historical rowstat_ns or store would count it twice.
  return result;
}

double Sm90ForwardCostModel::BodyCost(Sm90ForwardShape const& s,Sm90ForwardStage const& c,
    Sm90ForwardResourceUsage const& resource,json::Array& terms) const {
  auto const& g=s.dnn1_geometry;double result=0;
  if(Fused(s.body)) {
    // Source: forward.dwpw_depthwise_ns; production fused Depthwise already
    // includes cached scale/bias/Relu6, so do not add those DW steps again.
    auto const& ops=Steps(s,"depthwise_epilogue_steps");
    if(ops.size()!=3 || ops[0].AsString("scale")!="scale" ||
        ops[1].AsString("bias")!="bias" || ops[2].AsString("activation")!="relu6")
      throw Sm90ForwardMissing("DWPW depthwise chain differs from the measured primitive");
    for(int live:{64,32,16}) {
      long long blocks=live==64?s.channels/64:(s.channels%64==live?1:0);
      if(blocks)result+=Measured(forward_,"dwpw_depthwise_ns",{{"tw",S(c.tile_w)},{"th",S(c.tile_h)},
          {"ct",S(c.channel_block)},{"channels",S(live)},{"stride",S(s.stride)},
          {"pack",S(ForwardField(g,"depthwise_pack"))},{"chain","scale+bias+relu6"}},"ns/task/channel_block",blocks,terms);
    }
    if(s.channels%64!=0 && s.channels%64!=16 && s.channels%64!=32)
      throw Sm90ForwardMissing("unmeasured DWPW live channel tail");
    return result;
  }
  if(s.body==Sm90ForwardBody::kDepthwise) {
    auto live=std::min<long long>(s.channels,c.channel_block);
    Sm90ForwardSelector selector{{"tw",S(c.tile_w)},{"th",S(c.tile_h)},
        {"channels",S(live)},{"ct",S(c.channel_block)},{"stride",S(s.stride)},
        {"pack",S(ForwardField(g,"depthwise_pack"))}};
    if(g.At("gated").AsBool("gated")) {
      auto const& ops=Steps(s,"epilogue_steps");
      if(ops.size()!=2 || ops[0].AsString("bias")!="bias" || ops[1].AsString("gate")!="simple_gate")
        throw Sm90ForwardMissing("gated DW chain differs from measured compute primitive");
      selector["gate"]="simple_gate";selector["bias"]="1";
    }
    return Measured(forward_,"dw_tap_ns",selector,"ns/element/tap",
        double(c.tile_w)*c.tile_h*live*s.kernel*s.kernel,terms);
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
  if(abi_==Sm90ForwardAbi::kF3) {
    if(residency!=1)throw std::invalid_argument("F3 costs require residency1");
    return F3Stage(s,c,Matrix(s.body) && c.mode!=Sm90ForwardMode::kGather,terms);
  }
  c.Validate();
  auto const& g=s.dnn1_geometry;
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
    int activation_rows=backend::Sm90ForwardActivationRows(c.tile_m,s.body==Sm90ForwardBody::kConv);
    activation_task=2LL*activation_rows*(s.k/c.split_k);
    if(resource.slot_bytes!=2LL*64*(activation_rows+c.tile_n))
      throw std::invalid_argument("shared helper slot_bytes differs from production GEMM ring");
    async_ns=double(weight_task+activation_task)/native_.sm_gbps;
    rate=std::min(native_.CtaDramGbps(),double(c.slots)*resource.slot_bytes/native_.dram_latency);
    memory_ns=double(weight_task)/rate;
    data_ns=std::max(async_ns,memory_ns)+std::min(async_ns,memory_ns)/k_iter;
    if(c.tile_m==32) {
      // Source: forward.matrix_k64_ns[tm32,tn,tk64,repeat64,impl=mma_sync].
      // Resident operands isolate the actual new mma.sync instruction stream.
      // This is a task arithmetic limiter, not an eta fit or a claim that
      // historical TM64 WGMMA calibration used the same probe recipe.
      double math=Measured(forward_,"matrix_k64_ns",{{"tm","32"},{"tn",S(c.tile_n)},
          {"tk","64"},{"repeat","64"},{"impl","mma_sync"}},"ns/k64",k_iter,components);
      data_ns=std::max(data_ns,math);
    }
    if(g.At("has_input_scale").AsBool("has_input_scale")) {
      auto rows=ForwardField(g,"scale_rows"),channels=ForwardField(g,"scale_channels");
      if(rows%c.tile_m || ForwardField(g,"scale_stride")<channels ||
          channels>s.k || (channels!=32 && channels%64))
        throw Sm90ForwardMissing("SCA panel geometry differs from the measured cached path");
      long long full=channels/64,tail=channels%64;
      // Source: forward.sca_panel_ns, actual ScalePanel per K64. Shapes with
      // split-K have equal full-panel slices; the only tail is C32/Sk1.
      if(tail && c.split_k!=1)throw Sm90ForwardMissing("split SCA tail requires its own measured panel distribution");
      if(full)body_ns+=Measured(forward_,"sca_panel_ns",{{"tm",S(c.tile_m)},
          {"tk","64"},{"live_k","64"},{"implementation","typed_cached_per_lane_bf16"}},
          "ns/k64/cta",double(full)/c.split_k,components);
      if(tail)body_ns+=Measured(forward_,"sca_panel_ns",{{"tm",S(c.tile_m)},
          {"tk","64"},{"live_k",S(tail)},{"implementation","typed_cached_per_lane_bf16"}},
          "ns/k64/cta",1,components);
    }
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

double Sm90ForwardCostModel::F3Stage(Sm90ForwardShape const& s,Sm90ForwardStage const& c,
    bool preissued,json::Value* terms) const {
  if(forward_.F4())return F4Stage(s,c,preissued,terms);
  if(abi_!=Sm90ForwardAbi::kF3 || c.body!=s.body)
    throw std::invalid_argument("F3 cost called with a different ABI/body");
  auto resource=QuerySm90ForwardF3Resources(s,c,target_,1);
  if(!resource.feasible)
    throw std::invalid_argument("F3 cost called for infeasible resources: "+resource.reason);
  int grid=Grid(1);
  double waves=double(ForwardCeil(resource.task_count,grid));
  double barrier=native_.Barrier(grid),first=preissued?native_.l2_latency:native_.dram_latency;
  double data=0,math=0,epilogue=0,split=0,task=0;
  long long k_iter=0,a_bytes=0,b_bytes=0;
  json::Array components;
  if(Matrix(s.body)) {
    bool swap=c.mode==Sm90ForwardMode::kSwapAB;
    int tm=swap?(s.m<=8?8:16):c.tile_m,tn=swap?128:c.tile_n;
    k_iter=s.k/(64*c.split_k);
    a_bytes=2LL*tm*(s.k/c.split_k);b_bytes=2LL*tn*(s.k/c.split_k);
    data=double(a_bytes+b_bytes)/native_.sm_gbps;
    Term(components,"t_data",data,NativeKey(forward_,"sm_gbps"));
    math=Measured(forward_,"f3_k64_ns",{{"mode",kSm90ForwardModes[unsigned(c.mode)]},
        {"tm",S(tm)},{"tn",S(tn)}},"ns",k_iter,components);
    auto tile=Sm90ForwardSelector{{"tm",S(tm)},{"tn",S(tn)}};
    auto ep=tile;ep["class"]=ForwardString(s.dnn1_geometry,"f3_chain_class");
    epilogue=Measured(forward_,"f3_epilogue_ns",ep,"ns",1,components);
    if(c.split_k>1) {
      tile["slices"]=S(c.split_k);
      split=Measured(forward_,"f3_splitk_ns",tile,"ns",1,components);
    }else Term(components,"f3_splitk_ns",0,"structural: split_k=1");
    task=std::max(data,math)+epilogue+split;
  }else if(s.body==Sm90ForwardBody::kEncoderAttention) {
    task=Measured(forward_,"f3_attention_task_ns",{},"ns/task",1,components);
  }else {
    std::string kind;int items=256;
    switch(s.body) {
      case Sm90ForwardBody::kLayout:kind="layout";break;
      case Sm90ForwardBody::kPool:
        if(s.kernel!=2 && s.kernel!=3)throw Sm90ForwardMissing("unmeasured F3 pool window");
        kind="pool"+S(s.kernel);break;
      case Sm90ForwardBody::kGlobalPool:kind="global_pool";break;
      case Sm90ForwardBody::kLayerNorm:kind="layernorm";items=8;break;
      case Sm90ForwardBody::kEmbedding:kind="embedding";items=8;break;
      default:throw Sm90ForwardMissing("unmeasured F3 small body");
    }
    task=Measured(forward_,"f3_small_ns",{{"kind",kind}},"ns/work_item",items,components);
  }
  double floor=Matrix(s.body)?double(s.weight_bytes)/native_.dram_gbps:0;
  double raw=barrier+first+waves*task,total=std::max(raw,floor);
  if(!std::isfinite(total) || !(total>0))throw std::invalid_argument("invalid F3 prediction");
  if(terms)*terms=json::Object{{"abi","f3"},{"tasks",double(resource.task_count)},
      {"grid",grid},{"waves",waves},{"preissued",preissued},{"k_iter",double(k_iter)},
      {"a_task_bytes",double(a_bytes)},{"b_task_bytes",double(b_bytes)},
      {"t_data_ns",data},{"t_math_ns",math},{"epilogue_ns",epilogue},
      {"splitk_ns",split},{"t_task_ns",task},{"barrier_ns",barrier},{"first_ns",first},
      {"barrier_source",NativeKey(forward_,"c_bar_ns["+S(grid)+"]")},
      {"first_source",NativeKey(forward_,preissued?"l2_latency_ns":"dram_latency_ns")},
      {"weight_floor_ns",floor},{"weight_floor_source",NativeKey(forward_,"dram_gbps")},
      {"raw_ns",raw},{"stage_ns",total},{"components",std::move(components)},
      {"resources",ResourceReceipt(resource)}};
  return total;
}

double Sm90ForwardCostModel::F4Stage(Sm90ForwardShape const& s,Sm90ForwardStage const& c,
    bool preissued,json::Value* terms) const {
  if(abi_!=Sm90ForwardAbi::kF3 || c.body!=s.body || !forward_.F4())
    throw std::invalid_argument("F4 cost requires its measured F3 pipeline table");
  auto resource=QuerySm90ForwardF3Resources(s,c,target_,1);
  if(!resource.feasible)throw std::invalid_argument("infeasible F4 coordinate: "+resource.reason);
  int grid=Grid(1),active=int(std::min<long long>(resource.task_count,grid));
  double waves=double(ForwardCeil(resource.task_count,grid));
  json::Array components;
  double gate=Measured(forward_,"f4_gate_ns",{},"ns",1,components);
  double task=0,mma=0,ring=0,rate=0,data=0,epilogue=0,split=0,t_k=0;
  long long k_iter=0,dram_bytes=0,effective_slot=resource.slot_bytes;
  std::string source="l2";
  if(Matrix(s.body)) {
    auto const& geometry=s.dnn1_geometry;
    constexpr long long kL2=32LL*1024*1024;
    auto weights=ForwardField(geometry,"f4_model_weight_bytes",true);
    auto activation=ForwardField(geometry,"f4_activation_resident_bytes",true);
    auto prefetched=ForwardField(geometry,"f4_prefetch_weight_bytes",true);
    bool b_l2=weights<=kL2 || prefetched>=s.weight_bytes;
    bool a_l2=activation<=kL2;
    source=a_l2 && b_l2?"l2":"dram";
    dram_bytes=(b_l2?0:s.weight_bytes-prefetched)+std::max(0LL,activation-kL2);
    bool swap=c.mode==Sm90ForwardMode::kSwapAB;
    int tm=swap?(s.m<=8?8:16):c.tile_m,tn=swap?128:c.tile_n;
    k_iter=s.k/(c.tk*c.split_k);
    auto tile=Sm90ForwardSelector{{"tm",S(tm)},{"tn",S(tn)}};
    auto pipeline=tile;pipeline["tk"]=S(c.tk);pipeline["mode"]=kSm90ForwardModes[unsigned(c.mode)];
    mma=Measured(forward_,"f4_mma_ns",pipeline,"ns",1,components);
    pipeline["src"]=source;
    ring=Measured(forward_,"f4_ring_ns",pipeline,"ns",1,components)/(c.slots-1);
    int n=std::max(33,std::min(active,132)),lo=(n/33)*33,hi=std::min(132,lo+33);
    auto low=forward_.Get("f4_rate_gbps",{{"src",source},{"n",S(lo)}},"GB/s");
    auto high=forward_.Get("f4_rate_gbps",{{"src",source},{"n",S(hi)}},"GB/s");
    rate=lo==hi?low.ns:low.ns+(high.ns-low.ns)*double(n-lo)/(hi-lo);
    components.emplace_back(json::Object{{"name","f4_rate_gbps"},{"value",rate},
        {"active_ctas",active},{"interpolated_n",n},{"lower",low.ToJson()},{"upper",high.ToJson()}});
    if(c.mc)effective_slot-=1LL*tn*c.tk;
    data=double(effective_slot)/rate;t_k=std::max({mma,data,ring});
    auto ep=tile;ep["class"]=ForwardString(geometry,"f3_chain_class");
    ep["store"]=kSm90ForwardStores[unsigned(c.store)];
    epilogue=Measured(forward_,"f4_epilogue_ns",ep,"ns",1,components);
    if(c.split_k>1) {
      tile["slices"]=S(c.split_k);
      split=Measured(forward_,"f4_splitk_ns",tile,"ns",1,components);
    }
    task=k_iter*t_k+epilogue+split;
  }else if(s.body==Sm90ForwardBody::kEncoderAttention) {
    auto rows=std::min(128LL,16*ForwardField(s.dnn1_geometry,"batch"));
    task=Measured(forward_,"f4_attention_ns",{{"rows",S(rows)}},"ns/task",1,components);
  }else {
    std::string kind;int items=256;
    switch(s.body) {
      case Sm90ForwardBody::kLayout:kind="layout";break;
      case Sm90ForwardBody::kPool:kind="pool"+S(s.kernel);break;
      case Sm90ForwardBody::kGlobalPool:kind="global_pool";break;
      case Sm90ForwardBody::kLayerNorm:kind="layernorm";items=8;break;
      case Sm90ForwardBody::kEmbedding:kind="embedding";items=8;break;
      default:throw Sm90ForwardMissing("unmeasured F4 small body");
    }
    task=Measured(forward_,"f4_small_ns",{{"kind",kind}},"ns/work_item",items,components);
  }
  double first=Measured(forward_,"f4_first_ns",{{"src",preissued?"l2":source}},"ns",1,components);
  double first_dram=forward_.Get("f4_first_ns",{{"src","dram"}},"ns").ns;
  double raw=gate+first+waves*task;
  double floor=gate+first_dram+double(dram_bytes)/native_.dram_gbps;
  double total=std::max(raw,floor);
  if(!std::isfinite(total) || total<=0)throw std::invalid_argument("invalid F4 prediction");
  if(terms)*terms=json::Object{{"abi","f3"},{"model","f4"},
      {"tasks",double(resource.task_count)},{"grid",grid},{"waves",waves},{"source",source},
      {"preissued",preissued},{"k_iter",double(k_iter)},{"t_k_ns",t_k},
      {"mma_ns",mma},{"ring_ns",ring},{"data_ns",data},{"rate_gbps",rate},
      {"effective_slot_bytes",double(effective_slot)},{"epilogue_ns",epilogue},{"splitk_ns",split},
      {"t_task_ns",task},{"wave_task_ns",waves*task},{"gate_ns",gate},{"first_ns",first},
      {"dram_bytes",double(dram_bytes)},{"dram_floor_ns",floor},{"raw_ns",raw},{"stage_ns",total},
      {"components",std::move(components)},{"resources",ResourceReceipt(resource)}};
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

std::vector<Sm90ForwardStage> Sm90ForwardSearchSpace::Candidates(std::size_t index,int residency,bool f4)const {
  if(index>=shapes.size() || (residency!=1 && residency!=2))throw std::invalid_argument("bad forward candidate request");
  auto const& s=shapes[index];std::vector<Sm90ForwardStage> result;
  bool f3=abi==Sm90ForwardAbi::kF3;
  if(f3 && residency!=1)return result;
  // DT-SPILL: the measured register map keeps a spilling tile out of the legal
  // set entirely, so no candidate can reach a build that would spill.
  auto chain_class=Sm90F3ChainClass::kPlain;
  if(f3) {
    auto const& g=s.dnn1_geometry;
    bool residual=false,deferred=false,residual_norm=false,activation=false,row_stats=false;
    if(auto const* steps=g.Find("epilogue_steps"))
      for(auto const& step:steps->AsArray("epilogue steps")) {
        auto key=step.AsString("epilogue step");
        if(key=="residual" || key=="residual_scaled")residual=true;
        else if(key=="residual_ln")residual_norm=true;
        else if(key=="deferred_ln")deferred=true;
        else if(key!="bias" && key!="scale")activation=true;
      }
    if(auto const* flag=g.Find("has_row_stats"))row_stats=flag->AsBool("has_row_stats");
    chain_class=Sm90F3ClassOf(residual,deferred,residual_norm,row_stats,activation,
        s.mode==Sm90ForwardMode::kSwapAB);
  }
  auto push=[&](Sm90ForwardStage c){c.body=s.body;c.mode=f3?s.mode:Sm90ForwardMode::kNormal;
      if(f3 && Matrix(s.body) && c.mode!=Sm90ForwardMode::kSwapAB &&
         !Sm90F3TileFeasible(c.tile_m,c.tile_n,chain_class))return;
      try{c.Validate(abi);}catch(std::invalid_argument const&){return;}result.push_back(c);};
  if(Matrix(s.body)) {
    // F3 swap-AB fixes the tile at 128 weight rows by all M rows, and admits a
    // deeper split because each slice is only Mp x 64 of activation (§6.7).
    if(f3 && s.mode==Sm90ForwardMode::kSwapAB) {
      for(int tk:(f4?std::vector<int>{64,128}:std::vector<int>{64})) {
      auto iterations=s.k/tk;
      for(int split=1;split<=32;++split) {
        if(s.k%(tk*split) || split>iterations)continue;
        Sm90ForwardStage c;c.tile_n=128;c.split_k=split;c.tk=tk;
        for(int slots:{2,3,4,5,6,7,8}){c.slots=slots;push(c);}
      }
      }
      return result;
    }
    for(int m:(f3?std::vector<int>{64,128,256}:std::vector<int>{32,64,128}))
      for(int n:(f4?std::vector<int>{32,64,96,128,192,256}:std::vector<int>{32,64,128,256})) {
      if(!f3 && n>ForwardCeil(s.n,32)*32)continue;
      if(f3 && m==256 && s.m<512)continue;
      if(f3) {
        // Any divisor of the whole contraction, and a deferred normalization
        // no longer blocks the split: F3's last arriver runs the chain (§6.5).
        for(int tk:(f4?std::vector<int>{64,128}:std::vector<int>{64})) {
        auto iterations=s.k/tk;
        for(int split=1;split<=16;++split) {
          if(!iterations || s.k%(tk*split))continue;
          Sm90ForwardStage c;c.tile_m=m;c.tile_n=n;c.split_k=split;c.tk=tk;
          for(int slots:{2,3,4,5,6,7,8}) {
            c.slots=slots;push(c);
            if(f4 && m>=128 && n%64==0) {
              c.store=Sm90ForwardStore::kTma;push(c);c.store=Sm90ForwardStore::kReg;
            }
          }
        }
        }
        continue;
      }
      for(int split:{1,2,4,8,16}) {
        if(split>1 && (s.whole_row || s.k%(1LL*split*64) || s.k/split<128))continue;
        Sm90ForwardStage c;c.tile_m=m;c.tile_n=n;c.split_k=split;
        for(int slots:{2,3,4,6,8}){c.slots=slots;push(c);}
      }
    }
  }else if(s.body==Sm90ForwardBody::kDepthwise || Fused(s.body)) {
    for(auto tile:{std::pair{8,8},std::pair{8,16},std::pair{16,16}}) {
      if(Fused(s.body) && (!s.fusable || tile.first*tile.second>64))continue;
      for(int n:(Fused(s.body)?std::vector<int>{64,128,256}:std::vector<int>{64})) {
        if(Fused(s.body) && n>s.out_channels && n!=64)continue;
        Sm90ForwardStage c;c.tile_w=tile.first;c.tile_h=tile.second;c.tile_n=n;
        c.fused=Fused(s.body);c.channel_block=64;
        for(int slots:{2,3,4,6,8}){c.slots=slots;push(c);}
      }
    }
  }else if(s.body==Sm90ForwardBody::kEncoderAttention) {
    if(f3) {Sm90ForwardStage c;c.tile_m=f4?int(std::min(128LL,16*ForwardField(s.dnn1_geometry,"batch"))):128;push(c);return result;}
    auto seq=ForwardField(s.dnn1_geometry,"seq");
    for(int rows:{16,32,64,128})if(seq%rows==0) {Sm90ForwardStage c;c.tile_m=rows;push(c);}
  }else if(s.body==Sm90ForwardBody::kGemv) {
    for(int block:{64,128,256}) {Sm90ForwardStage c;c.tile_n=block;push(c);}
  }else {Sm90ForwardStage c;push(c);}
  // No [G/2,2G] window and no collapse-to-one fallback. Resource filtering
  // below is legality, never measured-constant-availability pruning.
  return result;
}

json::Value AuditSm90ForwardF4Candidates(Sm90ForwardSearchResult const& result) {
  json::Array pairs;bool passed=true;
  for(std::size_t i=0;i<result.ranked.size();++i)
    for(std::size_t j=i+1;j<result.ranked.size();++j) {
      auto const& a=result.ranked[i];auto const& b=result.ranked[j];
      bool identical=a.coordinates.ToJson().Dump(0)==b.coordinates.ToJson().Dump(0);
      double gap=std::abs(a.prediction.total_ns-b.prediction.total_ns)/
          std::min(a.prediction.total_ns,b.prediction.total_ns);
      bool ok=identical || gap>=.005;passed=passed && ok;
      pairs.emplace_back(json::Object{{"a",int(i)},{"b",int(j)},
          {"identical",identical},{"relative_gap",gap},{"passed",ok}});
    }
  return json::Object{{"passed",passed},{"candidate_count",int(result.ranked.size())},
      {"minimum_relative_gap",.005},{"pairs",std::move(pairs)}};
}

Sm90ForwardSearchResult SearchSm90Forward(Sm90ForwardSearchSpace const& space,
    Sm90ForwardCostModel const& cost,int top) {
  if(top!=4)throw std::invalid_argument("DNN1 compiles predicted top4; pass measure_top=4");
  if(space.shapes.empty())throw std::invalid_argument("forward search needs at least one stage");
  if(space.abi!=cost.Abi())throw std::invalid_argument("search ABI differs from cost model");
  bool f3=space.abi==Sm90ForwardAbi::kF3;
  bool f4=f3 && cost.ForwardConstants().F4();
  std::map<std::string,bool> sensitive;
  std::map<std::string,std::map<std::string,double>> sensitivity_groups;
  for(auto field:{"tile_m","tile_n","tk","split_k","slots","store","mc"})sensitive[field]=false;
  auto audit_field=[&](std::size_t stage,Sm90ForwardStage const& c,double price) {
    if(!f4)return;
    for(auto& [field,changed]:sensitive) {
      if(changed)continue;
      auto key=c.ToJson(space.abi);key.Set(field,"*");
      auto group=S(stage)+":"+key.Dump(0);
      auto [where,inserted]=sensitivity_groups[field].emplace(group,price);
      if(!inserted && std::abs(price-where->second)>1e-9*std::max(1.,std::abs(price)))changed=true;
    }
  };
  std::vector<std::vector<std::size_t>> consumers(space.shapes.size());
  std::vector<int> producers(space.shapes.size(),-1);
  for(std::size_t i=0;i<space.shapes.size();++i) {
    producers[i]=Sm90ForwardCostModel::PoolProducer(space.shapes,i);
    if(producers[i]>=0)consumers[producers[i]].push_back(i);
  }
  auto start=std::chrono::steady_clock::now();Sm90ForwardSearchResult result;
  std::vector<Sm90ForwardCandidate> pool;json::Array residency_receipts;
  for(int residency:(f3?std::vector<int>{1}:std::vector<int>{1,2})) {
    std::vector<std::vector<Sm90ForwardStage>> options(space.shapes.size());
    std::vector<std::vector<double>> prices(space.shapes.size());json::Array stage_receipts;
    bool complete=true;
    for(std::size_t i=0;i<space.shapes.size();++i) {
      auto raw=space.Candidates(i,residency,f4);int filtered=0;json::Array rejected;
      for(auto const& c:raw) {
        auto resources=QuerySm90ForwardResources(space.shapes[i],c,space.target,residency,space.abi);
        if(!resources.feasible) {++filtered;rejected.emplace_back(json::Object{{"coordinate",c.ToJson()},
            {"resources",ResourceReceipt(resources)}});continue;}
        options[i].push_back(c);
        // Missing constants abort the search; do not quietly make an
        // unmeasured candidate disappear and bias the selected coordinate.
        // A pool has no independent coordinate choice. Charge its exact
        // producer-selected partial count to that producer during local
        // ranking; Predict expands the linked group back into separate stages.
        double price=producers[i]<0?cost.Stage(space.shapes[i],c,residency):0;
        audit_field(i,c,price);
        for(auto pool_index:consumers[i]) {
          auto pool_options=space.Candidates(pool_index,residency,f4);
          if(pool_options.size()!=1)throw std::invalid_argument("linked pool must have exactly one coordinate");
          auto pool_shape=Sm90ForwardCostModel::PoolShape(space.shapes,pool_index,c,space.abi);
          price+=cost.Stage(pool_shape,pool_options.front(),residency);
        }
        prices[i].push_back(price);++result.evaluated;
        // DT-MC2 keeps multicast out of search. Audit its formula separately
        // so an inactive mechanism is distinguished from an ignored field.
        if(f4 && c.mode==Sm90ForwardMode::kNormal && Matrix(c.body)) {
          auto diagnostic=c;diagnostic.mc=1;
          if(QuerySm90ForwardF3Resources(space.shapes[i],diagnostic,space.target,1).feasible)
            audit_field(i,diagnostic,cost.F4Stage(space.shapes[i],diagnostic,false));
        }
      }
      if(options[i].empty()) {complete=false;stage_receipts.emplace_back(json::Object{{"stage",int(i)},
          {"legal_candidate_count",int(raw.size())},{"shared_filtered_count",filtered},
          {"remaining_candidate_count",0},{"rejected",rejected}});break;}
      std::vector<std::size_t> order(options[i].size());std::iota(order.begin(),order.end(),0);
      std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return prices[i][a]<prices[i][b];});
      std::vector<Sm90ForwardStage> sorted;std::vector<double> sorted_prices;
      auto ranked_count=order.size();
      if(f3 && !f4 && order.size()>16)order.resize(16);
      for(auto j:order){sorted.push_back(options[i][j]);sorted_prices.push_back(prices[i][j]);}
      options[i]=std::move(sorted);prices[i]=std::move(sorted_prices);
      json::Value chosen_terms=json::Object{{"resolution","program coordinates below"}};
      if(producers[i]<0)cost.Stage(space.shapes[i],options[i][0],residency,&chosen_terms);
      stage_receipts.emplace_back(json::Object{{"stage",int(i)},{"legal_candidate_count",int(raw.size())},
          {"shared_filtered_count",filtered},{"remaining_candidate_count",int(options[i].size())},
          {"ranked_candidate_count",int(ranked_count)},
          {"selected",options[i][0].ToJson(space.abi)},{"selected_terms",chosen_terms},
          {"linked_contribution_ns",prices[i][0]},{"pool_producer_stage",producers[i]},
          {"rejected",rejected}});
    }
    if(!complete){residency_receipts.emplace_back(json::Object{{"residency",residency},
        {"feasible",false},{"stages",stage_receipts}});continue;}
    if(f4) {
      std::vector<bool> spanning;
      for(std::size_t i=0;i<options.size();++i)
        spanning.push_back(QuerySm90ForwardF3Resources(space.shapes[i],options[i][0],space.target,1).exclusive);
      for(std::size_t i=0;i<options.size();++i) {
        std::vector<double> work(options[i].size());
        for(std::size_t j=0;j<options[i].size();++j) {
          auto const& c=options[i][j];json::Value terms;
          auto r=QuerySm90ForwardF3Resources(space.shapes[i],c,space.target,1);
          bool preissued=i>0 && Matrix(c.body) && c.mode!=Sm90ForwardMode::kGather &&
              !c.mc && !r.exclusive && !spanning[i-1];
          prices[i][j]=producers[i]<0?cost.F4Stage(space.shapes[i],c,preissued,&terms):0;
          work[j]=producers[i]<0?terms.At("wave_task_ns").AsNumber("wave task"):0;
          for(auto pool_index:consumers[i]) {
            auto pool_options=space.Candidates(pool_index,residency,true);
            auto shape=Sm90ForwardCostModel::PoolShape(space.shapes,pool_index,c,space.abi);
            prices[i][j]+=cost.F4Stage(shape,pool_options.front(),false,&terms);
            work[j]+=terms.At("wave_task_ns").AsNumber("pool wave task");
          }
        }
        std::vector<std::size_t> order(options[i].size());std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](auto a,auto b) {
          if(prices[i][a]!=prices[i][b])return prices[i][a]<prices[i][b];
          return options[i][a].ToJson(space.abi).Dump(0)<options[i][b].ToJson(space.abi).Dump(0);
        });
        // Anchor each tie window at its minimum. A pairwise fuzzy comparator
        // is nontransitive and would violate std::sort's ordering contract.
        for(std::size_t begin=0;begin<order.size();) {
          auto end=begin+1;
          while(end<order.size() && prices[i][order[end]]<1.005*prices[i][order[begin]])++end;
          std::sort(order.begin()+begin,order.begin()+end,[&](auto a,auto b) {
            if(work[a]!=work[b])return work[a]<work[b];
            auto const& x=options[i][a];auto const& y=options[i][b];
            if(x.slots!=y.slots)return x.slots>y.slots;
            if(x.split_k!=y.split_k)return x.split_k<y.split_k;
            return x.ToJson(space.abi).Dump(0)<y.ToJson(space.abi).Dump(0);
          });
          begin=end;
        }
        // P-TUNE must continue below the top8 when those contain near ties.
        // Keep six separated choices from this same enumeration; program
        // search still uses only the prescribed top8.
        json::Array tuning;std::vector<double> tuning_prices;
        for(auto j:order) {
          bool separated=true;
          for(double value:tuning_prices)
            if(std::abs(prices[i][j]-value)<.005*std::min(prices[i][j],value))separated=false;
          if(!separated)continue;
          tuning.emplace_back(json::Object{{"coordinate",options[i][j].ToJson(space.abi)},
              {"prediction_ns",prices[i][j]},{"wave_task_ns",work[j]}});
          tuning_prices.push_back(prices[i][j]);
          if(tuning.size()==6)break;
        }
        stage_receipts[i].Set("tuning_candidates",std::move(tuning));
        if(order.size()>8)order.resize(8);
        std::vector<Sm90ForwardStage> selected;std::vector<double> values;json::Array ranked;
        for(auto j:order) {
          selected.push_back(options[i][j]);values.push_back(prices[i][j]);
          ranked.emplace_back(json::Object{{"coordinate",options[i][j].ToJson(space.abi)},
              {"prediction_ns",prices[i][j]},{"wave_task_ns",work[j]}});
        }
        options[i]=std::move(selected);prices[i]=std::move(values);
        stage_receipts[i].Set("remaining_candidate_count",int(options[i].size()));
        stage_receipts[i].Set("ranked_candidates",std::move(ranked));
      }
    }
    // "At most nine" applies independently inside the required residency
    // outer loop: <=9 per residency, <=18 combined, global predicted top4.
    std::vector<std::size_t> contributions(space.shapes.size());
    std::iota(contributions.begin(),contributions.end(),0);
    std::stable_sort(contributions.begin(),contributions.end(),[&](auto a,auto b){return prices[a][0]>prices[b][0];});
    if(contributions.size()>8)contributions.resize(8);
    auto build=[&](std::size_t changed,std::string origin,std::size_t alternative=1) {
      Sm90ForwardCoordinates c;c.residency=residency;c.abi=space.abi;
      for(std::size_t i=0;i<options.size();++i)c.stages.push_back(options[i][i==changed?alternative:0]);
      auto prediction=cost.Predict(space.shapes,c);
      auto chosen_receipts=stage_receipts;
      for(std::size_t i=0;i<c.stages.size();++i) {
        chosen_receipts[i].Set("selected",c.stages[i].ToJson(space.abi));
        chosen_receipts[i].Set("selected_terms",prediction.stages[i].terms);
        double linked=producers[i]<0?prediction.stages[i].ns:0;
        for(auto pool_index:consumers[i])linked+=prediction.stages[pool_index].ns;
        chosen_receipts[i].Set("linked_contribution_ns",linked);
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
    for(auto stage:contributions)if(options[stage].size()>1) {
      std::size_t alternative=1;
      if(f4) {
        auto const& best=options[stage][0];
        while(alternative<options[stage].size()) {
          auto const& c=options[stage][alternative];
          if(c.tile_m!=best.tile_m || c.tile_n!=best.tile_n || c.tk!=best.tk || c.split_k!=best.split_k)break;
          ++alternative;
        }
      }
      if(alternative<options[stage].size()) {
        build(stage,prefix+"_stage"+S(stage)+"_alt"+S(alternative),alternative);++generated;
      }
    }
    residency_receipts.emplace_back(json::Object{{"residency",residency},{"feasible",true},
        {"program_candidate_count",generated},{"stages",stage_receipts}});
  }
  if(pool.empty())throw std::invalid_argument("forward search has no resource-legal program");
  if(f4) {
    std::vector<Sm90ForwardCandidate> unique;
    for(auto& p:pool) {
      auto coordinate=p.coordinates.ToJson().Dump(0);
      if(std::none_of(unique.begin(),unique.end(),[&](auto const& q) {
        return coordinate==q.coordinates.ToJson().Dump(0);
      }))unique.push_back(std::move(p));
    }
    pool=std::move(unique);
  }
  auto programs=pool.size();
  std::stable_sort(pool.begin(),pool.end(),[](auto const& a,auto const& b){return a.prediction.total_ns<b.prediction.total_ns;});
  if(pool.size()>4)pool.resize(4);
  result.ranked=std::move(pool);
  result.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::string bound=f3?"<=9 total; actual spanning first-byte rerank once; top4":
      "<=9 per residency; <=18 total; global top4";
  result.receipt=json::Object{{"residencies",residency_receipts},{"program_candidate_count",int(programs)},
      {"compiled_top_count",int(result.ranked.size())},{"evaluated_stage_candidates",double(result.evaluated)},
      {"bound_interpretation",bound},
      {"elapsed_ms",result.elapsed_ms},{"constants",cost.ForwardConstants().Receipt()}};
  if(f4) {
    result.receipt.Set("f4_candidate_selfcheck",AuditSm90ForwardF4Candidates(result));
    json::Object fields;for(auto const& [field,changed]:sensitive)fields.emplace_back(field,changed);
    result.receipt.Set("f4_coordinate_sensitivity",json::Object{{"scope","this_cell; OR across20 cells"},
        {"fields",std::move(fields)},{"mc_scope","diagnostic counterfactual; DT-MC2 admission remains disabled"}});
  }
  for(auto& selected:result.ranked) {
    selected.search_receipt.Set("program_candidate_count",int(programs));
    selected.search_receipt.Set("compiled_top_count",int(result.ranked.size()));
    selected.search_receipt.Set("bound_interpretation",bound);
    if(f4)selected.search_receipt.Set("f4_candidate_selfcheck",
        result.receipt.At("f4_candidate_selfcheck"));
    if(f4)selected.search_receipt.Set("f4_coordinate_sensitivity",
        result.receipt.At("f4_coordinate_sensitivity"));
    selected.prediction.search_receipt=selected.search_receipt;
  }
  return result;
}
} // namespace tilemega::solver
