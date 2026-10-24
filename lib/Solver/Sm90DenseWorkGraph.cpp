// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseWorkGraph.h>
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseResources.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace tilemega::solver {
namespace {
char const* Name(Sm90DenseTailOrder order) {
  switch(order) {
    case Sm90DenseTailOrder::kNone:return "none";
    case Sm90DenseTailOrder::kBeforeProduce:return "before_produce";
    case Sm90DenseTailOrder::kAfterProduce:return "after_produce";
    case Sm90DenseTailOrder::kRetireOperands:return "retire_operands";
  }
  throw std::logic_error("unknown native producer tail order");
}
struct StageInput {
  Sm90DenseStageWork work;
  Sm90DenseGemm const* gemm=nullptr;
  std::int64_t prefix=0,l2=0;
  bool late=false;
  int k=0;
};
std::pair<int,int> Range(int past,int chunk,int chunks,int row_work) {
  std::int64_t total=std::int64_t(past)+1+row_work;
  return {int(std::min<std::int64_t>(past,total*chunk/chunks)),
      chunk==chunks-1?past+1:int(std::min<std::int64_t>(past,total*(chunk+1)/chunks))};
}
}

std::map<int,int> Sm90DenseStageWork::LocalTaskHistogram() const {
  std::map<int,int> result;
  for(auto const& cta:ctas)++result[int(cta.tasks.size())];
  return result;
}

void Sm90DenseWorkGraph::RequireResident(Sm90DenseResidentEvidence const& evidence) const {
  if(evidence.binary_sha256.size()!=64 || evidence.source_digest.size()!=64 ||
      evidence.grid<=0 || evidence.resident_ctas_per_sm<=0)
    throw std::invalid_argument("native resource_pending: bound final-binary occupancy required");
  if(evidence.grid!=grid || evidence.dynamic_shared_bytes!=shared_bytes)
    throw std::invalid_argument("native occupancy witness describes a different launch");
  if(evidence.resident_ctas_per_sm<requested_residency)
    throw std::invalid_argument("native persistent grid cannot use hardware launch waves");
}

Sm90DenseWorkGraph BuildSm90DenseWorkGraph(Sm90DenseWorkload const& w,
    Sm90DenseCoordinates const& c,TargetSpec const& target,int past,int capacity) {
  if(target.arch_tag!="sm_90" || target.res.num_sms<=0 || w.layers<1 ||
      (w.head_dim!=64 && w.head_dim!=128) || w.queries<2 || w.queries>4 ||
      past<64 || past>1086 || capacity!=1088)
    throw std::invalid_argument("invalid native work graph workload");
  Sm90DenseWorkGraph graph;
  graph.num_sms=target.res.num_sms;graph.requested_residency=c.residency;
  graph.shared_budget=Sm90DenseSharedBudget(target,c.residency);
  if(!Sm90DenseSearchSpace{w.shapes,w.batch,w.head_dim,w.queries,target}.Legal(c))
    throw std::invalid_argument("unsupported native work graph coordinate");
  c.ValidateSchedule(w.shapes,w.batch,w.head_dim,w.queries,capacity,graph.shared_budget);
  graph.grid=target.res.num_sms*c.residency;graph.past=past;
  auto reuse=c.ActivationReuse(w.batch,w.head_dim,w.queries,capacity);
  graph.activation_reuse=reuse.valid;
  if(reuse.valid)graph.launch_inactive_zero_bytes_per_cta=2ll*reuse.slots*(16-w.batch)*reuse.tile_k;
  std::vector<StageInput> inputs;
  auto token=[&](char const* kind) {
    StageInput s;s.work.kind=kind;s.work.tasks=w.batch;
    s.work.layout=std::string(kind)=="argmax"?backend::Sm90DenseLayout::Argmax():
        backend::Sm90DenseLayout{};
    // Activation reuse can move argmax into otherwise unused arena space.
    if(std::string(kind)=="argmax" && reuse.valid)s.work.workspace_offset=reuse.ArgmaxShift();
    inputs.push_back(s);
  };
  auto gemm=[&](unsigned family,int layer) {
    auto const& g=c.gemms[family];StageInput s;
    s.work.kind=kSm90DenseFamilies[family];s.work.layer=layer;
    s.work.tasks=w.shapes[family][0]/g.tile_n;s.work.ring_slots=g.slots;
    if(!reuse.valid)s.work.inactive_zero_bytes_per_cta=2ll*g.slots*(16-w.batch)*g.TileK();
    s.work.workspace_offset=g.workspace_offset;
    s.work.layout=backend::Sm90DenseLayout::Gemm(g.tile_n,g.TileK(),g.slots,g.warps_k);
    s.work.prefix_payload=backend::Sm90DenseLayout::GemmPayload(g.tile_n,g.TileK(),g.slots,
        w.shapes[family][1],g.preissue_smem_bytes).Shift(g.workspace_offset);
    s.gemm=&g;s.k=w.shapes[family][1];s.prefix=g.preissue_smem_bytes;
    s.l2=g.prefetch_l2_bytes;s.late=g.preissue_late;inputs.push_back(s);
  };
  token("embedding");
  for(int layer=0;layer<w.layers;++layer) {
    gemm(0,layer);
    auto const& a=c.attention;StageInput s;
    s.work.kind="attention";s.work.layer=layer;s.work.tasks=w.batch*8*a.chunks;
    s.work.ring_slots=w.batch<=4?1:a.slots;s.work.merge_chunks=a.chunks;
    s.work.workspace_offset=a.workspace_offset;
    s.work.layout=backend::Sm90DenseLayout::Attention(w.batch,w.head_dim,w.queries,
        a.chunks,a.block_keys,a.slots,a.new_row_work,capacity);
    s.work.prefix_payload=backend::Sm90DenseLayout::AttentionPayload(w.batch,w.head_dim,w.queries,
        a.chunks,a.block_keys,a.slots,a.new_row_work,capacity,a.preissue_smem_bytes)
        .Shift(a.workspace_offset);
    s.prefix=a.preissue_smem_bytes;s.l2=a.prefetch_l2_bytes;s.late=a.preissue_late;
    inputs.push_back(s);
    for(unsigned family:{1,2,3})gemm(family,layer);
  }
  gemm(4,-1);token("argmax");

  using Generations=std::array<std::array<std::uint64_t,8>,2>;
  std::vector<Generations> generations(graph.grid);
  for(unsigned index=0;index<inputs.size();++index) {
    auto& input=inputs[index];auto& stage=input.work;stage.ring_bank=index%2;
    graph.shared_bytes=std::max(graph.shared_bytes,stage.workspace_offset+stage.layout.end);
    if(input.prefix || input.l2) {
      if(!index)throw std::logic_error("first native stage cannot receive preissue");
      auto const& previous=inputs[index-1].work;
      if(stage.prefix_payload.Overlaps(previous.layout.Scratch(previous.workspace_offset)))
        throw std::invalid_argument("native prefix overlaps predecessor scratch");
      stage.incoming_order=stage.prefix_payload.Overlaps(previous.layout.Live(previous.workspace_offset))?
          Sm90DenseTailOrder::kRetireOperands:input.late?Sm90DenseTailOrder::kAfterProduce:
          Sm90DenseTailOrder::kBeforeProduce;
      stage.prefix_predecessor=index-1;
    }
    stage.ctas.resize(graph.grid);
    for(int cta=0;cta<graph.grid;++cta) {
      auto& owner=stage.ctas[cta];owner.cta=cta;int sequence=0;
      for(int task=cta;task<stage.tasks;task+=graph.grid) {
        Sm90DenseTaskWork work;work.task=task;bool first=owner.tasks.empty();
        if(input.gemm) {
          auto const& g=*input.gemm;int iterations=input.k/g.TileK();
          for(int tile=0;tile<iterations;++tile,++sequence) {
            Sm90DenseBlockWork block;
            block.sequence=sequence;block.slot=sequence%g.slots;
            block.generation=generations[cta][stage.ring_bank][block.slot]++;
            block.previous_reader=sequence>=g.slots?sequence-g.slots:-1;
            block.weight_bytes=g.slot_bytes;block.activation_bytes=2ll*w.batch*g.TileK();
            block.prefix_bytes=first?std::clamp(input.prefix-std::int64_t(tile)*g.slot_bytes,
                std::int64_t(0),std::int64_t(g.slot_bytes)):0;
            block.prefix_requests=block.prefix_bytes?1:0;
            block.remainder_requests=block.prefix_bytes<block.weight_bytes?1:0;
            block.expected_transactions=1;work.blocks.push_back(block);
          }
          if(first) {
            owner.l2_hint_bytes=std::min<std::int64_t>(input.l2,2ll*g.tile_n*input.k-input.prefix);
            owner.l2_hint_requests=owner.l2_hint_bytes?1:0;
          }
        }else if(stage.kind=="attention") {
          auto const& a=c.attention;int dim=w.head_dim;
          work.group=task/a.chunks;
          auto range=Range(past,task%a.chunks,a.chunks,a.new_row_work);
          work.contains_current_row=range.first<=past && past<range.second;
          int prefix_rows=first?int(input.prefix/(4*dim)):0;
          int step=w.batch<=4?std::max(1,range.second-range.first):a.block_keys;
          // Low-B executes one barrier generation even for an empty chunk.
          for(int begin=range.first;begin<range.second ||
              (w.batch<=4 && work.blocks.empty());begin+=step,++sequence) {
            Sm90DenseBlockWork block;
            block.sequence=sequence;block.slot=sequence%stage.ring_slots;
            block.generation=generations[cta][stage.ring_bank][block.slot]++;
            block.previous_reader=sequence>=stage.ring_slots?sequence-stage.ring_slots:-1;
            int end=std::min(range.second,begin+step);
            block.history_rows=std::max(0,std::min(end,past)-begin);
            block.valid_rows=end-begin;
            block.compute_rows=w.batch<=4?block.valid_rows:a.block_keys;
            block.current_rows=begin<=past && past<end;
            block.kv_bytes=4ll*dim*block.history_rows;
            int rows=first && sequence<stage.ring_slots?std::min(prefix_rows,block.history_rows):0;
            prefix_rows-=rows;block.prefix_bytes=4ll*dim*rows;
            block.prefix_requests=rows?2:0;
            block.remainder_requests=rows<block.history_rows?2:0;
            block.expected_transactions=block.history_rows?1:0;
            block.explicit_bulk_arrivals=block.history_rows?0:1;
            work.blocks.push_back(block);
          }
          if(first) {
            int history=std::min(range.second,past)-range.first;
            int copied=std::min(history,int(input.prefix/(4*dim)));
            auto per_array=std::min<std::int64_t>(input.l2/2,2ll*dim*(history-copied))&~15ll;
            owner.l2_hint_bytes=2*per_array;owner.l2_hint_requests=per_array?2:0;
          }
        }
        owner.tasks.push_back(std::move(work));
      }
    }
    graph.stages.push_back(std::move(stage));
    // Keep the layout in inputs for the next edge after moving its task data.
    input.work.layout=graph.stages.back().layout;
    input.work.workspace_offset=graph.stages.back().workspace_offset;
  }
  if(graph.shared_bytes>graph.shared_budget)
    throw std::invalid_argument("native program shared reservation exceeds residency budget");
  return graph;
}

json::Value Sm90DenseWorkGraph::ToJson(bool include_blocks) const {
  json::Array rows;
  for(auto const& s:stages) {
    json::Array histogram;
    for(auto const& p:s.LocalTaskHistogram())histogram.emplace_back(json::Object{
        {"local_tasks",p.first},{"ctas",p.second}});
    std::int64_t weight=0,activation=0,kv=0,prefix=0,l2=0,current=0;
    int blocks=0,prefix_requests=0,remainder_requests=0;
    json::Array owners;
    for(auto const& cta:s.ctas) {
      l2+=cta.l2_hint_bytes;json::Array tasks;
      for(auto const& task:cta.tasks) {
        json::Array parts;
        for(auto const& b:task.blocks) {
          ++blocks;weight+=b.weight_bytes;activation+=b.activation_bytes;
          kv+=b.kv_bytes;prefix+=b.prefix_bytes;current+=b.current_rows;
          prefix_requests+=b.prefix_requests;remainder_requests+=b.remainder_requests;
          if(include_blocks)parts.emplace_back(json::Object{
              {"sequence",b.sequence},{"slot",b.slot},{"generation",double(b.generation)},
              {"previous_reader",b.previous_reader},{"history_rows",b.history_rows},
              {"valid_rows",b.valid_rows},{"compute_rows",b.compute_rows},{"current_rows",b.current_rows},
              {"weight_bytes",double(b.weight_bytes)},{"activation_bytes",double(b.activation_bytes)},
              {"kv_bytes",double(b.kv_bytes)},{"prefix_bytes",double(b.prefix_bytes)},
              {"prefix_requests",b.prefix_requests},{"remainder_requests",b.remainder_requests},
              {"expected_transactions",b.expected_transactions},
              {"explicit_bulk_arrivals",b.explicit_bulk_arrivals},
              {"activation_arrivals",b.activation_arrivals},{"empty_arrivals",b.empty_arrivals}});
        }
        if(include_blocks)tasks.emplace_back(json::Object{{"task",task.task},{"group",task.group},
            {"contains_current_row",task.contains_current_row},{"blocks",parts}});
      }
      if(include_blocks)owners.emplace_back(json::Object{{"cta",cta.cta},{"tasks",tasks},
          {"l2_hint_bytes",double(cta.l2_hint_bytes)},{"l2_hint_requests",cta.l2_hint_requests}});
    }
    json::Value row=json::Object{{"kind",s.kind},{"layer",s.layer},{"tasks",s.tasks},
        {"active_ctas",std::min(s.tasks,grid)},{"reserved_ctas",grid},
        {"local_task_histogram",histogram},{"ring_bank",s.ring_bank},{"ring_slots",s.ring_slots},
        {"prefix_predecessor",s.prefix_predecessor},{"incoming_order",Name(s.incoming_order)},
        {"inactive_zero_bytes_per_cta",double(s.inactive_zero_bytes_per_cta)},
        {"merge_chunks",s.merge_chunks},{"merge_winner",
            s.kind=="attention" && s.merge_chunks>1?"runtime_last_arrival":"none"},
        {"logical_weight_bytes",double(weight)},{"logical_activation_bytes",double(activation)},
        {"logical_kv_history_bytes",double(kv)},{"current_rows",double(current)},
        {"prefix_bytes",double(prefix)},{"l2_hint_bytes",double(l2)},{"blocks",blocks},
        {"prefix_requests",prefix_requests},{"remainder_requests",remainder_requests}};
    if(include_blocks)row.Set("ctas",owners);
    rows.push_back(row);
  }
  return json::Object{{"schema",1},{"status","logical_work_only"},{"prediction_available",false},
      {"occupancy_status","resource_pending"},{"grid",grid},{"num_sms",num_sms},
      {"requested_residency",requested_residency},{"shared_bytes",shared_bytes},
      {"shared_budget",shared_budget},{"past",past},{"stages",rows},
      {"activation_reuse",activation_reuse},
      {"launch_inactive_zero_bytes_per_cta",double(launch_inactive_zero_bytes_per_cta)},
      {"logical_bytes_are_hbm_transactions",false},{"l2_hints_guarantee_readiness",false},
      {"prefix_completes_before_predecessor_release",false},
      {"consumer_finalize_blocks_producer_refill",false},
      {"hardware_launch_waves_allowed",false}};
}
}  // namespace tilemega::solver
