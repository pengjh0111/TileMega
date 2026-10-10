// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ListScheduler.h>
#include <tilemega/Backend/ConvIteration.h>

#include <algorithm>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <set>
#include <limits>

#ifndef TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION
#define TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION 1
#endif

namespace tilemega::solver {
namespace {
std::string Ceil(std::string const& expression, int divisor) {
  if (divisor <= 0) throw std::invalid_argument("projection divisor must be positive");
  return "ceild((" + expression + ")," + std::to_string(divisor) + ")";
}
std::string Mul(std::string const& expression, int multiplier) {
  return "(" + expression + ")*" + std::to_string(multiplier);
}
std::string Join(std::vector<std::string> const& parts) {
  std::string text;
  for (auto const& part : parts) {
    if (!text.empty()) text += "; ";
    text += part;
  }
  return text;
}
}  // namespace

ProjectedPlacement BalanceProjectedQueues(RuntimeProjection const& projection,
    analysis::ParamBinding const& theta, int workers) {
  analysis::IslReferenceAudit audit(__func__);
  if (workers<=0) throw std::invalid_argument("projected placement requires workers");
  ProjectedPlacement out;
  for (auto const& [domain,task]:projection.tasks.BindParams(theta).Points())
    out.task_ids.push_back(task);
  std::sort(out.task_ids.begin(),out.task_ids.end());
  std::map<std::vector<long>,int> ids;
  std::vector<int> preferred,lengths(workers);
  for (std::size_t i=0;i<out.task_ids.size();++i) {
    auto const& task=out.task_ids[i];
    if (task.size()!=2 || task[0]<0 || task[1]<0 || !ids.emplace(task,i).second)
      throw std::invalid_argument("invalid projected runtime task identity");
    preferred.push_back(task[1]%workers);
    ++lengths[preferred.back()];
  }
  std::vector<std::vector<int>> successors(ids.size());
  for (auto const& [consumer,producer]:projection.dependencies.BindParams(theta).Points())
    successors.at(ids.at(producer)).push_back(ids.at(consumer));
  auto order=ListScheduler{}.Schedule(successors);
  out.placement=BalanceTaskPlacement(successors,order,preferred,workers,
      *std::max_element(lengths.begin(),lengths.end()));
  std::set<std::vector<long>> waits;
  for (auto const& [consumer,event]:projection.requested_events.BindParams(theta).Points()) {
    int worker=out.placement.worker.at(ids.at(consumer));
    if (event.size()!=4) throw std::invalid_argument("invalid projected event coordinate");
    // kind=2 is a singleton fine event; only these permit owner elision.
    if (event[2]==2) {
      int producer=ids.at({event[1],event[3]});
      if (out.placement.worker[producer]==worker) continue;
    }
    waits.insert({worker,event[1],event[2],event[3]});
  }
  out.wait_entries=waits.size();
  return out;
}

analysis::CouplingRelation ProjectScalarTaskOwnership(ModelTaskSemantics const& semantic,
    analysis::OperatorNode const& task,ModelStage const& stage,int threads) {
  if (stage.kind==StageKind::kGemm || task.output.axes.empty())
    throw std::invalid_argument("unsupported scalar ownership domain");
  return ProjectTaskOwnership(semantic,task,stage,threads);
}
analysis::CouplingRelation ProjectTaskOwnership(ModelTaskSemantics const& semantic,
    analysis::OperatorNode const& task,ModelStage const& stage,int threads) {
  analysis::IslReferenceAudit audit(__func__);
  if (threads<=0 || task.output.axes.empty() ||
      (semantic.element_chunk && task.output.axes.size()!=2))
    throw std::invalid_argument("unsupported runtime ownership domain");
  std::set<std::string> parameters;
  std::vector<std::string> bounds;
  std::string flat="0";
  long stride=1;
  for (int axis=int(task.output.axes.size())-1;axis>=0;--axis) {
    auto extent=task.CoordinateExtent(axis);
    for (auto const& parameter:extent.FreeSymbols()) parameters.insert(parameter);
    if (task.IsTiled(axis)) {
      auto const& name=task.output.axes[axis].name;
      bounds.push_back("0 <= "+name+" < ("+extent.ToIslText()+")");
      flat+="+"+std::to_string(stride)+"*"+name;
    }
    if (axis) stride*=extent.Eval({},{});
  }
  if (semantic.element_chunk) {
    if (!task.tile[0].IsLiteral(1) || !task.tile[1].IsLiteral(1))
      throw std::invalid_argument("element ownership requires unit logical tasks");
    if (stage.kind==StageKind::kRoPE) {
      if (stage.width<=0 || stage.width%2) throw std::invalid_argument("rotation head width must be positive and even");
      auto const& row=task.output.axes[0].name;
      auto const& col=task.output.axes[1].name;
      long cols=task.output.axes[1].extent.Eval({},{});
      flat=std::to_string(cols/2)+"*"+row+"+"+std::to_string(stage.width/2)+
          "*floord("+col+","+std::to_string(stage.width)+")+"+col+"%"+std::to_string(stage.width/2);
    }
    flat="floord(("+flat+"),"+std::to_string(threads)+")";
  }
  std::string prefix;
  for (auto const& name:parameters) prefix+=(prefix.empty() ? "" : ",")+name;
  if (!prefix.empty()) prefix="["+prefix+"] -> ";
  std::string coordinates;
  for (auto const& name:task.Coordinates()) coordinates+=(coordinates.empty() ? "" : ",")+name;
  std::string condition="q = "+flat;
  for (auto const& bound:bounds) condition+=" and "+bound;
  return analysis::CouplingRelation::FromIslText(prefix+"{ [q] -> ["+coordinates+"] : "+condition+" }");
}

RuntimeProjection ProjectRuntimeQueues(ModelDescription const& model,
                                      codegen::RuntimePlan const& plan,
                                      RuntimeProjectionOptions options) {
  analysis::IslReferenceAudit audit(__func__);
#if !TILEMEGA_SYMBOLIC_RUNTIME_PROJECTION
  throw std::runtime_error("symbolic runtime projection disabled");
#endif
  if (!options.stage_kappa.empty())
    for (int kappa : options.stage_kappa)
      if (kappa < 0) throw std::invalid_argument("negative per-stage kappa");
  if (options.grid <= 0 || options.threads <= 0 || options.kappa < 0 ||
      plan.gemms.size() != model.gemms.size() || model.stages.empty())
    throw std::invalid_argument("incomplete runtime projection configuration");
  if (!plan.attention.empty() && plan.attention.size()!=model.stages.size())
    throw std::invalid_argument("attention projection table has wrong length");
  auto dimension = [](std::string const& parameter, int value) {
    return parameter.empty() ? std::to_string(value) : parameter;
  };
  std::string seq = dimension(model.dims.seq_parameter, model.dims.seq);
  std::string past = dimension(model.dims.past_parameter, model.dims.past);
  std::string batch = dimension(model.batch_metric_parameter, model.dims.batch);
  std::string tokens = model.serving ? Mul(batch, model.dims.seq) : seq;
  std::vector<std::string> parameters;
  for (auto const& p : {model.dims.seq_parameter, model.dims.past_parameter,
                        model.serving ? model.batch_metric_parameter : std::string{}})
    if (!p.empty() && std::find(parameters.begin(), parameters.end(), p) == parameters.end())
      parameters.push_back(p);
  std::string prefix;
  if (!parameters.empty()) {
    prefix = "[";
    for (std::size_t i=0; i<parameters.size(); ++i) prefix += (i ? "," : "")+parameters[i];
    prefix += "] -> ";
  }
  std::string valid = "(" + seq + ") >= 1 and (" + past + ") >= 0";
  auto restrict_dimension = [&](std::string const& role, std::string const& expression,
                                std::string const& parameter, int concrete) {
    for (auto const& [name,range] : plan.parameter_ranges) {
      bool matches = name == role;
      for (auto const& [alias,canonical] : model.metric_aliases)
        matches = matches || (name == alias && role == canonical);
      if (!matches) continue;
      if (parameter.empty() && (concrete < range.first || concrete > range.second))
        throw std::invalid_argument("runtime dimension outside exported parameter domain");
      valid += " and " + std::to_string(range.first)+" <= ("+expression+") <= "+
               std::to_string(range.second);
    }
  };
  restrict_dimension(model.seq_metric_parameter,seq,model.dims.seq_parameter,model.dims.seq);
  restrict_dimension(model.past_metric_parameter,past,model.dims.past_parameter,model.dims.past);
  if (model.serving)
    restrict_dimension(model.batch_metric_parameter,batch,
                       model.batch_metric_parameter,model.dims.batch);
  for (auto const& name : parameters)
    if (plan.task_binding.Contains(name))
      valid += " and " + name + " = " + std::to_string(plan.task_binding.At(name));
  auto relation = [&](std::vector<std::string> const& pieces) {
    return analysis::CouplingRelation::FromIslText(prefix+"{ "+Join(pieces)+" }");
  };
  std::map<std::string,analysis::QuasiPolynomial> cardinalities;
  auto cardinality = [&](analysis::CouplingRelation const& map, char const* label, int stage) {
    auto image = map.Image();
    auto found = cardinalities.find(image.ToString());
    if (found != cardinalities.end()) return found->second;
    if (std::getenv("TILEMEGA_PROJECTION_TRACE") && std::string(label) != "worker_waits")
      std::fprintf(stderr,"RUNTIME_PROJECTION count=%s stage=%d\n",label,stage);
    auto count = image.ImageCard();
    if (options.split_count_periods) count = count.SplitPeriods(options.grid);
    cardinalities.emplace(image.ToString(),count);
    return count;
  };
  std::vector<int> chunks;
  for (std::size_t i=0; i<plan.gemms.size(); ++i) {
    auto const& g = plan.gemms[i];
    if (!g.tile_m || !g.tile_n || !g.tile_k || !g.split_k || model.gemms[i].k <= 0)
      throw std::invalid_argument("invalid GEMM in runtime projection");
    if(model.dm && !model.gemm_access.empty() &&
       model.gemm_access.at(i).a==codegen::DmAAccess::kIm2Col) {
      auto const& conv=model.convolutions.at(model.gemm_access.at(i).conv);
      auto geometry=backend::ConvIterationGeometry::Build(conv,
          model.buffer_layouts.at(conv.input_layout),g.tile_k);
      if(geometry.iterations%g.split_k)
        throw std::invalid_argument("runtime convolution split does not divide issued iterations");
      chunks.push_back(g.split_k);
    } else chunks.push_back(std::min<int>(g.split_k,
        (model.gemms[i].k+g.tile_k-1)/g.tile_k));
  }
  RuntimeProjection result;
  result.options=options;
  result.task_binding=plan.task_binding;
  std::vector<int> entry(model.stages.size()), done(model.stages.size());
  std::vector<std::string> counts, tiles(model.stages.size());
  std::vector<int> stage_chunks(model.stages.size(),1);
  std::vector<std::string> task_pieces;
  auto append = [&](int original, bool combine, std::string const& count) {
    int stage = static_cast<int>(counts.size());
    counts.push_back(count);
    auto piece = "[] -> [s="+std::to_string(stage)+",t] : "+valid+
                 " and 0 <= t < ("+count+")";
    auto set = relation({piece});
    result.stages.push_back({original,combine,cardinality(set,"tasks",stage)});
    result.runtime_task_refs = result.runtime_task_refs.Add(result.stages.back().task_count);
    task_pieces.push_back(piece);
    // For modulo ownership every stage's queue length decreases with the
    // placed worker id. Hence placed worker zero attains the global maximum.
    result.max_worker_task_refs = result.max_worker_task_refs.Add(
        cardinality(relation({piece+" and t % "+std::to_string(options.grid)+" = 0"}),
                    "longest",stage));
  };
  for (std::size_t i=0; i<model.stages.size(); ++i) {
    auto const& stage = model.stages[i];
    if (!plan.attention.empty() && plan.attention[i].chunks>1) {
      auto const& choice = plan.attention[i];
      if (stage.kind!=StageKind::kAttention || !choice.chunk_extent)
        throw std::invalid_argument("invalid attention stage in runtime projection");
      stage_chunks[i] = choice.chunks;
      entry[i] = counts.size();
      for (auto phase : codegen::kAttentionExpandedPhases) {
        int multiplier = codegen::AttentionPhaseTasks(phase,1,choice.chunks);
        append(i,false,Mul(Mul(seq,stage.extent),multiplier));
        result.stages.back().attention_phase = phase;
      }
      done[i] = counts.size()-1;
      continue;
    }
    std::string count;
    switch (stage.kind) {
      case StageKind::kAdd:
      case StageKind::kGemm: {
        if (stage.gemm < 0 || static_cast<std::size_t>(stage.gemm) >= plan.gemms.size())
          throw std::invalid_argument("stage GEMM index outside projection plan");
        auto const& g = plan.gemms[stage.gemm];
        int ntiles = (model.gemms[stage.gemm].n+g.tile_n-1)/g.tile_n;
        auto rows=stage.batch_rows ? batch : tokens;
        if (model.dm && !model.gemm_access.empty()) {
          if (model.gemm_access.size()!=model.gemms.size())
            throw std::invalid_argument("incomplete DM GEMM access table");
          if (auto per_batch=model.gemm_access[stage.gemm].rows_per_batch)
            rows=Mul(batch,per_batch);
          auto const& access=model.gemm_access[stage.gemm];
          if(access.b==codegen::DmBAccess::kExpertIndirect) {
            if(!access.binding_blocks || !access.block_rows)
              throw std::invalid_argument("incomplete virtual GEMM task capacity");
            rows=std::to_string(std::uint64_t(access.binding_blocks)*
                ((access.block_rows+g.tile_m-1)/g.tile_m)*g.tile_m);
          }
        }
        tiles[i] = Mul(Ceil(rows,g.tile_m),ntiles);
        stage_chunks[i] = stage.kind==StageKind::kAdd ? 1 : chunks[stage.gemm];
        count = Mul(tiles[i],stage_chunks[i]);
        break;
      }
      case StageKind::kMoETopK:
        if(stage.moe.step==codegen::DmMoeStep::kPrefix ||
            stage.moe.step==codegen::DmMoeStep::kSelectAndDispatch)count="1";
        else if(stage.moe.step==codegen::DmMoeStep::kSelect)count=Ceil(tokens,stage.group);
        else count=Ceil(tokens,stage.moe.chunk_tokens);
        break;
      case StageKind::kMoECombine:
        count=Mul(Ceil(tokens,stage.group),(stage.extent+stage.width-1)/stage.width);
        break;
      case StageKind::kLayerNorm:
      case StageKind::kEmbeddingSum:
      case StageKind::kLayoutConvert:
      case StageKind::kPool:
      case StageKind::kGlobalPoolReduce:
      case StageKind::kEncoderAttention: {
        if(!model.dm || stage.width<=0 || stage.group<=0)
          throw std::invalid_argument("incomplete DM primitive task geometry");
        if(stage.kind==StageKind::kGlobalPoolReduce)
          count=Mul(batch,(stage.extent+stage.width-1)/stage.width);
        else if(stage.kind==StageKind::kEncoderAttention)
          count=Mul(batch,stage.extent*((stage.width+stage.group-1)/stage.group));
        else {
          count=Ceil(stage.rows_per_batch?Mul(batch,stage.rows_per_batch):tokens,stage.group);
          if(stage.kind==StageKind::kPool)count=Mul(count,(stage.extent+stage.width-1)/stage.width);
        }
        break;
      }
      case StageKind::kDepthwiseConv: {
        auto const& conv=model.convolutions.at(stage.dm_conv);
        count=Mul(batch,((conv.p+stage.group-1)/stage.group)*
            ((stage.extent+stage.width-1)/stage.width));
        break;
      }
      case StageKind::kRMSNorm:
      case StageKind::kEmbedding:
        count = stage.batch_rows ? batch : tokens; break;
      case StageKind::kFusedAttention: {
        if (stage.attention_kv_block <= 0 || stage.attention_query_rows <= 0 ||
            stage.extent <= 0 || stage.group <= 0)
          throw std::invalid_argument("incomplete serving attention geometry");
        int qblocks = (model.dims.seq * stage.group +
                       stage.attention_query_rows - 1) /
                      stage.attention_query_rows;
        int cblocks = (model.serving_capacity + stage.attention_kv_block - 1) /
                      stage.attention_kv_block;
        count = Mul(batch, stage.extent * qblocks * cblocks);
        break;
      }
      case StageKind::kAttentionMerge:
        count = Mul(batch, stage.extent); break;
      case StageKind::kArgmaxReduce:
        count = batch; break;
      // One (token, head): the ownership the QK normalization declares.
      case StageKind::kQKNorm: count = Mul(seq,stage.extent); break;
      case StageKind::kRoPE:
        count = plan.ownership_flags & codegen::kRoPETileOwnership ? Mul(seq,stage.extent) :
            Ceil(Mul(seq,stage.extent*(stage.width/2)),options.threads);
        break;
      case StageKind::kKVAppend:
        if (plan.ownership_flags & codegen::kKVTileOwnership) count = Mul(seq,stage.extent);
        else {
          count = "max(("+seq+"),("+past+"))";
          count = Ceil(Mul(count,stage.extent*stage.width),options.threads);
        }
        break;
      case StageKind::kElementwise:
        count = plan.ownership_flags & codegen::kActivationTileOwnership ? seq :
            Ceil(Mul(seq,stage.extent),options.threads);
        break;
      case StageKind::kAttention: count = Mul(seq,stage.extent); break;
    }
    entry[i] = static_cast<int>(counts.size());
    append(static_cast<int>(i),false,count);
    done[i] = entry[i];
    if (stage_chunks[i] > 1) {
      done[i] = static_cast<int>(counts.size());
      auto combine = plan.ownership_flags & codegen::kCombinerTileOwnership ? tiles[i] :
          Ceil(Mul(seq,model.gemms[stage.gemm].n),options.threads);
      append(static_cast<int>(i),true,combine);
    }
  }
  struct Edge {
    int producer,consumer;
    analysis::WaitWindow window;
    std::string offset;
    std::optional<analysis::DependencyTable> table;
    std::optional<codegen::CountedWaitRecord> counted;
  };
  std::vector<Edge> edges;
  for (auto const& edge : plan.dependencies) {
    if (edge.producer >= entry.size() || edge.consumer >= entry.size())
      throw std::invalid_argument("dependency outside runtime projection stages");
    if ((edge.table || edge.counted) && ((model.stages[edge.consumer].kind==StageKind::kAttention &&
        stage_chunks[edge.consumer]>1) ||
        (done[edge.producer]!=entry[edge.producer] &&
         !(plan.ownership_flags & codegen::kCombinerTileOwnership))))
      throw std::invalid_argument("table dependency requires bound tile ownership");
    auto window = edge.window;
    if (model.stages[edge.producer].kind == StageKind::kGemm &&
        done[edge.producer] != entry[edge.producer] &&
        !(plan.ownership_flags & codegen::kCombinerTileOwnership))
      window = {};
    if (model.stages[edge.consumer].kind==StageKind::kAttention &&
        stage_chunks[edge.consumer]>1 && window.narrowed)
      window.div *= stage_chunks[edge.consumer];
    edges.push_back({done[edge.producer],entry[edge.consumer],window,
                     std::to_string(window.offset),edge.table,edge.counted});
  }
  for (std::size_t i=0; i<entry.size(); ++i) if (done[i] != entry[i]) {
    if (model.stages[i].kind==StageKind::kAttention) {
      for (auto const& dep : codegen::AttentionInternalDependencies(stage_chunks[i]))
        edges.push_back({entry[i]+dep.producer,entry[i]+dep.consumer,
            {true,dep.div,dep.scale,0,dep.count},"0"});
    } else if (plan.ownership_flags & codegen::kCombinerTileOwnership) {
      if (options.cg_split_task_order)
        edges.push_back({entry[i],done[i],{true,1,stage_chunks[i],0,stage_chunks[i]},"0"});
      else
        for (int chunk=0; chunk<stage_chunks[i]; ++chunk)
          edges.push_back({entry[i],done[i],{true,1,1,0,1},Mul(tiles[i],chunk)});
    } else edges.push_back({entry[i],done[i],{},"0"});
  }
  for (auto const& edge : edges)
    if (edge.counted)result.runtime_counted.push_back({edge.producer,edge.consumer,*edge.counted});
    else if (edge.table)
      result.runtime_tables.push_back({edge.producer,edge.consumer,*edge.table});
    else result.runtime_windows.push_back({edge.producer, edge.consumer,
                                           edge.window, edge.offset});
  std::vector<std::string> wait_pieces, dependency_pieces, requested_pieces;
  std::map<std::pair<int,int>,std::vector<std::string>> event_pieces;
  std::vector<analysis::CouplingRelation> table_dependencies,table_waits,table_requested;
  std::vector<analysis::CouplingRelation> ordinary_tables;
  std::map<std::pair<int,int>,std::vector<analysis::CouplingRelation>> table_events;
  std::uint64_t counted_offset=0;
  for (auto const& edge : edges) {
    if (edge.counted && !options.force_all_dependencies) {
      auto const& counted=*edge.counted;
      auto consumers=counted.contributions.expected.size();
      if(edge.table || !consumers || !counted.producers)
        throw std::invalid_argument("incomplete counted runtime ownership");
      for(auto count:counted.contributions.expected)
        if(!count)throw std::invalid_argument("nonpositive counted runtime threshold");
      for(auto [stage,count]:{std::pair{edge.producer,std::size_t(counted.producers)},
                              std::pair{edge.consumer,consumers}})
        if(cardinality(relation({"[] -> [t] : "+valid+" and 0<=t<("+counts[stage]+")"}),
                       "counted_domain",stage).Eval(plan.task_binding)!=long(count))
          throw std::invalid_argument("counted runtime ownership differs from projected task count");
      auto domain=analysis::CouplingRelation::FromIslText("{ [c] -> [p] : 0<=c<"+
          std::to_string(consumers)+" and 0<=p<"+std::to_string(counted.producers)+" }");
      if(!analysis::Contains(domain,counted.conservative_relation))
        throw std::invalid_argument("counted I2 coupling escapes physical ownership");
      auto consumer=relation({"[cs="+std::to_string(edge.consumer)+",c] -> [c] : "+
          valid+" and 0<=c<("+counts[edge.consumer]+")"});
      auto producer=relation({"[p] -> [ps="+std::to_string(edge.producer)+",p] : "+
          valid+" and 0<=p<("+counts[edge.producer]+")"});
      table_dependencies.push_back(consumer.ApplyRange(counted.conservative_relation).ApplyRange(producer));
      auto event=relation({"[cs="+std::to_string(edge.consumer)+",c] -> [w,pstage="+
          std::to_string(edge.producer)+",kind=3,g] : "+valid+" and 0<=c<("+
          counts[edge.consumer]+") and w=c%"+std::to_string(options.grid)+
          " and g="+std::to_string(counted_offset)+"+c"});
      table_waits.push_back(event);table_requested.push_back(event);
      table_events[{edge.producer,3}].push_back(event);
      counted_offset+=consumers;
      if(counted_offset>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("counted runtime counter storage overflows");
      continue;
    }
    if (edge.table && !options.force_all_dependencies) {
      auto const& table=*edge.table;
      auto proved=analysis::BuildDependencyTableLinear(table.linear_relation,table.producers,table.consumers);
      if (table.stride!=proved.stride || table.intervals.size()!=proved.intervals.size() ||
          !analysis::Contains(table.linear_relation,table.encoded_relation) ||
          !analysis::Contains(table.encoded_relation,table.linear_relation))
        throw std::invalid_argument("unproved runtime dependency table");
      for (std::size_t i=0;i<table.intervals.size();++i)
        if (table.intervals[i].first!=proved.intervals[i].first ||
            table.intervals[i].count!=proved.intervals[i].count)
          throw std::invalid_argument("runtime dependency intervals differ from their relation");
      for (auto [stage,count] : {std::pair{edge.producer,table.producers},
                                 std::pair{edge.consumer,table.consumers}})
        if (cardinality(relation({"[] -> [t] : "+valid+" and 0<=t<("+counts[stage]+")"}),
                        "table_domain",stage).Eval(plan.task_binding)!=count)
          throw std::invalid_argument("runtime table task count differs from projected ownership");
      auto consumer=relation({"[cs="+std::to_string(edge.consumer)+",c] -> [c] : "+
          valid+" and 0<=c<("+counts[edge.consumer]+")"});
      auto producer=relation({"[p] -> [ps="+std::to_string(edge.producer)+",p] : "+
          valid+" and 0<=p<("+counts[edge.producer]+")"});
      auto exact_table=consumer.ApplyRange(table.encoded_relation).ApplyRange(producer);
      table_dependencies.push_back(exact_table);
      ordinary_tables.push_back(std::move(exact_table));
      int kappa=ProducerKappa(options,edge.producer);
      auto identity=analysis::CouplingRelation::FromIslText("{ [c] -> [c] : 0<=c<"+
          std::to_string(table.consumers)+" }");
      auto pairs=consumer.ApplyRange(table.encoded_relation.RangeProduct(identity));
      auto events=[&](int kind) {
        return pairs.ApplyRange(relation({"[p,c] -> [w,pstage="+
            std::to_string(edge.producer)+",kind="+std::to_string(kind)+",g] : w=c%"+
            std::to_string(options.grid)+" and g="+(kind==0?"0":
            "floord(p,"+std::to_string(kappa)+")")}));
      };
      auto fine=events(kappa==0?0:1);
      table_events[{edge.producer,kappa==0?0:1}].push_back(fine);
      table_requested.push_back(kappa==1?events(2):fine);
      if (kappa==1) fine=fine.IntersectRange("{ [w,pstage,kind,g] : g%"+
          std::to_string(options.grid)+" != w }");
      table_waits.push_back(fine);
      continue;
    }
    std::string exact="[cs="+std::to_string(edge.consumer)+",c] -> [ps="+
        std::to_string(edge.producer)+",p] : "+valid+" and 0<=c<("+
        counts[edge.consumer]+") and 0<=p<("+counts[edge.producer]+")";
    if (edge.window.narrowed && !options.force_all_dependencies) {
      if (edge.window.div<=0) throw std::invalid_argument("nonpositive dependency divisor");
      auto at="floord(c,"+std::to_string(edge.window.div)+")*"+
          std::to_string(edge.window.scale)+"+("+edge.offset+")";
      exact+=" and ("+at+")<=p<("+at+")+"+std::to_string(edge.window.count);
    }
    dependency_pieces.push_back(exact);
    std::string base = "[cs="+std::to_string(edge.consumer)+",c] -> [w,pstage="+
        std::to_string(edge.producer)+",kind,g] : "+valid+" and 0 <= c < ("+
        counts[edge.consumer]+") and w = c % "+std::to_string(options.grid);
    int const edge_kappa = ProducerKappa(options, edge.producer);
    if (edge_kappa == 0 || options.force_all_dependencies || !edge.window.narrowed) {
      wait_pieces.push_back(base+" and kind=0 and g=0");
      requested_pieces.push_back(wait_pieces.back());
      event_pieces[{edge.producer,0}].push_back(wait_pieces.back());
    } else {
      auto const& window = edge.window;
      if (window.div <= 0) throw std::invalid_argument("nonpositive dependency divisor");
      std::string at = "floord(c,"+std::to_string(window.div)+")*"+
          std::to_string(window.scale)+"+("+edge.offset+")";
      auto fine = base+" and kind=1 and exists (p : 0 <= p < ("+
          counts[edge.producer]+") and ("+at+") <= p < ("+at+")+"+
          std::to_string(window.count)+" and g=floord(p,"+std::to_string(edge_kappa)+"))";
      event_pieces[{edge.producer,1}].push_back(fine);
      requested_pieces.push_back(edge_kappa==1 ?
          base+" and kind=2 and exists (p : 0<=p<("+counts[edge.producer]+
          ") and ("+at+")<=p<("+at+")+"+std::to_string(window.count)+" and g=p)" : fine);
      wait_pieces.push_back(fine+(edge_kappa == 1 ?
          " and g % "+std::to_string(options.grid)+" != w" : ""));
    }
  }
  result.tasks = relation(task_pieces);
  result.dependencies = relation(dependency_pieces.empty()
      ? std::vector<std::string>{"[cs,c] -> [ps,p] : false"} : dependency_pieces);
  result.ordinary_dependencies=result.dependencies;
  for(auto const& table:ordinary_tables)
    result.ordinary_dependencies=result.ordinary_dependencies->Union(table);
  result.requested_events=relation(requested_pieces.empty()
      ? std::vector<std::string>{"[cs,c] -> [w,pstage,kind,g] : false"} : requested_pieces);
  for (auto const& table : table_dependencies) result.dependencies=result.dependencies.Union(table);
  for (auto const& table : table_requested) result.requested_events=result.requested_events.Union(table);
  if (wait_pieces.empty() && table_waits.empty()) {
    result.runtime_wait_entries = analysis::QuasiPolynomial::Constant(0);
  } else {
    // Project away the consumer coordinate before counting: union cardinality
    // removes per-task duplicates AND repeated polls lifted along each queue.
    result.waits = relation(wait_pieces.empty()
        ? std::vector<std::string>{"[cs,c] -> [w,pstage,kind,g] : false"} : wait_pieces);
    for (auto const& table : table_waits) result.waits=result.waits.Union(table);
    if (!options.count_wait_entries) return result;
    std::vector<analysis::QuasiPolynomial> wait_counts;
    // Producer and event kind are disjoint keys. Count each image separately
    // so barvinok need not partition a union across unrelated stage planes.
    for (auto const& [key,maps] : table_events) event_pieces.try_emplace(key);
    for (auto const& [key,pieces] : event_pieces) {
      auto all=relation(pieces.empty()
          ? std::vector<std::string>{"[cs,c] -> [w,pstage,kind,g] : false"} : pieces);
      if (auto at=table_events.find(key);at!=table_events.end())
        for (auto const& table : at->second) all=all.Union(table);
      auto map = all.ApplyRange(analysis::CouplingRelation::FromIslText(
          "{ [w,pstage,kind,g] -> [w,g] }"));
      auto count_workers = [&](analysis::CouplingRelation const& events, char const* label) {
        if (!options.partition_worker_counts) return cardinality(events,label,key.first);
        // Worker ids form a finite disjoint partition. This enumerates no theta
        // values: each one-dimensional event image retains the full parameter
        // domain, avoiding a costly joint (worker,event) barvinok decomposition.
        std::vector<analysis::QuasiPolynomial> per_worker;
        for (int worker=0;worker<options.grid;++worker) {
          auto owned=events.IntersectRange("{ [w,g] : w="+std::to_string(worker)+" }")
              .ApplyRange(analysis::CouplingRelation::FromIslText("{ [w,g] -> [g] }"));
          per_worker.push_back(cardinality(owned,label,key.first));
        }
        return analysis::QuasiPolynomial::Sum(per_worker);
      };
      wait_counts.push_back(count_workers(map,"waits"));
      if (key.second == 1 && ProducerKappa(options, key.first) == 1) {
        // With singleton events, locality is a property of (worker,event),
        // independent of the consumer. Subtract that subset after the union.
        auto local = map.IntersectRange("{ [w,g] : w = g % "+
            std::to_string(options.grid)+" }");
        wait_counts.push_back(count_workers(local,"local_waits").Scale(-1));
      }
    }
    result.runtime_wait_entries = analysis::QuasiPolynomial::Sum(wait_counts);
  }
  return result;
}
void AttachRuntimeEventMetrics(ModelDescription& model, codegen::RuntimePlan const& plan,
                               RuntimeProjectionOptions options) {
  auto projection=ProjectRuntimeQueues(model,plan,options);
  AttachProjectedEventMetrics(model,plan,projection);
}
void AttachProjectedEventMetrics(ModelDescription& model,codegen::RuntimePlan const& plan,
                               RuntimeProjection const& projection) {
  auto const& options=projection.options;
  if (!options.count_wait_entries)
    throw std::invalid_argument("runtime wait cardinality was deferred by graph-only projection");
  ModelRuntimeEventMetrics metrics;
  metrics.task_refs=projection.runtime_task_refs;
  metrics.wait_entries=projection.runtime_wait_entries;
  metrics.max_worker_task_refs=projection.max_worker_task_refs;
  metrics.gemms=plan.gemms;
  metrics.grid=options.grid; metrics.threads=options.threads; metrics.kappa=options.kappa;
  metrics.stage_count=static_cast<int>(projection.stages.size());
  // Fusion and all-consumers-local producer counts are supplied by B, not
  // estimated from edge locality percentages in A.
  model.coupling_metrics.runtime=std::move(metrics);
}
}  // namespace tilemega::solver
