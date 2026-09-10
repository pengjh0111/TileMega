// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ListScheduler.h>

#include <algorithm>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <set>

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
  analysis::IslReferenceAudit audit(__func__);
  if (threads<=0 || task.output.axes.size()!=2 || stage.kind==StageKind::kGemm)
    throw std::invalid_argument("unsupported scalar ownership domain");
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
  std::vector<std::string> parameters;
  for (auto const& p : {model.dims.seq_parameter, model.dims.past_parameter})
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
    chunks.push_back(std::min<int>(g.split_k,
        (model.gemms[i].k+g.tile_k-1)/g.tile_k));
  }
  RuntimeProjection result;
  result.options=options;
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
      case StageKind::kGemm: {
        if (stage.gemm < 0 || static_cast<std::size_t>(stage.gemm) >= plan.gemms.size())
          throw std::invalid_argument("stage GEMM index outside projection plan");
        auto const& g = plan.gemms[stage.gemm];
        int ntiles = (model.gemms[stage.gemm].n+g.tile_n-1)/g.tile_n;
        tiles[i] = Mul(Ceil(seq,g.tile_m),ntiles);
        stage_chunks[i] = chunks[stage.gemm];
        count = Mul(tiles[i],stage_chunks[i]);
        break;
      }
      case StageKind::kRMSNorm: count = seq; break;
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
  struct Edge { int producer,consumer; analysis::WaitWindow window; std::string offset; };
  std::vector<Edge> edges;
  for (auto const& edge : plan.dependencies) {
    if (edge.producer >= entry.size() || edge.consumer >= entry.size())
      throw std::invalid_argument("dependency outside runtime projection stages");
    auto window = edge.window;
    if (model.stages[edge.producer].kind == StageKind::kGemm &&
        done[edge.producer] != entry[edge.producer] &&
        !(plan.ownership_flags & codegen::kCombinerTileOwnership))
      window = {};
    if (model.stages[edge.consumer].kind==StageKind::kAttention &&
        stage_chunks[edge.consumer]>1 && window.narrowed)
      window.div *= stage_chunks[edge.consumer];
    edges.push_back({done[edge.producer],entry[edge.consumer],window,
                     std::to_string(window.offset)});
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
  std::vector<std::string> wait_pieces, dependency_pieces, requested_pieces;
  std::map<std::pair<int,int>,std::vector<std::string>> event_pieces;
  for (auto const& edge : edges) {
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
    if (options.kappa == 0 || options.force_all_dependencies || !edge.window.narrowed) {
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
          std::to_string(window.count)+" and g=floord(p,"+std::to_string(options.kappa)+"))";
      event_pieces[{edge.producer,1}].push_back(fine);
      requested_pieces.push_back(options.kappa==1 ?
          base+" and kind=2 and exists (p : 0<=p<("+counts[edge.producer]+
          ") and ("+at+")<=p<("+at+")+"+std::to_string(window.count)+" and g=p)" : fine);
      wait_pieces.push_back(fine+(options.kappa == 1 ?
          " and g % "+std::to_string(options.grid)+" != w" : ""));
    }
  }
  result.tasks = relation(task_pieces);
  result.dependencies = relation(dependency_pieces.empty()
      ? std::vector<std::string>{"[cs,c] -> [ps,p] : false"} : dependency_pieces);
  result.requested_events=relation(requested_pieces.empty()
      ? std::vector<std::string>{"[cs,c] -> [w,pstage,kind,g] : false"} : requested_pieces);
  if (wait_pieces.empty()) {
    result.runtime_wait_entries = analysis::QuasiPolynomial::Constant(0);
  } else {
    // Project away the consumer coordinate before counting: union cardinality
    // removes per-task duplicates AND repeated polls lifted along each queue.
    result.waits = relation(wait_pieces);
    std::vector<analysis::QuasiPolynomial> wait_counts;
    // Producer and event kind are disjoint keys. Count each image separately
    // so barvinok need not partition a union across unrelated stage planes.
    for (auto const& [key,pieces] : event_pieces) {
      auto map = relation(pieces).ApplyRange(analysis::CouplingRelation::FromIslText(
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
      if (key.second == 1 && options.kappa == 1) {
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
