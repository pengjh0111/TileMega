// SPDX-License-Identifier: BSD-3-Clause
// A candidate's CG is re-imported at its actual tile/split, then projected by
// the same runtime ownership/split rules used by the generated executor.
#include "../SIMULATOR/cell_inputs.h"
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Solver/JointSearch.h>
#include <tilemega/Solver/EftPlacement.h>
#include <tilemega/Solver/ChainPlacement.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <set>
#include <functional>
#include <array>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/point.h>
#include <isl/val.h>
using namespace tilemega;using namespace tilemega::solver;
using namespace tilemega::experiments;
using Row=std::map<std::string,std::string>;
std::vector<Row> read(std::string path){
  std::ifstream f(path);if(!f)throw std::runtime_error(path);std::string line;
  std::getline(f,line);auto keys=Split(line,'\t');std::vector<Row> rows;
  while(std::getline(f,line)){auto v=Split(line,'\t');Row r;for(std::size_t i=0;i<keys.size();++i)r[keys[i]]=v.at(i);rows.push_back(r);}return rows;
}
void attach(mlir::ModuleOp module,Candidate const& c,dialect::PlacementTable const* t){
  mlir::OpBuilder b(module.getContext());
  for(auto p:module.getOps<dialect::PlacementOp>()){
    p->setAttr("mode",b.getStringAttr(dialect::PlacementModeName(c.mode)));
    p->setAttr("params",b.getDenseI64ArrayAttr(c.params));
    p->setAttr("window",b.getI64IntegerAttr(1));p->setAttr("policy",b.getStringAttr(dialect::kPlacementPolicyAot));
    p->setAttr("resident_only",b.getBoolAttr(true));
  }
  module->removeAttr(dialect::kPlacementTableAttr);
  if(t){
    llvm::SmallVector<std::int64_t> w(t->worker.begin(),t->worker.end()),s(t->slot.begin(),t->slot.end());
    module->setAttr(dialect::kPlacementTableAttr,b.getDictionaryAttr({
      b.getNamedAttr("worker",b.getDenseI64ArrayAttr(w)),b.getNamedAttr("slot",b.getDenseI64ArrayAttr(s)),
      b.getNamedAttr("seq",b.getI64IntegerAttr(t->seq)),b.getNamedAttr("past",b.getI64IntegerAttr(t->past)),b.getNamedAttr("grid",b.getI64IntegerAttr(t->grid))}));
  }
  if(mlir::failed(mlir::verify(module)))throw std::runtime_error("candidate Plan verifier");
}
// Stream finite relation components instead of building Points()'s vector of
// heap-allocated coordinate pairs. Components may overlap; clients deduplicate
// their integer adjacency/event keys exactly before using them.
void VisitRelation(analysis::IslContext& ctx,std::string const& text,int arity,
                   std::function<void(long const*)> const& visit) {
  struct Sink { int arity; std::function<void(long const*)> const* visit; std::string error; } sink{arity,&visit,{}};
  auto point=[](isl_point* p,void* data)->isl_stat {
    auto& s=*static_cast<Sink*>(data);std::array<long,6> coordinates{};
    for(int i=0;i<s.arity;++i){auto v=isl_point_get_coordinate_val(p,isl_dim_set,i);coordinates[i]=isl_val_get_num_si(v);isl_val_free(v);}
    isl_point_free(p);
    try { (*s.visit)(coordinates.data()); } catch(std::exception const& e){s.error=e.what();return isl_stat_error;}
    return isl_stat_ok;
  };
  struct Parts { Sink* sink; decltype(point)* callback; } parts{&sink,&point};
  auto component=[](isl_basic_set* b,void* data)->isl_stat {
    auto& p=*static_cast<Parts*>(data);auto set=isl_set_from_basic_set(b);
    auto status=isl_set_foreach_point(set,*p.callback,p.sink);isl_set_free(set);return status;
  };
  auto map=isl_map_read_from_str(ctx.raw(),text.c_str());
  if(!map || isl_map_dim(map,isl_dim_param)!=0){isl_map_free(map);throw std::runtime_error("relation must be finite and bound");}
  auto set=isl_map_wrap(map);
  if(isl_set_is_bounded(set)!=isl_bool_true){isl_set_free(set);throw std::runtime_error("unbounded task relation");}
  auto status=isl_set_foreach_basic_set(set,component,&parts);isl_set_free(set);
  if(status!=isl_stat_ok)throw std::runtime_error("relation stream: "+sink.error);
}
int main(int argc,char**argv) try{
  if(argc!=14)throw std::runtime_error("project REPO EXPORT MODEL SEQ M N K STAGES SPLIT KAPPA RESID OUT TOP_K");
  analysis::IslContext isl;std::string repo=argv[1],name=argv[3],out=argv[12],error;
  int seq=std::stoi(argv[4]),kappa=std::stoi(argv[10]),res=std::stoi(argv[11]);
  int past=std::getenv("JOINT_PAST")?std::stoi(std::getenv("JOINT_PAST")):3;
  int phase_seq=std::getenv("JOINT_PHASE_SEQ")?std::stoi(std::getenv("JOINT_PHASE_SEQ")):seq;
  GemmConfig g{std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9])};
  if(!TensorBF16ShapeLegal(g.tile_m,g.tile_n,g.tile_k,g.stages))throw std::runtime_error("backend illegal shape");
  auto target=TargetSpec::FromJson(std::getenv("JOINT_TARGET")?std::getenv("JOINT_TARGET"):repo+"/configs/targets/sm_89.json");int grid=target.res.num_sms*res;
  auto bridge=frontend::ReadExportBridge(argv[2]);auto logical=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  frontend::ImportOptions options;options.rope_tile_per_block=options.kv_tile_per_block=options.activation_tile_per_block=options.combiner_tile_per_block=true;
  options.gemms.assign(logical.gemms.size(),{g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  auto module=frontend::TorchExportImporter{}.Import(argv[2],context,nullptr,options);
  auto runtime=codegen::ReadRuntimePlan(*module);
  auto model=ModelDescription::FromCouplingGraph(*module,{seq,past,seq+past},name);
  std::cerr<<"PROJECT imported "<<name<<" s"<<seq<<std::endl;
  RuntimeProjectionOptions projection_options{grid,128,kappa};
  projection_options.partition_worker_counts=std::getenv("JOINT_PARTITION_WORKERS") && std::string(std::getenv("JOINT_PARTITION_WORKERS"))=="1";
  projection_options.count_wait_entries=false; // this driver consumes relations, not cardinality prices
  auto projection=ProjectRuntimeQueues(model,runtime,projection_options);
  std::vector<int> counts;codegen::RuntimeTaskGraph graph;graph.stage_offsets={0};
  for(auto const&s:projection.stages){int n=s.task_count.Eval({});counts.push_back(n);graph.stage_offsets.push_back(graph.stage_offsets.back()+n);}
  int nodes=graph.stage_offsets.back();graph.successors.resize(nodes);graph.preferred_worker.resize(nodes);
  auto relation_text=projection.dependencies.ToString();
  std::ofstream(out+"/dependencies.isl")<<relation_text;
  std::ostringstream count_text;for(int n:counts)count_text<<n<<'\n';
  std::ofstream(out+"/stage_counts.txt")<<count_text.str();
  if(std::getenv("JOINT_RELATION_ONLY")){std::cout<<"JOINT_RELATION_ONLY PASS nodes="<<nodes<<std::endl;return 0;}
  bool cached=false;
  if(auto cache=std::getenv("JOINT_GRAPH_CACHE")){
    auto contents=[](std::string const& path){std::ifstream f(path);return std::string(std::istreambuf_iterator<char>(f),{});};
    std::string dir=cache;
    if(contents(dir+"/dependencies.isl")==relation_text && contents(dir+"/stage_counts.txt")==count_text.str()){
      std::ifstream runs(dir+"/task_dag.runs.tsv");std::string header;std::getline(runs,header);
      if(header!="producer\tfirst\tlast")throw std::runtime_error("DAG cache header");
      int p,first,last;
      while(runs>>p>>first>>last){
        if(p<0 || p>=nodes || first<0 || last>=nodes || first>last)throw std::runtime_error("DAG cache bounds");
        for(int c=first;c<=last;++c)graph.successors[p].push_back(c);
      }
      if(!runs.eof())throw std::runtime_error("DAG cache parse");
      cached=true;std::cerr<<"PROJECT exact relation cache="<<dir<<std::endl;
    }
  }
  if(!cached){
    std::cerr<<"PROJECT projected; streaming dependencies"<<std::endl;
    VisitRelation(isl,relation_text,4,[&](long const* e){
      int p=graph.stage_offsets.at(e[2])+e[3],c=graph.stage_offsets.at(e[0])+e[1];graph.successors.at(p).push_back(c);
    });
  }
  for(auto& edges:graph.successors){std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());}
  std::vector<int> node_stage(nodes);std::vector<int> baseline_queue(grid);
  for(std::size_t s=0;s<counts.size();++s)for(int t=0;t<counts[s];++t){int n=graph.stage_offsets[s]+t;node_stage[n]=s;graph.preferred_worker[n]=t%grid;++baseline_queue[t%grid];}
  graph.baseline_max_queue=*std::max_element(baseline_queue.begin(),baseline_queue.end());
  std::cerr<<"PROJECT graph nodes="<<nodes<<std::endl;
  struct Phase{double fixed=0,loop=0,load=0;int count=0;};std::map<int,Phase> phases;
  std::string phase_root=std::getenv("JOINT_PHASE_ROOT")?std::getenv("JOINT_PHASE_ROOT"):repo+"/docs/experiments/PHASE/raw";
  for(auto r:read(phase_root+"/trace/"+name+"_s"+std::to_string(phase_seq)+"_p5/node_phases.tsv")){
    auto& p=phases[std::stoi(r.at("stage"))];++p.count;p.fixed+=std::stod(r.at("setup_ns"))+std::stod(r.at("epilogue_ns"));p.loop+=std::stod(r.at("mainloop_ns"));p.load+=std::stod(r.at("load_wait_ns"));}
  for(auto&[s,p]:phases){p.fixed/=p.count;p.loop/=p.count;p.load/=p.count;}
  SimulatorInput in;in.graph=&graph;in.task_ns.resize(nodes);
  std::ofstream weights(out+"/weights.tsv");weights<<"stage\tlogical_stage\tcombine\ttask_ns\tcount\n";
  for(std::size_t s=0;s<counts.size();++s){auto const& ps=projection.stages[s];auto p=phases.at(ps.logical_stage);double ns=p.fixed+p.loop+p.load;
    if(ps.combine)ns=std::max(1024.0,p.fixed)+g.tile_m*g.tile_n*std::min(g.split_k,32)*.02;
    else if(model.stages[ps.logical_stage].kind==StageKind::kGemm){auto op=model.gemms[model.stages[ps.logical_stage].gemm];
      int chunks=std::min(g.split_k,(op.k+g.tile_k-1)/g.tile_k);
      double padded=double((op.k+chunks*g.tile_k-1)/(chunks*g.tile_k)*g.tile_k)/op.k;
      double ratio=.7*g.tile_m*g.tile_n/(128.0*128.0)+.3*(g.tile_m+g.tile_n)/256.0;
      ns=p.fixed+p.load*(g.tile_m+g.tile_n)/256.0+p.loop*ratio*padded;
    }
    ns=std::max(1.0,ns);weights<<s<<'\t'<<ps.logical_stage<<'\t'<<ps.combine<<'\t'<<ns<<'\t'<<counts[s]<<'\n';
    for(int n=graph.stage_offsets[s];n<graph.stage_offsets[s+1];++n)in.task_ns[n]=ns;
  }
  HopCurve hop;if(!HopCurve::FromTsv(std::getenv("JOINT_HOP_TSV")?std::getenv("JOINT_HOP_TSV"):repo+"/docs/experiments/SIMULATOR/hop_ns.tsv",&hop,&error))throw std::runtime_error(error);
  SimulatorOptions sim;sim.observed_task_times=true;sim.flat_hop=true;
  sim.publication_ns=std::getenv("JOINT_PUBLICATION_NS")?std::stod(std::getenv("JOINT_PUBLICATION_NS")):1074.1284026707756;
  sim.consumer_wait_ns=std::getenv("JOINT_WAIT_NS")?std::stod(std::getenv("JOINT_WAIT_NS")):1764.3682109690692;
  PlanRequest request;request.grid=grid;request.graph=&graph;request.counts=counts;
  std::vector<codegen::RuntimeVariantModule> seed_variant{{*module,1,2048}};
  auto seed=codegen::CouplingGraphToCUDA{}.LowerVariants(seed_variant);
  std::ofstream(out+"/seed.cu")<<seed;
  auto schedule_begin=seed.find("constexpr ScheduleStageDesc kSchedule0[] = {");
  if(schedule_begin==std::string::npos)throw std::runtime_error("generated schedule missing");
  auto schedule=seed.substr(schedule_begin,seed.find("};",schedule_begin)-schedule_begin);
  std::regex row(R"(\{([0-9]+)u,)");
  for(std::sregex_iterator it(schedule.begin(),schedule.end(),row),end;it!=end;++it){
    int original=std::stoi((*it)[1]);
    for(std::size_t s=0;s<projection.stages.size();++s)if(projection.stages[s].logical_stage==original)request.stage_order.push_back(s);
  }
  if(request.stage_order.size()!=counts.size())throw std::runtime_error("expanded generated schedule mismatch");
  request.physical_worker.resize(grid);std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
  std::vector<Candidate> candidates={{"legacy_grid_stride",dialect::PlacementMode::kLegacyGridStride,{}},{"rotate",dialect::PlacementMode::kRotate,{}},{"balanced",dialect::PlacementMode::kBalanced,{}},{"eft",dialect::PlacementMode::kEft,{}},{"wavefront",dialect::PlacementMode::kTemplate,{long(dialect::PlacementTemplate::kWavefront)}},{"chain",dialect::PlacementMode::kEft,{}}};
  if(auto only=std::getenv("JOINT_ONLY_PLACEMENT")){
    candidates.erase(std::remove_if(candidates.begin(),candidates.end(),[&](auto const& c){return std::string(c.name)!=only;}),candidates.end());
    if(candidates.empty())throw std::runtime_error("unknown selected placement");
  }
  std::vector<MaterializedPlan> plans;std::vector<dialect::PlacementTable> tables;
  for(auto c:candidates){request.mode=c.mode;request.params=c.params;request.eft_worker.clear();request.eft_slot.clear();
    if(std::string(c.name)=="eft"){EftRequest e;e.graph=&graph;e.task_ns=in.task_ns;e.grid=grid;e.sms=grid;e.ctas_per_sm=1;e.hop=hop;EftSchedule s;
      if(!ScheduleByEarliestFinish(e,&s,&error))throw std::runtime_error(error);request.eft_worker=s.worker;request.eft_slot=s.slot;}
    if(std::string(c.name)=="chain"){ChainRequest e;e.graph=&graph;e.task_ns=in.task_ns;e.grid=grid;e.sms=grid;e.ctas_per_sm=1;e.hop=hop;e.cost_aware_extend=true;ChainSchedule s;
      if(!ScheduleByCriticalChain(e,&s,&error))throw std::runtime_error(error);request.eft_worker=s.worker;request.eft_slot=s.slot;}
    MaterializedPlan plan;if(!MaterializePlanPlacement(request,&plan,&error)||!CheckPlanLegality(graph,plan,&error))throw std::runtime_error(error);
    dialect::PlacementTable table;table.seq=seq;table.past=past;table.grid=grid;
    for(std::size_t s=0;s<counts.size();++s)for(int t=0;t<counts[s];++t){table.worker.push_back(plan.owner[s][t]);table.slot.push_back(plan.slot[s][t]);}
    plans.push_back(std::move(plan));tables.push_back(std::move(table));std::cerr<<"PROJECT placement="<<c.name<<std::endl;
  }
  PreparedPlanBounds bounds;if(!PreparePlanBounds(in,&bounds,&error))throw std::runtime_error(error);
  std::vector<MaterializedPlan const*> pointers;for(auto const&p:plans)pointers.push_back(&p);
  std::vector<RankedPlan> ranking;
  if(!RankPlans(in,bounds,pointers,sim,hop,std::stoul(argv[13]),&ranking,&error))throw std::runtime_error(error);
  auto requested=projection.requested_events.ToString();
  std::ofstream predicted(out+"/predicted.tsv");predicted<<std::setprecision(12)<<"placement\tsimulated\tmakespan_ns\tfloor_ns\tcp_ns\tqueue_lb_ns\tnodes\tgrid\tkappa\tgrouping_model\tstatus\n";
  for(auto r:ranking){
    std::string status="ok";auto const& plan=plans[r.index];
    bool validate_selected=std::getenv("JOINT_VALIDATE_PLACEMENT") && std::string(std::getenv("JOINT_VALIDATE_PLACEMENT"))==candidates[r.index].name;
    if(r.simulated || validate_selected){
      r.simulated=true;
      auto grouped=graph;SimulatorInput priced=in;priced.graph=&grouped;
      priced.publication_required.assign(nodes,0);priced.consumer_wait_required.assign(nodes,0);
      std::vector<std::set<std::pair<int,int>>> desired(nodes);
      std::set<int> publishing_stages;
      VisitRelation(isl,requested,6,[&](long const* event){
        int cs=event[0],ct=event[1],cn=graph.stage_offsets[cs]+ct;
        int ps=event[3],kind=event[4],group=event[5];
        if(kind==2 && plan.owner[ps][group]==plan.owner[cs][ct])return;
        publishing_stages.insert(ps);
        if(!desired[cn].insert({ps,kind==0?-1:group}).second)return;
        int begin=kind==0?0:group*kappa,end=kind==0?counts[ps]:std::min(counts[ps],begin+kappa);
        for(int pt=begin;pt<end;++pt)grouped.successors[graph.stage_offsets[ps]+pt].push_back(cn);
      });
      for(auto& edges:grouped.successors){std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());}
      for(int s:publishing_stages)for(int n=graph.stage_offsets[s];n<graph.stage_offsets[s+1];++n)priced.publication_required[n]=1;
      for(auto const& queue:plan.queue){std::set<std::pair<int,int>> seen;
        for(auto task:queue){int n=graph.stage_offsets[task.stage]+task.logical;
          for(auto event:desired[n])if(seen.insert(event).second)priced.consumer_wait_required[n]=1;}}
      SimulatorResult result;
      if(!CheckPlanLegality(grouped,plan,&error)||!SimulateExecution(priced,plan,sim,hop,&result,&error)){status="group_queue_rejected";r.simulated=false;std::cerr<<candidates[r.index].name<<": "<<error<<std::endl;}
      else r.makespan_ns=result.makespan_ns;
    }
    predicted<<candidates[r.index].name<<'\t'<<r.simulated<<'\t'<<r.makespan_ns<<'\t'<<r.bounds.lower_bound_ns<<'\t'<<r.bounds.critical_path_ns<<'\t'<<r.bounds.queue_lb_ns<<'\t'<<nodes<<'\t'<<grid<<'\t'<<kappa<<"\tprojected_events_and_sigma_lifting\t"<<status<<'\n';
    // Exact CG materialized table; kernel enforces the candidate's kappa.
    // Group readiness and singleton/sigma lifting are priced above using
    // the projection's requested-event relation and the candidate's own Plan.
    auto const& c=candidates[r.index];attach(*module,c,c.mode==dialect::PlacementMode::kEft?&tables[r.index]:nullptr);
    std::vector<codegen::RuntimeVariantModule> v{{*module,1,2048}};
    std::ofstream source(out+"/"+c.name+".cu");source<<codegen::CouplingGraphToCUDA{}.LowerVariants(v);
  }
  std::ofstream dag(out+"/task_dag.tsv");dag<<"producer\tconsumer\n";for(int p=0;p<nodes;++p)for(int c:graph.successors[p])dag<<p<<'\t'<<c<<'\n';
  std::cout<<"JOINT_PROJECT PASS model="<<name<<" seq="<<seq<<" nodes="<<nodes<<" plans="<<plans.size()<<std::endl;
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}
