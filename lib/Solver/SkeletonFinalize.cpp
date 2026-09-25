// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Analysis/VisitFiniteRelation.h>
#include <fstream>
#include <iomanip>
#include <queue>

namespace tilemega::solver {
CompilerSearchResult::ShortlistEntry FinalizeSkeletonPoint(SkeletonSolvedPoint&& point,
    SkeletonSearchOptions const& options,std::string const& prefix) {
  auto const& sk=point.skeleton;auto const& problem=point.problem;
  auto graph=codegen::MaterializeRuntimeTaskGraph(problem.counts,{},sk.grid);
  for(auto const& edge:sk.edges)for(int t=0;t<sk.spaces[edge.producer].count;++t)
    edge.oracle->forward.Query({t},sk.theta).ForEach([&](auto const& q){
      graph.successors[sk.spaces[edge.producer].offset+t].push_back(sk.spaces[edge.consumer].offset+int(q.at(0)));});
  PlanRequest request;request.mode=dialect::PlacementMode::kEft;request.grid=sk.grid;request.counts=problem.counts;request.graph=&graph;
  request.physical_worker.resize(sk.grid);std::iota(request.physical_worker.begin(),request.physical_worker.end(),0);
  for(int s:sk.stage_order)request.stage_order.push_back(s);
  request.eft_worker=point.schedule.worker;request.eft_slot=point.schedule.slot;
  PlacementEvaluation selected;selected.name="skeleton";selected.mode=dialect::PlacementMode::kEft;
  std::string error;
  if(!MaterializePlanPlacement(request,&selected.plan,&error) || !CheckPlanLegality(graph,selected.plan,&error))throw std::runtime_error(error);
  SimulatorInput input;input.graph=&graph;input.task_ns=problem.task_ns;
  input.publication_required.resize(problem.task_ns.size());input.consumer_wait_required.resize(problem.task_ns.size());
  std::vector<std::set<std::pair<int,int>>> desired(problem.task_ns.size());std::set<int> publishing;std::size_t omitted_local_events=0;
  auto node=[&](int s,int t){if(s<0 || s>=int(problem.counts.size()) || t<0 || t>=problem.counts[s])throw std::runtime_error("event outside projected tasks");return problem.offsets[s]+t;};
  analysis::VisitFiniteRelation(analysis::SharedIslContext(),problem.projection.requested_events.BindParams(sk.theta).ToString(),6,[&](long const* e){
    int cn=node(e[0],e[1]),ps=e[3],kind=e[4],group=e[5];
    if(kind==2 && selected.plan.owner.at(ps).at(group)==selected.plan.owner.at(e[0]).at(e[1])){++omitted_local_events;return;}
    publishing.insert(ps);if(!desired[cn].insert({ps,kind==0?-1:group}).second)return;
    int k=ProducerKappa(problem.projection.options,ps),begin=kind==0?0:group*k,end=kind==0?problem.counts[ps]:std::min(problem.counts[ps],begin+k);
    for(int t=begin;t<end;++t){int pn=node(ps,t);graph.successors[pn].push_back(cn);
      if(point.flow && kind!=2 && selected.plan.owner[ps][t]==selected.plan.owner[e[0]][e[1]])input.fluid_forced_local_hops.emplace_back(pn,cn);
    }
  });
  for(auto& row:graph.successors){std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());}
  if(!CheckPlanLegality(graph,selected.plan,&error))throw std::runtime_error("event-group legality: "+error);
  for(int stage:publishing)std::fill(input.publication_required.begin()+problem.offsets[stage],input.publication_required.begin()+problem.offsets[stage+1],1);
  for(auto const& queue:selected.plan.queue){std::set<std::pair<int,int>> seen;for(auto task:queue)for(auto event:desired[node(task.stage,task.logical)])
    if(seen.insert(event).second)input.consumer_wait_required[node(task.stage,task.logical)]=1;}
  SimulatorOptions sim;sim.observed_task_times=true;sim.flat_hop=true;
  if(point.flow) {
    input.task_price_parts=ExpandFlowPrices(*point.flow);
    sim.dram_fluid=true;sim.dram_gbps=point.flow->flow.dram_gbps;
    sim.dram_floor_ns=point.flow->flow.dram_floor_ns;sim.all_external_miss=point.flow->flow.all_external_miss;
  }
  auto const& rates=options.common.placement.target.EventCalibrationFor(problem.model.dtype==ScalarType::kBF16?"bf16":"f32");
  sim.publication_ns=rates.task_publication.ns.value_or(0);sim.consumer_wait_ns=rates.task_wait.ns.value_or(0);
  PreparedPlanBounds bounds;
  if(!PreparePlanBounds(input,&bounds,&error) || !EvaluatePlanBounds(bounds,selected.plan,&selected.bounds,&error))throw std::runtime_error(error);
  SimulatorResult simulated;
  {SolverPhase phase(options.common.timing,"simulate");if(!SimulateExecution(input,selected.plan,sim,options.common.placement.hop,&simulated,&error))throw std::runtime_error(error);}
  selected.predicted_ns=simulated.makespan_ns;
  auto placement=options.common.placement;placement.residency=point.candidate.residency;placement.verified_resident_limit=point.candidate.actual_limit?point.candidate.actual_limit:point.candidate.estimated_limit;placement.kappa=options.kappa;
  WritePlanSkeleton(*point.module,sk);dialect::WriteSolvedPlacement(*point.module,selected,placement);
  CompilerSearchResult::ShortlistEntry entry;
  entry.evaluation.candidate.config=point.candidate.config.front();entry.evaluation.candidate.key=point.candidate.key;
  entry.evaluation.candidate.kappa=options.kappa;entry.evaluation.candidate.ctas_per_sm=point.candidate.residency;
  entry.evaluation.placement="skeleton";entry.evaluation.status="ok";entry.evaluation.simulated=true;
  entry.evaluation.floor_ns=selected.bounds.lower_bound_ns;entry.evaluation.makespan_ns=simulated.makespan_ns;
  (*point.module)->setAttr("tmexec.sync_omitted_event_waits",mlir::IntegerAttr::get(mlir::IntegerType::get(point.module->getContext(),64),omitted_local_events));
  std::ofstream omissions(prefix+".omissions.tsv");
  omissions<<"key\tlocal_event_waits_omitted\n"<<point.candidate.key<<'\t'<<omitted_local_events<<'\n';
  auto const& st=point.candidate.placement;
  std::ofstream metrics(prefix+".metrics.tsv");metrics<<std::setprecision(17)<<"key\tgrid\tresidency\tflow_ns\tsimulated_ns\tfloor_ns\tcp_ns\tqueue_ns\tplaced\taffinity\thome\tspread_other\tcandidate_sum\ttransitions\tadjacent_slots\tinterleaving\n";
  metrics<<point.candidate.key<<'\t'<<sk.grid<<'\t'<<sk.residency<<'\t'<<point.candidate.score<<'\t'<<simulated.makespan_ns<<'\t'<<selected.bounds.lower_bound_ns<<'\t'<<selected.bounds.critical_path_ns<<'\t'<<selected.bounds.queue_lb_ns<<'\t'<<st.placed<<'\t'<<st.affinity<<'\t'<<st.home<<'\t'<<st.spread_other<<'\t'<<st.candidate_sum<<'\t'<<st.transitions<<'\t'<<st.adjacent_slots<<'\t'<<st.interleaving<<'\n';
  std::ofstream oracle(prefix+".oracles.tsv");oracle<<"producer\tconsumer\tstructure\tpredecessor\tsuccessor\tall_producer\n";
  for(auto const& e:sk.edges)oracle<<e.producer<<'\t'<<e.consumer<<'\t'<<analysis::ToString(e.oracle->structure)<<'\t'<<analysis::ToString(e.oracle->reverse.kind())<<'\t'<<analysis::ToString(e.oracle->forward.kind())<<'\t'<<e.all_producer<<'\n';
  std::ofstream tasks(prefix+".tasks.tsv"),edges(prefix+".edges.tsv");tasks<<"node\tstage\ttask\tworker\tslot\tstart_ns\tend_ns\tattention\n";tasks<<std::setprecision(17);edges<<"producer\tconsumer\tkind\n";
  for(std::size_t s=0;s<problem.counts.size();++s)for(int t=0;t<problem.counts[s];++t){int n=node(s,t);auto const& measured=simulated.tasks[n];
    bool attention=problem.model.stages[problem.projection.stages[s].logical_stage].kind==StageKind::kAttention;
    tasks<<n<<'\t'<<s<<'\t'<<t<<'\t'<<selected.plan.owner[s][t]<<'\t'<<selected.plan.slot[s][t]<<'\t'<<measured.start_ns<<'\t'<<measured.end_ns<<'\t'<<attention<<'\n';
    for(int next:graph.successors[n])edges<<n<<'\t'<<next<<"\tdependency\n";}
  for(auto const& queue:selected.plan.queue)for(std::size_t i=1;i<queue.size();++i)edges<<node(queue[i-1].stage,queue[i-1].logical)<<'\t'<<node(queue[i].stage,queue[i].logical)<<"\tqueue\n";
  if(point.flow) {
    auto const& flow=point.flow->flow;FlowDecomposition d;
    {SolverPhase phase(options.common.timing,"flow_report");d=DecomposeFlow(flow);}
    std::vector<int> depth(flow.spaces.size(),1);int chain_depth=0;
    for(int stage:sk.stage_order){for(auto const& edge:flow.edges)if(edge.consumer==stage)depth[stage]=std::max(depth[stage],depth[edge.producer]+1);chain_depth=std::max(chain_depth,depth[stage]);}
    int colocated=0,omitted=0;for(auto const& edge:flow.edges){colocated+=edge.colocated;omitted+=edge.colocated && edge.kappa==1;}
    std::ofstream values(prefix+".flow.tsv");values<<std::setprecision(17)<<"key\tT\tT_floor\tT_dram\tT_s\tT_sf\tT_sf_infinite\tT_np0\tsynchronization\tfixed\tcontention\tchain_delay\tpg_upper_bound\tchain_depth\tbubble_ns\tcolocated_edges\tsync_omitted_edges\n";
    values<<point.candidate.key<<'\t'<<d.original.makespan_ns<<'\t'<<flow.floor_ns<<'\t'<<flow.dram_floor_ns<<'\t'<<d.no_sync<<'\t'<<d.no_fixed<<'\t'<<d.infinite<<'\t'<<d.no_external<<'\t'<<d.synchronization<<'\t'<<d.fixed<<'\t'<<d.contention<<'\t'<<d.chain<<'\t'<<d.pg_upper_bound<<'\t'<<chain_depth<<'\t'<<(d.original.makespan_ns-flow.floor_ns)/chain_depth<<'\t'<<colocated<<'\t'<<omitted<<'\n';
    std::ofstream links(prefix+".flow_chain.tsv");links<<std::setprecision(17)<<"stage\ttask\tcategory\tstart_ns\tend_ns\twait_ns\tfixed_ns\tmainloop_ns\tpublication_ns\thop_ns\n";
    for(auto const& link:d.original.critical_links)links<<link.space<<'\t'<<link.task<<'\t'<<flow.spaces[link.space].category<<'\t'<<link.start_ns<<'\t'<<link.end_ns<<'\t'<<link.wait_ns<<'\t'<<link.fixed_ns<<'\t'<<link.mainloop_ns<<'\t'<<link.publication_ns<<'\t'<<link.hop_ns<<'\n';
    std::ofstream spaces(prefix+".flow_spaces.tsv");spaces<<std::setprecision(17)<<"stage\tname\tcategory\ttasks\tfirst_start\tlast_end\twait_sum\tfixed_sum\tmainloop_sum\tpublication_sum\n";
    for(std::size_t i=0;i<flow.spaces.size();++i){auto const& s=flow.spaces[i];auto const& r=d.original.spaces[i];spaces<<i<<'\t'<<s.name<<'\t'<<s.category<<'\t'<<s.count<<'\t'<<r.first_start<<'\t'<<r.last_end<<'\t'<<r.wait_ns<<'\t'<<r.fixed_ns<<'\t'<<r.mainloop_ns<<'\t'<<r.publication_ns<<'\n';}
  }
  entry.module=std::move(point.module);return entry;
}
}
