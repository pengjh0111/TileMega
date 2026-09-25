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
  std::vector<std::set<std::pair<int,int>>> desired(problem.task_ns.size());std::set<int> publishing;
  auto node=[&](int s,int t){if(s<0 || s>=int(problem.counts.size()) || t<0 || t>=problem.counts[s])throw std::runtime_error("event outside projected tasks");return problem.offsets[s]+t;};
  analysis::VisitFiniteRelation(analysis::SharedIslContext(),problem.projection.requested_events.BindParams(sk.theta).ToString(),6,[&](long const* e){
    int cn=node(e[0],e[1]),ps=e[3],kind=e[4],group=e[5];
    if(kind==2 && selected.plan.owner.at(ps).at(group)==selected.plan.owner.at(e[0]).at(e[1]))return;
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
  entry.module=std::move(point.module);return entry;
}
}
