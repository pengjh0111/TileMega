// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/SkeletonSearch.h>
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Analysis/ExactMemo.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <cstring>
#include <numeric>
#include <tilemega/Codegen/tasks/TaskResources.h>
using namespace tilemega;
namespace {
using Clock=std::chrono::steady_clock;
double Ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
solver::VariantResources ReadResource(std::filesystem::path const& root,solver::GemmConfig const* g) {
  auto name=g?std::to_string(g->tile_m)+"_"+std::to_string(g->tile_n)+"_"+std::to_string(g->tile_k)+"_"+std::to_string(g->stages):"nongemm";
  auto path=root/(name+".json");auto file=llvm::MemoryBuffer::getFile(path.string());
  if(!file)throw std::runtime_error("resource not yet probed: "+path.string());
  auto json=llvm::json::parse(file.get()->getBuffer());auto* o=json?json->getAsObject():nullptr;
  if(!o)throw std::runtime_error("bad resource JSON");
  return {int(*o->getInteger("registers")),int(*o->getInteger("shared_bytes")),int(*o->getInteger("threads")),false};
}
void AuditPrices(solver::SymbolicProblem const& problem,solver::PreparedFlow const& flow,
    analysis::DramFloor const& floor,TargetSpec const& target,int residency,int sample,std::ostream& out) {
  auto model=problem.model;model.metric_bindings.values.erase("Tm");model.metric_bindings.values.erase("Tn");
  auto theta=model.MetricBindings();auto graph=solver::InstantiateModelTasks(model,problem.geometry);
  solver::CostModelOptions options;options.regime_a=true;options.physical_fixed=false;
  solver::CostModel cost(target,model.dtype,options);double fair=target.CalibrationFor("bf16").dram_gbps/(target.res.num_sms*residency);
  for(std::size_t s=0;s<problem.counts.size();++s) {
    auto const& projected=problem.projection.stages[s];auto const& stage=model.stages[projected.logical_stage];
    auto semantic=*std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[&](auto const& sem){return sem.stage==projected.logical_stage && (!stage.IsCollective() || sem.op.kind==analysis::OperatorKind::kMatmul);});
    auto const& g=problem.geometry.at(stage.IsCollective()?stage.gemm:0);solver::DerivedTaskInput input;solver::BackendTraits traits;int chunks=1;
    if(projected.combine) {
      input=solver::DeriveCombineTaskInput(model,projected.logical_stage,g,graph,problem.threads,problem.runtime.ownership_flags & codegen::kCombinerTileOwnership,true);
      auto resource=codegen::ReadSimtTaskResources(codegen::TaskKind::kGemmCombine,problem.threads);traits.threads=resource.threads;traits.smem_bytes=resource.shared_bytes;traits.shape_legal=true;
      semantic.op.name=input.task.name;semantic.op.kind=analysis::OperatorKind::kReduction;semantic.op.element_reads.clear();
    } else {input=solver::DeriveModelTaskInput(model,semantic,graph,stage.IsCollective()?&g:nullptr);traits=solver::ModelTaskTraits(model,projected.logical_stage,g);chunks=stage.IsCollective()?cost.Chunks(model.gemms[stage.gemm],g):1;}
    solver::BindTaskDramProvenance(input,semantic,floor,theta);
    std::vector<std::pair<std::string,long>> axes;
    long count=input.work.task_count.Eval(theta);
    if(input.scalar_access)axes.push_back({"q",count});
    else for(std::size_t a=0;a<input.task.output.axes.size();++a)if(input.task.IsTiled(a))axes.push_back({input.task.output.axes[a].name,input.task.CoordinateExtent(a).Eval(theta,theta)});
    std::vector<analysis::ParamBinding> points(count);
    for(long t=0;t<count;++t){long rest=t;for(auto a=axes.rbegin();a!=axes.rend();++a){points[t].Bind(a->first,rest%a->second);rest/=a->second;}}
    auto values=solver::PriceTaskInstances(cost,input,traits,{residency},model,chunks,points,residency);
    double direct=std::accumulate(values.begin(),values.end(),0.0),piece=flow.prices[s].total_isolated_ns;
    double relative=direct?std::abs(piece/direct-1):std::abs(piece);bool exact=true;
    auto const& pieces=flow.prices[s].pieces;
    // The sum above checks every task. Exercise the constructor identity at
    // the two boundaries and midpoint; repeated identical price components
    // add no coverage to the separate bitwise instance identity test.
    std::set<std::size_t> probes;
    if(!pieces.empty())probes={0,pieces.size()/2,pieces.size()-1};
    for(auto index:probes) {
      auto const& part=pieces[index];
      auto value=cost.PriceParts(input,traits,{residency},model,chunks,part.representative,residency);
      double a=solver::IsolatedNs(value,fair),b=cost.TaskInstanceNs(input,traits,{residency},model,chunks,part.representative,residency);
      exact &= std::memcmp(&a,&b,sizeof(double))==0;
    }
    out<<sample<<'\t'<<s<<'\t'<<input.task.name<<'\t'<<count<<'\t'<<flow.prices[s].pieces.size()<<'\t'<<piece<<'\t'<<direct<<'\t'<<relative<<'\t'<<exact<<'\t'<<probes.size()<<'\n';out.flush();
    if(relative>1e-9 || !exact)throw std::runtime_error("piece audit failed: "+input.task.name);
  }
}
}
int main(int argc,char** argv) try {
  if(argc!=9 && argc!=10)throw std::invalid_argument("usage: tilemega-flow-validation export.json target.json fixture resources out count seed colocate");
  std::filesystem::path out=argv[5];std::filesystem::create_directories(out);
  bool resume=argc==10 && std::string(argv[9])=="--resume";
  int completed=0;
  if(std::filesystem::exists(out/"samples.tsv")) {
    if(!resume)throw std::runtime_error("refusing to overwrite samples");
    std::ifstream prior(out/"samples.tsv");std::string line;std::getline(prior,line);
    while(std::getline(prior,line))if(!line.empty()){int index=std::stoi(line.substr(0,line.find('\t')));if(index!=completed++)throw std::runtime_error("noncontiguous resume samples");}
  } else if(resume)throw std::runtime_error("no samples to resume");
  analysis::IslContext isl;analysis::ScopedExactAnalysisMemo memo;mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto target=TargetSpec::FromJson(argv[2]);auto bridge=frontend::ReadExportBridge(argv[1]);
  auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);frontend::TorchExportImporter importer;
  auto imported=importer.ImportSemantics(argv[1],plan,context);auto classes=solver::BuildOperatorClasses(imported);
  std::vector<std::vector<solver::GemmConfig>> domains;
  for(auto const& c:classes)domains.push_back(solver::ClassCandidates(c,imported,target,solver::ScalarType::kBF16));
  analysis::CouplingCache couplings;solver::FlowPreparationCache cache;
  solver::VariantResourceCache resources([&](auto const&,auto const* g,auto){return ReadResource(argv[4],g);});
  solver::HopCurve hop;std::string error;
  if(!solver::HopCurve::FromTsv(std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/SIMULATOR/hop_ns.tsv",&hop,&error))throw std::runtime_error(error);
  bool prices_only=argc==10 && std::string(argv[9])=="--prices-only";
  auto mode=resume?std::ios::app:std::ios::out;
  std::ofstream audits(out/"price_checks.tsv",mode);if(!resume)audits<<std::setprecision(17)<<"configuration\tstage\tspace\ttasks\tpieces\tpiece_ns\ttile_ns\trelative_error\tbit_exact\tbit_coordinates\n";
  int count=std::stoi(argv[6]);bool colocate=std::stoi(argv[8])!=0;std::mt19937 rng(std::stoul(argv[7]));
  std::ofstream samples(out/"samples.tsv",mode),configs(out/"configs.tsv",mode);
  samples<<std::setprecision(17);if(!resume)samples<<"sample\tresidency\tkappa\tlimit\tflow_ns\tfluid_ns\tflow_ms\tfluid_ms\tprepare_ms\tfloor_ns\tnonprefix_edges\tvarying_spaces\n";
  if(!resume)configs<<"sample\tclass\tm\tn\tk\tstages\tsplit\n";
  std::optional<analysis::DramFloor> floor;
  for(int sample=0;sample<count;) {
    std::vector<solver::GemmConfig> config;
    for(auto const& domain:domains)config.push_back(domain[std::uniform_int_distribution<std::size_t>(0,domain.size()-1)(rng)]);
    auto estimate=resources.Estimate(classes,config,target,solver::ScalarType::kBF16);
    if(estimate.resident_limit<1){std::cout<<"REJECT resource_limit=0\n";continue;}
    int residency=std::uniform_int_distribution<int>(1,estimate.resident_limit)(rng),kappa=1<<std::uniform_int_distribution<int>(0,2)(rng);
    if(sample<completed){++sample;continue;}
    auto start=Clock::now();auto module=importer.InstantiateForGranularity(imported,context,solver::ClassGranularity(imported,classes,config),&couplings);
    auto problem=solver::PrepareSymbolicProblem(*module,target,{4,3,7},target.res.num_sms*residency,residency,kappa,nullptr,false);
    if(!floor)floor=solver::DeriveModelDramFloor(*module,problem.model,target,argv[3]);
    auto flow=solver::PrepareFlow(problem,*floor,target,residency,hop,couplings,cache,colocate);
    if(prices_only){AuditPrices(problem,flow,*floor,target,residency,sample,audits);++sample;continue;}
    double prepare_ms=Ms(start);auto value=solver::EvaluateFlow(flow.flow);start=Clock::now();
    auto second=solver::EvaluateFlow(flow.flow);double flow_ms=Ms(start);
    if(second.makespan_ns!=value.makespan_ns)throw std::runtime_error("flow nondeterministic");
    solver::ApplyFlowPrices(problem,flow,target,residency);
    auto skeleton=solver::BuildPlanSkeleton(problem,target.res.num_sms*residency,residency,8,false,couplings);
    if(!colocate)for(auto& s:skeleton.spaces){s.colocation.reset();s.colocated_producer=-1;}
    solver::SkeletonRequest request;request.skeleton=&skeleton;request.pure_template=true;request.hop=hop;request.sms=skeleton.grid;
    solver::SkeletonSolvedPoint point;point.candidate.config=config;point.candidate.key=std::to_string(sample);point.candidate.residency=residency;point.candidate.estimated_limit=estimate.resident_limit;point.candidate.kappa=kappa;
    if(!solver::ScheduleBySkeleton(request,&point.schedule,&point.candidate.placement,&error))throw std::runtime_error(error);
    point.module=std::move(module);point.problem=std::move(problem);point.skeleton=std::move(skeleton);point.flow=flow;
    solver::SkeletonSearchOptions options;options.kappa=kappa;options.common.placement.target=target;options.common.placement.hop=hop;options.common.placement.dims={4,3,7};
    solver::SolverTiming timing;options.common.timing=&timing;start=Clock::now();
    auto entry=solver::FinalizeSkeletonPoint(std::move(point),options,(out/("sample_"+std::to_string(sample))).string());
    std::ofstream phases(out/("sample_"+std::to_string(sample)+".timing.tsv"));timing.Write(phases,"fluid-validation",argv[1],4);
    samples<<sample<<'\t'<<residency<<'\t'<<kappa<<'\t'<<estimate.resident_limit<<'\t'<<value.makespan_ns<<'\t'<<entry.evaluation.makespan_ns<<'\t'<<flow_ms<<'\t'<<Ms(start)<<'\t'<<prepare_ms<<'\t'<<flow.flow.floor_ns<<'\t'<<flow.nonprefix_edges<<'\t'<<flow.varying_spaces.size()<<'\n';samples.flush();
    for(std::size_t c=0;c<config.size();++c){auto const& g=config[c];configs<<sample<<'\t'<<c<<'\t'<<g.tile_m<<'\t'<<g.tile_n<<'\t'<<g.tile_k<<'\t'<<g.stages<<'\t'<<g.split_k<<'\n';}configs.flush();
    std::cout<<"SAMPLE "<<sample<<" flow="<<value.makespan_ns<<" fluid="<<entry.evaluation.makespan_ns<<std::endl;++sample;
  }
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
