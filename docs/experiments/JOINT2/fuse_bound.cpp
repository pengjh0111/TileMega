// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/FusionResources.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/BuiltinOps.h>
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace tilemega;
using namespace tilemega::solver;
// Zero-cost internal storage is optimistic: no extra fused barrier, copies,
// recomputation or residency loss. External traffic and arithmetic stay priced
// by TaskInstanceNs. This is a model upper bound, not a predicted speedup.
static double TrafficFloor(ModelFusionCandidate const& candidate, CostModel const& cost,
    BackendTraits const& producer, BackendTraits const& consumer, Residency residency,
    ModelDescription const& model) {
  auto theta=model.MetricBindings();auto relation=candidate.accesses.consumer_to_producer.BindParams(theta);
  std::vector<std::vector<long>> ps,cs;
  for(auto const& [c,p]:relation.Points()){ps.push_back(p);cs.push_back(c);}
  std::sort(ps.begin(),ps.end());ps.erase(std::unique(ps.begin(),ps.end()),ps.end());
  std::sort(cs.begin(),cs.end());cs.erase(std::unique(cs.begin(),cs.end()),cs.end());
  auto phase=[&](DerivedTaskInput const& input,BackendTraits const& traits,
      auto const& accesses,std::vector<std::vector<long>> const& coords,bool is_producer) {
    std::vector<analysis::ParamBinding> points;
    for(auto const& v:coords){analysis::ParamBinding p;for(std::size_t j=0;j<v.size();++j)p.Bind(input.cost_coordinates[j],v[j]);points.push_back(p);}
    std::vector<TaskMemoryTraffic> memory(coords.size());double bytes=model.dtype==ScalarType::kBF16 ? 2 : 4;
    for(auto const& [name,map]:accesses.reads){
      auto n=map.Card().EvalPoints(theta,points);
      bool local=!is_producer && candidate.accesses.intermediate_tiles.count(name);
      for(std::size_t i=0;i<n.size();++i)if(!local)memory[i].global_read_bytes+=bytes*n[i];
    }
    for(auto const& [name,map]:accesses.writes){
      auto n=map.Card().EvalPoints(theta,points);
      bool removed=is_producer && candidate.accesses.intermediate_tiles.count(name) && !candidate.accesses.retained_intermediates.count(name);
      for(std::size_t i=0;i<n.size();++i)if(!removed)memory[i].global_write_bytes+=bytes*n[i];
    }
    if(!is_producer)for(auto& m:memory)for(std::size_t j=0;j<input.task.operands.size();++j)
      if(candidate.accesses.intermediate_tiles.count(input.task.operands[j].tensor.name))m.local_read_operands.insert(j);
    long grid=long(cost.target().res.num_sms)*residency.ctas_per_sm;double total=0;
    for(long first=0;first<long(coords.size());first+=grid){
      long active=std::min(grid,long(coords.size())-first);double wave=0;
      double occupancy=cost.options().wave_tail ? std::max(1.,double(active)/cost.target().res.num_sms) : residency.ctas_per_sm;
      for(long i=first;i<first+active;++i)wave=std::max(wave,cost.TaskInstanceNs(input,traits,residency,model,1,points[i],occupancy,&memory[i]));
      total+=wave;
    }
    return total;
  };
  return phase(candidate.producer,producer,candidate.producer_accesses,ps,true)+phase(candidate.consumer,consumer,candidate.consumer_accesses,cs,false);
}
int main(int argc,char** argv) try {
  if(argc!=5)throw std::invalid_argument("fuse_bound SOLVED_CG TARGET SEQ OUT.tsv");
  analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);
  if(!module)throw std::invalid_argument("invalid CG");
  int seq=std::stoi(argv[3]);auto target=TargetSpec::FromJson(argv[2]);
  auto model=ModelDescription::FromCouplingGraph(*module,{seq,3,seq+3},"fusion-upper-bound");
  auto runtime=codegen::ReadRuntimePlan(*module);std::vector<GemmConfig> configs;
  for(auto const& g:runtime.gemms)configs.push_back({g.tile_m,g.tile_n,g.tile_k,g.stages,g.split_k});
  auto theta=model.MetricBindings();CostModel cost(target,model.dtype);
  int residency=1;if(auto a=module->getOperation()->getAttrOfType<mlir::IntegerAttr>("tilemega.solved_residency"))residency=a.getInt();
  std::ofstream out(argv[4]);out<<std::setprecision(12)<<"producer\tconsumer\tproducer_stage\tconsumer_stage\tstatus\treason\tproducer_nodes\tconsumer_nodes\teliminated_nodes\tremoved_global_bytes\tseparate_task_ns\tfused_task_ns\trecompute_ns\tfixed_share_input\tfixed_upper_ns\ttraffic_floor_ns\ttraffic_upper_ns\toptimistic_upper_ns\n";
  for(std::size_t i=1;i<model.task_semantics.size();++i){
    auto const& p=model.task_semantics[i-1];auto const& c=model.task_semantics[i];
    auto prefix=[&]{out<<p.op.name<<'\t'<<c.op.name<<'\t'<<p.stage<<'\t'<<c.stage<<'\t';};
    try{
      auto candidate=DeriveLogicalFusionCandidate(model,configs,p.op.name,c.op.name);
      if(p.stage==c.stage){prefix();out<<"ALREADY_FUSED\tno additional runtime node to eliminate\t0\t0\t0\t0\t0\t0\t0\t0.232\t0\t0\t0\t0\n";continue;}
      auto trait=[&](int s){auto const& stage=model.stages[s];return ModelTaskTraits(model,s,configs.at(stage.IsCollective()?stage.gemm:0));};
      auto price=PriceFusionTasks(candidate,cost,trait(p.stage),trait(c.stage),{residency},model);
      double fixed=.232*price.separate_ns;
      double floor=TrafficFloor(candidate,cost,trait(p.stage),trait(c.stage),{residency},model);
      double traffic=std::max(0.,price.separate_ns-floor);
      double upper=std::min(price.separate_ns,fixed+traffic);
      prefix();out<<"SUPPORTED\tlogical adjacency and single-producer ownership\t"<<price.producer_tasks<<'\t'<<price.consumer_tasks<<'\t'<<price.producer_tasks<<'\t'<<std::max(0.,price.global_bytes_before-price.global_bytes_after)<<'\t'<<price.separate_ns<<'\t'<<price.fused_ns<<'\t'<<price.recompute_ns<<"\t0.232\t"<<fixed<<'\t'<<floor<<'\t'<<traffic<<'\t'<<upper<<'\n';
    }catch(std::exception const& e){prefix();std::string error=e.what();std::replace(error.begin(),error.end(),'\n',' ');std::replace(error.begin(),error.end(),'\t',' ');out<<"UNSUPPORTED\t"<<error<<"\t0\t0\t0\t0\t0\t0\t0\t0.232\t0\t0\t0\t0\n";}
    out.flush();
  }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
