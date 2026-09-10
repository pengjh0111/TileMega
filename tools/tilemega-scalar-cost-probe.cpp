// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <tuple>

namespace {
int CheckIndices(tilemega::solver::ModelStage const& stage,bool tiles,int threads,
    tilemega::solver::ModelDescription const& model,tilemega::solver::DerivedTaskInput const& input,long count) {
  using tilemega::solver::StageKind;
  int seq=model.dims.seq,past=model.dims.past,total=model.dims.total;
  int checks=0;
  for (long q:std::set<long>{0,count/2,count-1}) {
    std::set<std::vector<long>> expected;
    std::set<std::tuple<int,long>> reads;
    int dim=stage.width,heads=stage.extent;
    if (stage.kind==StageKind::kRMSNorm) {
      for (int d=0;d<dim;++d) {
        expected.insert({q,d}); reads.emplace(0,q*dim+d); reads.emplace(1,d);
      }
    } else if (stage.kind==StageKind::kRoPE) {
      long begin=tiles ? q*(dim/2) : q*threads;
      long end=std::min<long>(seq*heads*(dim/2),begin+(tiles ? dim/2 : threads));
      for (long pair=begin;pair<end;++pair) {
        long half=pair%(dim/2),row=pair/(dim/2),token=row/heads,head=row%heads;
        for (long column:{head*dim+half,head*dim+half+dim/2}) {
          expected.insert({token,column}); reads.emplace(0,token*heads*dim+column);
        }
        reads.emplace(1,half);
      }
    } else if (stage.kind==StageKind::kKVAppend) {
      long begin=tiles ? q*dim : q*threads;
      long end=begin+(tiles ? dim : threads);
      for (long index=begin;index<end;++index) {
        if (index<long(seq)*heads*dim) {
          long d=index%dim,head=(index/dim)%heads,token=index/(dim*heads);
          expected.insert({past+token,head*dim+d}); reads.emplace(0,index);
        }
        if (!tiles && index<long(past)*heads*dim) {
          long d=index%dim,pos=(index/dim)%past,head=index/(dim*past);
          expected.insert({pos,head*dim+d}); reads.emplace(1,index);
        }
      }
    } else if (stage.kind==StageKind::kElementwise) {
      long begin=tiles ? q*heads : q*threads;
      long end=std::min<long>(seq*heads,begin+(tiles ? heads : threads));
      for (long index=begin;index<end;++index) {
        expected.insert({index/heads,index%heads}); reads.emplace(0,index); reads.emplace(1,index);
      }
    } else if (stage.kind==StageKind::kAttention) {
      long token=q/heads,head=q%heads,kv=head/stage.group;
      for (int d=0;d<dim;++d) {
        expected.insert({token,head*dim+d}); reads.emplace(0,(token*heads+head)*dim+d);
        for (int j=0;j<total;++j) {
          if (j<=past+token) reads.emplace(1,(kv*total+j)*dim+d);
          reads.emplace(2,(kv*total+j)*dim+d);
        }
      }
    } else throw std::runtime_error("missing independent TaskBody index trace");
    auto actual=input.scalar_access->writes.BindParams(model.MetricBindings())
        .IntersectDomain("{ [q] : q="+std::to_string(q)+" }").Points();
    std::set<std::vector<long>> written;
    for (auto const& [task,point]:actual) written.insert(point);
    tilemega::analysis::ParamBinding point; point.Bind("q",q);
    auto read_count=input.work.read_elements.BindCoordinates(point).SubstituteParams(model.MetricBindings()).Eval({});
    if (written!=expected || read_count!=long(reads.size()))
      throw std::runtime_error("runtime scalar access differs from independent body index trace at q="+std::to_string(q));
    ++checks;
  }
  return checks;
}
}

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=6) throw std::invalid_argument("usage: tilemega-scalar-cost-probe REPO MODEL DTYPE SEQ PAST");
  analysis::IslContext context;
  std::string root=argv[1],name=argv[2],dtype=argv[3];
  int seq=std::stoi(argv[4]),past=std::stoi(argv[5]);
  auto target=TargetSpec::FromJson(root+"/configs/targets/sm_89.json");
  mlir::MLIRContext mlir;
  mlir.getOrLoadDialect<dialect::CGDialect>();
  std::string source=dtype=="bf16" ? root+"/docs/experiments/SEQSCAN/raw/export/"+name+".json"
      : root+"/docs/experiments/"+(name=="gqa2" ? "E2E_GEN" : "P3_GENERALIZATION")+"/raw/export_bridge.json";
  std::cout << std::setprecision(17) << "dtype\tmodel\townership\tseq\tpast\tstage\top\ttasks\tread_elements\twrite_elements\tdepth\tbarriers\told_ns\tnew_ns\tratio\n";
  int index_checks=0;
  for (bool tiles:{false,true}) {
    frontend::ImportOptions options;
    options.rope_tile_per_block=options.kv_tile_per_block=options.activation_tile_per_block=
        options.combiner_tile_per_block=tiles;
    auto cg=frontend::TorchExportImporter{}.Import(source,mlir,nullptr,options);
    auto model=ModelDescription::FromCouplingGraph(*cg,{seq,past,seq+past},name);
    auto plan=codegen::ReadRuntimePlan(*cg);
    std::vector<GemmConfig> configs;
    for (auto const& g:plan.gemms) configs.push_back({int(g.tile_m),int(g.tile_n),int(g.tile_k),int(g.stages),int(g.split_k)});
    auto graph=InstantiateModelTasks(model,configs);
    int threads=model.dtype==ScalarType::kBF16 ? kTensorBF16Threads : kSimtF32Threads;
    auto symbolic=model; symbolic.dims=ModelDims::Symbolic("S",past);
    auto projection=ProjectRuntimeQueues(symbolic,plan,{target.res.num_sms*2,threads,0});
    CostModel cost(target,model.dtype);
    for (auto const& semantic:model.task_semantics) {
      auto const& stage=model.stages.at(semantic.stage);
      if (stage.kind==StageKind::kGemm) continue;
      auto input=DeriveModelTaskInput(model,semantic,graph,nullptr);
      auto bindings=model.MetricBindings();
      auto count=input.work.task_count.SubstituteParams(bindings).Eval({});
      analysis::ParamBinding theta; theta.Bind("S",seq);
      auto projected=projection.stages.at(semantic.stage).task_count.SubstituteParams(theta).Eval({});
      if (projected!=count) throw std::runtime_error("scalar ownership task count disagrees with runtime projection: "+semantic.op.name);
      index_checks+=CheckIndices(stage,tiles,threads,model,input,count);
      BackendTraits traits; traits.threads=threads;
      traits.smem_bytes=sizeof(float)*codegen::SimtSharedElements(static_cast<codegen::TaskKind>(stage.kind),threads,TILEMEGA_ATTENTION_MAX_TOTAL);
      auto [depth,barriers]=input.scalar_flow->MemoryDepthAndBarriers(threads);
      double old=cost.NonGemmStageNs(stage,model.dims,{2});
      double current=cost.TaskCostNs(input,traits,{2},model,1);
      std::cout << dtype << '\t' << name << '\t' << (tiles ? "tile" : "element") << '\t'
          << seq << '\t' << past << '\t' << semantic.stage << '\t' << semantic.op.name << '\t'
          << count << '\t' << input.work.read_elements.SumDomain().SubstituteParams(bindings).Eval({}) << '\t'
          << input.work.write_elements.SumDomain().SubstituteParams(bindings).Eval({}) << '\t'
          << depth << '\t' << barriers << '\t' << old << '\t' << current << '\t' << current/old << '\n';
    }
  }
  std::cerr << "SCALAR_INDEX_CHECKS task_points=" << index_checks << " exact_write_sets=1 read_counts=1\n";
  if (context.ReferenceCount()) throw std::runtime_error("scalar probe retained isl objects");
  std::cerr << "ISL_CONTEXT remaining=0\n";
} catch (std::exception const& error) {
  std::cerr << "tilemega-scalar-cost-probe: " << error.what() << '\n'; return 2;
}
