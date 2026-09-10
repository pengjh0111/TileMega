// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/AttentionWork.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>
#include <iomanip>

int main(int argc,char** argv) try {
  using namespace tilemega;
  using namespace tilemega::solver;
  if (argc!=2) throw std::invalid_argument("usage: tilemega-attention-work REPO");
  analysis::IslContext isl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>();
  long checks=0;
  int errors=0;
  auto target=TargetSpec::FromJson(std::string(argv[1])+"/configs/targets/sm_89.json");
  std::cout << std::setprecision(17) << "model\tseq\tpast\tchunks\tphase\ttasks\tshared_bytes\tprice_ns\n";
  for (std::string model_name:{"gqa2","mha4"}) {
    auto module=frontend::TorchExportImporter{}.Import(std::string(argv[1])+
        "/docs/experiments/SEQSCAN/raw/export/"+model_name+".json",context);
    auto model=ModelDescription::FromCouplingGraph(*module,{4,3,7},model_name);
    CostModel cost(target,model.dtype);
    auto reject=[&](auto action) {
      auto before=isl.ReferenceCount(); bool caught=false;
      try { action(); } catch (std::invalid_argument const&) { caught=true; }
      if (!caught || before!=isl.ReferenceCount()) throw std::runtime_error("attention rejection not audited");
      ++errors;
    };
    for (auto const& semantic:model.task_semantics) {
      auto const& stage=model.stages.at(semantic.stage);
      if (stage.kind!=StageKind::kAttention) continue;
      int H=stage.extent,W=stage.width;
      reject([&]{DeriveAttentionPhaseWork(model,semantic,{1,64},kTensorBF16Threads);});
      reject([&]{DeriveAttentionPhaseWork(model,semantic,{2,0},kTensorBF16Threads);});
      reject([&]{DeriveAttentionPhaseWork(model,semantic,{2,1},kTensorBF16Threads);});
      reject([&]{DeriveAttentionPhaseWork(model,semantic,{2,64},3);});
      reject([&]{auto m=model; ApplyAttentionCostPlan(m,{{2,64}},kTensorBF16Threads);});
      reject([&]{auto m=model; ApplyAttentionCostPlan(m,{},0);});
      reject([&]{auto m=model; std::vector<codegen::AttentionRuntimeRecord> bad(model.stages.size());
        bad.at(semantic.stage).chunks=0; ApplyAttentionCostPlan(m,bad,kTensorBF16Threads);});
      for (unsigned chunks:{2,4,8}) {
        auto phases=DeriveAttentionPhaseWork(model,semantic,{chunks,64},kTensorBF16Threads);
        for (int seq:{1,4}) for (int past:{0,3}) {
          auto bound=model; bound.dims={seq,past,seq+past};
          auto theta=bound.MetricBindings();
          for (auto const& phase:phases) {
            using codegen::AttentionPhase;
            bool partitioned=phase.phase==AttentionPhase::kScores || phase.phase==AttentionPhase::kPartialValue;
            int tasks=seq*H*(partitioned ? chunks : 1);
            if (phase.task_count.Eval(theta)!=tasks) throw std::runtime_error("phase task count mismatch");
            BackendTraits traits; traits.threads=kTensorBF16Threads; traits.smem_bytes=phase.shared_bytes;
            double price=cost.TaskCostNs(phase,traits,{2},bound);
            if (!(price>0)) throw std::runtime_error("nonpositive attention phase price");
            std::cout << model_name << '\t' << seq << '\t' << past << '\t' << chunks << '\t'
                      << static_cast<int>(phase.phase) << '\t' << tasks << '\t' << phase.shared_bytes
                      << '\t' << price << '\n';
            auto bound_reads=phase.read_bytes.SubstituteParams(theta),bound_writes=phase.write_bytes.SubstituteParams(theta);
            auto bound_flops=phase.flops.SubstituteParams(theta),bound_transc=phase.transcendental.SubstituteParams(theta);
            for (int q=0;q<tasks;++q) {
              int query=partitioned ? q/chunks : q,chunk=partitioned ? q%chunks : 0;
              int total=seq+past,token=query/H;
              int begin=partitioned ? total*chunk/chunks : 0;
              int end=partitioned ? total*(chunk+1)/chunks : total;
              int length=end-begin,valid=std::max(0,std::min(end,past+token+1)-begin);
              long reads=0,writes=0,flops=0,transc=0;
              if (phase.phase==AttentionPhase::kScores) {
                reads=(valid ? W : 0)*2+valid*W*2; writes=length*4;
                flops=valid*(2*W+1);
              } else if (phase.phase==AttentionPhase::kNormalize) {
                reads=writes=total*4; flops=3*total; transc=total;
              } else if (phase.phase==AttentionPhase::kPartialValue) {
                reads=length*4+length*W*2; writes=W*4; flops=2*length*W;
              } else { reads=chunks*W*4; writes=W*2; flops=chunks*W; }
              analysis::ParamBinding point; point.Bind("q",q);
              auto eval=[&](auto const& work) { return work.BindCoordinates(point).Eval({}); };
              if (eval(bound_reads)!=reads || eval(bound_writes)!=writes ||
                  eval(bound_flops)!=flops || eval(bound_transc)!=transc)
                throw std::runtime_error("attention phase work mismatch: "+model_name+" phase="+
                    std::to_string(static_cast<int>(phase.phase))+" q="+std::to_string(q));
              ++checks;
            }
          }
        }
      }
      break;
    }
  }
  if (checks==0 || isl.ReferenceCount()!=0) throw std::runtime_error("incomplete attention work audit");
  std::cerr << "ATTENTION_WORK per_task_checks=" << checks << " errors=" << errors << " reference_delta=0\n";
} catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 2; }
