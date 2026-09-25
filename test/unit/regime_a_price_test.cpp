// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Solver/TaskModel.h>
#include <mlir/IR/MLIRContext.h>
#include <cstring>
#include <iostream>
using namespace tilemega;
static void Require(bool ok,char const* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv) try {
  if(argc!=3)throw std::invalid_argument("regime_a_price_test repo target");
  analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto target=TargetSpec::FromJson(argv[2]);auto const& cal=target.CalibrationFor("bf16");
  solver::CacheServiceCurve curve(cal.l2_curve_bytes,cal.l2_curve_gbps);
  Require(curve.HitFraction(2.*1024*1024*1024,cal.l2_gbps,cal.dram_gbps)==0,"2 GiB external stream must miss L2");
  int checks=0;
  for(std::string name:{"gqa2","mha4"}) {
    auto module=frontend::TorchExportImporter{}.Import(std::string(argv[1])+"/docs/experiments/SEQSCAN/raw/export/"+name+".json",context);
    auto model=solver::ModelDescription::FromCouplingGraph(*module,{4,3,7},name);
    solver::CostModelOptions options;options.regime_a=true;solver::CostModel cost(target,model.dtype,options);
    for(int k:{16,32,64}) {
      solver::GemmConfig config{32,16,k,2,1};auto graph=solver::InstantiateModelTasks(model,std::vector<solver::GemmConfig>(model.gemms.size(),config));
      auto semantic=std::find_if(model.task_semantics.begin(),model.task_semantics.end(),[](auto const& s){return s.op.kind==analysis::OperatorKind::kMatmul;});
      auto input=solver::DeriveModelTaskInput(model,*semantic,graph,&config);
      analysis::ParamBinding p;for(auto const& n:input.cost_coordinates)p.Bind(n,0);
      double last=1e300;
      for(int stages:{2,3,4}) {
        auto traits=solver::TensorBF16Traits(32,16,k,stages);
        auto parts=cost.PriceParts(input,traits,{1},model,1,p,1);
        Require(parts.compute_ns<=last,"stages increase iteration time");last=parts.compute_ns;
        auto expected=solver::IsolatedNs(parts,cal.dram_gbps/target.res.num_sms);
        auto actual=cost.TaskInstanceNs(input,traits,{1},model,1,p,1);
        Require(std::memcmp(&expected,&actual,sizeof(double))==0,"PriceParts decomposition is not bit exact");++checks;
      }
      input.no_producer_read_bytes=input.work.read_elements.Scale(2);input.external_write_bytes=analysis::QuasiPolynomial::Constant(0);input.stream_bytes=2.*1024*1024*1024;
      auto parts=cost.PriceParts(input,solver::TensorBF16Traits(32,16,k,2),{1},model,1,p,1);
      auto traffic=solver::DeriveTaskMemoryTraffic(input,model.MetricBindings(),p,2,2);
      Require(parts.dram_bytes==traffic.global_read_bytes,"large external task lost DRAM bytes");
    }
  }
  std::cout<<"REGIME_A_PRICE bit_exact="<<checks<<" stage_monotonic=PASS external_df_2GiB=1\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
