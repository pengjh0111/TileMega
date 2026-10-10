// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnDwPwFusion.h>
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Solver/DmGemmCandidates.h>
#include <tilemega/Solver/DmOperatorClasses.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
namespace tilemega::tests::dnn_dwpw_fusion_test {
int TestDnnDwPwFusion(int argc,char** argv) {
  using namespace frontend;using namespace analysis;
  if(argc<2)throw std::invalid_argument("dw-pw fixture requires an exported bridge");
  IslContext isl;auto bridge=ReadExportBridge(argv[1]);DnnPlanOptions opts;opts.batch=2;
  auto plan=BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,opts);
  auto before=plan.stages.size();auto disabled=plan;std::set<unsigned> empty;
  assert(!ApplyDnnDwPwFusion(disabled,&empty) && disabled.stages.size()==before);
  auto retained=plan;
  auto dw=std::find_if(retained.stages.begin(),retained.stages.end(),
      [](auto const& stage){return stage.kind==PlanTaskKind::kDepthwiseConv;});
  assert(dw!=retained.stages.end());retained.outputs.push_back({dw->operands[2],""});
  assert(ApplyDnnDwPwFusion(retained) && retained.stages.size()==before);
  auto count=ApplyDnnDwPwFusion(plan);assert(count && plan.stages.size()<before);
  TargetSpec target;target.caps.cp_async=true;
  for(auto const& stage:plan.stages)if(stage.kind==PlanTaskKind::kDwPwFused) {
    auto const& gemm=plan.gemms.at(stage.gemm);
    auto required=solver::DmServingBF16SmemBytes(16,16,16,2)+2*16*stage.width;
    target.res.max_dynamic_smem_per_cta=required-1;
    assert(!solver::DmGemmCandidateRejection(gemm,plan,{16,16,16,2,2},target).empty());
    target.res.max_dynamic_smem_per_cta=required;
    assert(solver::DmGemmCandidateRejection(gemm,plan,{16,16,16,2,2},target).empty());
  }
  LiftOptions lift;lift.forward=true;lift.static_seq=plan.serving_seq;lift.batch_symbol="B";
  auto sem=LiftDnnSemantics(plan,lift);assert(sem.ops.size()==plan.stages.size());
  std::vector<GemmGranularity> shapes(plan.gemms.size(),{16,16,16,2,2});
  auto graph=Instantiate(sem.sem,LaunchGranularity(sem,plan,shapes));
  ParamBinding known;known.Bind("B",2);
  for(auto const& op:sem.sem.ops)if(plan.stages.at(sem.ops.at(&op-sem.sem.ops.data()).stage).kind==PlanTaskKind::kDwPwFused) {
    assert(op.operands.size()>=3 && op.reduction.dim=="k");
    auto modified=plan;auto& fused=modified.stages.at(sem.ops.at(&op-sem.sem.ops.data()).stage);
    auto original=solver::GemmSemanticSignature(op,plan.gemms.at(fused.gemm),plan);
    bool changed=false;
    for(unsigned i=0;i<fused.chain.count;++i)if(fused.chain.operations[i].kind==codegen::DmEpilogueKind::kActivation) {
      fused.chain.operations[i].activation=codegen::DmActivation::kTanh;changed=true;
    }
    assert(changed && original!=solver::GemmSemanticSignature(op,modified.gemms.at(fused.gemm),modified));
    auto const& task=*graph.Find(op.name);auto const& access=*task.element_access;
    auto const& read=op.operands[0];
    auto relation=ProjectTaskRead(access.semantic,task,access.partition,read.tensor,read.map,{},known);
    using Pair=std::pair<std::vector<long>,std::vector<long>>;
    std::set<Pair> expected;
    auto const& stage=plan.stages[sem.ops[&op-sem.sem.ops.data()].stage];
    auto const& conv=plan.convolutions.at(stage.conv);auto const& layout=plan.buffers[conv.input_layout].layout;
    for(unsigned mt=0;mt<(2*conv.p*conv.q+15)/16;++mt)
      for(unsigned nt=0;nt<(plan.gemms[stage.gemm].n+15)/16;++nt)
        for(unsigned chunk=0;chunk<2;++chunk)
          for(unsigned row=mt*16;row<std::min((mt+1)*16,2*conv.p*conv.q);++row)
            for(unsigned c=0;c<conv.c;++c)for(unsigned r=0;r<conv.r;++r)for(unsigned s=0;s<conv.s;++s) {
              auto pixel=row%(conv.p*conv.q);
              expected.insert({{long(mt),long(nt),long(chunk)},
                  {long(row/(conv.p*conv.q)),long(pixel/conv.q*conv.stride_h+r*conv.dilation_h+layout.halo_top)-conv.pad_h,
                   long(pixel%conv.q*conv.stride_w+s*conv.dilation_w+layout.halo_left)-conv.pad_w,long(c)}});
            }
    auto points=relation.Points();assert(std::set<Pair>(points.begin(),points.end())==expected);
    auto work=DeriveTaskWork(op,task,known);assert(work.reduce_extent.Eval(known)==plan.gemms[plan.stages[sem.ops[&op-sem.sem.ops.data()].stage].gemm].k);
  }
  mlir::MLIRContext context;ImportOptions geometry;geometry.phase_batch=2;geometry.gemms=shapes;
  auto module=TorchExportImporter{}.ImportPlan(argv[1],plan,context,nullptr,geometry);
  solver::ModelDims dims;dims.seq=dims.total=1;dims.batch=2;
  auto model=solver::ModelDescription::FromCouplingGraph(*module,dims,"dw-pw");
  std::vector<solver::GemmConfig> configs(plan.gemms.size(),{16,16,16,2,2});
  auto priced_graph=solver::InstantiateModelTasks(model,configs);
  target=TargetSpec::FromJson(std::string(TILEMEGA_SOURCE_DIR)+
      "/docs/experiments/DNN_MOE_R1/inputs/regression/llama_B1/prefill/target.json");
  solver::CostModelOptions pricing;pricing.regime_a=true;
  solver::CostModel cost(target,model.dtype,pricing);
  for(auto const& semantic:model.task_semantics)
    if(model.stages.at(semantic.stage).kind==solver::StageKind::kDwPwFused) {
      auto decoded=DecodeSemanticOp(EncodeSemanticOp(semantic.op));
      assert(decoded.compute_prologue.size()==2);
      auto input=solver::DeriveModelTaskInput(model,semantic,priced_graph,&configs[0]);
      auto names=input.task.Coordinates();assert(names.size()==3 && input.compute_prologue.size()==2);
      auto const& stage=plan.stages.at(semantic.stage);auto const& conv=plan.convolutions.at(stage.conv);
      auto rows=2*conv.p*conv.q;auto traits=solver::ModelTaskTraits(model,semantic.stage,configs[0]);
      for(unsigned mt=0;mt<(rows+15)/16;++mt)for(unsigned nt=0;nt<3;++nt)
        for(unsigned chunk=0;chunk<2;++chunk) {
          ParamBinding point;point.Bind(names[0],mt);point.Bind(names[1],nt);point.Bind(names[2],chunk);
          for(auto const& phase:input.compute_prologue)
            assert(phase.output_elements.BindCoordinates(point).Eval(model.MetricBindings())==
                long(std::min(16u,rows-mt*16)*conv.c));
          auto base=input;base.compute_prologue.clear();
          auto extra=cost.PriceParts(input,traits,{1},model,2,point,1);
          auto plain=cost.PriceParts(base,traits,{1},model,2,point,1);
          auto prologue=cost.PrivateComputeNs(input,model.MetricBindings(),point,1);
          assert(prologue>0 && std::abs(extra.compute_ns-plain.compute_ns-prologue)<1e-9);
        }
    }
  if(argc==3) {
    auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,unsigned(plan.serving_seq),unsigned(plan.serving_seq)}});
    std::ofstream stream(argv[2]);stream<<source;assert(stream);
  }
  std::cout<<"Dw-pw fusion: selected pairs, exact window reads and split-K generation PASS\n";
  return 0;
}
}
