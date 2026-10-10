// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/OperatorClasses.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <iostream>
#include <regex>
#include <fstream>
#include <cassert>

namespace tilemega::tests::operator_classes_test {
using namespace tilemega;
int TestDmOperatorClasses(int, char**) {
  using namespace codegen;
  analysis::IslContext isl;
  frontend::ImportedSemantics imported;
  auto& plan=imported.plan;plan.dm=true;plan.buffers.resize(7);
  frontend::PlanGemm gemm;gemm.a=0;gemm.b=1;gemm.d=2;gemm.n=64;gemm.k=27;
  plan.gemms={gemm,gemm};plan.stages.resize(2);plan.stages[1].gemm=1;
  analysis::SemanticOp op;op.name="conv0";op.kind=analysis::OperatorKind::kMatmul;
  op.result.name="output0";
  op.result.axes={{"m",analysis::ClosedForm::Constant(49)},
                  {"n",analysis::ClosedForm::Constant(64)}};
  imported.lifted.sem.ops={op,op};imported.lifted.sem.ops[1].name="conv1";
  imported.lifted.ops.resize(2);imported.lifted.ops[1].stage=1;
  auto key=[&](frontend::ModelPlan const& p) {
    return solver::GemmSemanticSignature(op,p.gemms[0],p);
  };
  auto legacy=analysis::SemanticSignature(op);
  assert(key(plan)==legacy && solver::BuildOperatorClasses(imported).size()==1);
  plan.gemms[0].access.a_row_stride=1;assert(key(plan)==legacy);
  plan.gemms[0].access.a_row_stride=128;
  assert(key(plan)!=legacy && solver::BuildOperatorClasses(imported).size()==2);
  plan.gemms[0].access={};
  plan.buffers[0].layout={DmLayout::kNHWC,4,{1,7,7,3},{1,9,9,8},
                         {648,72,8,1},1,1,1,1,DmFill::kZero};
  plan.buffers[2].layout={DmLayout::kNHWC,4,{1,7,7,64},{1,7,7,64},
                         {3136,448,64,1}};
  plan.convolutions.push_back({1,7,7,3,64,3,3,1,1,1,1,1,1,7,7,0,2});
  auto& access=plan.gemms[0].access;access.a=DmAAccess::kIm2Col;access.conv=0;
  access.rows_per_batch=49;
  auto convolution=key(plan);unsigned distinctions=0;
  auto differs=[&](auto change) {
    auto candidate=plan;change(candidate);
    assert(key(candidate)!=convolution);++distinctions;
  };
  differs([](auto& p){p.gemms[0].access.a=DmAAccess::kDense;});
  differs([](auto& p){p.gemms[0].access.a_row_offset=1;});
  differs([](auto& p){p.gemms[0].access.rows_per_batch=48;});
  differs([](auto& p){p.convolutions[0].stride_h=2;});
  differs([](auto& p){p.convolutions[0].dilation_w=2;});
  differs([](auto& p){p.convolutions[0].pad_h=0;});
  differs([](auto& p){p.convolutions[0].r=7;});
  differs([](auto& p){p.buffers[0].layout.physical[3]=4;});
  differs([](auto& p){p.buffers[0].layout.fill=DmFill::kNegativeInfinity;});
  differs([](auto& p){p.gemms[0].access.a_scale=3;});
  differs([](auto& p){p.gemms[0].access.write={DmWriteKind::kPixelShuffle,2,2,kDmNoIndex};});
  differs([](auto& p){p.gemms[0].chain.count=1;p.gemms[0].chain.operations[0].kind=DmEpilogueKind::kActivation;});
  differs([](auto& p){p.gemms[0].chain.side_count=1;p.gemms[0].chain.side[0]={DmSideOutputKind::kRowStats,3};});
  auto renamed=plan;
  for(auto& b:renamed.buffers)b.name="new-fqn";
  std::swap(renamed.buffers[0],renamed.buffers[6]);
  renamed.gemms[0].a=6;renamed.convolutions[0].input_layout=6;
  renamed.convolutions.insert(renamed.convolutions.begin(),codegen::ConvDesc{});
  renamed.gemms[0].access.conv=1;
  assert(key(renamed)==convolution);
  auto renamed_op=op;renamed_op.name="other";renamed_op.result.name="other-output";
  assert(solver::GemmSemanticSignature(renamed_op,plan.gemms[0],plan)==convolution);
  plan.gemms[1]=plan.gemms[0];
  assert(solver::BuildOperatorClasses(imported).size()==1);
  plan.gemms[1].access.conv=1;plan.convolutions.push_back(plan.convolutions[0]);
  assert(solver::BuildOperatorClasses(imported).size()==1);
  plan.convolutions[1].stride_w=2;
  assert(solver::BuildOperatorClasses(imported).size()==2);

  plan.gemms[0].access={};access.a=DmAAccess::kRowGather;
  access.b=DmBAccess::kExpertIndirect;access.rows=3;access.binding=4;
  access.expert_stride=128*128;access.binding_blocks=16;access.binding_rows=256;
  access.experts=128;access.block_rows=16;
  auto expert=key(plan);
  for(unsigned mode=0;mode<5;++mode) {
    auto candidate=plan;auto& a=candidate.gemms[0].access;
    if(mode==0)a.b=DmBAccess::kDense;
    if(mode==1)a.expert_stride*=2;
    if(mode==2)a.block_rows=32;
    if(mode==3)a.experts=16;
    if(mode==4)a.binding_blocks=17;
    assert(key(candidate)!=expert);++distinctions;
  }
  auto rejected=[&](auto change) {
    auto candidate=plan;change(candidate);bool failed=false;
    try {key(candidate);}catch(std::exception const&) {failed=true;}
    assert(failed);
  };
  rejected([](auto& p){p.gemms[0].access.conv=99;});
  rejected([](auto& p){p.gemms[0].access.rows=99;});
  rejected([](auto& p){p.gemms[0].chain.count=9;});
  rejected([](auto& p){p.dm=false;});
  std::cout<<"DM classes: legacy keys, canonical IDs, "<<distinctions
           <<" descriptor distinctions, partition and bounds PASS\n";
  return 0;
}
int TestOperatorClasses(int argc, char** argv) {
 try {
  analysis::IslContext isl;mlir::MLIRContext context;
  std::string path=std::string(TILEMEGA_SOURCE_DIR)+"/docs/experiments/E2E_GEN/raw/export_bridge.json";
  auto bridge=frontend::ReadExportBridge(path);auto plan=frontend::BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
  frontend::TorchExportImporter importer;auto imported=importer.ImportSemantics(path,plan,context);
  auto classes=solver::BuildOperatorClasses(imported);
  if(classes.size()<2)throw std::runtime_error("fixture needs distinct semantic classes");
  std::vector<solver::GemmConfig> selected(classes.size(),{32,16,16,2,1});selected.back()={32,32,32,2,1};
  auto options=solver::ClassGranularity(imported,classes,selected);
  analysis::CouplingCache cache;auto module=importer.InstantiateForGranularity(imported,context,options,&cache);
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1u,4u}});
  if(source.find("#define TILEMEGA_GEMM_VARIANT_COUNT 2\n")==std::string::npos)throw std::runtime_error("variant count mismatch");
  auto begin=source.find("constexpr GemmRuntimeDesc kRuntimeGemms0[]");
  if(begin==std::string::npos)throw std::runtime_error("missing GEMM invocation table");
  auto end=source.find("};",begin);auto table=source.substr(begin,end-begin);
  std::regex row(R"(\{(\d+)u, (\d+)u, (\d+)u, (\d+)u, (\d+)u, (\d+)u\})");
  std::map<std::tuple<int,int,int,int>,int> index;std::size_t i=0;
  for(std::sregex_iterator it(table.begin(),table.end(),row),stop;it!=stop;++it,++i) {
    auto const& g=options.gemms.at(i);auto key=std::make_tuple(g.tile_m,g.tile_n,g.tile_k,g.stages);
    auto found=index.emplace(key,index.size()).first;auto const& match=*it;
    if(std::stoi(match[1])!=found->second || std::stoi(match[2])!=g.split_k ||
       std::stoi(match[3])!=g.tile_m || std::stoi(match[4])!=g.tile_n ||
       std::stoi(match[5])!=g.tile_k || std::stoi(match[6])!=g.stages)
      throw std::runtime_error("wrong invocation variant");
  }
  if(i!=plan.gemms.size())throw std::runtime_error("incomplete invocation verification");
  if(argc>1)std::ofstream(argv[1])<<source;
  std::cout<<"VARIANTS count=2 classes="<<classes.size()<<" invocations="<<i<<" PASS\n";
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}

  return 0;
}

}  // namespace tilemega::tests::operator_classes_test
