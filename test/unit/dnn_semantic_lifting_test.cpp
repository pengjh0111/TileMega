// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Codegen/CouplingGraphToCUDA.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Solver/TaskModel.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Codegen/RuntimePlan.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>

namespace tilemega::tests::dnn_semantic_lifting_test {
namespace {
using namespace analysis;
using namespace frontend;
using Pair=std::pair<std::vector<long>,std::vector<long>>;
using Points=std::set<Pair>;
codegen::DmBufferLayout Image(unsigned batch,unsigned h,unsigned w,unsigned c,unsigned cp,unsigned pad) {
  codegen::DmBufferLayout l;l.kind=codegen::DmLayout::kNHWC;l.rank=4;
  l.logical[0]=l.physical[0]=batch;l.logical[1]=h;l.logical[2]=w;l.logical[3]=c;
  l.physical[1]=h+2*pad;l.physical[2]=w+2*pad;l.physical[3]=cp;
  l.strides[3]=1;l.strides[2]=cp;l.strides[1]=l.physical[2]*cp;l.strides[0]=l.physical[1]*l.strides[1];
  l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=pad;return l;
}
ModelPlan Fixture(unsigned batch) {
  ModelPlan p;p.dm=p.forward=true;p.dtype="bf16";p.serving_seq=1;
  p.buffers.resize(8);
  for(unsigned i=0;i<8;++i)p.buffers[i].name="buffer"+std::to_string(i);
  p.buffers[1].layout=Image(batch,7,11,3,8,1);
  p.buffers[3].layout=Image(batch,4,6,19,24,0);
  codegen::ConvDesc c;c.n=batch;c.h=7;c.w=11;c.c=3;c.k=19;c.r=c.s=3;
  c.pad_h=c.pad_w=1;c.stride_h=c.stride_w=2;c.p=4;c.q=6;c.input_layout=1;c.output_layout=3;
  p.convolutions={c};
  PlanGemm g;g.a=1;g.b=2;g.c=g.d=3;g.n=19;g.k=27;
  g.access.a=codegen::DmAAccess::kIm2Col;g.access.conv=0;
  g.access.rows_per_batch=24;g.access.write.layout=3;p.gemms={g};
  PlanStage convert;convert.kind=PlanTaskKind::kLayoutConvert;
  convert.width=3;convert.group=17;convert.rows_per_batch=77;
  convert.operands.fill(codegen::kDmNoIndex);convert.operands[0]=0;convert.operands[1]=1;
  PlanStage conv;conv.kind=PlanTaskKind::kGemm;conv.gemm=0;
  PlanStage norm;norm.kind=PlanTaskKind::kLayerNorm;norm.width=19;norm.group=4;
  norm.rows_per_batch=24;norm.norm_epsilon=1e-6;norm.operands.fill(codegen::kDmNoIndex);
  norm.operands[0]=3;norm.operands[1]=4;norm.operands[2]=5;norm.operands[3]=6;norm.operands[4]=7;
  p.stages={convert,conv,norm};return p;
}
std::vector<long> Coords(OperatorNode const& node,std::vector<long> const& full) {
  assert(node.output.axes.size()==full.size());std::vector<long> out;
  for(unsigned i=0;i<full.size();++i)if(node.IsTiled(i))out.push_back(full[i]);
  return out;
}
Points Listed(CouplingRelation const& relation) {
  auto points=relation.Points();return {points.begin(),points.end()};
}
CouplingRelation Relation(Points const& points,unsigned left,unsigned right) {
  auto tuple=[](std::vector<long> const& p) {
    std::string s="[";for(auto x:p){if(s.size()>1)s+=",";s+=std::to_string(x);}return s+"]";
  };
  std::string text="{ ";
  for(auto const& [a,b]:points)text+=tuple(a)+" -> "+tuple(b)+"; ";
  if(points.empty())text+=tuple(std::vector<long>(left))+" -> "+tuple(std::vector<long>(right))+" : false";
  return CouplingRelation::FromIslText(text+" }");
}
void Equal(CouplingRelation const& actual,Points const& points) {
  auto oracle=Relation(points,actual.DomainDimNames().size(),actual.RangeDimNames().size());
  assert(Contains(actual,oracle) && Contains(oracle,actual));assert(Listed(actual)==points);
}
void PoolAndGlobal() {
  LiftOptions options;options.forward=true;options.batch_symbol="B";options.static_seq=1;
  unsigned cases=0;
  for(unsigned batch:{1,2,3})for(unsigned stride:{1,2})for(unsigned dilation:{1,2}) {
    auto plan=Fixture(batch);plan.stages.resize(2);plan.gemms.clear();
    auto& conv=plan.convolutions.front();conv.c=conv.k=3;
    conv.pad_h=conv.pad_w=dilation;conv.dilation_h=conv.dilation_w=dilation;
    conv.stride_h=conv.stride_w=stride;conv.p=(7+stride-1)/stride;conv.q=(11+stride-1)/stride;
    plan.buffers[3].layout=Image(batch,conv.p,conv.q,3,8,1);
    auto& pool=plan.stages[1];pool.kind=PlanTaskKind::kPool;pool.width=32;pool.group=17;
    pool.extent=3;pool.conv=0;pool.rows_per_batch=conv.p*conv.q;
    pool.operands.fill(codegen::kDmNoIndex);pool.operands[0]=1;pool.operands[1]=3;
    auto model=LiftSemantics(plan,options);auto graph=Instantiate(model.sem,LaunchGranularity(model,plan,{}));
    ParamBinding known;known.Bind("B",batch);
    auto const& task=graph.nodes.back();auto const& access=*task.element_access;
    auto const& read=access.semantic.element_reads.front();
    auto reads=ProjectTaskRead(access.semantic,task,access.partition,read.tensor,read.map,read.nonnegative,known);
    Points expected;
    for(unsigned row=0;row<batch*conv.p*conv.q;++row)for(unsigned c=0;c<3;++c)
      for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s) {
        long image=row/(conv.p*conv.q),y=(row%(conv.p*conv.q))/conv.q*stride+r*dilation-long(dilation);
        long x=row%conv.q*stride+s*dilation-long(dilation);
        if(y<0 || y>=7 || x<0 || x>=11)continue;
        expected.insert({Coords(task,{long(row/17),0}),{image,y+1,x+1,long(c)}});
      }
    Equal(reads,expected);++cases;
  }
  for(unsigned batch:{1,2,5})for(unsigned area:{1,49,77})for(unsigned tile:{16,64,128})
    for(unsigned channels:{19,144}) {
      ModelPlan plan;plan.dm=plan.forward=true;plan.serving_seq=1;plan.buffers.resize(2);
      for(unsigned i=0;i<2;++i) {plan.buffers[i].name="partial"+std::to_string(i);plan.buffers[i].dtype="f32";}
      PlanStage stage;stage.kind=PlanTaskKind::kGlobalPoolReduce;
      stage.width=64;stage.group=tile;stage.extent=channels;stage.rows_per_batch=area;
      stage.operands.fill(codegen::kDmNoIndex);stage.operands[0]=0;stage.operands[1]=1;plan.stages={stage};
      auto model=LiftSemantics(plan,options);auto const& op=model.sem.ops.front();
      assert(op.dtype==ScalarType::kF32 && op.domain_nonnegative.size()==2);
      assert(EncodeSemanticOp(DecodeSemanticOp(EncodeSemanticOp(op)))==EncodeSemanticOp(op));
      auto graph=Instantiate(model.sem,LaunchGranularity(model,plan,{}));auto const& task=graph.nodes.front();
      ParamBinding known;known.Bind("B",batch);
      auto const& input=op.operands.front();auto const& partition=task.element_access->partition;
      auto reads=ProjectTaskRead(op,task,partition,input.tensor,input.map,{},known);
      auto work=DeriveTaskWork(op,task,known);Points expected;
      unsigned total_parts=0;
      for(unsigned image=0;image<batch;++image) {
        unsigned first=image*area/tile,last=((image+1)*area+tile-1)/tile;
        total_parts+=last-first;
        for(unsigned c=0;c<channels;++c)for(unsigned part=first;part<last;++part)
          expected.insert({Coords(task,{long(image),long(c/64)}),{long(image),long(part),long(c)}});
        for(unsigned c=0;c<(channels+63)/64;++c) {
          ParamBinding point;point.Bind("m",image).Bind("n",c);
          assert(work.task_reduce_extent.BindCoordinates(point).Eval(known)==last-first);
          assert(work.nominal_task_reduce_extent.BindCoordinates(point).Eval(known)==last-first);
        }
      }
      Equal(reads,expected);
      assert(work.read_elements.SumDomain().Eval(known)==total_parts*channels);
      assert(work.write_elements.SumDomain().Eval(known)==batch*channels);
      for(unsigned invalid=0;invalid<3;++invalid) {
        auto bad=op;
        if(invalid==0)bad.exact_task_access=false;
        if(invalid==1)bad.domain_nonnegative.front()=IndexResult::FullRange();
        if(invalid==2)bad.domain_nonnegative.front()=IndexResult::Dim("unknown");
        bool rejected=false;try {Instantiate(SemanticGraph{{bad}},{});}catch(std::invalid_argument const&){rejected=true;}
        assert(rejected);
        rejected=false;try {DecodeSemanticOp(EncodeSemanticOp(bad));}catch(std::invalid_argument const&){rejected=true;}
        assert(rejected);
      }
      ++cases;
    }
  std::cout<<"DNN_POOL_GLOBAL exact_read_domain_work_codec cases="<<cases<<" PASS\n";
}
void EncoderIntegration(char const* output_path) {
  ModelPlan plan;plan.dm=plan.forward=true;plan.dtype="bf16";plan.serving_seq=128;
  plan.buffers.resize(3);
  for(unsigned i=0;i<3;++i) {
    auto& buffer=plan.buffers[i];buffer.name=buffer.external_name=i==0?"qkv":i==1?"context":"mask";
    buffer.role="external";buffer.dtype=i==2?"i64":"bf16";
    buffer.per_batch=128*(i==0?576:i==1?192:1);
  }
  PlanStage stage;stage.kind=PlanTaskKind::kEncoderAttention;
  stage.width=128;stage.group=64;stage.extent=3;stage.rows_per_batch=128;
  stage.operands.fill(codegen::kDmNoIndex);stage.operands[0]=0;stage.operands[1]=1;stage.operands[2]=2;
  stage.representative="attention";stage.representative_index=2;
  plan.stages={stage};plan.outputs={{1,""}};plan.node_buffer={{"qkv",0},{"mask",2},{"attention",1}};
  LiftOptions lift;lift.forward=true;lift.static_seq=128;lift.batch_symbol="s0";
  auto lifted=LiftSemantics(plan,lift);auto graph=Instantiate(lifted.sem,LaunchGranularity(lifted,plan,{}));
  ParamBinding known;known.Bind("s0",2);
  auto const& task=graph.nodes.front();auto const& access=*task.element_access;
  assert(task.Count().Eval(known,{})==12);
  std::vector<CouplingRelation> reads;
  for(unsigned i=0;i<3;++i) {
    auto const& read=access.semantic.operands[i];
    auto relation=ProjectTaskRead(access.semantic,task,access.partition,read.tensor,read.map,{},known);
    auto only=relation.IntersectDomain("{ [m] : m=9 }");
    Points expected;
    unsigned image=1,head=1,query_begin=64;
    for(unsigned row=0;row<(i?128:64);++row)for(unsigned d=0;d<64;++d)
      expected.insert({Coords(task,{9,0}),{long(image*128+(i?row:query_begin+row)),long(head*192+i*64+d)}});
    auto oracle=CouplingRelation::FromIslText("{ [m] -> [row,col] : m=9 and "+
        std::to_string(image*128+(i?0:query_begin))+" <= row < "+
        std::to_string(image*128+(i?128:query_begin+64))+" and "+
        std::to_string(head*192+i*64)+" <= col < "+std::to_string(head*192+(i+1)*64)+" }");
    assert(Contains(only,oracle) && Contains(oracle,only));assert(Listed(only)==expected);
    reads.push_back(std::move(relation));
  }
  assert(CouplingRelation::UnionAll(reads).Card().SumDomain().Eval({})==12*(64+256)*64);
  llvm::json::Value json=llvm::json::Object{
    {"schema","tilemega.exported_program.v1"},{"guards",llvm::json::Array{}},
    {"range_constraints",llvm::json::Object{{"s0","VR[1, 64]"}}},
    {"nodes",llvm::json::Array{
      llvm::json::Object{{"index",0},{"name","qkv"},{"op","placeholder"},{"target","qkv"},
        {"inputs",llvm::json::Array{}},{"shape",llvm::json::Array{"s0","128","576"}},{"dtype","torch.bfloat16"}},
      llvm::json::Object{{"index",1},{"name","mask"},{"op","placeholder"},{"target","mask"},
        {"inputs",llvm::json::Array{}},{"shape",llvm::json::Array{"s0","128"}},{"dtype","torch.int64"}},
      llvm::json::Object{{"index",2},{"name","attention"},{"op","call_function"},{"target","aten.scaled_dot_product_attention.default"},
        {"inputs",llvm::json::Array{"qkv","mask"}},{"shape",llvm::json::Array{"s0","128","192"}},{"dtype","torch.bfloat16"}}}},
    {"signature",llvm::json::Object{{"inputs",llvm::json::Array{
      llvm::json::Object{{"name","qkv"},{"kind","USER_INPUT"},{"target",""}},
      llvm::json::Object{{"name","mask"},{"kind","USER_INPUT"},{"target",""}}}},
      {"outputs",llvm::json::Array{llvm::json::Object{{"name","attention"},{"kind","USER_OUTPUT"}}}}}}};
  int fd;llvm::SmallString<128> filename;
  assert(!llvm::sys::fs::createTemporaryFile("dm-encoder-cg","json",fd,filename));
  {llvm::raw_fd_ostream out(fd,true);out<<llvm::formatv("{0:2}",json);}
  mlir::MLIRContext context;ImportOptions options;options.phase_batch=2;
  auto module=TorchExportImporter{}.ImportPlan(filename.str().str(),plan,context,nullptr,options);
  solver::ModelDims dims{128,0,128};dims.batch=2;
  auto description=solver::ModelDescription::FromCouplingGraph(*module,dims,"encoder");
  assert(description.forward && description.task_semantics.size()==1);
  auto candidate_graph=solver::InstantiateModelTasks(description,{});
  auto input=solver::DeriveModelTaskInput(description,description.task_semantics.front(),candidate_graph,nullptr,false);
  assert(input.arithmetic.runtime_implemented && input.serving_body_kind=="encoder_attention");
  assert(input.arithmetic.flops_per_output_element.Eval(description.MetricBindings())==512);
  auto traits=solver::ModelTaskTraits(description,0,{16,16,16,3,1});
  assert(traits.threads==128 && traits.smem_bytes==codegen::EncoderAttentionSharedBytes());
  auto runtime=solver::ProjectRuntimeQueues(description,codegen::ReadRuntimePlan(*module),{1,128,1});
  assert(runtime.runtime_task_refs.Eval(description.MetricBindings())==12);
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,1}});
  assert(source.find("#define TILEMEGA_DM_BOUND_BATCH 2")!=std::string::npos);
  assert(source.find("Run<TaskKind::kEncoderAttention, 128, 64>")!=std::string::npos);
  assert(source.find("#define TILEMEGA_DM_STAGE_SHARED_BYTES 35520")!=std::string::npos);
  std::error_code error;llvm::raw_fd_ostream out(output_path,error);assert(!error);out<<source;
  assert(!llvm::sys::fs::remove(filename));
  std::cout<<"ENCODER_CG exact_noncausal_read_ownership_arithmetic_codegen PASS\n";
}
void GlobalIntegration(char const* output_path) {
  ModelPlan plan;plan.dm=plan.forward=true;plan.dtype="bf16";plan.serving_seq=1;
  plan.buffers.resize(2);
  for(unsigned i=0;i<2;++i) {
    auto& buffer=plan.buffers[i];buffer.name=buffer.external_name=i?"mean":"partials";
    buffer.role="external";buffer.dtype="f32";buffer.per_batch=i?19:7*19;
  }
  auto& l=plan.buffers[0].layout;l.rank=3;
  l.logical[0]=l.physical[0]=2;l.logical[1]=l.physical[1]=7;l.logical[2]=l.physical[2]=19;
  l.strides[2]=1;l.strides[1]=19;l.strides[0]=7*19;
  PlanStage stage;stage.kind=PlanTaskKind::kGlobalPoolReduce;
  stage.width=32;stage.group=16;stage.extent=19;stage.rows_per_batch=49;
  stage.operands.fill(codegen::kDmNoIndex);stage.operands[0]=0;stage.operands[1]=1;
  stage.representative="mean";stage.representative_index=1;
  plan.stages={stage};plan.outputs={{1,""}};plan.node_buffer={{"partials",0},{"mean",1}};
  llvm::json::Value json=llvm::json::Object{
    {"schema","tilemega.exported_program.v1"},{"guards",llvm::json::Array{}},
    {"range_constraints",llvm::json::Object{{"s0","VR[1, 64]"}}},
    {"nodes",llvm::json::Array{
      llvm::json::Object{{"index",0},{"name","partials"},{"op","placeholder"},{"target","partials"},
        {"inputs",llvm::json::Array{}},{"shape",llvm::json::Array{"s0","7","19"}},{"dtype","torch.float32"}},
      llvm::json::Object{{"index",1},{"name","mean"},{"op","call_function"},{"target","aten.mean.dim"},
        {"inputs",llvm::json::Array{"partials"}},{"shape",llvm::json::Array{"s0","19"}},{"dtype","torch.float32"}}}},
    {"signature",llvm::json::Object{{"inputs",llvm::json::Array{
      llvm::json::Object{{"name","partials"},{"kind","USER_INPUT"},{"target",""}}}},
      {"outputs",llvm::json::Array{llvm::json::Object{{"name","mean"},{"kind","USER_OUTPUT"}}}}}}};
  int fd;llvm::SmallString<128> filename;
  assert(!llvm::sys::fs::createTemporaryFile("dm-global-cg","json",fd,filename));
  {llvm::raw_fd_ostream out(fd,true);out<<llvm::formatv("{0:2}",json);}
  mlir::MLIRContext context;ImportOptions options;options.phase_batch=2;
  auto module=TorchExportImporter{}.ImportPlan(filename.str().str(),plan,context,nullptr,options);
  solver::ModelDims dims{1,0,1};dims.batch=2;
  auto description=solver::ModelDescription::FromCouplingGraph(*module,dims,"global");
  auto runtime=solver::ProjectRuntimeQueues(description,codegen::ReadRuntimePlan(*module),{1,128,1});
  assert(runtime.runtime_task_refs.Eval(description.MetricBindings())==2);
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,1}});
  assert(source.find("#define TILEMEGA_DM_BOUND_BATCH 2")!=std::string::npos);
  assert(source.find("Run<TaskKind::kGlobalPoolReduce, 32, 16>")!=std::string::npos);
  std::error_code error;llvm::raw_fd_ostream out(output_path,error);assert(!error);out<<source;
  assert(!llvm::sys::fs::remove(filename));
  std::cout<<"GLOBAL_CG segmented_partial_ownership_codegen PASS\n";
}
void Integration(int argc,char** argv) {
  bool pool=argc==3 && std::string(argv[1])=="--emit-pool";
  auto plan=Fixture(2);
  unsigned counts[]={231,936,1368,576,19,19,456,48};
  for(unsigned i=0;i<8;++i) {
    auto& buffer=plan.buffers[i];
    if(i==2 || i==4 || i==5) {
      buffer.constant=counts[i];buffer.role="external";
      buffer.external_name=buffer.name;
      buffer.pack_json="{\"kind\":\"alias\",\"source\":\""+buffer.name+"\"}";
    }else buffer.per_batch=counts[i];
  }
  plan.buffers[0].role=plan.buffers[6].role="external";
  plan.buffers[0].external_name="input";plan.buffers[6].external_name="output";
  plan.buffers[7].dtype="f32";
  if(pool) {
    plan.buffers.resize(9);plan.buffers[8].name="buffer8";
    plan.buffers[8].layout=Image(2,2,3,19,24,0);plan.buffers[8].per_batch=144;
    plan.buffers[6].per_batch=114;plan.buffers[7].per_batch=12;
    auto window=plan.convolutions.front();window.h=4;window.w=6;window.c=window.k=19;
    window.p=2;window.q=3;window.input_layout=3;window.output_layout=8;
    plan.convolutions.push_back(window);
    PlanStage stage;stage.kind=PlanTaskKind::kPool;stage.width=32;stage.group=5;
    stage.extent=19;stage.conv=1;stage.rows_per_batch=6;stage.operands.fill(codegen::kDmNoIndex);
    stage.operands[0]=3;stage.operands[1]=8;plan.stages.insert(plan.stages.begin()+2,stage);
    plan.stages.back().operands[0]=8;plan.stages.back().rows_per_batch=6;
  }
  plan.outputs.push_back({6,""});
  auto node=[](int index,char const* name,char const* op,char const* target,
      std::initializer_list<llvm::json::Value> inputs,
      std::initializer_list<llvm::json::Value> shape) {
    return llvm::json::Object{{"index",index},{"name",name},{"op",op},{"target",target},
        {"inputs",llvm::json::Array(inputs)},{"shape",llvm::json::Array(shape)},
        {"dtype","torch.bfloat16"}};
  };
  llvm::json::Array nodes{
        node(0,"input","placeholder","input",{}, {"s0","3","7","11"}),
        node(1,"weight","placeholder","weight",{}, {"19","3","3","3"}),
        node(2,"gamma","placeholder","gamma",{}, {"19"}),
        node(3,"beta","placeholder","beta",{}, {"19"}),
        node(4,"layout","call_function","aten.permute.default",{"input"},{"s0","7","11","3"}),
        node(5,"conv","call_function","aten.convolution.default",{"layout","weight"},{"s0","4","6","19"})};
  if(pool)nodes.push_back(node(6,"pool","call_function","aten.max_pool2d.default",{"conv"},{"s0","2","3","19"}));
  nodes.push_back(node(pool?7:6,"norm","call_function","aten.layer_norm.default",
      {pool?"pool":"conv","gamma","beta"},{"s0",pool?"6":"24","19"}));
  llvm::json::Value json=llvm::json::Object{
      {"schema","tilemega.exported_program.v1"},{"guards",llvm::json::Array{}},
      {"range_constraints",llvm::json::Object{{"s0","VR[1, 64]"}}},
      {"nodes",std::move(nodes)},
      {"signature",llvm::json::Object{
        {"inputs",llvm::json::Array{
          llvm::json::Object{{"name","input"},{"kind","USER_INPUT"},{"target",""}},
          llvm::json::Object{{"name","weight"},{"kind","PARAMETER"},{"target","buffer2"}},
          llvm::json::Object{{"name","gamma"},{"kind","PARAMETER"},{"target","buffer4"}},
          llvm::json::Object{{"name","beta"},{"kind","PARAMETER"},{"target","buffer5"}}}},
        {"outputs",llvm::json::Array{llvm::json::Object{{"name","norm"},{"kind","USER_OUTPUT"}}}}}}};
  for(unsigned i=0;i<plan.stages.size();++i) {
    plan.stages[i].representative_index=4+i;
    plan.stages[i].representative=i==0?"layout":i==1?"conv":pool&&i==2?"pool":"norm";
  }
  plan.node_buffer={{"input",0},{"weight",2},{"gamma",4},{"beta",5},
      {"layout",1},{"conv",3},{"norm",6}};
  if(pool)plan.node_buffer["pool"]=8;
  int fd;llvm::SmallString<128> filename;
  assert(!llvm::sys::fs::createTemporaryFile("dm-dnn-cg","json",fd,filename));
  {llvm::raw_fd_ostream out(fd,true);out<<llvm::formatv("{0:2}",json);}
  mlir::MLIRContext context;
  ImportOptions options;options.gemms={{16,16,16,3,1}};options.phase_batch=2;
  auto trace=[](char const* phase) {
    if(std::getenv("TILEMEGA_DNN_IMPORT_TRACE"))std::cerr<<"DNN_IMPORT "<<phase<<std::endl;
  };
  trace("import");
  auto module=TorchExportImporter{}.ImportPlan(filename.str().str(),plan,context,nullptr,options);
  trace("model_description");
  solver::ModelDims dims{1,0,1};dims.batch=2;
  auto description=solver::ModelDescription::FromCouplingGraph(*module,dims,"dnn-primitives");
  assert(description.dm && description.serving && description.forward && description.stages.size()==plan.stages.size());
  assert(description.task_semantics.size()==plan.stages.size() && description.batch_metric_parameter=="s0");
  auto candidate_graph=solver::InstantiateModelTasks(description,{{16,16,16,3,1}});
  auto projected=solver::ProjectRuntimeQueues(description,codegen::ReadRuntimePlan(*module),{1,128,1});
  long expected_tasks=0;
  for(auto const& task:candidate_graph.nodes)expected_tasks+=task.Count().Eval(description.MetricBindings(),{});
  assert(projected.runtime_task_refs.Eval(description.MetricBindings())==expected_tasks);
  for(auto const& semantic:description.task_semantics) {
    trace(semantic.op.name.c_str());
    auto graph=Instantiate(SemanticGraph{{semantic.op}},Granularity{});
    auto const& task=graph.nodes.front();
    ParamBinding known;known.Bind("s0",2);
    auto work=DeriveTaskWork(semantic.op,task,known);
    assert(work.task_count.SumDomain().Eval(known)>0);
    auto traits=solver::ModelTaskTraits(description,semantic.stage,{16,16,16,3,1});
    assert(traits.threads==128 && traits.shape_legal);
    if(description.stages[semantic.stage].kind!=solver::StageKind::kGemm) {
      assert(traits.smem_bytes==0);
      auto flow=codegen::ScalarTaskDataflow(static_cast<codegen::TaskKind>(description.stages[semantic.stage].kind));
      assert(flow.MemoryDepthAndBarriers(128)==std::make_pair(2,0));
      auto candidate=solver::DeriveModelTaskInput(description,semantic,candidate_graph,nullptr,false);
      assert(candidate.arithmetic.runtime_implemented);
      if(description.stages[semantic.stage].kind==solver::StageKind::kLayerNorm)
        assert(std::abs(candidate.arithmetic.flops_per_output_element.Eval(known)-(8.0+1.0/19))<1e-12);
    }
  }
  trace("codegen");
  auto source=codegen::CouplingGraphToCUDA{}.LowerVariants({{*module,1,1}});
  assert(source.find("Run<TaskKind::kLayerNorm, 19, 4>")!=std::string::npos);
  assert(source.find("Run<TaskKind::kLayoutConvert, 3, 17>")!=std::string::npos);
  if(pool)assert(source.find("Run<TaskKind::kPool, 32, 5>")!=std::string::npos);
  if(argc==3 && (std::string(argv[1])=="--emit" || pool)) {
    std::error_code error;llvm::raw_fd_ostream output(argv[2],error);assert(!error);output<<source;
  }
  assert(!llvm::sys::fs::remove(filename));
}
}
int TestDnnSemanticLifting(int argc,char** argv) {
  IslContext isl;unsigned cases=0;
  if(argc==3 && std::string(argv[1])=="--emit-global") {GlobalIntegration(argv[2]);return 0;}
  if(argc==3 && std::string(argv[1])=="--emit-encoder") {EncoderIntegration(argv[2]);return 0;}
  if(argc==2 && std::string(argv[1])=="--pool-global-only") {PoolAndGlobal();return 0;}
  if(argc==3 && std::string(argv[1])=="--emit-pool") {Integration(argc,argv);return 0;}
  if(argc==2 && std::string(argv[1])=="--integration-only") {
    Integration(argc,argv);return 0;
  }
  LiftOptions options;options.forward=true;options.batch_symbol="B";options.static_seq=1;
  for(unsigned batch:{1,2,3})for(int split:{1,5}) {
    ParamBinding known;known.Bind("B",batch);
    auto p=Fixture(batch);auto model=LiftSemantics(p,options);
    assert(model.has_plan && model.degraded.empty() && model.sem.ops.size()==3);
    auto text=model.sem.Serialize();
    for(auto const& op:model.sem.ops)assert(EncodeSemanticOp(DecodeSemanticOp(EncodeSemanticOp(op)))==EncodeSemanticOp(op));
    auto partition=LaunchGranularity(model,p,{{16,16,16,3,split}});
    auto graph=Instantiate(model.sem,partition);
    assert(model.sem.Serialize()==text);
    auto const& writer=*graph.Find("dnn.s0");auto const& reader=*graph.Find("dnn.s1");
    auto const& access=*reader.element_access;auto const& operand=access.semantic.operands.front();
    auto reads=ProjectTaskRead(access.semantic,reader,access.partition,operand.tensor,operand.map,{},known);
    Points expected;
    for(unsigned row=0;row<batch*24;++row)for(unsigned n=0;n<19;++n)
      for(unsigned r=0;r<3;++r)for(unsigned s=0;s<3;++s)for(unsigned c=0;c<3;++c) {
        std::vector<long> task{long(row/16),long(n/16)};
        if(split>1)task.push_back(((r*3+s)*8+c)/16);
        expected.insert({Coords(reader,task),{long(row/24),long((row%24)/6*2+r),long(row%6*2+s),long(c)}});
      }
    Equal(reads,expected);
    auto const& wa=*writer.element_access;
    auto writes=ProjectTaskWrite(wa.semantic,writer,wa.partition,wa.semantic.result,wa.semantic.result_map,{},known);
    Points expected_edges;
    for(auto const& [task,element]:expected) {
      auto image=element[0],y=element[1]-1,x=element[2]-1;
      if(y<0 || y>=7 || x<0 || x>=11)continue;
      expected_edges.insert({task,Coords(writer,{(image*77+y*11+x)/17,0})});
    }
    auto coupling=DeriveExactTaskCoupling(writes,reads,reader,known);
    Equal(coupling.relation,expected_edges);
    auto table=BuildDependencyTable(coupling.relation,writer,reader,known);
    assert(Contains(table.encoded_relation,table.linear_relation) && Contains(table.linear_relation,table.encoded_relation));
    auto const& normalized=model.sem.ops.back();
    assert(normalized.arithmetic=="layernorm" && normalized.additional_writes.size()==2);
    assert(model.written[1] && model.written[3] && model.written[6] && model.written[7]);
    assert(Instantiate(model.sem,LaunchGranularity(model,p,{{32,32,16,3,1}})).nodes.size()==3);
    ++cases;
  }
  ModelPlan bert;bert.dm=bert.forward=true;bert.dtype="bf16";bert.serving_seq=128;
  bert.buffers.resize(7);for(unsigned i=0;i<7;++i)bert.buffers[i].name="bert"+std::to_string(i);
  for(unsigned i=2;i<5;++i) {auto& l=bert.buffers[i].layout;l.rank=2;l.logical[0]=i==2?257:i==3?2:128;l.logical[1]=768;}
  PlanStage embed;embed.kind=PlanTaskKind::kEmbeddingSum;embed.width=768;embed.extent=257;
  for(unsigned i=0;i<7;++i)embed.operands[i]=i;
  bert.stages={embed};options.static_seq=128;
  auto embedding=LiftSemantics(bert,options);auto const& op=embedding.sem.ops.front();
  assert(op.arithmetic=="embedding_sum" && op.Dim("table")->extent.IsLiteral(3));
  for(unsigned i=0;i<2;++i) {
    auto const& index=op.operands[2+i].map.results.front();
    assert(index.kind==IndexResult::Kind::kDataDependent && index.binding_source==bert.buffers[i].name);
    assert(index.request_dims==std::vector<std::string>{"m"});
  }
  assert(op.operands[4].map.results.front().kind==IndexResult::Kind::kAffine);
  auto graph=Instantiate(embedding.sem,LaunchGranularity(embedding,bert,{}));
  ParamBinding known;known.Bind("B",2);
  assert(graph.nodes.front().Count().Eval(known,{})==256);
  auto const& access=*graph.nodes.front().element_access;
  auto const& table=access.semantic.operands[2];
  auto requests=ProjectTaskRequests(access.semantic,graph.nodes.front(),access.partition,
      table.tensor,table.map,{},known);
  assert(requests.Card().SumDomain().Eval({})==256*768);
  Integration(argc,argv);
  std::cout<<"DNN_LIFT cases="<<cases<<" exact_halo_split_ownership_binding_requests_CG_codegen PASS\n";
  return 0;
}
} // namespace tilemega::tests::dnn_semantic_lifting_test
