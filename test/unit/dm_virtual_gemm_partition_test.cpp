// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DmVirtualGemmPartition.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <cassert>
#include <iostream>
#include <map>

namespace tilemega::tests::dm_virtual_gemm_partition_test {
int TestDmVirtualGemmPartition(int,char**) {
  using namespace analysis;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  unsigned cases=0;
  for(long blocks:{1,5,17})for(unsigned bm:{16,32,64,128})
      for(int tm:{16,32,64,128})for(int tn:{16,32,64,128})for(int factor:{1,2}) {
    SemanticOp op;op.name="expert";op.kind=OperatorKind::kMatmul;
    op.dtype=ScalarType::kBF16;op.arithmetic=factor==2?"swiglu_gemm":"gemm";
    op.exact_task_access=true;
    long columns=70/factor;
    op.domain={{"block",f(blocks)},{"row",f(bm)},{"column",f(columns)},
        {"k",f(48),f(0),IteratorType::kReduction}};
    auto& v=op.domain.front();v.runtime=true;v.capacity=f(blocks);
    v.binding_source="bindings";v.binding_requirement="prefix_sum";
    op.task_space={"owners",{{"virtual_owner",f(blocks)},
        {"row_owner",f(bm)},{"column_owner",f(columns)}}};
    op.task_map.results={IndexResult::Dim("block"),IndexResult::Dim("row"),IndexResult::Dim("column")};
    op.result={"partial",{{"token",f(19)},{"rank",f(3)},{"n",f(columns)}}};
    op.result_map.results={IndexResult::DataDependent("rows",{"block","row"}),
        IndexResult::DataDependent("rows",{"block","row"}),IndexResult::Dim("column")};
    codegen::DmGemmAccess access;access.b=codegen::DmBAccess::kExpertIndirect;
    access.block_rows=bm;
    solver::GemmConfig geometry{tm,tn,16,2,1};
    Granularity partition;solver::PartitionDmVirtualGemm(op,access,geometry,partition);
    auto graph=Instantiate({{op}},partition);auto const& task=graph.nodes.front();
    long rows=(bm+tm-1)/tm,width=tn/factor,cols=(columns+width-1)/width;
    assert(task.Count().Eval({},{})==blocks*rows*cols);
    assert(task.tile[0].IsLiteral(1) && task.tile[1].IsLiteral(tm) && task.tile[2].IsLiteral(width));
    auto writes=ProjectTaskRequests(op,task,task.element_access->partition,
        op.result,op.result_map,{},{});
    auto requests=writes.BoundTaskCard();
    assert(requests.SumDomain().Eval({})==blocks*bm*columns);
    solver::ModelTaskSemantics semantic;semantic.op=op;
    solver::ModelStage stage;stage.kind=solver::StageKind::kGemm;
    auto owner=solver::ProjectTaskOwnership(semantic,task,stage,128);
    assert(owner.IsSingleValued());
    assert(owner.ImageCard().Eval({})==blocks*rows*cols);
    std::map<std::vector<long>,long> numbering;
    for(auto const& [q,coordinate]:owner.Points()) {
      assert(q.size()==1 && numbering.emplace(coordinate,q[0]).second);
    }
    // Enumerated executor order is virtual id, row subtile, then N subtile.
    for(long b=0;b<blocks;++b)for(long r=0;r<rows;++r)for(long n=0;n<cols;++n) {
      ParamBinding at;
      at.Bind("virtual_owner",b).Bind("row_owner",r).Bind("column_owner",n);
      std::vector<long> coordinate;
      for(auto const& name:owner.RangeDimNames())coordinate.push_back(at.At(name));
      assert(numbering.at(coordinate)==(b*rows+r)*cols+n);
      auto count=requests.BindCoordinates(at).Eval({});
      assert(count==std::min<long>(tm,bm-r*tm)*std::min<long>(width,columns-n*width));
    }
    auto reject=[&](SemanticOp bad,codegen::DmGemmAccess description) {
      bool failed=false;try{Granularity wrong;
        solver::PartitionDmVirtualGemm(bad,description,geometry,wrong);}
      catch(std::invalid_argument const&){failed=true;}assert(failed);
    };
    auto bad=op;bad.domain[0].capacity.reset();reject(bad,access);
    bad=op;bad.domain[0].runtime=false;reject(bad,access);
    bad=op;bad.domain[0].binding_source.clear();reject(bad,access);
    bad=op;bad.task_map.results[1]=bad.task_map.results[0];reject(bad,access);
    auto wrong=access;++wrong.block_rows;reject(op,wrong);
    ++cases;
  }
  assert(cases==384);
  unsigned splits=0;
  for(int k:{17,27,65})for(int tk:{16,32})for(int requested:{1,2,3,5})for(int pair:{1,2}) {
    frontend::ModelPlan plan;plan.dm=true;plan.buffers.resize(3);plan.gemms.resize(1);plan.stages.resize(1);
    auto& gemm=plan.gemms[0];gemm.n=48*pair;gemm.k=k;
    gemm.access.b=codegen::DmBAccess::kExpertIndirect;gemm.access.block_rows=32;
    SemanticOp op;op.name="expert.split";op.kind=OperatorKind::kMatmul;op.exact_task_access=true;
    op.arithmetic=pair==2?"swiglu_gemm":"gemm";
    op.domain={{"v",f(3)},{"row",f(32)},{"column",f(48)},
        {"k",f(k),f(0),IteratorType::kReduction}};
    op.domain[0].runtime=true;op.domain[0].capacity=f(3);
    op.domain[0].binding_source="bindings";op.domain[0].binding_requirement="prefix_sum";
    op.task_space={"owners",{{"virtual_owner",f(3)},{"row_owner",f(32)},{"column_owner",f(48)}}};
    op.task_map.results={IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("column")};
    op.result={"output",{{"v",f(3)},{"row",f(32)},{"column",f(48)}}};op.result_map=op.task_map;
    op.operands={{"",{"input",{{"v",f(3)},{"row",f(32)},{"k",f(k)}}},
        {{IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("k")}}, {}}};
    op.reduction={"k","add","partial","combined",true,{"v","row","column"},unsigned(pair)};
    frontend::LiftedModel lifted;lifted.sem.ops={op};
    lifted.ops={{op.name,frontend::OpRole::kProjection,frontend::OwnershipKind::kTilePerBlock,0,0,{}}};
    auto geometry=frontend::LaunchGranularity(lifted,plan,{{16,32,tk,2,requested}});
    auto graph=Instantiate(lifted.sem,geometry);auto const& partial=graph.nodes.front();
    int tiles=(k+tk-1)/tk,chunks=std::min(requested,tiles),columns=(48+32/pair-1)/(32/pair);
    assert(graph.nodes.size()==unsigned(chunks>1?2:1));
    assert(partial.Count().Eval({},{})==3*2*columns*(chunks>1?chunks:1));
    auto const& access=*partial.element_access;auto const& read=access.semantic.operands[0];
    auto relation=ProjectTaskRead(access.semantic,partial,access.partition,read.tensor,read.map,{},{});
    std::string expected="{ ";
    for(int c=0;c<(chunks>1?chunks:1);++c) {
      if(c)expected+="; ";
      expected+="[v_owner,row_owner,column_owner"+std::string(chunks>1?",chunk":"")+
          "] -> [v,row,k] : 0<=v_owner<3 and v=v_owner and 0<=row_owner<2 and "+
          "16*row_owner<=row<16*row_owner+16 and 0<=column_owner<"+std::to_string(columns)+
          " and "+std::to_string(c*tiles/chunks*tk)+"<=k<"+
          std::to_string(std::min(k,(c+1)*tiles/chunks*tk));
      if(chunks>1)expected+=" and chunk="+std::to_string(c);
    }
    auto reference=CouplingRelation::FromIslText(expected+" }");
    assert(Contains(reference,relation) && Contains(relation,reference));
    auto work=DeriveTaskWork(access.semantic,partial,{});
    assert(work.nominal_task_reduce_extent.SumDomain().Eval({})==3*2*columns*tiles*tk);
    if(chunks>1 && pair==2) {
      auto const& combine=*graph.nodes.back().element_access;
      assert(combine.semantic.arithmetic=="swiglu_combine");
      auto combined_work=DeriveTaskWork(combine.semantic,graph.nodes.back(),{});
      assert(combined_work.write_elements.SumDomain().Eval({})==3*32*48);
    }
    ++splits;
  }
  assert(splits==48);
  std::cout<<"DM virtual GEMM partition: 384 ownership and 48 issued split/gate cases PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_virtual_gemm_partition_test
