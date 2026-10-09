// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DmVirtualGemmPartition.h>
#include <tilemega/Solver/RuntimeProjection.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskWork.h>
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
  std::cout<<"DM virtual GEMM partition: 384 independent BM/TM, gate and task-order cases PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_virtual_gemm_partition_test
