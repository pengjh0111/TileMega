// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/DependencyTable.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <cassert>
#include <algorithm>

namespace tilemega::tests::exact_task_metadata_test {
namespace {
using namespace analysis;
ClosedForm F(long value) { return ClosedForm::Constant(value); }
SemanticOp Identity(char const* name, char const* tensor, ClosedForm rows) {
  SemanticOp op; op.name = name; op.exact_task_access = true;
  op.domain = {{"row", rows}, {"col", F(8)}};
  op.result = {tensor, {{"row", rows}, {"col", F(8)}}};
  op.result_map.results = {IndexResult::Dim("row"), IndexResult::Dim("col")};
  op.task_space = op.result; op.task_map = op.result_map; return op;
}
void DynamicAxis() {
  auto producer = Identity("producer", "hidden", F(4));
  auto consumer = Identity("consumer", "output", F(4));
  consumer.operands.push_back({producer.name, producer.result,
      {{IndexResult::DataDependent(), IndexResult::Dim("col")}}, {}});
  Granularity g;
  g.Tile(producer.name, "row", F(2)).Tile(producer.name, "col", F(4));
  g.Tile(consumer.name, "row", F(2)).Tile(consumer.name, "col", F(4));
  auto graph = Instantiate({{producer, consumer}}, g);
  auto edges = CouplingDerivation{}.Derive(graph, {});
  assert(edges.size() == 1);
  auto const& edge = edges[0];
  assert(!edge.exact && edge.tier == Tier::kDataDependent);
  assert(edge.attributes.relation_kind == RelationKind::kDataDependent);
  assert(edge.attributes.runtime_requirement == RuntimeRequirement::kTensorValues);
  // The unknown row ranges over all rows; its affine column block remains
  // exact. Four producer tasks exist, but each consumer needs only two.
  assert(edge.metrics.wait.Eval({}) == 2);
  auto points = edge.C.Points(); assert(points.size() == 8);
  for (auto const& [to, from] : points) assert(to[1] == from[1]);
  auto table = BuildDependencyTable(edge.C, graph.nodes[0], graph.nodes[1], {});
  assert(table.stride == 2);
  consumer.operands[0].map.results[0] = IndexResult::Dim("row");
  auto exact = CouplingDerivation{}.Derive(Instantiate({{producer, consumer}}, g), {});
  assert(Contains(edge.C, exact[0].C));
  assert(!Contains(exact[0].C, edge.C));
}
void SymbolicAndOrigin() {
  auto sem = Identity("identity", "output", ClosedForm::Symbol("B") * F(3));
  sem.domain[0].origin = F(5); sem.result.axes[0].origin = F(5);
  sem.task_space = sem.result;
  auto graph = Instantiate({{sem}}, Granularity{}.Tile(sem.name, "row", F(2)).Tile(sem.name, "col", F(4)));
  auto const& task = graph.nodes[0];
  auto write = ProjectTaskElements(sem, task, task.element_access->partition,
                                   sem.result, sem.result_map, {}, {});
  for (long batch : {1, 2, 5}) {
    ParamBinding known; known.Bind("B", batch);
    auto points = write.BindParams(known).Points(); assert(points.size() == batch * 3 * 8);
    for (auto const& [owner, element] : points) {
      assert(owner[0] == (element[0] - 5) / 2 && owner[1] == element[1] / 4);
    }
    auto work = DeriveTaskWork(sem, task, known);
    assert(work.write_elements.SumDomain().Eval({}) == batch * 3 * 8);
  }
}
void AttributeProof() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<dialect::CGDialect>(); context.getOrLoadDialect<dialect::ExecDialect>();
  auto text = [](bool correct) {
    return std::string("module { ") +
      "tmcg.tile_space @a {granularity = {}, kind = #tmcg.task_kind<\"gemm\">, stage = 0 : i64, operator_name = \"a\", write_map = #tmcg.access_map<{}>} "
      "tmcg.tile_space @b {granularity = {}, kind = #tmcg.task_kind<\"gemm\">, stage = 1 : i64, operator_name = \"b\", write_map = #tmcg.access_map<{}>} "
      "tmcg.event_tensor @e : tensor<1xi32> {extent = #tmcg.metric<\"{ 1 }\">} "
      "tmcg.coupling @c from @a to @b {count = #tmcg.metric<\"{ 1 }\">, event = @e, "
      "fanout = #tmcg.metric<\"{ [0] -> 1 }\">, read_map = #tmcg.access_map<{}>, "
      "relation = #tmcg.coupling_map<\"{ [0] -> [0] }\">, sync_kind = #tmcg.sync<\"global\">, tier = #tmcg.tier<0>, "
      "shared_elements = #tmcg.coupling_map<\"{ [0,0] -> [i] : 0 <= i < 3 }\">, "
      "coupled_reads = #tmcg.coupling_map<\"{ [0] -> [i] : 0 <= i < 3 }\">, "
      "interface_elements = #tmcg.metric<\"{ 0 }\">, volume = #tmcg.metric<\"{ [0,0] -> " +
      (correct ? "3" : "4") + " }\">, wait = #tmcg.metric<\"{ [0] -> 1 }\">} }";
  };
  assert(mlir::parseSourceString<mlir::ModuleOp>(text(true), &context));
  assert(!mlir::parseSourceString<mlir::ModuleOp>(text(false), &context));
}
void SplitOrigin() {
  auto sem = Identity("split", "result", F(7));
  sem.domain[1].name = "j"; sem.result.axes[1].name = "j";
  sem.result_map.results[1] = IndexResult::Dim("j");
  sem.task_space = sem.result; sem.task_map = sem.result_map;
  sem.domain.push_back({"k", F(5), F(5), IteratorType::kReduction});
  TensorSpace input{"input", {{"row", F(7)}, {"k", F(5), F(5)}}};
  sem.operands.push_back({"", input, {{IndexResult::Dim("row"), IndexResult::Dim("k")}}, {}});
  sem.reduction = {"k", "add", "partials", "combine", true};
  Granularity g; g.Tile(sem.name, "row", F(2)).Tile(sem.name, "j", F(4)).Split(sem.name, F(2));
  auto graph = Instantiate({{sem}}, g); assert(graph.nodes.size() == 2);
  auto const& task = graph.nodes[0];
  auto const& access = *task.element_access;
  assert(task.output.axes.back().name == "j_");
  assert(DecodeSemanticOp(EncodeSemanticOp(access.semantic)).Serialize() == access.semantic.Serialize());
  auto writes = ProjectTaskElements(access.semantic, task, access.partition,
      access.semantic.result, access.semantic.result_map, {}, {});
  assert(writes.Reverse().IsSingleValued());
  for (auto const& [owner, element] : writes.Points()) assert(owner.back() == element.back());
  auto reads = ProjectTaskRead(access.semantic, task, access.partition, input, sem.operands[0].map, {});
  for (auto const& [owner, element] : reads.Points())
    assert(owner.back() == (element.back() - 5) / 2);
  auto work = DeriveTaskWork(sem, task, {});
  assert(work.write_elements.SumDomain().Eval({}) == 7 * 8 * 3);
  assert(work.read_elements.SumDomain().Eval({}) == 7 * 5 * 2);
  assert(work.reduce_extent.Eval({}) == 5 && work.parallel_extent.Eval({}) == 7 * 8);
  auto final_work = DeriveTaskWork(sem, graph.nodes[1], {});
  assert(final_work.write_elements.SumDomain().Eval({}) == 7 * 8);
  assert(final_work.read_elements.SumDomain().Eval({}) == 7 * 8 * 3);
  auto edge = CouplingDerivation{}.Derive(graph, {});
  assert(edge.size() == 1 && edge[0].metrics.wait.Eval({}) == 3);
  bool rejected = false;
  try { (void)Instantiate({{sem}}, Granularity{}.Split(sem.name, F(0))); }
  catch (std::invalid_argument const&) { rejected = true; }
  assert(rejected);
}
void FiniteFibers() {
  for (auto const* text : {"{ [m] -> [i] : 0 <= m < 7 and m <= i < m+3 }",
       "{ [m,n] -> [i] : 0 <= m < 7 and 0 <= n < 3 and 0 <= i < m+n+1 }",
       "[B] -> { [m] -> [i] : 0 <= m < B and 0 <= i <= m }",
       "{ [m] -> [i] : false }", "{ [] -> [i] : 0 <= i < 1000000 }"}) {
    auto relation = CouplingRelation::FromIslText(text);
    assert(relation.BoundTaskCard().SemanticallyEqual(relation.Card(), {}));
    assert(relation.BoundTaskCard(1).SemanticallyEqual(relation.Card(), {}));
  }
  auto exact = CouplingRelation::FromIslText("{ [t] -> [i,j] : 0 <= t < 3 and 0 <= i,j < 4 and j=i }");
  auto large=CouplingRelation::FromIslText(
      "{ [m] -> [i] : 0<=m<32768 and m%3!=0 and 0<=i<2+floor(m/16384) }");
  auto finite=large.BoundTaskCard();
  auto expected=QuasiPolynomial::FromIslText(
      "{ [m] -> (2+floor(m/16384)) : 0<=m<32768 and m%3!=0 }");
  assert(finite.SemanticallyEqual(expected,{}));
  assert(finite.SumDomain().Eval({})==54613);
  std::vector<ParamBinding> coordinates(32768);
  for(unsigned m=0;m<coordinates.size();++m)coordinates[m].Bind("m",m);
  auto values=finite.EvalPoints({},coordinates);
  for(unsigned m=0;m<values.size();++m)assert(values[m]==(m%3?2+m/16384:0));
  auto alternating=CouplingRelation::FromIslText(
      "{ [m] -> [i] : 0<=m<64 and 0<=i<2+m%2 }");
  assert(alternating.BoundTaskCard().SemanticallyEqual(alternating.Card(),{}));
  auto pixels=CouplingRelation::FromIslText(
      "{ [m] -> [p,q,c] : 0<=m<3 and 0<=p<224 and 0<=q<224 and 0<=c<512 }");
  assert(pixels.BoundTaskCard().Eval({})==224L*224*512);
  auto holes=CouplingRelation::FromIslText(
      "{ [m] -> [i] : 0<=m<3 and 0<=i<100000000 and i%2=0 }");
  assert(holes.BoundTaskCard().Eval({})==50000000);
  auto triangle=CouplingRelation::FromIslText(
      "{ [m] -> [p,q] : 0<=m<3 and 0<=p<1000 and 0<=q<=p }");
  assert(triangle.BoundTaskCard().Eval({})==500500);
  auto envelope = DescribeTaskElementBox(exact);
  assert(std::string(envelope.exactness) == "over");
  assert(Contains(envelope.relation, exact) && !Contains(exact, envelope.relation));
  auto symbolic = CouplingRelation::FromIslText("[B] -> { [t] -> [i] : 0 <= t < B and t <= i < t+3 }");
  auto rereads = symbolic.BoundTaskCard().SumDomain().Add(symbolic.Image().BoundTaskCard().Scale(-1));
  for (long batch : {1,2,8}) {
    ParamBinding known; known.Bind("B",batch);
    assert(rereads.Eval(known) == 2*batch-2);
    assert(symbolic.Image().BoundTaskCard().Eval(known) == batch+2);
  }
}
void BindingRequests() {
  for(long capacity:{1,5,17})for(long tokens:{1,2,19}) {
    SemanticOp sem;sem.name="expert";sem.exact_task_access=true;
    IterationDim v{"v",ClosedForm::Symbol("live")};v.runtime=true;v.capacity=F(capacity);
    v.binding_source="bindings";v.binding_requirement="prefix_sum";
    sem.domain={v,{"row",F(4)},{"n",F(7)},{"k",F(5),F(0),IteratorType::kReduction}};
    sem.task_space={"virtual",{{"v",v.extent,F(0),true},{"row",F(4)},{"n",F(7)}}};
    sem.task_map.results={IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("n")};
    sem.result={"partial",{{"t",F(tokens)},{"rank",F(3)},{"n",F(7)}}};
    sem.result_map.results={IndexResult::DataDependent("rows",{"v","row"}),
        IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("n")};
    TensorSpace a{"hidden",{{"t",F(tokens)},{"k",F(5)}}};
    TensorSpace b{"experts",{{"e",F(2)},{"n",F(7)},{"k",F(5)}}};
    sem.operands={{"input",a,{{IndexResult::DataDependent("rows",{"v","row"}),IndexResult::Dim("k")}},{}},
        {"",b,{{IndexResult::DataDependent("bindings",{"v"}),IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
    auto encoded=EncodeSemanticOp(sem);auto decoded=DecodeSemanticOp(encoded);
    assert(EncodeSemanticOp(decoded)==encoded && decoded.Serialize()==sem.Serialize());
    auto graph=Instantiate({{decoded}},Granularity{}.Tile("expert","v",F(1))
        .Tile("expert","row",F(3)).Tile("expert","n",F(4)));
    auto const& task=graph.nodes[0];auto const& partition=task.element_access->partition;
    auto work=DeriveTaskWork(decoded,task,{});
    assert(work.task_count.Eval({})==capacity*4);
    assert(work.parallel_extent.Eval({})==capacity*4*7 && work.reduce_extent.Eval({})==5);
    assert(work.write_elements.SumDomain().Eval({})==capacity*4*7);
    assert(work.nominal_write_elements.SumDomain().Eval({})==capacity*4*3*4);
    auto requests_a=ProjectTaskRequests(decoded,task,partition,a,decoded.operands[0].map,{},{});
    auto requests_b=ProjectTaskRequests(decoded,task,partition,b,decoded.operands[1].map,{},{});
    auto conservative=ProjectTaskRead(decoded,task,partition,a,decoded.operands[0].map,{},{});
    for(long vt=0;vt<capacity;++vt)for(long rt=0;rt<2;++rt)for(long nt=0;nt<2;++nt) {
      ParamBinding owner;owner.Bind("v",vt);owner.Bind("row",rt);owner.Bind("n",nt);
      long rows=rt==0?3:1,cols=nt==0?4:3;
      assert(requests_a.BoundTaskCard().BindCoordinates(owner).Eval({})==rows*5);
      assert(requests_b.BoundTaskCard().BindCoordinates(owner).Eval({})==cols*5);
      assert(conservative.BoundTaskCard().BindCoordinates(owner).Eval({})==tokens*5);
      assert(work.read_elements.BindCoordinates(owner).Eval({})==(rows+cols)*5);
      assert(work.frontier_read_elements.BindCoordinates(owner).Eval({})==cols*5);
      assert(work.write_elements.BindCoordinates(owner).Eval({})==rows*cols);
    }
    // Enumeration counts binding requests, even when several requests name
    // the same physical row. I2 retains the complete legal address envelope.
    auto coordinates=task.Coordinates();
    for(auto const& [owner,key]:requests_a.Points()) {
      auto coordinate=[&](char const* name) {
        auto found=std::find(coordinates.begin(),coordinates.end(),name);
        return found==coordinates.end()?0L:owner.at(found-coordinates.begin());
      };
      assert(key[0]==coordinate("v") && key[1]/3==coordinate("row") && key[2]>=0 && key[2]<5);
    }
    auto reject=[&](SemanticOp wrong) {
      bool caught=false;try{(void)DecodeSemanticOp(EncodeSemanticOp(wrong));}
      catch(std::invalid_argument const&){caught=true;}assert(caught);
    };
    auto wrong=sem;wrong.operands[0].map.results[0].request_dims={"unknown"};reject(wrong);
    wrong=sem;wrong.operands[0].map.results[0].request_dims={"v","v"};reject(wrong);
    wrong=sem;wrong.operands[0].map.results[0].binding_source.clear();reject(wrong);
    wrong=sem;wrong.operands[0].map.results[0].kind=IndexResult::Kind::kAffine;reject(wrong);
  }
}
}
int TestExactTaskMetadata(int, char**) {
  IslContext isl; DynamicAxis(); SymbolicAndOrigin(); AttributeProof(); SplitOrigin(); FiniteFibers(); BindingRequests(); return 0;
}
}
