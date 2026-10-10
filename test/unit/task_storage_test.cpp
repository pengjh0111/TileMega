// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskStorage.h>
#include <tilemega/Analysis/TaskElementRelation.h>
#include <tilemega/Analysis/CouplingDerivation.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
#include <set>

namespace tilemega::tests::task_storage_test {
namespace {
using namespace analysis;
ClosedForm C(long x){return ClosedForm::Constant(x);}
IndexResult I(char const* dim,long group=1){return IndexResult::Dim(dim,C(1),C(group));}
using Points=std::set<std::pair<std::vector<long>,std::vector<long>>>;
void Equal(CouplingRelation const& map,Points const& expected) {
  auto points=map.Points();assert(Points(points.begin(),points.end())==expected);
  auto tuple=[](std::vector<long> const& values) {
    std::string text="[";for(auto value:values)text+=(text.size()>1?",":"")+std::to_string(value);
    return text+"]";
  };
  std::string text="{ ";for(auto const& [a,b]:expected)text+=tuple(a)+" -> "+tuple(b)+"; ";
  auto oracle=CouplingRelation::FromIslText(text+" }");
  assert(Contains(map,oracle) && Contains(oracle,map));
}
SemanticGraph Graph() {
  auto rows=ClosedForm::Symbol("B")*C(15);
  SemanticOp p;p.name="projection";p.kind=OperatorKind::kMatmul;p.arithmetic="gemm";
  p.domain={{"m",rows},{"n",C(6)},{"k",C(3),C(0),IteratorType::kReduction}};
  p.result={"output",{{"row",rows},{"channel",C(6)}}};p.result_map.results={I("m"),I("n")};
  p.exact_task_access=true;p.task_space={"owners",{{"m",rows},{"n",C(6)}}};p.task_map=p.result_map;
  p.reduction={"k","add","partial","combine",true,{"m","n"}};
  TensorSpace stats{"stats",{{"row",rows},{"stat",C(2)}}};
  for(int stat=0;stat<2;++stat)
    p.additional_writes.push_back({stats,{{I("m"),IndexResult::Affine({},C(stat))}},{},
        {EffectKind::kWrite}});
  TensorSpace channels{"channels",{{"image",ClosedForm::Symbol("B")},{"channel",C(6)}}};
  p.additional_writes.push_back({channels,{{I("m",15),I("n")}},{},{EffectKind::kWrite}});
  p.tile_storage={{"stats","n",1},{"channels","m",1}};
  SemanticOp norm;norm.name="normalization";norm.arithmetic="add";
  norm.domain={{"m",rows},{"n",C(2)}};norm.result={"norm",{{"row",rows},{"channel",C(2)}}};
  norm.result_map.results={I("m"),I("n")};norm.task_map=norm.result_map;
  norm.task_space={"norm.owners",norm.result.axes};norm.task_space.axes[0].name="m";
  norm.task_space.axes[1].name="n";norm.exact_task_access=true;
  norm.operands={{p.name,stats,{{I("m"),I("n")}}, {}}};
  SemanticOp pool;pool.name="pool";pool.arithmetic="global_pool_reduce";pool.kind=OperatorKind::kReduction;
  pool.domain={{"m",ClosedForm::Symbol("B")},{"n",C(6)},
      {"r",C(15),C(0),IteratorType::kReduction}};
  pool.result={"mean",channels.axes};pool.result_map.results={I("m"),I("n")};
  pool.task_map=pool.result_map;pool.task_space={"pool.owners",pool.result.axes};
  pool.task_space.axes[0].name="m";pool.task_space.axes[1].name="n";pool.exact_task_access=true;
  pool.operands={{p.name,channels,{{I("m"),I("n")}}, {}}};
  pool.tile_storage_reads={{"channels","r","m",C(15)}};
  return {{p,norm,pool}};
}
template<class F>void Reject(F const& f) {
  bool fail=false;try{f();}catch(std::invalid_argument const&){fail=true;}assert(fail);
}
}
int TestTaskStorage(int,char**) {
  IslContext isl;auto source=Graph();auto raw=source.Serialize();
  ParamBinding known;known.Bind("B",3);unsigned cases=0;
  for(auto const& op:source.ops)assert(EncodeSemanticOp(DecodeSemanticOp(EncodeSemanticOp(op)))==EncodeSemanticOp(op));
  for(unsigned tm:{8,16,32})for(unsigned tn:{4,8})for(bool split:{false,true}) {
    Granularity g;g.Tile("projection","m",C(tm)).Tile("projection","n",C(tn));
    g.Tile("normalization","m",C(8)).Tile("normalization","n",C(2));
    g.Tile("pool","m",C(1)).Tile("pool","n",C(4));
    if(split)g.Split("projection",C(2));
    auto graph=Instantiate(source,g);assert(raw==source.Serialize());
    auto const& final=*graph.Find(split?"combine":"projection");
    assert(final.element_access->semantic.tile_storage.empty());
    if(split)assert(graph.Find("projection")->element_access->semantic.additional_writes.empty());
    auto owner=[&](unsigned row,unsigned col) {
      std::vector<long> v;if(final.IsTiled(0))v.push_back(row/tm);
      if(final.IsTiled(1))v.push_back(col/tn);return v;
    };
    for(unsigned store=0;store<3;++store) {
      auto const& write=final.element_access->semantic.additional_writes[store];Points expected;
      for(unsigned row=0;row<45;++row)for(unsigned col=0;col<6;++col)
        expected.insert({owner(row,col),store<2?std::vector<long>{long(row),long(col/tn),long(store)}:
            std::vector<long>{long(row/15),long(row/tm),long(col)}});
      Equal(ProjectTaskWrite(final.element_access->semantic,final,final.element_access->partition,
          write.tensor,write.map,{},known),expected);
    }
    auto const& norm=*graph.Find("normalization");auto const& ns=norm.element_access->semantic;
    Points reads;
    for(unsigned row=0;row<45;++row)for(unsigned stat=0;stat<2;++stat)
      for(unsigned part=0;part<(6+tn-1)/tn;++part)
        reads.insert({{long(row/8)},{long(row),long(part),long(stat)}});
    Equal(ProjectTaskRead(ns,norm,norm.element_access->partition,ns.operands[0].tensor,
        ns.operands[0].map,{},known),reads);
    auto const& pool=*graph.Find("pool");auto const& ps=pool.element_access->semantic;reads.clear();
    for(unsigned image=0;image<3;++image)for(unsigned col=0;col<6;++col)
      for(unsigned part=image*15/tm;part<((image+1)*15+tm-1)/tm;++part)
        reads.insert({{long(image),long(col/4)},{long(image),long(part),long(col)}});
    Equal(ProjectTaskRead(ps,pool,pool.element_access->partition,ps.operands[0].tensor,
        ps.operands[0].map,{},known),reads);
    auto edges=CouplingDerivation{}.Derive(graph,known);
    for(auto const& edge:edges)if(edge.dst.name=="pool" || edge.dst.name=="normalization")
      assert(edge.src.name==final.name);
    ++cases;
  }
  DramFloorOptions options;options.dram_gbps=100;options.tc_gflops=100;
  auto floor=DeriveDramFloor(source,options,known);assert(!floor.tensors.at("stats").output);
  auto invalid=source;invalid.ops[0].tile_storage[0].owner_axis="k";
  Reject([&]{Instantiate(invalid,{});});
  invalid=source;invalid.ops[0].tile_storage[0].tensor_axis=3;
  Reject([&]{DecodeSemanticOp(EncodeSemanticOp(invalid.ops[0]));});
  invalid=source;invalid.ops[2].tile_storage_reads[0].tensor="missing";
  Reject([&]{Instantiate(invalid,{});});
  invalid=source;invalid.ops[0].additional_writes[0].tensor.axes[1].extent=C(3);
  Reject([&]{Instantiate(invalid,{});});
  std::cout<<"TASK_STORAGE exact segmented writes/reads split="<<cases<<" PASS\n";return 0;
}
} // namespace tilemega::tests::task_storage_test
