// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/ISLContext.h>
#include <iostream>
#include <stdexcept>
using namespace tilemega::analysis;
static void Check(bool v, char const* why) { if(!v)throw std::runtime_error(why); }
int main() {
  IslContext context;
  auto c=[](long n){return ClosedForm::Constant(n);};
  auto S=ClosedForm::Symbol("seq"),P=ClosedForm::Symbol("past"),B=ClosedForm::Symbol("batch");
  SemanticOp mm;mm.name="mm";mm.kind=OperatorKind::kMatmul;mm.dtype=ScalarType::kBF16;
  mm.domain={{"b",B},{"m",S},{"n",c(7)},{"k",c(11),c(0),IteratorType::kReduction}};
  mm.result={"out",{{"b",B},{"m",S},{"n",c(7)}}};
  mm.result_map.results={IndexResult::Dim("b"),IndexResult::Dim("m"),IndexResult::Dim("n")};
  SemanticOperand a,w;a.tensor={"input",{{"b",B},{"m",S},{"k",c(11)}}};
  a.map.results={IndexResult::Dim("b"),IndexResult::Dim("m"),IndexResult::Dim("k")};
  w.tensor={"weight",{{"k",c(11)},{"n",c(7)}}};w.map.results={IndexResult::Dim("k"),IndexResult::Dim("n")};
  mm.operands={a,w};
  auto mm2=mm;mm2.name="mm2";mm2.result.name="out2";
  DramFloorOptions opts;opts.dram_gbps=1000;opts.tc_gflops=100000;
  auto floor=DeriveDramFloor({{mm,mm2}},opts);
  for(int batch:{1,3})for(int seq:{1,4,64}) {
    ParamBinding theta;theta.Bind("batch",batch).Bind("seq",seq);
    auto v=floor.Evaluate(theta);
    Check(v.read_bytes==2*(77+batch*seq*11),"unique shared weights and input");
    Check(v.write_bytes==4*batch*seq*7,"physical outputs");
    Check(v.flops==4*batch*seq*77,"matmul work");
  }
  SemanticOp append;append.name="append";append.dtype=ScalarType::kBF16;
  append.domain={{"row",S},{"h",c(8)}};
  append.result={"cache",{{"row",S,P},{"h",c(8)}}};
  append.result_map.results={IndexResult::Dim("row"),IndexResult::Dim("h")};
  append.result_effect={EffectKind::kReadWrite,"kv","kv"};
  SemanticOperand current;current.tensor={"current",{{"row",S},{"h",c(8)}}};current.map=append.result_map;append.operands={current};
  SemanticOp use=append;use.name="use";use.domain[0].extent=S+P;use.result.name="readout";use.result.axes[0].extent=S+P;use.result.axes[0].origin=c(0);use.result_effect={};
  use.operands[0].tensor={"cache",{{"row",S+P},{"h",c(8)}}};
  floor=DeriveDramFloor({{append,use}},opts);
  for(int seq:{1,4})for(int past:{0,3,512}) {
    ParamBinding theta;theta.Bind("seq",seq).Bind("past",past);
    Check(floor.tensors.at("cache").read_bytes.Eval(theta)==16*past,"only historical KV is external read");
    Check(floor.tensors.at("cache").write_bytes.Eval(theta)==16*seq,"only current KV is state write");
  }
  auto gather=use;gather.name="gather";gather.domain[0].extent=S;
  gather.operands[0].tensor={"table",{{"row",c(100)},{"h",c(8)}}};
  gather.operands[0].map.results[0]=IndexResult::DataDependent();
  bool rejected=false;try{DeriveDramFloor({{gather}},opts);}catch(std::invalid_argument const&){rejected=true;}
  Check(rejected,"indirect reads cannot silently use a cardinality placeholder");
  opts.indirect_read_images["table"]=CouplingRelation::FromIslText("{ [] -> [row,h] : (row=3 or row=7) and 0<=h<8 }");
  floor=DeriveDramFloor({{gather}},opts);ParamBinding theta;theta.Bind("seq",4).Bind("past",0);
  Check(floor.tensors.at("table").read_bytes.Eval(theta)==32,"duplicate gather indices count once");
  std::cout<<"DRAM_FLOOR symbolic_batch_seq=PASS weight_dedup=PASS partial_KV=PASS indirect_image=PASS\n";
}
