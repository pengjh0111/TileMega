// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/SemanticCodec.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <tilemega/Solver/TaskModel.h>
#include <iostream>
#include <stdexcept>

using namespace tilemega::analysis;
void Require(bool value) { if (!value) throw std::runtime_error("element work assertion"); }

int main(int argc, char** argv) {
  IslContext context;
  auto c=[](long value){return ClosedForm::Constant(value);};
  auto S=ClosedForm::Symbol("S"),P=ClosedForm::Symbol("past");
  SemanticOp rope;
  rope.name="rope"; rope.domain={{"m",S},{"hh",c(8)}};
  rope.result={"out",{{"m",S},{"hh",c(8)}}};
  rope.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("hh")};
  SemanticOperand input;
  input.tensor={"in",{{"m",S},{"hh",c(8)}}}; input.map=rope.result_map;
  rope.operands={input};
  SetRotationElementReads(rope,{"frequency",{{"half",c(2)}}},c(4));
  int cells=0;
  for (int tile:{1,4}) {
    Granularity g; g.Tile("rope","m",c(1)).Tile("rope","hh",c(tile));
    auto graph=Instantiate(SemanticGraph{{rope}},g);
    auto work=DeriveTaskWork(rope,*graph.Find("rope"),{});
    for (int seq:{1,4,128}) for (int column=0;column<8/tile;++column) {
      ParamBinding theta; theta.Bind("S",seq).Bind("m",seq-1).Bind("hh",column);
      Require(work.read_elements.BindCoordinates(theta).Eval(theta)==(tile==1 ? 3 : 6));
      Require(work.write_elements.Eval(theta)==tile); ++cells;
    }
  }
  SemanticOp attention;
  attention.name="attention";
  attention.domain={{"s",S},{"h",c(8)},{"kv",S+P}};
  attention.result={"out",{{"s",S},{"h",c(8)}}};
  attention.result_map.results={IndexResult::Dim("s"),IndexResult::Dim("h")};
  SemanticOperand q,k,v;
  q.tensor={"q",{{"s",S},{"h",c(8)}}}; q.map=attention.result_map;
  k.tensor={"k",{{"kv",S+P},{"h",c(4)}}};
  k.map.results={IndexResult::Dim("kv"),IndexResult::Dim("h",c(1),c(2))};
  v=k; v.tensor.name="v"; attention.operands={q,k,v};
  attention.reduction.dim="kv";
  attention.reduction.splittable=true;
  attention.reduction.partial_tensor="attention.partial";
  attention.reduction.combiner="attention.combine";
  attention.reduction.reduction_operator="flash_combine";
  SetCausalAttentionReads(attention,c(4),c(2),P);
  Granularity g; g.Tile("attention","s",c(1)).Tile("attention","h",c(4));
  auto graph=Instantiate(SemanticGraph{{attention}},g);
  auto const& task=*graph.Find("attention");
  auto work=DeriveTaskWork(attention,task,{});
  std::cerr << "CAUSAL_READ_QP " << work.read_elements.ToString() << '\n';
  auto reads=ExactElementRead(attention,task,attention.element_reads[1],{});
  tilemega::solver::ModelTaskSemantics model_semantic;
  model_semantic.op=attention;
  tilemega::solver::DerivedTaskInput model_input;
  model_input.task=task;
  auto fusion_access=tilemega::solver::DeriveModelTaskAccesses(model_semantic,model_input);
  Require(fusion_access.reads.at("k").IsSubset(reads) &&
          reads.IsSubset(fusion_access.reads.at("k")));
  auto expected=CouplingRelation::FromIslText(
      "[S,past] -> { [s,h] -> [key,d] : 0<=s<S and 0<=h<2 and "
      "0<=key<S+past and key<=past+s and 0<=d<4 }");
  Require(reads.IsSubset(expected) && expected.IsSubset(reads));
  auto split_g=g;
  split_g.Split("attention",c(4));
  auto split_graph=Instantiate(SemanticGraph{{attention}},split_g);
  auto split_reads=ExactElementRead(attention,*split_graph.Find("attention"),attention.element_reads[1],{});
  auto expected_split=CouplingRelation::FromIslText(
      "[S,past] -> { [s,h,j] -> [key,d] : 0<=s<S and 0<=h<2 and "
      "0<=j<ceild(S+past,4) and 4*j<=key<4*j+4 and 0<=key<S+past and "
      "key<=past+s and 0<=d<4 }");
  Require(split_reads.IsSubset(expected_split) && expected_split.IsSubset(split_reads));
  for (int seq:{1,4,128}) for (int past:{0,3,512}) for (int token:{0,seq-1}) {
    ParamBinding theta; theta.Bind("S",seq).Bind("past",past).Bind("s",token).Bind("h",1);
    Require(work.read_elements.BindCoordinates(theta).Eval(theta)==4*(1+(past+token+1)+(past+seq)));
    Require(work.write_elements.Eval(theta)==4); ++cells;
  }
  int rejected=0;
  auto reject=[&](auto action) {
    int before=context.ReferenceCount(); bool failed=false;
    try { action(); } catch (std::exception const&) { failed=true; }
    Require(failed && context.ReferenceCount()==before); ++rejected;
  };
  auto invalid_input=model_input;
  invalid_input.task.output.name.clear();
  reject([&]{tilemega::solver::DeriveModelTaskAccesses(model_semantic,invalid_input);});
  auto invalid_semantic=model_semantic;
  invalid_semantic.op.element_reads[0].tensor.name.clear();
  reject([&]{tilemega::solver::DeriveModelTaskAccesses(invalid_semantic,model_input);});
  invalid_semantic=model_semantic;
  auto duplicate=invalid_semantic.op.element_reads.front();
  duplicate.tensor.layout_id="different";
  invalid_semantic.op.element_reads.push_back(duplicate);
  reject([&]{tilemega::solver::DeriveModelTaskAccesses(invalid_semantic,model_input);});
  model_input.scalar_access.emplace();
  model_input.scalar_access->writes=CouplingRelation::FromIslText("{ [q] -> [i] : 0<=q<4 and i=q }");
  model_input.scalar_access->reads.emplace("runtime_only",model_input.scalar_access->writes);
  auto runtime_access=tilemega::solver::DeriveModelTaskAccesses(model_semantic,model_input);
  Require(runtime_access.reads.size()==1 && runtime_access.reads.count("runtime_only") &&
          runtime_access.writes.at(task.output.name).IsSubset(model_input.scalar_access->writes));
  auto bad=attention.element_reads[1]; bad.map.results.pop_back();
  reject([&]{work.read_elements.BindCoordinates({});});
  reject([&]{ExactElementRead(attention,task,bad,{});});
  bad=attention.element_reads[1]; bad.map.results[0]=IndexResult::DataDependent();
  reject([&]{ExactElementRead(attention,task,bad,{});});
  bad=attention.element_reads[1]; bad.map.results[0].terms[0].dim="unknown";
  reject([&]{ExactElementRead(attention,task,bad,{});});
  bad=attention.element_reads[1]; bad.map.results[0].terms[0].group=c(0);
  reject([&]{ExactElementRead(attention,task,bad,{});});
  int codec_rejected=0;
  auto reject_codec=[&](std::string const& payload) {
    auto before=rejected;
    reject([&]{DecodeSemanticOp(payload);});
    Require(rejected==before+1); ++codec_rejected;
  };
  reject_codec("{}");
  auto invalid=attention;
  invalid.domain.push_back(invalid.domain.front());
  reject_codec(EncodeSemanticOp(invalid));
  invalid=attention; invalid.result_map.results.clear();
  reject_codec(EncodeSemanticOp(invalid));
  invalid=attention; invalid.operands[0].map.results[0].terms[0].dim="unknown";
  reject_codec(EncodeSemanticOp(invalid));
  invalid=attention; invalid.reduction.dim="unknown";
  reject_codec(EncodeSemanticOp(invalid));
  Require(context.ReferenceCount()==0);
  std::cout << "ELEMENT_WORK cells=" << cells << " causal_set_equivalence=PASS error_branches="
            << rejected << " codec_errors=" << codec_rejected
            << " split_set_equivalence=PASS remaining=" << context.ReferenceCount() << '\n';
  for (int input=1; input<argc; ++input) {
    using namespace tilemega::frontend;
    auto bridge=ReadExportBridge(argv[input]);
    auto plan=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
    auto lifted=LiftSemantics(plan,{});
    auto production=Instantiate(lifted.sem,LaunchGranularity(lifted,plan,{}));
    int rotations=0, attentions=0, roundtrips=0;
    for (std::size_t i=0;i<lifted.ops.size();++i) {
      auto const& op=lifted.sem.ops[i];
      Require(DecodeSemanticOp(EncodeSemanticOp(op)).Serialize()==op.Serialize());
      ++roundtrips;
      auto const& stage=plan.stages.at(lifted.ops[i].stage);
      if (op.arithmetic!="rope" && op.arithmetic!="attention") continue;
      auto const& node=*production.Find(op.name);
      tilemega::solver::ModelTaskSemantics production_semantic;
      production_semantic.op=op;
      tilemega::solver::DerivedTaskInput production_input;
      production_input.task=node;
      auto accesses=tilemega::solver::DeriveModelTaskAccesses(production_semantic,production_input);
      for (auto const& read:op.element_reads)
        Require(ExactElementRead(op,node,read,{}).IsSubset(accesses.reads.at(read.tensor.name)));
      auto derived=DeriveTaskWork(op,node,{});
      Require(!op.element_reads.empty());
      for (int seq:{1,4,128}) for (int past:{0,3,512}) for (int token:{0,seq-1}) {
        ParamBinding theta; theta.Bind("S",seq).Bind("past",past);
        if (op.arithmetic=="rope") {
          for (int column:{0,int(stage.extent*stage.width)-1}) {
            theta.Bind("m",token).Bind("hh",column);
            Require(derived.read_elements.BindCoordinates(theta).Eval(theta)==3);
            Require(derived.write_elements.BindCoordinates(theta).Eval(theta)==1);
            ++rotations;
          }
        } else {
          for (int head:{0,int(stage.extent)-1}) {
            theta.Bind("s",token).Bind("h",head);
            Require(derived.read_elements.BindCoordinates(theta).Eval(theta)==
                    long(stage.width)*(1+(past+token+1)+(past+seq)));
            auto keys=ExactElementRead(op,node,op.element_reads[1],theta);
            auto expected_keys=CouplingRelation::FromIslText(
                "{ [s,h] -> [key,d] : s="+std::to_string(token)+" and h="+
                std::to_string(head)+" and 0<=key<="+std::to_string(past+token)+
                " and "+std::to_string((head/stage.group)*stage.width)+"<=d<"+
                std::to_string((head/stage.group+1)*stage.width)+" }");
            auto points=CouplingRelation::FromIslText("{ [s,h] -> [s,h] : s="+
                std::to_string(token)+" and h="+std::to_string(head)+" }");
            keys=points.ApplyRange(keys);
            Require(keys.IsSubset(expected_keys) && expected_keys.IsSubset(keys));
            ++attentions;
          }
        }
      }
    }
    Require(rotations>0 && attentions>0 && context.ReferenceCount()==0);
    std::cout << "PRODUCTION_ELEMENT_WORK file=" << argv[input]
              << " rotation_cells=" << rotations << " attention_cells=" << attentions
              << " semantic_roundtrips=" << roundtrips
              << " exact_key_sets=PASS remaining=" << context.ReferenceCount() << '\n';
  }
}
