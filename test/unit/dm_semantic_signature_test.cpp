// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/DmSemanticSignature.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::dm_semantic_signature_test {
int TestDmSemanticSignature(int,char**) {
  using namespace analysis;
  IslContext isl;
  auto f=[](long n){return ClosedForm::Constant(n);};
  auto fixture=[&](std::string const& prefix) {
    SemanticOp op;op.name=prefix+".expert";op.kind=OperatorKind::kMatmul;
    op.exact_task_access=true;op.arithmetic="gemm";op.dtype=ScalarType::kBF16;
    op.domain={{"v",f(17)},{"row",f(32)},{"n",f(19)},{"k",f(23),f(0),IteratorType::kReduction}};
    auto& v=op.domain.front();v.runtime=true;v.capacity=f(17);
    v.binding_source=prefix+".bindings";v.binding_requirement="prefix_sum";
    op.result={prefix+".out",{{"v",f(17)},{"row",f(32)},{"n",f(19)}},prefix+".nhwc"};
    op.result_map.results={IndexResult::Dim("v"),IndexResult::Dim("row"),IndexResult::Dim("n")};
    op.task_space=op.result;op.task_space.name=prefix+".owners";op.task_map=op.result_map;
    op.result_effect={EffectKind::kWrite,prefix+".arena",prefix+".state"};
    TensorSpace hidden{prefix+".hidden",{{"t",f(19)},{"k",f(23)}},prefix+".nhwc"};
    TensorSpace weights{prefix+".weights",{{"e",f(128)},{"n",f(19)},{"k",f(23)}}};
    op.operands={{prefix+".producer",hidden,
        {{IndexResult::DataDependent(prefix+".rows",{"v","row"}),IndexResult::Dim("k")}},
        {EffectKind::kRead,prefix+".arena",{}}},
        {{},weights,{{IndexResult::DataDependent(prefix+".bindings",{"v"}),IndexResult::Dim("n"),IndexResult::Dim("k")}}, {}}};
    for(auto const& input:op.operands)op.element_reads.push_back({input.tensor,input.map,{}});
    auto side=op.result;side.name=prefix+".side";
    op.additional_writes={{side,op.result_map,{}, {EffectKind::kWrite,prefix+".stats",{}}}};
    op.reduction={"k","add",prefix+".partial",prefix+".combine",true};
    return op;
  };
  auto original=fixture("layer0");auto key=solver::DmSemanticSignature(original);
  for(unsigned i=1;i<48;++i)assert(solver::DmSemanticSignature(fixture("layer"+std::to_string(i)))==key);
  auto distinct=[&](SemanticOp const& op){assert(solver::DmSemanticSignature(op)!=key);};
  auto changed=original;changed.domain[0].capacity=f(18);distinct(changed);
  changed=original;changed.additional_writes[0].tensor.axes[2].extent=f(20);distinct(changed);
  changed=original;changed.additional_writes[0].effect.alias_set=changed.result_effect.alias_set;distinct(changed);
  changed=original;changed.operands[1].map.results[0].binding_source=changed.operands[0].map.results[0].binding_source;
  changed.element_reads[1].map=changed.operands[1].map;distinct(changed);
  changed=original;changed.operands[0].tensor.layout_id="different-layout";
  changed.element_reads[0].tensor.layout_id="different-layout";distinct(changed);
  changed=original;changed.operands[0].map.results[0].request_dims={"v"};
  changed.element_reads[0].map=changed.operands[0].map;distinct(changed);
  changed=original;changed.task_map.results[1]=IndexResult::Dim("row",f(1),f(2));distinct(changed);
  assert(original.Serialize()==fixture("layer0").Serialize());
  std::cout<<"DM semantic signature: 48 layer identities, seven semantic distinctions and input immutability PASS\n";
  return 0;
}
} // namespace tilemega::tests::dm_semantic_signature_test
