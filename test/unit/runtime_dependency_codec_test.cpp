// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/RuntimeDependencyCodec.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/CGOps.h>
#include <tilemega/Analysis/ISLContext.h>
#include <mlir/IR/MLIRContext.h>
#include <cassert>
#include <iostream>

namespace tilemega::tests::runtime_dependency_codec_test {
int TestRuntimeDependencyCodec(int,char**) {
  using namespace tilemega;
  using R=analysis::CouplingRelation;
  analysis::IslContext isl;
  mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
  mlir::OpBuilder builder(&context);
  auto equal=[](R const& a,R const& b){return a.IsSubset(b) && b.IsSubset(a);};
  unsigned checks=0,rejected=0;
  auto reject=[&](auto const& record) {
    auto refs=isl.ReferenceCount();bool caught=false;
    try{(void)codegen::DecodeRuntimeDependency(record);}catch(std::invalid_argument const&){caught=true;}
    assert(caught && refs==isl.ReferenceCount());++rejected;
  };
  auto set=[&](mlir::DictionaryAttr attr,char const* key,mlir::Attribute value) {
    mlir::NamedAttrList result(attr);result.set(key,value);return result.getDictionary(&context);
  };
  for(unsigned consumers:{1,3,17})for(unsigned producers:{1,5,20}) {
    codegen::DependencyRecord seed{2,5,{true,2,3,-1,4}};
    auto legacy=codegen::EncodeRuntimeDependency(builder,seed);
    assert(legacy.size()==3);
    auto restored=codegen::DecodeRuntimeDependency(legacy);
    assert(restored.producer==2 && restored.consumer==5 && restored.window==seed.window);
    assert(!restored.table && !restored.counted && !restored.phase_window);++checks;
    for(bool main:{false,true})for(bool done:{false,true}) {
      seed.producer_main=main;seed.consumer_done=done;
      auto encoded=codegen::EncodeRuntimeDependency(builder,seed);
      auto endpoint=codegen::DecodeRuntimeDependency(encoded);
      assert(endpoint.producer_main==main && endpoint.consumer_done==done);
      assert(encoded.size()==3+unsigned(main)+unsigned(done));++checks;
    }
    reject(set(legacy,"producer_main",builder.getStringAttr("true")));
    reject(set(legacy,"consumer_done",builder.getI64IntegerAttr(1)));
    seed.producer_main=seed.consumer_done=false;
    seed.phase_window=analysis::WaitWindow{true,1,2,0,3};seed.phase_tiles=7;
    restored=codegen::DecodeRuntimeDependency(codegen::EncodeRuntimeDependency(builder,seed));
    assert(restored.phase_window==seed.phase_window && restored.phase_tiles==7);++checks;
    seed.phase_window.reset();seed.phase_tiles=0;
    auto relation=R::FromIslText("{ [c] -> [p] : 0<=c<"+std::to_string(consumers)+
        " and 0<=p<"+std::to_string(producers)+" and (c+2*p)%3=0 }");
    for(auto table_relation:{relation,relation.Subtract(relation)}) {
      seed.table=analysis::BuildDependencyTableLinear(table_relation,producers,consumers);
      auto encoded=codegen::EncodeRuntimeDependency(builder,seed);
      restored=codegen::DecodeRuntimeDependency(encoded);
      assert(restored.table && equal(restored.table->linear_relation,table_relation));
      assert(restored.table->stride==seed.table->stride && restored.table->intervals.size()==seed.table->intervals.size());
      auto fields=encoded.getAs<mlir::DictionaryAttr>("table");
      reject(set(encoded,"table",set(fields,"stride",builder.getI64IntegerAttr(seed.table->stride+1))));
      reject(set(encoded,"table",set(fields,"encoded_relation",dialect::CouplingMapAttr::get(&context,
          R::FromIslText("{ [c] -> [p] : c=0 and p="+std::to_string(producers)+" }")))));
      ++checks;
    }
    seed.table.reset();
    codegen::CountedWaitRecord counted;
    counted.producers=producers;counted.tensor="partial";counted.unit_axes={0,1};
    counted.contributions.binding_source="rows";
    counted.contributions.target_units=R::FromIslText("{ [target] -> [t,r] : 0<=target<"+
        std::to_string(consumers)+" and 2*target<=t<2*target+2 and 0<=r<3 }");
    counted.contributions.expected.assign(consumers,6);
    // The final target owns one logical row, independently of producer count.
    counted.contributions.target_units=counted.contributions.target_units.IntersectRange(
        "{ [t,r] : t<"+std::to_string(2*consumers-1)+" }");
    counted.contributions.expected.back()=3;
    counted.conservative_relation=R::FromIslText("{ [c] -> [p] : 0<=c<"+
        std::to_string(consumers)+" and 0<=p<"+std::to_string(producers)+" }");
    seed.counted=counted;
    auto encoded=codegen::EncodeRuntimeDependency(builder,seed);
    restored=codegen::DecodeRuntimeDependency(encoded);
    assert(restored.counted && restored.counted->producers==producers);
    auto const& actual=*restored.counted;
    assert(actual.tensor==counted.tensor && actual.unit_axes==counted.unit_axes);
    assert(actual.contributions.binding_source=="rows" && actual.contributions.expected==counted.contributions.expected);
    assert(equal(actual.contributions.target_units,counted.contributions.target_units));
    assert(equal(actual.conservative_relation,counted.conservative_relation));++checks;
    auto fields=encoded.getAs<mlir::DictionaryAttr>("counted");
    llvm::SmallVector<std::int64_t> wrong(counted.contributions.expected.begin(),counted.contributions.expected.end());++wrong.back();
    reject(set(encoded,"counted",set(fields,"expected",builder.getDenseI64ArrayAttr(wrong))));
    reject(set(encoded,"counted",set(fields,"unit_axes",builder.getDenseI64ArrayAttr({1,0}))));
    reject(set(encoded,"counted",set(fields,"target_units",dialect::CouplingMapAttr::get(&context,
        counted.contributions.target_units.Subtract(counted.contributions.target_units)))));
    reject(set(encoded,"counted",set(fields,"producers",builder.getI64IntegerAttr(producers-1))));
    reject(set(encoded,"consumer",builder.getI64IntegerAttr(2)));
  }
  assert(checks==45+4*9 && rejected==81+2*9);
  std::cout<<"Runtime dependency codec: "<<checks<<" roundtrips and "
           <<rejected<<" corruption rejections PASS\n";
  return 0;
}
} // namespace tilemega::tests::runtime_dependency_codec_test
