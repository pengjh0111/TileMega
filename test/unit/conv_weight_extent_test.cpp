// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <cassert>
#include <iostream>
using namespace tilemega;
int main() {
  mlir::MLIRContext context;mlir::Builder b(&context);unsigned cases=0;
  for(unsigned c:{3u,24u,27u})for(unsigned cp:{4u,8u,24u,32u}) {
    if((c==3 && cp>8)||(c==24 && cp!=24)||(c==27 && cp!=32))continue;
    for(unsigned r:{1u,3u,7u})for(unsigned tk:{16u,32u,64u,128u}) {
      if(cp<tk && tk%cp)continue;
      auto module=mlir::ModuleOp::create(b.getUnknownLoc());
      codegen::DmBufferLayout layout;layout.kind=codegen::DmLayout::kNHWC;
      layout.rank=4;layout.logical[0]=layout.physical[0]=1;
      layout.logical[1]=layout.physical[1]=r;layout.logical[2]=layout.physical[2]=r;
      layout.logical[3]=c;layout.physical[3]=cp;layout.strides[0]=r*r*cp;
      layout.strides[1]=r*cp;layout.strides[2]=cp;layout.strides[3]=1;
      codegen::ConvDesc conv;conv.c=c;conv.k=19;conv.r=conv.s=r;conv.input_layout=0;conv.output_layout=2;
      conv.n=1;conv.h=conv.w=r;conv.p=conv.q=1;conv.stride_h=conv.stride_w=1;conv.dilation_h=conv.dilation_w=1;
      codegen::DmGemmAccess access;access.a=codegen::DmAAccess::kIm2Col;access.conv=0;
      auto integer=[&](int value){return b.getI64IntegerAttr(value);};
      auto input=b.getDictionaryAttr({b.getNamedAttr("dm_layout",frontend::EncodeDm(b,layout))});
      auto weight=b.getDictionaryAttr({b.getNamedAttr("name",b.getStringAttr("weight")),
        b.getNamedAttr("constant",integer(19*r*r*cp)),b.getNamedAttr("pack_json",b.getStringAttr("{\"kind\":\"conv_krsc\",\"source\":\"weight\",\"padded_channels\":"+std::to_string(cp)+"}"))});
      mlir::NamedAttrList gemm;
      for(auto [name,value]:std::initializer_list<std::pair<char const*,int>>{{"a",0},{"b",1},{"c",2},{"d",2},{"n",19},{"k",int(c*r*r)}})gemm.set(name,integer(value));
      gemm.set("dm_access",frontend::EncodeDm(b,access));
      module->setAttr("tilemega.model_plan",b.getDictionaryAttr({b.getNamedAttr("dm",b.getBoolAttr(true)),
        b.getNamedAttr("buffers",b.getArrayAttr({input,weight,b.getDictionaryAttr({})})),
        b.getNamedAttr("gemms",b.getArrayAttr({gemm.getDictionary(&context)})),
        b.getNamedAttr("stages",b.getArrayAttr({})),b.getNamedAttr("outputs",b.getArrayAttr({})),
        b.getNamedAttr("dm_convolutions",b.getArrayAttr({frontend::EncodeDm(b,conv)}))}));
      mlir::NamedAttrList geometry;
      for(auto [name,value]:std::initializer_list<std::pair<char const*,int>>{{"tile_m",16},{"tile_n",64},{"tile_k",int(tk)},{"stages",2},{"split_k",1}})geometry.set(name,integer(value));
      module->setAttr("tilemega.gemm_runtime",b.getArrayAttr({geometry.getDictionary(&context)}));
      module->setAttr("tmexec.nonpaged_weight_layout_tiled",b.getBoolAttr(true));
      unsigned physical_k=cp>=tk?r*r*((cp+tk-1)/tk)*tk:((r*r*cp+tk-1)/tk)*tk;
      for(unsigned repeat=0;repeat<2;++repeat) {
        codegen::ResolveServingWeightPacking(module);
        auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
        auto buffers=plan.getAs<mlir::ArrayAttr>("buffers");
        assert(mlir::cast<mlir::DictionaryAttr>(buffers[1]).getAs<mlir::IntegerAttr>("constant").getInt()==64*physical_k);
      }
      module.erase();++cases;
    }
  }
  std::cout<<"packed convolution physical K and replay: "<<cases<<" cases PASS\n";
}
