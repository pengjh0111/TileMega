// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Frontend/ModelPlan.h>
#include <tilemega/Codegen/tasks/TaskBase.h>
#include <tilemega/Analysis/OpArithmetic.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Codegen/TaskBodyEmitter.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace tilemega::tests::dm_descriptor_test {
int TestDmDescriptor(int, char**) {
  using namespace tilemega::codegen;
  using namespace tilemega::frontend;
  mlir::MLIRContext context; mlir::Builder builder(&context);
  auto rejects=[](auto action) {
    bool rejected=false; try {action();} catch(std::invalid_argument const&) {rejected=true;}
    assert(rejected);
  };
  ConvDesc conv{2,224,224,3,64,7,7,2,2,3,3,1,1,112,112,0,1};
  auto encoded=EncodeDm(builder,conv); auto decoded=DecodeDmConv(encoded);
  assert(encoded==EncodeDm(builder,decoded));
  assert(decoded.stride_h==2 && decoded.pad_w==3 && decoded.q==112);
  conv.p=111; rejects([&]{EncodeDm(builder,conv);}); conv.p=112;
  conv.dilation_h=0; rejects([&]{EncodeDm(builder,conv);}); conv.dilation_h=1;
  conv.r=conv.dilation_h=std::numeric_limits<std::uint32_t>::max();
  rejects([&]{EncodeDm(builder,conv);});
  rejects([&]{DecodeDmConv(builder.getDenseI64ArrayAttr({1,2,3}));});

  DmBufferLayout layout{DmLayout::kNHWC,4,{2,112,112,3},{2,118,118,8},
                        {118*118*8,118*8,8,1},3,3,3,3,DmFill::kZero};
  auto l=DecodeDmLayout(EncodeDm(builder,layout));
  assert(l.logical[3]==3 && l.physical[3]==8 && l.halo_top==3);
  assert(EncodeDm(builder,l)==EncodeDm(builder,layout));
  layout.physical[3]=4; assert(DecodeDmLayout(EncodeDm(builder,layout)).physical[3]==4);
  layout.strides[2]=4; rejects([&]{EncodeDm(builder,layout);}); layout.strides[2]=8;
  layout.halo_top=4; rejects([&]{EncodeDm(builder,layout);}); layout.halo_top=3;
  layout.strides[0]=1; rejects([&]{EncodeDm(builder,layout);});

  for(auto a:{DmAAccess::kDense,DmAAccess::kIm2Col,DmAAccess::kRowGather})
    for(auto b:{DmBAccess::kDense,DmBAccess::kExpertIndirect})
      for(auto w:{DmWriteKind::kDense,DmWriteKind::kNCHW,
                  DmWriteKind::kPixelShuffle,DmWriteKind::kRowScatter}) {
        DmGemmAccess access; access.a=a; access.b=b; access.rows_per_batch=128;
        access.a_row_stride=128; access.a_row_offset=7; access.conv=0;
        access.rows=9; access.binding=10; access.a_scale=11;
        access.expert_stride=std::uint64_t(1)<<33;
        access.write={w,w==DmWriteKind::kPixelShuffle ? 2u : 1u,1,9};
        auto value=EncodeDm(builder,access);
        assert(value==EncodeDm(builder,DecodeDmAccess(value)));
        assert(value.size()==14);
        access.binding_blocks=113;access.binding_rows=512;
        access.experts=128;access.block_rows=16;
        value=EncodeDm(builder,access);
        assert(value.size()==18 && value==EncodeDm(builder,DecodeDmAccess(value)));
        if(b==DmBAccess::kExpertIndirect) {
          auto routed=access;routed.routing_topk=8;
          value=EncodeDm(builder,routed);
          assert(value.size()==19 && value==EncodeDm(builder,DecodeDmAccess(value)));
          auto bad=routed;bad.routing_topk=129;
          rejects([&]{EncodeDm(builder,bad);});
        }
        auto invalid=access;invalid.binding_rows=0;
        rejects([&]{EncodeDm(builder,invalid);});
        invalid=access;invalid.binding_blocks=std::numeric_limits<std::uint32_t>::max();
        rejects([&]{EncodeDm(builder,invalid);});
      }
  DmGemmAccess missing; missing.a=DmAAccess::kIm2Col;
  rejects([&]{EncodeDm(builder,missing);});
  missing.a=DmAAccess::kDense; missing.b=DmBAccess::kExpertIndirect;
  rejects([&]{EncodeDm(builder,missing);});
  missing.b=DmBAccess::kDense; missing.expert_stride=std::numeric_limits<std::uint64_t>::max();
  rejects([&]{EncodeDm(builder,missing);});
  DmMoeStage moe;moe.step=DmMoeStep::kScatter;moe.experts=128;moe.top_k=8;
  moe.block_rows=32;moe.binding_capacity=384;moe.row_capacity=8192;moe.router_gemm=0;moe.grouped=true;
  assert(EncodeDm(builder,moe)==EncodeDm(builder,DecodeDmMoeStage(EncodeDm(builder,moe))));
  auto bad_moe=moe;bad_moe.top_k=129;rejects([&]{EncodeDm(builder,bad_moe);});
  bad_moe=moe;bad_moe.grouped=false;rejects([&]{EncodeDm(builder,bad_moe);});
  ModelPlan binding_plan;binding_plan.dm=true;binding_plan.forward=true;
  binding_plan.forward_token_axis=true;binding_plan.serving_seq=1;binding_plan.buffers.resize(9);
  binding_plan.gemms.resize(1);binding_plan.stages.resize(2);
  binding_plan.stages[0].kind=PlanTaskKind::kMoETopK;
  auto& source_stage=binding_plan.stages[0];source_stage.extent=8;source_stage.width=8;source_stage.group=1;
  source_stage.moe={DmMoeStep::kScatter,8,8,1,8,8,0,128,false};
  for(unsigned j=0;j<9;++j)source_stage.operands[j]=j;
  for(unsigned j:{1u,2u,4u,5u,6u,7u,8u})binding_plan.buffers[j].dtype="i32";
  binding_plan.buffers[0].dtype="f32";
  auto& binding_access=binding_plan.gemms[0].access;
  binding_access.b=DmBAccess::kExpertIndirect;binding_access.binding=4;binding_access.rows=5;
  binding_access.expert_stride=128*128;binding_access.binding_blocks=16;
  binding_access.binding_rows=256;binding_access.experts=8;binding_access.block_rows=16;
  binding_plan.stages[1].binding_producer=0;
  ValidateDmModelPlan(binding_plan);
  for(unsigned source:{1u,2u}) {
    binding_plan.stages[1].binding_producer=source;
    rejects([&]{ValidateDmModelPlan(binding_plan);});
  }
  binding_plan.stages[1].binding_producer=0;binding_access.b=DmBAccess::kDense;
  rejects([&]{ValidateDmModelPlan(binding_plan);});

  DmEpilogueChain chain; chain.count=8; chain.side_count=5;
  for(unsigned i=0;i<8;++i) {
    auto& op=chain.operations[i]; op.kind=static_cast<DmEpilogueKind>(i);
    for(unsigned p=0;p<4;++p)op.parameter[p]=p+1;
    op.activation=DmActivation::kGeluTanh; op.gate=DmGatePair::kSimpleGate;
    op.input_rounding=DmRounding::kBF16; op.output_rounding=DmRounding::kBF16;
  }
  for(unsigned i=0;i<5;++i)chain.side[i]={static_cast<DmSideOutputKind>(i),i+1,7,8};
  auto c=EncodeDm(builder,chain); assert(c==EncodeDm(builder,DecodeDmChain(c)));
  chain.count=9; rejects([&]{EncodeDm(builder,chain);}); chain.count=8;
  chain.side[1].kind=chain.side[0].kind; rejects([&]{EncodeDm(builder,chain);});
  chain.side[1].kind=DmSideOutputKind::kChannelPartialSums;
  chain.operations[4].unit=3; rejects([&]{EncodeDm(builder,chain);});
  auto words=builder.getDenseI64ArrayAttr({-1,0,0,0,0,0,0,0,0,0,0,1,0,0});
  rejects([&]{DecodeDmAccess(words);});
  words=builder.getDenseI64ArrayAttr({3,0,0,0,0,0,0,0,0,0,0,1,0,0});
  rejects([&]{DecodeDmAccess(words);});
  for(unsigned value=16;value<=25;++value)
    assert(OwnershipOf(static_cast<TaskKind>(value))==TaskOwnershipKind::kTilePerBlock);
  assert(unsigned(TaskKind::kArgmaxReduce)==15 && unsigned(TaskKind::kGemm)==0);

  ModelPlan scalar;scalar.dm=scalar.forward=true;scalar.serving_seq=128;
  scalar.dtype="bf16";scalar.buffers.resize(10);scalar.stages.resize(2);
  for(unsigned i:{0u,1u})scalar.buffers[i].dtype="i64";
  scalar.buffers[9].dtype="f32";
  for(unsigned i:{2u,3u,4u}) {
    auto& l=scalar.buffers[i].layout;l.rank=2;l.logical[1]=l.physical[1]=768;
    l.logical[0]=l.physical[0]=i==2?30522:i==3?2:512;
    l.strides[0]=768;l.strides[1]=1;
  }
  auto& embedding=scalar.stages[0];embedding.kind=PlanTaskKind::kEmbeddingSum;
  embedding.width=768;embedding.group=1;embedding.extent=30522;
  for(unsigned i=0;i<6;++i)embedding.operands[i]=i;
  embedding.operands[6]=kDmNoIndex;
  auto& norm=scalar.stages[1];norm.kind=PlanTaskKind::kLayerNorm;
  norm.width=768;norm.group=4;norm.norm_epsilon=1e-12f;
  norm.operands={5,6,7,8,9};ValidateDmModelPlan(scalar);
  norm.norm_epsilon=0;rejects([&]{ValidateDmModelPlan(scalar);});norm.norm_epsilon=1e-12f;
  norm.group=1;rejects([&]{ValidateDmModelPlan(scalar);});norm.group=4;
  scalar.buffers[0].dtype="i32";rejects([&]{ValidateDmModelPlan(scalar);});scalar.buffers[0].dtype="i64";
  scalar.buffers[4].layout.logical[0]=127;rejects([&]{ValidateDmModelPlan(scalar);});
  scalar.buffers[4].layout.logical[0]=512;
  ModelPlan encoder;encoder.dm=encoder.forward=true;encoder.serving_seq=128;
  encoder.buffers.resize(3);encoder.buffers[2].dtype="i64";
  PlanStage attention;attention.kind=PlanTaskKind::kEncoderAttention;
  attention.width=128;attention.group=64;attention.extent=12;attention.rows_per_batch=128;
  attention.operands.fill(kDmNoIndex);attention.operands[0]=0;
  attention.operands[1]=1;attention.operands[2]=2;encoder.stages={attention};
  ValidateDmModelPlan(encoder);
  for(unsigned field=0;field<5;++field) {
    auto bad=encoder;
    if(field==0)bad.stages[0].width=64;
    if(field==1)bad.stages[0].group=32;
    if(field==2)bad.stages[0].extent=0;
    if(field==3)bad.stages[0].rows_per_batch=64;
    if(field==4)bad.buffers[2].dtype="i32";
    rejects([&]{ValidateDmModelPlan(bad);});
  }
  encoder.stages[0].operands[2]=kDmNoIndex;ValidateDmModelPlan(encoder);
  auto module=mlir::ModuleOp::create(builder.getUnknownLoc());
  auto stage_attr=[&](char const* kind,unsigned width,unsigned group) {
    return builder.getDictionaryAttr({builder.getNamedAttr("kind",builder.getStringAttr(kind)),
        builder.getNamedAttr("width",builder.getI64IntegerAttr(width)),
        builder.getNamedAttr("group",builder.getI64IntegerAttr(group))});
  };
  auto attrs=builder.getArrayAttr({stage_attr("kEmbeddingSum",768,1),
      stage_attr("kLayerNorm",768,4),stage_attr("kLayerNorm",768,4),
      stage_attr("kLayoutConvert",3,128)});
  module->setAttr("tilemega.model_plan",builder.getDictionaryAttr({
      builder.getNamedAttr("dm",builder.getBoolAttr(true)),
      builder.getNamedAttr("gemms",builder.getArrayAttr({})),
      builder.getNamedAttr("stages",attrs)}));
  auto emitted=TaskBodyEmitter{}.Emit(module);
  auto call=std::string("runner.template Run<TaskKind::kLayerNorm, 768, 4>()");
  auto at=emitted.find(call);assert(at!=std::string::npos && emitted.find(call,at+1)==std::string::npos);
  assert(emitted.find("Run<TaskKind::kEmbeddingSum, 768, 1>()")!=std::string::npos);
  assert(emitted.find("Run<TaskKind::kLayoutConvert, 3, 128>()")!=std::string::npos);
  module.erase();

  tilemega::analysis::IslContext isl;
  tilemega::analysis::ArithmeticInputs inputs;
  inputs.reduction=tilemega::analysis::QuasiPolynomial::Constant(8);
  inputs.total=tilemega::analysis::QuasiPolynomial::Constant(128);
  inputs.width=64; inputs.dtype=tilemega::analysis::ScalarType::kBF16;
  for(auto const* name:{"layernorm","embedding_sum","layout_convert","pool","global_pool_reduce","depthwise_conv","encoder_attention","moe_topk","moe_combine"}) {
    auto arithmetic=tilemega::analysis::InstantiateArithmetic(name,inputs);
    assert(arithmetic.runtime_implemented);
    tilemega::analysis::RequireArithmeticImplementation(arithmetic);
  }
  for(auto const* name:{"dwpw_depthwise","moe_router"}) {
    auto arithmetic=tilemega::analysis::InstantiateArithmetic(name,inputs);
    assert(!arithmetic.runtime_implemented);
    rejects([&]{tilemega::analysis::RequireArithmeticImplementation(arithmetic);});
  }
  assert(tilemega::analysis::InstantiateArithmetic("depthwise_conv",inputs).
      flops_per_output_element.Eval({})==16);
  assert(tilemega::analysis::InstantiateArithmetic("moe_combine",inputs).
      flops_per_output_element.Eval({})==8+7+1); // products, rank sum, residual
  return 0;
}
}  // namespace tilemega::tests::dm_descriptor_test
