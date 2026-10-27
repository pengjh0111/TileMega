// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90ForwardPlan.h>

#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Codegen/ServingPages.h>
#include <tilemega/Frontend/DmDescriptorCodec.h>
#include <tilemega/Frontend/DnnModelPlan.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Solver/Sm90DenseResources.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

#include <algorithm>
#include <stdexcept>

namespace tilemega::solver {
namespace {
using frontend::PlanTaskKind;

std::uint64_t Elements(frontend::PlanBuffer const& buffer,int batch) {
  return std::uint64_t(buffer.constant)+std::uint64_t(buffer.per_batch)*unsigned(batch);
}
}  // namespace

frontend::ModelPlan BuildSm90ForwardPlan(std::string const& export_path,int batch,bool fuse) {
  if(batch<1)throw std::invalid_argument("a native forward plan needs a positive batch");
  auto bridge=frontend::ReadExportBridge(export_path);
  frontend::DnnPlanOptions options;
  options.batch=unsigned(batch);
  // Every native A panel is a whole 64-channel block, including the image.
  options.nhwc_channel_block=64;
  options.global_pool_channels=128;
  options.dwpw_fuse=fuse;
  options.deferred_layernorm=true;
  auto plan=frontend::BuildDnnModelPlan(bridge.nodes,bridge.inputs,bridge.outputs,options);
  if(!plan.dm || !plan.forward)
    throw std::invalid_argument("the export is not a DNN forward model");
  return plan;
}

Sm90ForwardBody Sm90ForwardBodyOf(frontend::ModelPlan const& plan,std::size_t index) {
  auto const& stage=plan.stages.at(index);
  switch(stage.kind) {
    case PlanTaskKind::kGemm: {
      auto const& gemm=plan.gemms.at(stage.gemm);
      if(gemm.access.a==codegen::DmAAccess::kIm2Col)return Sm90ForwardBody::kConv;
      if(gemm.access.a!=codegen::DmAAccess::kDense)
        throw std::invalid_argument("native forward A access must be dense or im2col");
      // A projection whose A is a single pooled row per image is a GEMV: the
      // WGMMA bodies would run one row of a 64-row tile.
      auto rows=plan.buffers.at(gemm.a).layout.physical[0];
      return rows<16?Sm90ForwardBody::kGemv:Sm90ForwardBody::kGemm;
    }
    case PlanTaskKind::kDepthwiseConv:return Sm90ForwardBody::kDepthwise;
    case PlanTaskKind::kDwPwFused:return Sm90ForwardBody::kDwPw;
    case PlanTaskKind::kEncoderAttention:return Sm90ForwardBody::kEncoderAttention;
    case PlanTaskKind::kLayoutConvert:return Sm90ForwardBody::kLayout;
    case PlanTaskKind::kPool:return Sm90ForwardBody::kPool;
    case PlanTaskKind::kGlobalPoolReduce:return Sm90ForwardBody::kGlobalPool;
    case PlanTaskKind::kLayerNorm:return Sm90ForwardBody::kLayerNorm;
    case PlanTaskKind::kEmbeddingSum:return Sm90ForwardBody::kEmbedding;
    default:
      throw std::invalid_argument("native forward does not implement this plan stage kind");
  }
}

std::vector<Sm90ForwardShape> Sm90ForwardShapesFromPlan(frontend::ModelPlan const& plan) {
  std::vector<Sm90ForwardShape> shapes;
  for(std::size_t index=0;index<plan.stages.size();++index) {
    auto const& stage=plan.stages[index];
    Sm90ForwardShape shape;
    shape.body=Sm90ForwardBodyOf(plan,index);
    auto conv=[&]() -> codegen::ConvDesc const& {
      if(stage.conv==codegen::kDmNoIndex || stage.conv>=plan.convolutions.size())
        throw std::invalid_argument("a native forward conv stage has no descriptor");
      return plan.convolutions[stage.conv];
    };
    switch(shape.body) {
      case Sm90ForwardBody::kGemm:
      case Sm90ForwardBody::kConv:
      case Sm90ForwardBody::kGemv: {
        auto const& gemm=plan.gemms.at(stage.gemm);
        shape.n=gemm.n;
        // The native bodies contract whole 64-element k tiles: a conv's K is
        // its issued iteration count, a dense K its padded extent.
        shape.k=(std::int64_t(gemm.k)+63)/64*64;
        if(shape.body==Sm90ForwardBody::kConv) {
          auto const& c=plan.convolutions.at(gemm.access.conv);
          shape.k=std::int64_t(backend::ConvIterationGeometry::Build(c,
              plan.buffers.at(c.input_layout).layout,64).iterations)*64;
          shape.m=std::int64_t(c.n)*c.p*c.q;
          shape.pixels=shape.m;shape.channels=c.c;shape.out_channels=c.k;
          shape.kernel=int(c.r);shape.stride=int(c.stride_h);
        }else {
          shape.m=plan.buffers.at(gemm.a).layout.physical[0];
          shape.pixels=shape.m;shape.channels=gemm.k;shape.out_channels=gemm.n;
        }
        shape.weight_bytes=2*shape.n*shape.k;
        shape.activation_bytes=2*shape.m*shape.k;
        shape.output_bytes=2*shape.m*shape.n;
        shape.epilogue_ops=int(gemm.chain.count);
        for(unsigned i=0;i<gemm.chain.count;++i) {
          auto kind=gemm.chain.operations[i].kind;
          if(kind==codegen::DmEpilogueKind::kDeferredLayerNorm ||
             kind==codegen::DmEpilogueKind::kResidualLN ||
             kind==codegen::DmEpilogueKind::kDeferredRMSNorm)shape.whole_row=true;
        }
        break;
      }
      case Sm90ForwardBody::kDepthwise: {
        auto const& c=conv();
        shape.pixels=std::int64_t(c.n)*c.p*c.q;shape.channels=c.c;shape.out_channels=c.c;
        shape.kernel=int(c.r);shape.stride=int(c.stride_h);
        shape.m=shape.pixels;shape.n=c.c;shape.k=c.r*c.s;
        shape.weight_bytes=2*std::int64_t(c.r)*c.s*c.c;
        shape.activation_bytes=2*std::int64_t(c.n)*c.h*c.w*c.c;
        shape.output_bytes=2*shape.pixels*c.c;
        shape.epilogue_ops=int(stage.chain.count);
        break;
      }
      case Sm90ForwardBody::kDwPw:
      case Sm90ForwardBody::kInvertedResidual: {
        auto const& c=conv();
        auto const& gemm=plan.gemms.at(stage.gemm);
        shape.pixels=std::int64_t(c.n)*c.p*c.q;shape.channels=c.c;shape.out_channels=gemm.n;
        shape.kernel=int(c.r);shape.stride=int(c.stride_h);
        shape.m=shape.pixels;shape.n=gemm.n;shape.k=gemm.k;
        shape.fusable=true;
        shape.weight_bytes=2*(std::int64_t(gemm.n)*gemm.k+std::int64_t(c.r)*c.s*c.c);
        shape.activation_bytes=2*std::int64_t(c.n)*c.h*c.w*c.c;
        shape.output_bytes=2*shape.pixels*gemm.n;
        shape.epilogue_ops=int(gemm.chain.count);
        break;
      }
      case Sm90ForwardBody::kEncoderAttention: {
        int sequence=int(stage.width),heads=int(stage.extent),dim=int(stage.group);
        shape.pixels=std::int64_t(heads)*(sequence/16);
        shape.m=sequence;shape.n=sequence;shape.k=dim;
        shape.channels=heads*dim;shape.out_channels=heads*dim;
        // Q K^T and P V over every head.
        shape.weight_bytes=0;
        shape.activation_bytes=2*std::int64_t(sequence)*heads*dim*3;
        shape.output_bytes=2*std::int64_t(sequence)*heads*dim;
        break;
      }
      case Sm90ForwardBody::kEmbedding: {
        shape.pixels=stage.rows_per_batch;shape.channels=stage.width;
        shape.m=shape.pixels;shape.n=shape.channels;shape.k=1;
        shape.activation_bytes=2*shape.pixels*shape.channels*3;
        shape.output_bytes=2*shape.pixels*shape.channels;
        break;
      }
      default: {
        auto input=stage.operands[0],output=stage.operands[1];
        auto const& source=plan.buffers.at(input);
        shape.channels=source.layout.rank?source.layout.physical[source.layout.rank-1]:0;
        shape.pixels=source.layout.rank==4?
            std::int64_t(source.layout.logical[0])*source.layout.logical[1]*source.layout.logical[2]:
            std::int64_t(source.layout.physical[0]);
        shape.m=shape.pixels;shape.n=shape.channels;shape.k=1;
        shape.activation_bytes=2*shape.pixels*shape.channels;
        shape.output_bytes=2*std::int64_t(Elements(plan.buffers.at(output),1));
        break;
      }
    }
    if(shape.body==Sm90ForwardBody::kEncoderAttention) {
      // FLOPs of both matmuls, which the cost model reads through m, n, k.
      shape.m=std::int64_t(stage.extent)*stage.width;shape.n=stage.width;shape.k=stage.group;
    }
    shapes.push_back(shape);
  }
  return shapes;
}

mlir::OwningOpRef<mlir::ModuleOp> MaterializeSm90Forward(frontend::ModelPlan const& plan,
    Sm90ForwardCoordinates const& coordinates,int batch,TargetSpec const& target,
    mlir::MLIRContext& context,std::string const& origin) {
  if(target.arch_tag!="sm_90" || target.NvccArch()!="sm_90a" || target.res.num_sms<=0 ||
     target.res.max_dynamic_smem_per_cta<=0 || (origin!="fixed" && origin!="solver"))
    throw std::invalid_argument("native forward materialization requires a complete sm_90a target");
  if(plan.stages.size()!=coordinates.stages.size())
    throw std::invalid_argument("native forward coordinates do not cover every stage");
  for(std::size_t i=0;i<plan.stages.size();++i) {
    coordinates.stages[i].Validate();
    if(coordinates.stages[i].body!=Sm90ForwardBodyOf(plan,i))
      throw std::invalid_argument("native forward coordinates name another body for a stage");
  }
  auto budget=Sm90DenseSharedBudget(target,coordinates.residency);
  // D5: a 1 KB header and two equal, 1 KB-aligned arenas.
  int arena=(budget-1024)/2/1024*1024;
  if(arena<1024)throw std::invalid_argument("the target leaves no native forward arena");

  mlir::OpBuilder b(&context);
  auto module=mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  std::vector<std::uint8_t> written(plan.buffers.size(),1);
  (*module)->setAttr("tilemega.model_plan",frontend::EncodeModelPlan(b,plan,written));
  // The weight packer reads one record per GEMM: the native B layout is the
  // K-major tile pages of that GEMM's own coordinate.
  llvm::SmallVector<mlir::Attribute> records(plan.gemms.size());
  std::vector<bool> bound(plan.gemms.size(),false);
  for(std::size_t i=0;i<plan.stages.size();++i) {
    auto const& stage=plan.stages[i];
    if(!frontend::IsGemmStage(stage.kind))continue;
    auto const& c=coordinates.stages[i];
    int tile_n=c.body==Sm90ForwardBody::kGemv?8:c.tile_n;
    records[stage.gemm]=b.getDictionaryAttr({
        b.getNamedAttr("tile_m",b.getI64IntegerAttr(64)),
        b.getNamedAttr("tile_n",b.getI64IntegerAttr(tile_n)),
        b.getNamedAttr("tile_k",b.getI64IntegerAttr(64)),
        b.getNamedAttr("stages",b.getI64IntegerAttr(c.slots)),
        b.getNamedAttr("split_k",b.getI64IntegerAttr(c.split_k)),
        b.getNamedAttr("impl",b.getI64IntegerAttr(0))});
    bound[stage.gemm]=true;
  }
  for(std::size_t i=0;i<records.size();++i)
    if(!bound[i])throw std::invalid_argument("a native forward GEMM has no stage");
  (*module)->setAttr("tilemega.gemm_runtime",b.getArrayAttr(records));
  (*module)->setAttr("tmexec.nonpaged_weight_layout_tiled",b.getBoolAttr(true));
  codegen::ResolveServingWeightPacking(*module);
  (*module)->setAttr("tmexec.sm90_forward",b.getStringAttr(coordinates.ToJson().Dump(0)));
  (*module)->setAttr("tmexec.sm90_forward_batch",b.getI64IntegerAttr(batch));
  (*module)->setAttr("tmexec.sm90_forward_arena",b.getI64IntegerAttr(arena));
  (*module)->setAttr("tmexec.sm90_forward_origin",b.getStringAttr(origin));
  (*module)->setAttr("tmexec.sm90_forward_target",b.getStringAttr(target.ToJson()));
  (*module)->setAttr("tmexec.solved_arch",b.getStringAttr("sm_90"));
  (*module)->setAttr("tmexec.solved_residency",b.getI64IntegerAttr(coordinates.residency));
  (*module)->setAttr("tmexec.sm90_forward_grid",
      b.getI64IntegerAttr(target.res.num_sms*coordinates.residency));
  if(mlir::failed(mlir::verify(*module)))
    throw std::invalid_argument("native forward module verification failed");
  return module;
}

}  // namespace tilemega::solver
