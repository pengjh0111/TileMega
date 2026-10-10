// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/DnnResourceProbe.h>
#include <tilemega/Codegen/tasks/TaskResources.h>
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace tilemega::frontend {
std::uint64_t DnnNonGemmSharedBytes(ModelPlan const& plan) {
  if(!plan.dm || !plan.forward || plan.forward_token_axis)
    throw std::invalid_argument("DNN resource probe requires a DNN forward plan");
  std::uint64_t result=0;
  for(auto const& stage:plan.stages) {
    if(stage.kind==PlanTaskKind::kEncoderAttention)
      result=std::max<std::uint64_t>(result,codegen::EncoderAttentionSharedBytes());
    if(stage.kind==PlanTaskKind::kEmbeddingSum)result=std::max<std::uint64_t>(result,32);
    if(stage.kind==PlanTaskKind::kDepthwiseConv) {
      auto const& conv=plan.convolutions.at(stage.conv);
      if(!stage.group || !stage.width)throw std::invalid_argument("invalid depthwise resource geometry");
      bool gated=false;
      for(unsigned i=0;i<stage.chain.count;++i)
        gated|=stage.chain.operations[i].kind==codegen::DmEpilogueKind::kGatePair;
      auto rows=std::uint64_t(stage.group-1)*conv.stride_h+
          std::uint64_t(conv.r-1)*conv.dilation_h+1;
      auto bytes=rows*plan.buffers.at(conv.input_layout).layout.physical[2]*
          stage.width*(gated?2:1)*2;
      result=std::max(result,std::max<std::uint64_t>(bytes,4096));
    }
  }
  return result;
}
std::string DnnNonGemmProbeSource(ModelPlan const& plan) {
  auto shared=DnnNonGemmSharedBytes(plan);
  std::ostringstream out;
  out<<"#include <tilemega/Codegen/tasks/DmStageTaskBody.h>\n"
      <<"using namespace tilemega::codegen;\n"
      <<"extern \"C\" __global__ __launch_bounds__(128) void probe_dm_bounds(Params const* p) { if(p->dims.batch<=0) asm volatile(\"trap;\"); }\n";
  unsigned index=0;
  for(auto const& stage:plan.stages) {
    if(IsGemmStage(stage.kind))continue;
    std::string kind;
    switch(stage.kind) {
      case PlanTaskKind::kLayerNorm:kind="kLayerNorm";break;
      case PlanTaskKind::kEmbeddingSum:kind="kEmbeddingSum";break;
      case PlanTaskKind::kLayoutConvert:kind="kLayoutConvert";break;
      case PlanTaskKind::kPool:kind="kPool";break;
      case PlanTaskKind::kGlobalPoolReduce:kind="kGlobalPoolReduce";break;
      case PlanTaskKind::kEncoderAttention:kind="kEncoderAttention";break;
      case PlanTaskKind::kDepthwiseConv:kind="kDepthwiseConv";break;
      default:throw std::invalid_argument("unsupported DNN resource probe stage");
    }
    if(stage.kind==PlanTaskKind::kDepthwiseConv) {
      out<<"using Primitive"<<index<<"=DmEpilogueProgram<";
      for(unsigned i=0;i<stage.chain.count;++i) {
        auto const& op=stage.chain.operations[i];if(i)out<<',';
        out<<"DmEpilogueStep<static_cast<DmEpilogueKind>("<<unsigned(op.kind)
            <<"),static_cast<DmActivation>("<<unsigned(op.activation)
            <<"),static_cast<DmGatePair>("<<unsigned(op.gate)<<"),"<<op.unit
            <<",static_cast<DmRounding>("<<unsigned(op.input_rounding)
            <<"),static_cast<DmRounding>("<<unsigned(op.output_rounding)
            <<"),static_cast<DmWriteKind>("<<unsigned(op.residual_map.kind)
            <<"),"<<op.residual_map.factor<<">";
      }
      out<<">;\n";
    }
    out<<"extern \"C\" __global__ __launch_bounds__(128) void probe_dm_"<<index
        <<"(Params const* p,StageDesc const* s,unsigned task) { extern __shared__ char bytes[]; "
        <<"DmStageRunner<ProbeArch> runner{*p,*s,task,bytes};runner.template ";
    if(stage.kind==PlanTaskKind::kDepthwiseConv)
      out<<"RunDepthwise<"<<stage.group<<','<<stage.width<<",Primitive"<<index<<">";
    else out<<"Run<TaskKind::"<<kind<<','<<stage.width<<','<<stage.group<<">";
    out<<"(); }\n";++index;
  }
  out<<"int main() { std::printf(\"%zu 128\\n\",std::size_t("<<shared<<")); }\n";
  return out.str();
}
}
