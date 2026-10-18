// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/executor/ServingLaunch.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <tilemega/Codegen/executor/TensorIm2col.h>
#include <tilemega/Codegen/executor/TensorMap3D.h>
#include <tilemega/Codegen/executor/SharedMemoryBudget.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>

#ifndef TILEMEGA_SERVING_BATCH_LO
#define TILEMEGA_SERVING_BATCH_LO 1
#endif
#ifndef TILEMEGA_SERVING_BATCH_HI
#define TILEMEGA_SERVING_BATCH_HI TILEMEGA_SERVING_BATCH_LO
#endif
#if defined(TILEMEGA_DM_BOUND_BATCH)
static_assert(TILEMEGA_SERVING_BATCH_LO == TILEMEGA_DM_BOUND_BATCH &&
              TILEMEGA_SERVING_BATCH_HI == TILEMEGA_DM_BOUND_BATCH,
              "exact task dependencies bind one batch per plan");
#endif
#ifndef TILEMEGA_SERVING_PAST_LO
#define TILEMEGA_SERVING_PAST_LO 0
#endif
#ifndef TILEMEGA_SERVING_PAST_HI
#define TILEMEGA_SERVING_PAST_HI TILEMEGA_SERVING_PAST_LO
#endif
#if defined(TILEMEGA_SERVING_PHASE) && TILEMEGA_SERVING_PHASE == 2
static_assert(TILEMEGA_SERVING_PAST_LO == 0 && TILEMEGA_SERVING_PAST_HI == 0,
              "forward plans have no past state");
static_assert(kModel.dims.capacity == 0, "forward plans have no KV capacity");
#if defined(TILEMEGA_FORWARD_TOKEN_AXIS) && TILEMEGA_FORWARD_TOKEN_AXIS
static_assert(TILEMEGA_SERVING_BATCH_LO == 1 && TILEMEGA_SERVING_BATCH_HI == 1,
              "token-axis forward plans bind batch to one");
#endif
#endif

namespace tilemega::codegen::serving {

struct Plan {
  harness::DeviceModel model;
  Params* ring = nullptr;
  std::uint64_t* step_ns = nullptr;
  std::uint32_t steps = 0;
  int grid = 0;
  bool pdl = false;
  unsigned loop_modes = 0;
  executor::TensorMap* tensor_maps = nullptr;
#if TILEMEGA_PREFILL_ATTENTION_TMA
  std::uint32_t* prefill_map_offsets=nullptr;
#endif
  LagDependency* lag_dependencies = nullptr;
  PageTraceRecord* page_trace = nullptr;
#if TILEMEGA_TRACE_TASK
  ServingTaskProfile* task_profile=nullptr;
  unsigned* task_offsets=nullptr;
  unsigned long long* task_epoch=nullptr;
  unsigned task_stride=0;
#endif
#if TILEMEGA_TRACE_STAGE
  StageTraceRecord* stage_trace=nullptr;
#endif
#if TILEMEGA_TRACE_STEP
  StepTraceRecord* step_trace=nullptr;
#endif
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_TASK
  unsigned trace_launches=0;
#endif
  WatchdogRecord* watchdog = nullptr;
  // L1 grid-barrier rows and L2 task-event rows are disjoint. Ticket equality
  // requires a gap-free sequence for each mode, even when launches alternate.
  std::uint64_t next_iteration[2] = {0, 0};
};

inline int Count(ModelSpec const& spec, RuntimeVariantDesc const& variant,
                 StageDesc const& stage, ModelDims dims) {
  switch (stage.kind) {
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    case TaskKind::kLayerNorm:
    case TaskKind::kEmbeddingSum:
    case TaskKind::kPool:
    case TaskKind::kGlobalPoolReduce:
    case TaskKind::kDepthwiseConv:
    case TaskKind::kEncoderAttention:
    case TaskKind::kMoETopK:
    case TaskKind::kMoECombine:
    case TaskKind::kLayoutConvert: return DmStageTaskCount(stage,dims);
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    case TaskKind::kDwPwFused:
#endif
    case TaskKind::kGemm:
    case TaskKind::kGemmCombine: {
      auto const& gemm = spec.gemms[stage.gemm];
      auto const& geometry = variant.gemms[stage.gemm];
      int rows = stage.batch_rows ? dims.batch : dims.tokens();
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
      if (gemm.access.rows_per_batch)
        rows = dims.batch * int(gemm.access.rows_per_batch);
      if (gemm.access.b==DmBAccess::kExpertIndirect) {
        auto const& access=gemm.access;
        if(!geometry.tile_m || !access.block_rows || !access.binding_blocks)return -1;
        auto virtual_rows=std::uint64_t(access.binding_blocks)*
            CeilDiv(access.block_rows,geometry.tile_m)*geometry.tile_m;
        if(virtual_rows>std::uint64_t(std::numeric_limits<int>::max()))return -1;
        rows=int(virtual_rows);
      }
#endif
      int tiles = CeilDiv(rows, geometry.tile_m) *
                  CeilDiv(gemm.n, geometry.tile_n);
      return IsGemmStage(stage.kind) ? tiles * geometry.split_k : tiles;
    }
    case TaskKind::kEmbedding:
      return dims.tokens();
    case TaskKind::kRMSNorm:
      return stage.batch_rows ? dims.batch : dims.tokens();
    case TaskKind::kFusedAttention:
      return dims.batch * int(stage.extent) *
          CeilDiv(int(stage.group) * dims.seq,
                           stage.attention_query_rows) *
          CeilDiv(dims.capacity, stage.attention_kv_block);
    case TaskKind::kAttentionMerge:
      return dims.batch * int(stage.extent);
    case TaskKind::kArgmaxReduce:
      return dims.batch;
    default:
      return -1;
  }
}

inline bool StructureInvariant(ModelSpec const& spec,
                               RuntimeVariantDesc const& variant,
                               ModelDims lo, ModelDims hi) {
  std::vector<int> lower_counts(spec.stage_count);
  std::vector<int> upper_counts(spec.stage_count);
  for (std::uint32_t stage = 0; stage < spec.stage_count; ++stage) {
    int lower = Count(spec, variant, spec.stages[stage], lo);
    int upper = Count(spec, variant, spec.stages[stage], hi);
    if (lower < 0 || lower != upper) return false;
    lower_counts[stage] = lower;
    upper_counts[stage] = upper;
  }
  // A StageDependency is a literal window in the generated table. If both
  // endpoint task cardinalities agree, its window is identical at every past.
  for (std::uint32_t edge = 0; edge < variant.dependency_count; ++edge) {
    auto const& dep = variant.dependencies[edge];
    if (dep.producer >= lower_counts.size() ||
        dep.consumer >= lower_counts.size())
      return false;
    for (int task : {0, lower_counts[dep.consumer] - 1}) {
      if (task < 0) continue;
      auto lower = RuntimeDependencyBounds(
          task, lower_counts[dep.producer],
          dep.map == StageDependency::Map::kAll ||
              dep.map == StageDependency::Map::kPhase,
          dep.div, dep.scale, dep.offset, dep.count);
      auto upper = RuntimeDependencyBounds(
          task, upper_counts[dep.producer],
          dep.map == StageDependency::Map::kAll ||
              dep.map == StageDependency::Map::kPhase,
          dep.div, dep.scale, dep.offset, dep.count);
      if (lower.first != upper.first || lower.past != upper.past)
        return false;
    }
  }
  return true;
}

#if TILEMEGA_FORWARD_GEMM_WGMMA
template<int Variant=0>
inline bool ForwardWgmmaVariant(int index) {
  if(index==Variant)return GemmVariant<Variant>::kWgmma &&
      (GemmVariant<Variant>::kTileM>=64 ||
       (TILEMEGA_ACTIVATION_TMA && (GemmVariant<Variant>::kTileM==8 || GemmVariant<Variant>::kTileM==16)));
  if constexpr(Variant+1<TILEMEGA_GEMM_VARIANT_COUNT)return ForwardWgmmaVariant<Variant+1>(index);
  return false;
}
#endif
#if TILEMEGA_DWPW_WGMMA
template<int Variant=0>
inline bool DwPwWgmmaVariant(int index) {
  if(index==Variant)return GemmVariant<Variant>::kWgmma && GemmVariant<Variant>::kTileM==64 &&
      (GemmVariant<Variant>::kTileN==64 || GemmVariant<Variant>::kTileN==128) &&
      GemmVariant<Variant>::kTileK==64 && GemmVariant<Variant>::kStages==2;
  if constexpr(Variant+1<TILEMEGA_GEMM_VARIANT_COUNT)return DwPwWgmmaVariant<Variant+1>(index);
  return false;
}
#endif
inline void CreateTensorMaps(Plan& plan,ModelSpec const& spec,TargetSpec const& target) {
#if TILEMEGA_PAGED || TILEMEGA_ATTENTION_IMPL==4 || TILEMEGA_FORWARD_GEMM_WGMMA || TILEMEGA_ENCODER_ATTENTION_WGMMA || TILEMEGA_DEPTHWISE_TMA || TILEMEGA_DWPW_WGMMA || TILEMEGA_PREFILL_ATTENTION_TMA
  if(!target.caps.tma || TILEMEGA_ARCH_PATH_SM80)return;
  unsigned encoder_heads=0,encoder_map_count=0;
#if TILEMEGA_ENCODER_ATTENTION_WGMMA
  for(unsigned i=0;i<spec.stage_count;++i)if(spec.stages[i].kind==TaskKind::kEncoderAttention)
    encoder_heads=std::max(encoder_heads,spec.stages[i].extent);
  encoder_map_count=spec.buffer_count*plan.model.params.dims.batch*encoder_heads*3;
#endif
  std::vector<executor::TensorMap> maps(spec.buffer_count+encoder_map_count
#if TILEMEGA_FORWARD_GEMM_WGMMA
      +2*plan.model.gemm_invocation_count
#endif
      );
  std::vector<bool> encoded(spec.buffer_count,false);
  auto encode=[&](unsigned id,executor::TensorMapShape shape) {
    if(encoded.at(id))return;
    auto code=executor::EncodeTensorMap(maps[id],plan.model.buffers.at(id),shape);
    if(code!=CUDA_SUCCESS)throw std::runtime_error("serving tensor map encoding failed: "+std::to_string(int(code)));
    encoded[id]=true;
  };
#if TILEMEGA_PAGED && !TILEMEGA_WEIGHT_LAYOUT_TILED
  for(unsigned i=0;i<spec.gemm_count;++i) {
    auto const& g=spec.gemms[i];auto const& tile=spec.runtime_variants[0].gemms[i];
    encode(g.b,{std::uint64_t(g.k),std::uint64_t(g.n),std::uint64_t(g.k)*2,
        64,std::uint32_t(std::min(int(tile.tile_n),TILEMEGA_PAGE_BYTES/128))});
  }
#endif
  for(unsigned i=0;i<spec.stage_count;++i) {
    auto const& s=spec.stages[i];if(s.kind!=TaskKind::kFusedAttention)continue;
    auto dims=plan.model.params.dims;
    executor::TensorMapShape shape{std::uint64_t(s.width),
        std::uint64_t(dims.batch)*s.extent*dims.capacity,std::uint64_t(s.width)*2,
        64,TILEMEGA_ATTENTION_IMPL==4?64u:16u};
    encode(s.operand[1],shape);encode(s.operand[2],shape);
  }
#if TILEMEGA_FORWARD_GEMM_WGMMA
  std::vector<GemmInvocation> gemms(plan.model.gemm_invocation_count);
  TILEMEGA_CUDA_CHECK(cudaMemcpy(gemms.data(),plan.model.device_gemms,
      gemms.size()*sizeof(gemms[0]),cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<gemms.size();++i) {
    auto const& g=gemms[i];auto [m,n,k,batch]=g.problem;(void)k;(void)batch;
    if(!ForwardWgmmaVariant(g.variant))continue;
#if TILEMEGA_DWPW_WGMMA
    if(g.fused_program!=kDmNoIndex)continue;
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    if((g.access.a!=DmAAccess::kDense && g.access.a!=DmAAccess::kIm2Col) ||
       g.access.b!=DmBAccess::kDense || g.a_scale)continue;
#endif
    auto encode_gemm=[&](unsigned index,void* address,executor::TensorMapShape shape) {
      auto code=executor::EncodeTensorMap(maps[index],address,shape);
      if(code!=CUDA_SUCCESS)throw std::runtime_error("forward GEMM tensor map encoding failed: "+std::to_string(int(code)));
    };
    auto a_pitch=std::uint64_t(cute::get<0>(g.mainloop.dA));
    auto b_pitch=std::uint64_t(cute::get<0>(g.mainloop.dB));
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    if(g.access.a==DmAAccess::kIm2Col) {
      auto const& conv=spec.convolutions[g.access.conv];
      auto shape=executor::TensorIm2colShape::Build(conv,spec.buffers[conv.input_layout].layout);
      auto code=executor::EncodeTensorIm2col(maps[spec.buffer_count+2*i],
          plan.model.buffers.at(g.serving_activation_buffer),shape);
      if(code!=CUDA_SUCCESS)throw std::runtime_error("convolution im2col map encoding failed: "+std::to_string(int(code)));
    }else
#endif
    encode_gemm(spec.buffer_count+2*i,plan.model.buffers.at(g.serving_activation_buffer),
        {std::uint64_t(g.k_total),std::uint64_t(m),a_pitch*2,64,std::uint32_t(g.tile_m)});
#if !TILEMEGA_NONPAGED_TILED
    encode_gemm(spec.buffer_count+2*i+1,plan.model.buffers.at(g.serving_weight_buffer),
        {std::uint64_t(g.k_total),std::uint64_t(n),b_pitch*2,64,std::uint32_t(g.tile_n)});
#endif
  }
#endif
#if TILEMEGA_ENCODER_ATTENTION_WGMMA
  unsigned encoder_offset=maps.size()-encoder_map_count;
  for(unsigned i=0;i<spec.stage_count;++i) {
    auto const& stage=spec.stages[i];if(stage.kind!=TaskKind::kEncoderAttention)continue;
    if(stage.width!=128 || stage.group!=64 || stage.operand[2]!=kNoOperand)
      throw std::invalid_argument("Hopper encoder attention requires S128, query64, and no padding mask");
    for(unsigned b=0;b<unsigned(plan.model.params.dims.batch);++b)
      for(unsigned h=0;h<stage.extent;++h)for(unsigned slot=0;slot<3;++slot) {
        unsigned index=encoder_offset+((stage.operand[0]*plan.model.params.dims.batch+b)*encoder_heads+h)*3+slot;
        auto* source=static_cast<ModelElement*>(plan.model.buffers.at(stage.operand[0]))+
            (std::size_t(b)*128*stage.extent+h)*192+slot*64;
        auto code=executor::EncodeTensorMap(maps[index],source,
            {64,128,std::uint64_t(stage.extent)*192*2,64,slot==0?64u:128u});
        if(code!=CUDA_SUCCESS)throw std::runtime_error("encoder tensor map encoding failed: "+std::to_string(int(code)));
      }
  }
#endif
#if TILEMEGA_DEPTHWISE_TMA
  unsigned depthwise_offset=maps.size(),depthwise_batch=plan.model.params.dims.batch;
  maps.resize(maps.size()+spec.stage_count*depthwise_batch);
  bool has_depthwise=false;
  for(unsigned i=0;i<spec.stage_count;++i) {
    auto const& stage=spec.stages[i];if(stage.kind!=TaskKind::kDepthwiseConv)continue;
    has_depthwise=true;
    if(stage.dm_program>=spec.stage_count)throw std::invalid_argument("invalid depthwise map program index");
    auto const& conv=spec.convolutions[stage.conv];auto const& layout=spec.buffers[stage.operand[0]].layout;
    auto shape=executor::TensorMap3DShape::Depthwise(conv,layout,stage.group,stage.width);
    for(unsigned image=0;image<depthwise_batch;++image) {
      auto* source=static_cast<ModelElement*>(plan.model.buffers.at(stage.operand[0]))+image*layout.strides[0];
      auto index=depthwise_offset+stage.dm_program*depthwise_batch+image;
      auto code=executor::EncodeTensorMap3D(maps[index],source,shape);
      if(code!=CUDA_SUCCESS)throw std::runtime_error("depthwise 3D map encoding failed: "+std::to_string(int(code)));
    }
  }
  if(!has_depthwise)throw std::invalid_argument("depthwise TMA candidate requires an active depthwise stage");
#endif
#if TILEMEGA_DWPW_WGMMA
  unsigned dwpw_offset=maps.size();
  std::vector<GemmInvocation> fused_gemms(plan.model.gemm_invocation_count);
  TILEMEGA_CUDA_CHECK(cudaMemcpy(fused_gemms.data(),plan.model.device_gemms,
      fused_gemms.size()*sizeof(fused_gemms[0]),cudaMemcpyDeviceToHost));
  maps.resize(maps.size()+fused_gemms.size());bool has_dwpw_wgmma=false;
  for(unsigned i=0;i<fused_gemms.size();++i) {
    auto const& g=fused_gemms[i];if(g.fused_program==kDmNoIndex || !DwPwWgmmaVariant(g.variant))continue;
    auto const& c=g.fused_depthwise.convolution;
    if(g.chunks!=1 || c.c<64 || c.c%64 || g.serving_k_begin || g.k_total!=int(c.c))
      throw std::invalid_argument("streamed DW/PW WGMMA requires unsplit complete K64 channel blocks");
    for(unsigned j=0;j<g.fused_depthwise.chain.count;++j)
      if(g.fused_depthwise.chain.operations[j].kind==DmEpilogueKind::kGatePair)
        throw std::invalid_argument("streamed DW/PW WGMMA does not support a depthwise gate");
    has_dwpw_wgmma=true;
#if !TILEMEGA_NONPAGED_TILED
    auto [m,n,k,batch]=g.problem;(void)m;(void)k;(void)batch;
    if(int(cute::get<0>(g.mainloop.dB))!=int(c.c))
      throw std::invalid_argument("streamed DW/PW WGMMA requires contiguous complete weight rows");
    auto code=executor::EncodeTensorMap(maps[dwpw_offset+i],
        plan.model.buffers.at(g.serving_weight_buffer),
        {c.c,std::uint64_t(n),std::uint64_t(c.c)*2,64,std::uint32_t(g.tile_n)});
    if(code!=CUDA_SUCCESS)throw std::runtime_error("DW/PW weight map encoding failed: "+std::to_string(int(code)));
#endif
  }
  if(!has_dwpw_wgmma)throw std::invalid_argument("DW/PW WGMMA requires an active eligible fused projection");
#endif
#if TILEMEGA_PREFILL_ATTENTION_TMA
  unsigned prefill_offset=maps.size();
  std::vector<std::uint32_t> prefill_offsets(spec.buffer_count,~std::uint32_t(0));
  auto dims=plan.model.params.dims;
  for(unsigned i=0;i<spec.stage_count;++i) {
    auto const& s=spec.stages[i];if(s.kind!=TaskKind::kFusedAttention)continue;
    if(dims.seq<=1 || dims.past+dims.seq>dims.capacity || s.group<1 || s.group>8 ||
        (s.width!=64 && s.width!=128) || s.attention_kv_block<dims.capacity)
      throw std::invalid_argument("prefill TMA maps require complete preprocessed QKV and a single KV chunk");
    if(prefill_offsets.at(s.operand[0])!=~std::uint32_t(0))
      throw std::invalid_argument("prefill TMA QKV buffer has multiple attention consumers");
    prefill_offsets[s.operand[0]]=maps.size()-prefill_offset;
    auto append=[&](void* address,executor::TensorMapShape shape) {
      maps.emplace_back();auto status=executor::EncodeTensorMap(maps.back(),address,shape);
      if(status!=CUDA_SUCCESS)throw std::invalid_argument("prefill attention tensor map encoding failed: "+std::to_string(int(status)));
    };
    unsigned columns=s.extent*(s.group+2)*s.width;
    for(int batch=0;batch<dims.batch;++batch)for(unsigned group=0;group<s.extent;++group) {
      for(unsigned head=0;head<s.group;++head) {
        auto offset=std::size_t(batch)*dims.seq*columns+(group*(s.group+2)+head)*s.width;
        append(plan.model.buffers.at(s.operand[0])+offset,
            {s.width,std::uint64_t(dims.seq),std::uint64_t(columns)*2,64,
             unsigned(ServingPrefillQueryMapRows(s.group))});
      }
      auto offset=(std::size_t(batch)*s.extent+group)*dims.capacity*s.width;
      for(unsigned operand:{1u,2u})append(plan.model.buffers.at(s.operand[operand])+offset,
          {s.width,std::uint64_t(dims.capacity),std::uint64_t(s.width)*2,64,64});
    }
  }
  if(maps.size()==prefill_offset)throw std::invalid_argument("prefill TMA has no attention stage");
  TILEMEGA_CUDA_CHECK(cudaMalloc(&plan.prefill_map_offsets,prefill_offsets.size()*sizeof(prefill_offsets[0])));
  TILEMEGA_CUDA_CHECK(cudaMemcpy(plan.prefill_map_offsets,prefill_offsets.data(),
      prefill_offsets.size()*sizeof(prefill_offsets[0]),cudaMemcpyHostToDevice));
#endif
  TILEMEGA_CUDA_CHECK(cudaMalloc(&plan.tensor_maps,maps.size()*sizeof(maps[0])));
  TILEMEGA_CUDA_CHECK(cudaMemcpy(plan.tensor_maps,maps.data(),maps.size()*sizeof(maps[0]),cudaMemcpyHostToDevice));
  plan.model.params.serving_tensor_maps=plan.tensor_maps;
#if TILEMEGA_PREFILL_ATTENTION_TMA
  plan.model.params.prefill_tensor_maps=plan.tensor_maps+prefill_offset;
  plan.model.params.prefill_tensor_map_offsets=plan.prefill_map_offsets;
#endif
#if TILEMEGA_DEPTHWISE_TMA
  plan.model.params.depthwise_tensor_maps=plan.tensor_maps+depthwise_offset;
#endif
#if TILEMEGA_ENCODER_ATTENTION_WGMMA
  plan.model.params.encoder_tensor_maps=plan.tensor_maps+encoder_offset;
  plan.model.params.encoder_tensor_map_heads=encoder_heads;
#endif
#if TILEMEGA_FORWARD_GEMM_WGMMA
  for(unsigned i=0;i<gemms.size();++i) {
    auto& g=gemms[i];if(!ForwardWgmmaVariant(g.variant))continue;
#if TILEMEGA_DWPW_WGMMA
    if(g.fused_program!=kDmNoIndex)continue;
#endif
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    if((g.access.a!=DmAAccess::kDense && g.access.a!=DmAAccess::kIm2Col) ||
       g.access.b!=DmBAccess::kDense || g.a_scale)continue;
#endif
    g.serving_activation_tensor_map=plan.tensor_maps+spec.buffer_count+2*i;
#if !TILEMEGA_NONPAGED_TILED
    g.serving_weight_tensor_map=plan.tensor_maps+spec.buffer_count+2*i+1;
#endif
  }
  TILEMEGA_CUDA_CHECK(cudaMemcpy(plan.model.device_gemms,gemms.data(),
      gemms.size()*sizeof(gemms[0]),cudaMemcpyHostToDevice));
#endif
#if TILEMEGA_DWPW_WGMMA
  // Patch only this coordinate's field; preserve any activation/weight maps
  // installed by the independent forward/decode TMA initialization above.
  for(unsigned i=0;i<fused_gemms.size();++i) {
    auto const& g=fused_gemms[i];if(g.fused_program==kDmNoIndex || !DwPwWgmmaVariant(g.variant))continue;
    void const* pointer=plan.tensor_maps+dwpw_offset+i;
    auto offset=reinterpret_cast<char const*>(&g.serving_fused_weight_tensor_map)-reinterpret_cast<char const*>(&g);
    TILEMEGA_CUDA_CHECK(cudaMemcpy(reinterpret_cast<char*>(plan.model.device_gemms)+
        i*sizeof(GemmInvocation)+offset,&pointer,sizeof(pointer),cudaMemcpyHostToDevice));
  }
#endif
#endif
}

inline void Destroy(Plan* plan) {
  if (!plan) return;
  if(plan->page_trace) {
    if(auto path=std::getenv("TILEMEGA_PAGE_TRACE_OUT")) {
      cudaDeviceSynchronize();
      std::vector<PageTraceRecord> rows(std::size_t(plan->grid)*plan->steps);
      if(cudaMemcpy(rows.data(),plan->page_trace,rows.size()*sizeof(rows[0]),
                    cudaMemcpyDeviceToHost)==cudaSuccess)
        if(auto* out=std::fopen(path,"w")) {
          std::fprintf(out,"step\tworker\tkernel_begin_ns\tkernel_end_ns\tdependency_wait_ns\tpage_full_ns\tfull_and_wait_ns\tdependency_episodes\tpage_full_episodes\tloader_issue_ns\n");
          for(std::size_t i=0;i<rows.size();++i)
            std::fprintf(out,"%zu\t%zu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n",i/std::size_t(plan->grid),i%std::size_t(plan->grid),
                rows[i].kernel_begin_ns,rows[i].kernel_end_ns,
                rows[i].dependency_wait_ns,rows[i].page_full_ns,rows[i].full_and_wait_ns,
                rows[i].dependency_episodes,rows[i].page_full_episodes,rows[i].loader_issue_ns);
          std::fclose(out);
        }
    }
    cudaFree(plan->page_trace);
  }
#if TILEMEGA_TRACE_TASK
  if(plan->task_profile)cudaFree(plan->task_profile);
  if(plan->task_offsets)cudaFree(plan->task_offsets);
  if(plan->task_epoch)cudaFree(plan->task_epoch);
#endif
#if TILEMEGA_TRACE_STAGE
  if(plan->stage_trace)cudaFree(plan->stage_trace);
#endif
#if TILEMEGA_TRACE_STEP
  if(plan->step_trace)cudaFree(plan->step_trace);
#endif
  if (plan->ring) cudaFree(plan->ring);
  if (plan->step_ns) cudaFree(plan->step_ns);
  if (plan->tensor_maps) cudaFree(plan->tensor_maps);
#if TILEMEGA_PREFILL_ATTENTION_TMA
  if(plan->prefill_map_offsets)cudaFree(plan->prefill_map_offsets);
#endif
  if (plan->lag_dependencies) cudaFree(plan->lag_dependencies);
  if (plan->watchdog) cudaFreeHost(plan->watchdog);
  auto& model = plan->model;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  if(model.device_dm_convolutions)cudaFree(model.device_dm_convolutions);
  if(model.device_dm_layouts)cudaFree(model.device_dm_layouts);
  if(model.device_dm_dtypes)cudaFree(model.device_dm_dtypes);
  if(model.device_dm_buffers)cudaFree(model.device_dm_buffers);
  if(model.device_memory_arena)cudaFree(model.device_memory_arena);
  if(model.device_dependency_intervals)cudaFree(model.device_dependency_intervals);
  if(model.device_counted_thresholds)cudaFree(model.device_counted_thresholds);
  if(model.params.dm_reductions.stages)cudaFree(const_cast<DmReductionStage*>(model.params.dm_reductions.stages));
  if(model.params.dm_reductions.offsets)cudaFree(const_cast<unsigned*>(model.params.dm_reductions.offsets));
  if(model.params.dm_reductions.arrivals)cudaFree(const_cast<DmReductionArrival*>(model.params.dm_reductions.arrivals));
  if(model.params.dm_reductions.tickets)cudaFree(model.params.dm_reductions.tickets);
  if(model.params.counted_dependencies)cudaFree(model.params.counted_dependencies);
  if(model.device_counted_l2_params)cudaFree(model.device_counted_l2_params);
#if TILEMEGA_MOE_DYNAMIC
  if(model.device_dynamic_ranges)cudaFree(model.device_dynamic_ranges);
  if(model.device_dynamic_canonical)cudaFree(model.device_dynamic_canonical);
  if(model.params.dynamic_claims)cudaFree(model.params.dynamic_claims);
#endif
#endif
  if(model.params.serving_handoff_tickets)cudaFree(model.params.serving_handoff_tickets);
#if TILEMEGA_ATTENTION_FRONTIER
  if(model.params.serving_attention_frontier)cudaFree(model.params.serving_attention_frontier);
#endif
#if TILEMEGA_NONPAGED_LA || (TILEMEGA_DM_REDUCTIONS && !TILEMEGA_PAGED)
  if(model.params.serving_nonpaged_tickets)cudaFree(model.params.serving_nonpaged_tickets);
#endif
  if(model.params.serving_no_producer)cudaFree(const_cast<std::uint8_t*>(model.params.serving_no_producer));
#if TILEMEGA_TRACE_V2
  if (model.device_reducer_trace) cudaFree(model.device_reducer_trace);
  if (model.device_task_trace_v2) cudaFree(model.device_task_trace_v2);
  if (model.device_event_publish) cudaFree(model.device_event_publish);
#endif
  for (std::size_t i = 0; i < model.buffers.size(); ++i)
    if (i < model.owned_buffers.size() && model.owned_buffers[i])
      cudaFree(model.buffers[i]);
  for (void* pointer : {
           static_cast<void*>(model.device_buffers),
           static_cast<void*>(model.device_gemms),
           static_cast<void*>(model.device_stages),
           static_cast<void*>(model.device_dependencies),
           static_cast<void*>(model.device_dependency_offsets),
           static_cast<void*>(model.device_schedule),
           static_cast<void*>(model.device_schedule_offsets),
           static_cast<void*>(model.device_task_waits),
           static_cast<void*>(model.device_event_offsets),
           static_cast<void*>(model.device_event_flags),
           static_cast<void*>(model.device_event_fanin),
           static_cast<void*>(model.device_shard_arrivals),
           static_cast<void*>(model.device_shard_targets),
           static_cast<void*>(model.device_shard_local_offsets),
           static_cast<void*>(model.device_cluster_shard_offsets),
           static_cast<void*>(model.device_cluster_shard_indices),
           static_cast<void*>(model.device_params),
           static_cast<void*>(model.events)})
    if (pointer) cudaFree(pointer);
  delete plan;
}

inline cudaError_t ConfigureServingDynamicShared(void const* kernel,
                                                std::size_t dynamic_bytes) {
  cudaFuncAttributes attributes{};
  cudaError_t status=cudaFuncGetAttributes(&attributes,kernel);
  if(status!=cudaSuccess)return status;
  int device=0,default_limit=0;
  status=cudaGetDevice(&device);
  if(status!=cudaSuccess)return status;
  status=cudaDeviceGetAttribute(&default_limit,cudaDevAttrMaxSharedMemoryPerBlock,device);
  if(status!=cudaSuccess)return status;
  if(executor::SharedMemoryRequiresOptIn(dynamic_bytes,attributes.sharedSizeBytes,
                                        static_cast<std::size_t>(default_limit)))
    return cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(dynamic_bytes));
  return cudaSuccess;
}

inline int ResidentServingGrid(tilemega::TargetSpec const& target) {
  int pinned=kModel.runtime_variants[0].plan.eft_grid;
  if constexpr(TILEMEGA_SERVING_CLUSTER_DIM==1) {
    if constexpr(TILEMEGA_SERVING_RESIDENCY_REQUEST>0) {
      int requested=target.res.num_sms*TILEMEGA_SERVING_RESIDENCY_REQUEST;
      if(pinned && pinned!=requested)
        throw std::invalid_argument("requested CTA residency differs from the pinned EFT grid");
      return requested;
    }
    return pinned?pinned:target.res.num_sms;
  } else {
    static_assert(TILEMEGA_SERVING_CLUSTER_DIM==1 || arch::Caps<GemmVariantArch>::kCluster,
                  "serving cluster launch requires native cluster capability");
    if(!target.caps.cluster || target.res.max_cluster_size<TILEMEGA_SERVING_CLUSTER_DIM)
      throw std::invalid_argument("serving cluster dimension exceeds runtime device capability");
    int candidates=target.res.num_sms/TILEMEGA_SERVING_CLUSTER_DIM;
    if(!candidates)throw std::invalid_argument("serving grid is smaller than one cluster");
    int grid=candidates*TILEMEGA_SERVING_CLUSTER_DIM;
    int resident=std::numeric_limits<int>::max();
    auto check=[&](auto kernel) {
      TILEMEGA_CUDA_CHECK(ConfigureServingDynamicShared(
          reinterpret_cast<void const*>(kernel),kServingSharedBytes));
      int clusters=0;
      TILEMEGA_CUDA_CHECK(executor::ServingActiveClusters(&clusters,kernel,
          kServingThreads,kServingSharedBytes,grid));
      resident=std::min(resident,clusters);
    };
    check(tilemega_l1_kernel);check(tilemega_l2_kernel);
#if TILEMEGA_PAGED
    check(tilemega_loop_kernel);
#elif TILEMEGA_SERVING_SEQ==1
    check(tilemega_l1_loop_kernel);
#endif
    if(resident<=0 || resident>std::numeric_limits<int>::max()/TILEMEGA_SERVING_CLUSTER_DIM)
      throw std::invalid_argument("serving kernels have no resident cluster grid");
    grid=resident*TILEMEGA_SERVING_CLUSTER_DIM;
    if constexpr(TILEMEGA_SERVING_RESIDENCY_REQUEST>0) {
      int requested=target.res.num_sms*TILEMEGA_SERVING_RESIDENCY_REQUEST;
      if(requested%TILEMEGA_SERVING_CLUSTER_DIM || requested>grid)
        throw std::invalid_argument("cluster occupancy cannot satisfy requested CTA residency");
      grid=requested;
    }
    // EFT worker IDs are part of sigma; changing their grid changes the plan.
    if(pinned && pinned!=grid)
      throw std::invalid_argument("serving cluster occupancy differs from the pinned EFT grid");
    return grid;
  }
}

}  // namespace tilemega::codegen::serving

extern "C" int tm_plan_query(tm_plan_info* output) {
  if (!output) return -1;
  auto target = tilemega::TargetSpec::Probe();
  int grid=0;
  try { grid=tilemega::codegen::serving::ResidentServingGrid(target); }
  catch(std::exception const& error) {
    std::fprintf(stderr,"tm_plan_query: %s\n",error.what());return -5;
  }
  *output = {TM_SERVING_ABI_VERSION,
#if defined(TILEMEGA_SERVING_PHASE)
             TILEMEGA_SERVING_PHASE,
#else
             TILEMEGA_SERVING_SEQ == 1 ? TM_SERVING_DECODE : TM_SERVING_PREFILL,
#endif
             TILEMEGA_SERVING_BATCH_LO, TILEMEGA_SERVING_BATCH_HI,
             TILEMEGA_SERVING_SEQ, TILEMEGA_SERVING_PAST_LO,
             TILEMEGA_SERVING_PAST_HI, kModel.dims.capacity,
             grid, (grid+target.res.num_sms-1)/target.res.num_sms,
             TM_SERVING_L1 | TM_SERVING_L2,
             kModel.buffer_count};
  return 0;
}

extern "C" int tm_plan_buffer(std::uint32_t index,
                                      tm_buffer_info* output) {
  if (!output || index >= kModel.buffer_count) return -1;
  auto const& desc = kModel.buffers[index];
  if (desc.per_past || desc.per_total) return -2;
  *output = {desc.external_name ? desc.external_name : desc.name,
             desc.role, desc.dtype,
             desc.constant + std::uint64_t(desc.per_seq) * TILEMEGA_SERVING_SEQ,
             desc.per_batch, desc.pack_json};
  return 0;
}

// Optional diagnostic export. The caller synchronizes its launch stream before
// copying; these host reads never run on the inference or timing path.
extern "C" int tm_plan_read_buffer(void* opaque, std::uint32_t index,
                                    void* host, std::uint64_t bytes) {
  using namespace tilemega::codegen;
  auto* plan=static_cast<serving::Plan*>(opaque);
  if(!plan || !host || index>=plan->model.buffers.size())return -1;
  auto const& desc=plan->model.spec->buffers[index];
  std::uint64_t element_bytes=desc.dtype==0?sizeof(ModelElement):(desc.dtype==3?8:4);
  std::uint64_t elements=desc.Elements(plan->model.params.dims);
  if(elements>std::numeric_limits<std::uint64_t>::max()/element_bytes ||
     bytes!=elements*element_bytes || !plan->model.buffers[index])return -2;
  return cudaMemcpy(host,plan->model.buffers[index],bytes,cudaMemcpyDeviceToHost)==cudaSuccess?0:-3;
}

extern "C" void* tm_plan_create(int batch, void* const* external,
                                        int device) {
  using namespace tilemega::codegen;
  if (!external || batch < TILEMEGA_SERVING_BATCH_LO ||
      batch > TILEMEGA_SERVING_BATCH_HI) return nullptr;
  if (cudaSetDevice(device) != cudaSuccess) return nullptr;
  try {
    auto target = tilemega::TargetSpec::Probe();
    ModelDims dims = kModel.dims;
    dims.batch = batch;
    dims.seq = TILEMEGA_SERVING_SEQ;
    dims.past = TILEMEGA_SERVING_PAST_LO;
    dims.total = dims.seq + dims.past;
    if (!serving::StructureInvariant(kModel, kModel.runtime_variants[0],
          dims, ModelDims{dims.seq, TILEMEGA_SERVING_PAST_HI,
                          dims.seq + TILEMEGA_SERVING_PAST_HI,
                          batch, dims.capacity}))
      return nullptr;
    std::unique_ptr<serving::Plan, void (*)(serving::Plan*)> plan(
        new serving::Plan, serving::Destroy);
    int const grid=serving::ResidentServingGrid(target);
    std::size_t const smem = kServingSharedBytes;
    if (smem > target.res.max_dynamic_smem_per_cta) return nullptr;
    {
      if (serving::ConfigureServingDynamicShared(
            reinterpret_cast<void const*>(tilemega_l1_kernel),smem) != cudaSuccess ||
          serving::ConfigureServingDynamicShared(
            reinterpret_cast<void const*>(tilemega_l2_kernel),smem) != cudaSuccess)
        return nullptr;
#if TILEMEGA_PAGED
      if (serving::ConfigureServingDynamicShared(
            reinterpret_cast<void const*>(tilemega_loop_kernel),smem) != cudaSuccess)
        return nullptr;
#elif TILEMEGA_SERVING_SEQ==1
      if(serving::ConfigureServingDynamicShared(
          reinterpret_cast<void const*>(tilemega_l1_loop_kernel),smem)!=cudaSuccess)
        return nullptr;
#endif
    }
    int l1 = target.ActiveBlocksPerSM(
        reinterpret_cast<void const*>(tilemega_l1_kernel),
        kServingThreads, smem);
    int l2 = target.ActiveBlocksPerSM(
        reinterpret_cast<void const*>(tilemega_l2_kernel),
        kServingThreads, smem);
    if (grid > target.res.num_sms * std::min(l1, l2)) {
      if constexpr(TILEMEGA_SERVING_RESIDENCY_REQUEST>0) {
        std::fprintf(stderr,"E2E_SERVING_RESIDENCY_REJECT requested=%d grid=%d l1_limit=%d l2_limit=%d shared=%zu\n",
            TILEMEGA_SERVING_RESIDENCY_REQUEST,grid,l1,l2,smem);
        throw std::invalid_argument("requested serving residency exceeds actual CUDA kernel occupancy");
      }
      return nullptr;
    }
#if TILEMEGA_PAGED
    if(grid>target.res.num_sms*(TILEMEGA_SERVING_RESIDENCY_REQUEST?
          TILEMEGA_SERVING_RESIDENCY_REQUEST:1))return nullptr;
    if constexpr(TILEMEGA_SERVING_RESIDENCY_REQUEST>0) {
      int loop=target.ActiveBlocksPerSM(reinterpret_cast<void const*>(tilemega_loop_kernel),
          kServingThreads,smem);
      if(grid>target.res.num_sms*loop)
        throw std::invalid_argument("paged loop occupancy cannot satisfy requested CTA residency");
    }
#endif
    plan->model = harness::Create(
        kModel, kModel.runtime_variants[0], 0, dims, "", grid,
        std::min(l1, l2), target, smem, external);
#if TILEMEGA_ATTENTION_FRONTIER
    if(TILEMEGA_PAGED && TILEMEGA_SERVING_SEQ==1 && batch==1) {
      unsigned stride=0;
      for(auto const& stage:plan->model.stages)
        if(stage.kind==TaskKind::kFusedAttention)stride=std::max(stride,stage.extent);
      auto& p=plan->model.params;p.serving_attention_frontier_stride=stride;
      std::size_t bytes=plan->model.stages.size()*stride*sizeof(EventCounter);
      if(bytes) {
        TILEMEGA_CUDA_CHECK(cudaMalloc(&p.serving_attention_frontier,bytes));
        TILEMEGA_CUDA_CHECK(cudaMemset(p.serving_attention_frontier,0,bytes));
      }
    }
#endif
    {
      auto const env=std::getenv("TILEMEGA_WATCHDOG_MS");
      unsigned long long ms=env?std::strtoull(env,nullptr,10):10000ull;
      if(ms) {
        TILEMEGA_CUDA_CHECK(cudaHostAlloc(&plan->watchdog,sizeof(WatchdogRecord),
            cudaHostAllocMapped));
        *plan->watchdog={};
        TILEMEGA_CUDA_CHECK(cudaHostGetDevicePointer(
            &plan->model.params.serving_watchdog,plan->watchdog,0));
        plan->model.params.serving_watchdog_ns=ms*1000000ull;
      }
    }
#if TILEMEGA_PAGED
    if(!plan->model.lag_dependencies.empty()) {
      TILEMEGA_CUDA_CHECK(cudaMalloc(&plan->lag_dependencies,
          plan->model.lag_dependencies.size()*sizeof(LagDependency)));
      TILEMEGA_CUDA_CHECK(cudaMemcpy(plan->lag_dependencies,plan->model.lag_dependencies.data(),
          plan->model.lag_dependencies.size()*sizeof(LagDependency),cudaMemcpyHostToDevice));
      plan->model.params.lag_dependencies=plan->lag_dependencies;
      plan->model.params.lag_dependency_count=plan->model.lag_dependencies.size();
    }
#endif
    bool has_handoff=false;
    std::vector<unsigned> reduce_users(plan->model.stages.size(),0);
    for(std::size_t i=0;i<plan->model.stages.size();++i) {
      auto const& stage=plan->model.stages[i];
      if(stage.handoff_reduce_stage==kNoOperand)continue;
      if((!TILEMEGA_PAGED && !TILEMEGA_NONPAGED_LA) ||
#if !defined(TILEMEGA_DM_SUPPORT) || !TILEMEGA_DM_SUPPORT
         TILEMEGA_SERVING_SEQ!=1 ||
#endif
         stage.handoff_reduce_stage<=i ||
         stage.handoff_reduce_stage>=plan->model.stages.size())
        throw std::invalid_argument("last-arriver requires a later decode reducer and enabled execution support");
      auto const& reduce=plan->model.stages[stage.handoff_reduce_stage];
      if(!reduce.handoff_elided ||
         !((stage.kind==TaskKind::kGemm && reduce.kind==TaskKind::kGemmCombine &&
            stage.gemm==reduce.gemm) ||
           (TILEMEGA_PAGED && stage.kind==TaskKind::kGemm && reduce.kind==TaskKind::kArgmaxReduce) ||
           (stage.kind==TaskKind::kFusedAttention && reduce.kind==TaskKind::kAttentionMerge)))
        throw std::invalid_argument("last-arriver stage pair is not a complete GEMM or attention reduction");
      ++reduce_users[stage.handoff_reduce_stage];
      has_handoff=true;
    }
    for(std::size_t i=0;i<reduce_users.size();++i)
      if(reduce_users[i]>1)
        throw std::invalid_argument("a reducer cannot have multiple last-arriver owners");
    if(has_handoff && TILEMEGA_PAGED) {
      std::uint32_t stride=1;
      for(std::uint32_t i=0;i<kModel.stage_count;++i) {
        auto const& stage=kModel.stages[i];
        stride=std::max(stride,std::uint32_t(serving::Count(
            kModel,kModel.runtime_variants[0],stage,dims)));
      }
      auto count=std::size_t(plan->model.stages.size())*stride;
      if(count>std::numeric_limits<std::size_t>::max()/sizeof(unsigned))
        throw std::overflow_error("handoff ticket allocation overflow");
      TILEMEGA_CUDA_CHECK(cudaMalloc(&plan->model.params.serving_handoff_tickets,
          count*sizeof(unsigned)));
      TILEMEGA_CUDA_CHECK(cudaMemset(plan->model.params.serving_handoff_tickets,0,
          count*sizeof(unsigned)));
      plan->model.params.serving_handoff_ticket_stride=stride;
    }
    serving::CreateTensorMaps(*plan,kModel,target);
#if TILEMEGA_L2_PREFETCH
    std::uint8_t* frontier=nullptr;
    TILEMEGA_CUDA_CHECK(cudaMalloc(&frontier,sizeof(kServingFrontier)));
    TILEMEGA_CUDA_CHECK(cudaMemcpy(frontier,kServingFrontier,sizeof(kServingFrontier),cudaMemcpyHostToDevice));
    plan->model.params.serving_no_producer=frontier;
#endif
    harness::PrepareEvents(plan->model, grid);
    if (cudaMemset(plan->model.events, 0,
                   plan->model.event_count * sizeof(EventCounter)) != cudaSuccess)
      return nullptr;
    plan->grid = grid;
    if constexpr(TILEMEGA_SERVING_RESIDENCY_REQUEST>0)
      std::fprintf(stderr,"E2E_SERVING_RESIDENCY requested=%d grid=%d l1_limit=%d l2_limit=%d\n",
          TILEMEGA_SERVING_RESIDENCY_REQUEST,grid,l1,l2);
    if constexpr(TILEMEGA_SERVING_CLUSTER_DIM>1)
      std::fprintf(stderr,"E2E_SERVING_CLUSTER dim=%d resident_grid=%d\n",
          TILEMEGA_SERVING_CLUSTER_DIM,grid);
#if TILEMEGA_PAGED
    plan->loop_modes=2;
#elif TILEMEGA_SERVING_SEQ==1
    int loop_resident=target.ActiveBlocksPerSM(
        reinterpret_cast<void const*>(tilemega_l1_loop_kernel),kServingThreads,smem);
    if(grid<=target.res.num_sms*loop_resident)plan->loop_modes=1;
#endif
#if defined(TILEMEGA_SERVING_PHASE) && TILEMEGA_SERVING_PHASE == 2
    plan->loop_modes = 0;
#endif
    std::fprintf(stderr,"E2E_LOOP_MODES available=%u\n",plan->loop_modes);
    plan->pdl = TILEMEGA_SERVING_SEQ==1 && TILEMEGA_PDL &&
        !TILEMEGA_ARCH_PATH_SM80 && target.caps.pdl;
#if defined(TILEMEGA_SERVING_PHASE) && TILEMEGA_SERVING_PHASE == 2
    plan->pdl = false;
#endif
    std::fprintf(stderr,"E2E_PDL enabled=%d caps=%d\n",int(plan->pdl),int(target.caps.pdl));
    return plan.release();
  } catch (std::exception const& error) {
    std::fprintf(stderr, "tm_plan_create: %s\n", error.what());
    return nullptr;
  }
}

extern "C" int tm_plan_set_steps(void* opaque,
                                         std::int32_t const* past,
                                         std::uint32_t count) {
  using namespace tilemega::codegen;
  auto* plan = static_cast<serving::Plan*>(opaque);
  if (!plan || !past || !count || plan->ring) return -1;
#if defined(TILEMEGA_SERVING_PHASE) && TILEMEGA_SERVING_PHASE == 2
  if (count != 1 || past[0] != 0) return -2;
#endif
  std::vector<Params> host(count, plan->model.params);
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_TASK
  auto capacity_env=std::getenv("TILEMEGA_TRACE_LAUNCHES");
  plan->trace_launches=capacity_env?unsigned(std::strtoul(capacity_env,nullptr,10)):64;
  if(!plan->trace_launches || plan->trace_launches>4096)return -7;
#if TILEMEGA_TRACE_STAGE
  std::size_t stage_bytes=std::size_t(plan->trace_launches)*plan->grid*
      plan->model.stages.size()*sizeof(StageTraceRecord);
  if(cudaMalloc(&plan->stage_trace,stage_bytes)!=cudaSuccess)return -5;
  if(cudaMemset(plan->stage_trace,0,stage_bytes)!=cudaSuccess)return -6;
  for(auto& p:host)p.serving_stage_trace=plan->stage_trace;
#endif
#if TILEMEGA_TRACE_STEP
  std::size_t step_bytes=std::size_t(plan->trace_launches)*plan->grid*sizeof(StepTraceRecord);
  if(cudaMalloc(&plan->step_trace,step_bytes)!=cudaSuccess)return -5;
  if(cudaMemset(plan->step_trace,0,step_bytes)!=cudaSuccess)return -6;
  for(auto& p:host)p.serving_step_trace=plan->step_trace;
#endif
#if TILEMEGA_TRACE_TASK
  std::vector<unsigned> offsets(plan->model.stages.size()+1);
  for(auto const& task:plan->model.schedule)
    offsets[task.stage+1]=std::max(offsets[task.stage+1],task.logical_task+1);
  for(unsigned i=1;i<offsets.size();++i)offsets[i]+=offsets[i-1];
  plan->task_stride=offsets.back();
  std::size_t profile_bytes=std::size_t(plan->trace_launches)*plan->task_stride*sizeof(ServingTaskProfile);
  if(cudaMalloc(&plan->task_profile,profile_bytes)!=cudaSuccess ||
     cudaMalloc(&plan->task_offsets,offsets.size()*sizeof(unsigned))!=cudaSuccess ||
     cudaMalloc(&plan->task_epoch,plan->grid*sizeof(unsigned long long))!=cudaSuccess)return -5;
  if(cudaMemset(plan->task_profile,0,profile_bytes)!=cudaSuccess ||
     cudaMemset(plan->task_epoch,0,plan->grid*sizeof(unsigned long long))!=cudaSuccess ||
     cudaMemcpy(plan->task_offsets,offsets.data(),offsets.size()*sizeof(unsigned),cudaMemcpyHostToDevice)!=cudaSuccess)return -6;
  for(auto& p:host){p.serving_task_profile=plan->task_profile;p.serving_task_offsets=plan->task_offsets;
    p.serving_task_stride=plan->task_stride;p.serving_task_epoch=plan->task_epoch;}
#endif
  for(auto& p:host)p.serving_trace_launches=plan->trace_launches;
#endif
#if TILEMEGA_PAGED && TILEMEGA_PAGE_TRACE
  if(std::getenv("TILEMEGA_PAGE_TRACE_OUT")) {
    std::size_t bytes=std::size_t(count)*plan->grid*sizeof(PageTraceRecord);
    if(cudaMalloc(&plan->page_trace,bytes)!=cudaSuccess)return -5;
    if(cudaMemset(plan->page_trace,0,bytes)!=cudaSuccess)return -6;
    for(std::uint32_t i=0;i<count;++i)
      host[i].serving_page_trace=plan->page_trace+std::size_t(i)*plan->grid;
  }
#endif
  for (std::uint32_t i = 0; i < count; ++i) {
    if (past[i] < TILEMEGA_SERVING_PAST_LO ||
        past[i] > TILEMEGA_SERVING_PAST_HI) return -2;
    host[i].dims.past = past[i];
    host[i].dims.total = past[i] + host[i].dims.seq;
  }
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  if (plan->model.params.counted_dependencies || plan->model.params.dm_reductions.tickets) {
    // Bind the execution bank once on the host; per-thread Params copies
    // would turn a dependency extension into large local-memory traffic.
    host.reserve(2ull * count);
    for (unsigned i = 0; i < count; ++i) {
      Params l2 = host[i];
      if(l2.counted_dependencies)l2.counted_dependencies += plan->model.params.counted_dependency_count;
      if(l2.dm_reductions.tickets)l2.dm_reductions.tickets+=l2.dm_reductions.ticket_count;
      host.push_back(l2);
    }
  }
#endif
  if (cudaMalloc(&plan->ring, host.size() * sizeof(Params)) != cudaSuccess)
    return -3;
  if (cudaMemcpy(plan->ring, host.data(), host.size() * sizeof(Params),
                 cudaMemcpyHostToDevice) != cudaSuccess) return -4;
  if (cudaMalloc(&plan->step_ns, (std::size_t(count)+1)*sizeof(std::uint64_t))
      != cudaSuccess) return -5;
  plan->steps = count;
  return 0;
}

extern "C" int tm_plan_launch(void* opaque, std::uint32_t step,
                                      std::uint32_t mode,
                                      std::uint64_t iteration,
                                      void* stream) {
  using namespace tilemega::codegen;
  auto* plan = static_cast<serving::Plan*>(opaque);
  if (!plan || !plan->ring || step >= plan->steps ||
      (mode != TM_SERVING_L1 && mode != TM_SERVING_L2)) return -1;
  std::uint32_t mode_index = mode == TM_SERVING_L1 ? 0 : 1;
  if (iteration != plan->next_iteration[mode_index]) return -2;
  auto* params = plan->ring + step;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  if ((plan->model.params.counted_dependencies || plan->model.params.dm_reductions.tickets) && mode_index) params += plan->steps;
#endif
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  cudaError_t status;
  if (mode == TM_SERVING_L1)
    status=executor::LaunchServing(tilemega_l1_kernel,plan->grid,kServingThreads,
        kServingSharedBytes,cuda_stream,plan->pdl,params,plan->model.events,iteration);
  else
    status=executor::LaunchServing(tilemega_l2_kernel,plan->grid,kServingThreads,
        plan->model.l2_smem_bytes,cuda_stream,plan->pdl,params,plan->model.events,iteration);
  if (status != cudaSuccess) return int(status);
  ++plan->next_iteration[mode_index];
  return 0;
}

extern "C" unsigned tm_plan_loop_modes(void* opaque) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  return plan?plan->loop_modes:0;
}

// Host-only identity queries: no context creation, allocation or GPU probe.
extern "C" std::uint64_t tm_plan_shared_bytes() {
  return tilemega::codegen::kServingSharedBytes;
}
extern "C" unsigned tm_plan_cluster_dim() { return TILEMEGA_SERVING_CLUSTER_DIM; }
extern "C" unsigned tm_plan_requested_residency() { return TILEMEGA_SERVING_RESIDENCY_REQUEST; }
extern "C" unsigned tm_plan_pdl_enabled(void* opaque) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  return plan && plan->pdl;
}
extern "C" std::uint64_t tm_plan_execution_smem(void* opaque,unsigned mode) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  if(!plan)return 0;
  return mode==TM_SERVING_L1?tilemega::codegen::kServingSharedBytes:plan->model.l2_smem_bytes;
}

extern "C" int tm_plan_launch_steps(void* opaque, std::uint32_t first_step,
    std::uint32_t steps, std::uint32_t mode, std::uint64_t base_iteration,
    void* stream) {
  using namespace tilemega::codegen;
  auto* plan=static_cast<serving::Plan*>(opaque);
  if(!plan || !plan->ring || !steps || first_step>plan->steps ||
     steps>plan->steps-first_step ||
     (mode!=TM_SERVING_L1 && mode!=TM_SERVING_L2))return -1;
  if(!(plan->loop_modes&mode))return -3;
  unsigned index=mode==TM_SERVING_L1?0:1;
  if(base_iteration!=plan->next_iteration[index])return -2;
  auto cuda_stream=static_cast<cudaStream_t>(stream);
  cudaError_t status;
  auto* loop_params = plan->ring + first_step;
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
  if ((plan->model.params.counted_dependencies || plan->model.params.dm_reductions.tickets) && index) loop_params += plan->steps;
#endif
#if TILEMEGA_PAGED
  status=executor::LaunchServing(tilemega_loop_kernel,plan->grid,
        kServingThreads,plan->model.l2_smem_bytes,cuda_stream,plan->pdl,
        static_cast<Params const*>(loop_params),steps,
        plan->model.events,base_iteration,
        reinterpret_cast<unsigned long long*>(plan->step_ns+first_step));
#elif TILEMEGA_SERVING_SEQ==1
#if TILEMEGA_NONPAGED_LA && !TILEMEGA_PAGED
  auto* loop_step_ns=plan->step_ns?
      reinterpret_cast<unsigned long long*>(plan->step_ns+first_step):nullptr;
#endif
  status=executor::LaunchServing(tilemega_l1_loop_kernel,plan->grid,
        kServingThreads,kServingSharedBytes,cuda_stream,plan->pdl,
        static_cast<Params const*>(loop_params),steps,
        plan->model.events,base_iteration,
#if TILEMEGA_NONPAGED_LA && !TILEMEGA_PAGED
        loop_step_ns);
#else
        reinterpret_cast<unsigned long long*>(plan->step_ns+first_step));
#endif
#else
  return -3;
#endif
#if TILEMEGA_PAGED || TILEMEGA_SERVING_SEQ==1
  if(status!=cudaSuccess)return int(status);
  plan->next_iteration[index]+=steps;
  return 0;
#endif
}

extern "C" int tm_plan_read_step_ns(void* opaque,std::uint32_t first,
    std::uint32_t count,std::uint64_t* host) {
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  if(!plan || !plan->step_ns || !host || first>plan->steps ||
     count>plan->steps-first+1)return -1;
  return int(cudaMemcpy(host,plan->step_ns+first,
      std::size_t(count)*sizeof(std::uint64_t),cudaMemcpyDeviceToHost));
}

extern "C" int tm_plan_watchdog(void* opaque,std::uint64_t out[12]) {
#if !TILEMEGA_WATCHDOG
  return 0;
#else
  auto* plan=static_cast<tilemega::codegen::serving::Plan*>(opaque);
  if(!plan || !out || !plan->watchdog || !plan->watchdog->fired)return 0;
  auto const* words=reinterpret_cast<unsigned long long const*>(plan->watchdog);
  for(int i=0;i<12;++i)out[i]=words[i];
  return 1;
#endif
}

#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_TASK
extern "C" int tm_plan_dump_serving_trace(void* opaque,char const* directory) {
  using namespace tilemega::codegen;
  auto* plan=static_cast<serving::Plan*>(opaque);
  if(!plan || !directory || cudaDeviceSynchronize()!=cudaSuccess)return -1;
  auto dump=[&](char const* name,auto* device,std::size_t count,char const* header) {
    using Row=std::remove_pointer_t<decltype(device)>;
    std::vector<Row> host(count);
    if(cudaMemcpy(host.data(),device,count*sizeof(Row),cudaMemcpyDeviceToHost)!=cudaSuccess)
      return false;
    auto path=std::string(directory)+"/"+name;
    auto* out=std::fopen(path.c_str(),"w");if(!out)return false;
    std::fprintf(out,"%s\n",header);
    for(std::size_t i=0;i<count;++i) {
      auto* words=reinterpret_cast<unsigned long long const*>(&host[i]);
      // A zero begin stamp is an unused launch/stage slot.
      if(!words[2])continue;
      std::fprintf(out,"%zu",i%plan->grid);
      for(unsigned j=0;j<sizeof(Row)/sizeof(*words);++j)
        std::fprintf(out,"\t%llu",words[j]);
      std::fprintf(out,"\n");
    }
    std::fclose(out);return true;
  };
#if TILEMEGA_TRACE_STAGE
  if(!dump("stage_trace.tsv",plan->stage_trace,std::size_t(plan->trace_launches)*
      plan->model.stages.size()*plan->grid,
      "worker\titeration\tpast\tt_begin\tt_tasks_end\tt_release\ttasks\tsmid\tstage"))return -2;
#endif
#if TILEMEGA_TRACE_STEP
  if(!dump("step_trace.tsv",plan->step_trace,std::size_t(plan->trace_launches)*plan->grid,
      "worker\titeration\tpast\tkernel_begin\tkernel_end\tfirst_task\tlast_task\ttoken_lag\tkv_lag\tlast_barrier_wait\tstage_has_tasks"))return -2;
#endif
#if TILEMEGA_TRACE_TASK
  {
    std::vector<ServingTaskProfile> rows(std::size_t(plan->trace_launches)*plan->task_stride);
    if(cudaMemcpy(rows.data(),plan->task_profile,rows.size()*sizeof(rows[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)return -3;
    auto* out=std::fopen((std::string(directory)+"/task_profile.tsv").c_str(),"w");if(!out)return -4;
    std::fprintf(out,"kind\titeration\tpast\trun_begin\trun_end\tstage\ttask\tworker\tbytes\tfirst_ready\tquery_ns\tfirst_page_wait_ns\tlater_page_wait_ns\twave_compute_ns\tla_ns\tepilogue_ns\n");
    for(auto const& row:rows)if(row.run_begin) {
      std::fprintf(out,"%s",plan->model.stages[row.stage].kind==TaskKind::kGemm?"gemm":"attention");
      auto* words=reinterpret_cast<unsigned long long const*>(&row);
      for(unsigned i=0;i<sizeof(row)/sizeof(*words);++i)std::fprintf(out,"\t%llu",words[i]);
      std::fputc('\n',out);
    }
    std::fclose(out);
  }
#endif
  std::size_t gemm_count=0;
  for(auto const& stage:plan->model.stages)
    if(stage.kind==TaskKind::kGemm || stage.kind==TaskKind::kGemmCombine)
      gemm_count=std::max(gemm_count,std::size_t(stage.gemm)+1);
  std::vector<GemmInvocation> gemms(gemm_count);
  if(gemm_count && cudaMemcpy(gemms.data(),plan->model.device_gemms,
      gemm_count*sizeof(gemms[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)return -3;
  auto* out=std::fopen((std::string(directory)+"/runtime_stages.tsv").c_str(),"w");
  if(!out)return -4;
  std::fprintf(out,"stage\tkind\tkind_name\tname\tweight_bytes\textent\tgroup\twidth\tkv_block\telided\treducer\n");
  for(unsigned i=0;i<plan->model.stages.size();++i) {
    auto const& stage=plan->model.stages[i];unsigned long long bytes=0;
    char const* name="";
    if(stage.kind==TaskKind::kGemm || stage.kind==TaskKind::kGemmCombine) {
      auto const& inv=gemms[stage.gemm];auto [m,n,k,l]=inv.problem;
      if(stage.kind==TaskKind::kGemm)bytes=2ull*n*inv.serving_k_total_full;
      if(inv.serving_weight_buffer<plan->model.spec->buffer_count)
        name=plan->model.spec->buffers[inv.serving_weight_buffer].name;
    }
    char const* kind_name="unknown";switch(stage.kind) {case TaskKind::kGemm:kind_name="kGemm";break;case TaskKind::kRMSNorm:kind_name="kRMSNorm";break;case TaskKind::kRoPE:kind_name="kRoPE";break;case TaskKind::kKVAppend:kind_name="kKVAppend";break;case TaskKind::kElementwise:kind_name="kElementwise";break;case TaskKind::kAttention:kind_name="kAttention";break;case TaskKind::kGemmCombine:kind_name="kGemmCombine";break;case TaskKind::kGemmAdd:kind_name="kGemmAdd";break;case TaskKind::kGemmRMSNorm:kind_name="kGemmRMSNorm";break;case TaskKind::kRoPEKVAppend:kind_name="kRoPEKVAppend";break;case TaskKind::kAdd:kind_name="kAdd";break;case TaskKind::kEmbedding:kind_name="kEmbedding";break;case TaskKind::kQKNorm:kind_name="kQKNorm";break;case TaskKind::kFusedAttention:kind_name="kFusedAttention";break;case TaskKind::kAttentionMerge:kind_name="kAttentionMerge";break;case TaskKind::kArgmaxReduce:kind_name="kArgmaxReduce";break;}
#if defined(TILEMEGA_DM_SUPPORT) && TILEMEGA_DM_SUPPORT
    switch(stage.kind) {
      case TaskKind::kDepthwiseConv:kind_name="kDepthwiseConv";break;
      case TaskKind::kPool:kind_name="kPool";break;
      case TaskKind::kGlobalPoolReduce:kind_name="kGlobalPoolReduce";break;
      case TaskKind::kLayerNorm:kind_name="kLayerNorm";break;
      case TaskKind::kEncoderAttention:kind_name="kEncoderAttention";break;
      case TaskKind::kEmbeddingSum:kind_name="kEmbeddingSum";break;
      case TaskKind::kDwPwFused:kind_name="kDwPwFused";break;
      case TaskKind::kMoETopK:kind_name="kMoETopK";break;
      case TaskKind::kMoECombine:kind_name="kMoECombine";break;
      case TaskKind::kLayoutConvert:kind_name="kLayoutConvert";break;
      default:break;
    }
#endif
    std::fprintf(out,"%u\t%u\t%s\t%s\t%llu\t%u\t%u\t%u\t%u\t%u\t%u\n",i,
        unsigned(stage.kind),kind_name,name,bytes,stage.extent,stage.group,stage.width,
        unsigned(stage.attention_kv_block),unsigned(stage.handoff_elided),
        unsigned(stage.handoff_reduce_stage));
  }
  std::fclose(out);return 0;
}
#endif

#if TILEMEGA_TRACE_V2
// Diagnostic-only entry point. A trace build is separate from the measured
// serving library; dumping synchronizes only after its selected launch.
extern "C" int tm_plan_dump_trace_v2(void* opaque, float step_ms) {
  auto* plan = static_cast<tilemega::codegen::serving::Plan*>(opaque);
  if (!plan || !plan->model.trace_v2_enabled) return -1;
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  tilemega::codegen::harness::DumpTraceV2(
      plan->model,
#if defined(TILEMEGA_SERVING_PHASE) && TILEMEGA_SERVING_PHASE == 2
      "forward", plan->grid, 0.0f, step_ms, "forward_l2_last_diagnostic_launch");
#else
      "serving", plan->grid, 0.0f, step_ms, "serving_decode_last_diagnostic_launch");
#endif
  return 0;
}
#endif

extern "C" void tm_plan_destroy(void* opaque) {
  tilemega::codegen::serving::Destroy(
      static_cast<tilemega::codegen::serving::Plan*>(opaque));
}
