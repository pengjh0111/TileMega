// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/Sm90F3Program.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <tilemega/Codegen/executor/TensorIm2col.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

// Host runtime of an F3 forward plan (DNN3 prompt §5.7). The generated
// translation unit defines the buffer table, the tensor-map table and the
// F3ForwardProgram type, then includes this header for the forward ABI.
//
// Unlike the legacy runtime this launch issues **no** memset: the stage gate
// counts against an epoch that grows with every launch, and every side output
// is overwritten rather than accumulated, so nothing has to start from zero.
#ifndef TM_DTYPE_I64
#define TM_DTYPE_I64 3
#endif

namespace tilemega::codegen::forward_f3 {

inline void Check(cudaError_t status) {
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}

struct Plan {
  typename F3ForwardProgram::Binding bindings{};
  std::vector<Element*> host_buffers;
  Element** device_buffers=nullptr;
  ForwardCounter* events=nullptr;
  std::vector<void*> owned;
  // The gate's epoch: the host starts at one and adds one per launch, so a
  // stage's counter is "seen >= epoch * grid" without ever being cleared.
  unsigned long long epoch=0;
  int grid=0;
  ~Plan() {
    for(void* pointer:owned)cudaFree(pointer);
    cudaFree(events);
    cudaFree(device_buffers);
  }
};

template<class P>
inline void InstallRepeatTables(Plan& plan) {
  if constexpr(P::kRepeated) {
    auto const& indices=P::RepeatIndices();
    auto const& weights=P::RepeatWeights();
    void* index_device=nullptr;void* weight_device=nullptr;
    Check(cudaMalloc(&index_device,sizeof(indices)));
    plan.owned.push_back(index_device);
    Check(cudaMemcpy(index_device,indices.data(),sizeof(indices),cudaMemcpyHostToDevice));
    Check(cudaMalloc(&weight_device,sizeof(weights)));
    plan.owned.push_back(weight_device);
    Check(cudaMemcpy(weight_device,weights.data(),sizeof(weights),cudaMemcpyHostToDevice));
    plan.bindings.repeat_indices=static_cast<unsigned const*>(index_device);
    plan.bindings.repeat_weights=static_cast<RepeatWeight const*>(weight_device);
  }
}

inline cudaLaunchConfig_t LaunchConfig(Plan const& plan,cudaStream_t stream,
    cudaLaunchAttribute& cluster) {
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(plan.grid);config.blockDim=dim3(F3ForwardProgram::kThreads);
  config.dynamicSmemBytes=F3ForwardProgram::kSharedBytes;config.stream=stream;
  if constexpr(F3ForwardProgram::kCluster==2) {
    cluster.id=cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim={2,1,1};config.attrs=&cluster;config.numAttrs=1;
  }
  return config;
}

inline int Grid() {
  int device=0;cudaDeviceProp properties{};
  Check(cudaGetDevice(&device));Check(cudaGetDeviceProperties(&properties,device));
  if(properties.major!=9 || properties.minor!=0)
    throw std::invalid_argument("F3 forward plans require Hopper");
  return properties.multiProcessorCount;
}

}  // namespace tilemega::codegen::forward_f3

extern "C" char const* tm_plan_identity() {
  return tilemega::codegen::forward_f3::kF3ForwardIdentity;
}
extern "C" int tm_plan_static_dense_abi() {return 0;}
extern "C" unsigned tm_plan_pdl_enabled(void*) {return 0;}
extern "C" unsigned tm_plan_cluster_dim() {
  return tilemega::codegen::forward_f3::F3ForwardProgram::kCluster;
}
extern "C" unsigned tm_plan_loop_modes(void*) {return 0;}
extern "C" unsigned long long tm_plan_shared_bytes() {
  return tilemega::codegen::forward_f3::F3ForwardProgram::kSharedBytes;
}
extern "C" unsigned long long tm_plan_execution_smem(void*,unsigned mode) {
  return mode==TM_SERVING_L1 || mode==TM_SERVING_L2?tm_plan_shared_bytes():0;
}
extern "C" int tm_plan_watchdog(void*,std::uint64_t[12]) {return 0;}

extern "C" int tm_plan_query(tm_plan_info* output) {
  using namespace tilemega::codegen::forward_f3;
  if(!output)return -1;
  try {
    *output={TM_SERVING_ABI_VERSION,TM_SERVING_FORWARD,
        F3ForwardProgram::kBatch,F3ForwardProgram::kBatch,1,0,0,0,
        Grid(),F3ForwardProgram::kResidency,TM_SERVING_L1|TM_SERVING_L2,
        F3ForwardProgram::kBuffers};
    return 0;
  }catch(std::exception const& error) {
    std::fprintf(stderr,"tm_plan_query: %s\n",error.what());return -2;
  }
}

extern "C" int tm_plan_buffer(unsigned index,tm_buffer_info* output) {
  using namespace tilemega::codegen::forward_f3;
  if(!output || index>=kF3ForwardBuffers.size())return -1;
  *output=kF3ForwardBuffers[index];return 0;
}

extern "C" int tm_plan_set_steps(void*,const std::int32_t*,unsigned count) {
  return count<=1?0:-1;
}

extern "C" void* tm_plan_create(int batch,void* const* external,int device) {
  using namespace tilemega::codegen::forward_f3;
  if(batch!=F3ForwardProgram::kBatch || !external)return nullptr;
  try {
    Check(cudaSetDevice(device));
    auto plan=std::make_unique<Plan>();
    plan->grid=Grid();
    plan->host_buffers.assign(kF3ForwardBuffers.size(),nullptr);
    auto kernel=sm90_forward_f3_kernel<F3ForwardProgram>;
    Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,
        F3ForwardProgram::kSharedBytes));
    int resident=0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,
        F3ForwardProgram::kThreads,F3ForwardProgram::kSharedBytes));
    if(resident<1)
      throw std::invalid_argument("an F3 forward CTA does not fit one SM");
    if constexpr(F3ForwardProgram::kCluster==2) {
      if(plan->grid!=132)throw std::invalid_argument("F3 multicast requires132 H800 SMs");
      cudaLaunchAttribute cluster{};
      auto config=LaunchConfig(*plan,nullptr,cluster);
      int clusters=0;
      Check(cudaOccupancyMaxActiveClusters(&clusters,kernel,&config));
      if(clusters<66)throw std::invalid_argument("F3 multicast cannot keep66 clusters resident");
    }
    for(unsigned i=0;i<kF3ForwardBuffers.size();++i) {
      auto const& desc=kF3ForwardBuffers[i];
      if(desc.role==TM_BUFFER_EXTERNAL) {
        plan->host_buffers[i]=static_cast<Element*>(external[i]);
        continue;
      }
      std::size_t elements=std::size_t(desc.elements_constant)+
          std::size_t(desc.elements_per_batch)*std::size_t(batch);
      std::size_t bytes=elements*
          (desc.dtype==TM_DTYPE_F32 || desc.dtype==TM_DTYPE_I32?4u:
           desc.dtype==TM_DTYPE_I64?8u:2u);
      void* allocation=nullptr;
      Check(cudaMalloc(&allocation,bytes?bytes:16));
      // The one-time clear a split-K counter and a row counter need before
      // their first self-resetting use; no launch repeats it.
      Check(cudaMemset(allocation,0,bytes?bytes:16));
      plan->owned.push_back(allocation);
      plan->host_buffers[i]=static_cast<Element*>(allocation);
    }
    Check(cudaMalloc(&plan->device_buffers,plan->host_buffers.size()*sizeof(Element*)));
    Check(cudaMemcpy(plan->device_buffers,plan->host_buffers.data(),
        plan->host_buffers.size()*sizeof(Element*),cudaMemcpyHostToDevice));
    plan->bindings.buffers=plan->device_buffers;
    InstallRepeatTables<F3ForwardProgram>(*plan);
    for(unsigned i=0;i<kF3ForwardMaps.size();++i) {
      auto const& m=kF3ForwardMaps[i];
      namespace ex=tilemega::codegen::executor;
      auto* address=plan->host_buffers[m.buffer]+m.offset;
      if(m.kind==1) {
        auto status=m.box[0]==8?
            tilemega::backend::Sm90F3Async::EncodeStemMap(plan->bindings.maps[i],address,m.conv,m.layout):
            ex::EncodeTensorIm2col(*reinterpret_cast<ex::TensorMap*>(&plan->bindings.maps[i]),
                address,ex::TensorIm2colShape::Build(m.conv,m.layout));
        if(status!=CUDA_SUCCESS)
          throw std::invalid_argument("the driver rejected an F3 im2col map");
      }else if(m.rank==2)ex::EncodeTiledMap<2>(plan->bindings.maps[i],address,
          {{m.dims[0],m.dims[1]},{m.strides[0]},{m.box[0],m.box[1]}},m.swizzle!=0);
      else if(m.rank==3)ex::EncodeTiledMap<3>(plan->bindings.maps[i],address,
          {{m.dims[0],m.dims[1],m.dims[2]},{m.strides[0],m.strides[1]},
           {m.box[0],m.box[1],m.box[2]}},m.swizzle!=0);
      else throw std::invalid_argument("an F3 tensor map has rank 2, 3 or im2col 4");
    }
    Check(cudaMalloc(&plan->events,F3ForwardProgram::kStages*sizeof(ForwardCounter)));
    Check(cudaMemset(plan->events,0,F3ForwardProgram::kStages*sizeof(ForwardCounter)));
    static_assert(alignof(decltype(F3ForwardProgram::Binding::maps))>=64,
        "an im2col descriptor is written through a 64-byte aligned TensorMap view");
    return plan.release();
  }catch(std::exception const& error) {
    std::fprintf(stderr,"tm_plan_create: %s\n",error.what());return nullptr;
  }
}

extern "C" int tm_plan_launch(void* opaque,unsigned,unsigned mode,
                              std::uint64_t,void* stream) {
  using namespace tilemega::codegen::forward_f3;
  if(!opaque || (mode!=TM_SERVING_L1 && mode!=TM_SERVING_L2))return -1;
  auto& plan=*static_cast<Plan*>(opaque);
#if TILEMEGA_F3_H800_GRID
  if(plan.grid!=132)return -1;
#endif
  // The whole of §5.7 item 1: one launch, no memset of any kind.
  ++plan.epoch;
  if constexpr(F3ForwardProgram::kCluster==2) {
    cudaLaunchAttribute cluster{};
    auto config=LaunchConfig(plan,static_cast<cudaStream_t>(stream),cluster);
    return int(cudaLaunchKernelEx(&config,sm90_forward_f3_kernel<F3ForwardProgram>,
        plan.bindings,plan.events,plan.epoch));
  }else sm90_forward_f3_kernel<F3ForwardProgram><<<plan.grid,F3ForwardProgram::kThreads,
      F3ForwardProgram::kSharedBytes,static_cast<cudaStream_t>(stream)>>>(
      plan.bindings,plan.events,plan.epoch);
  return int(cudaGetLastError());
}

extern "C" int tm_plan_read_buffer(void* opaque,std::uint32_t index,void* host,
                                   std::uint64_t bytes) {
  using namespace tilemega::codegen::forward_f3;
  if(!opaque || !host || index>=kF3ForwardBuffers.size())return -1;
  auto& plan=*static_cast<Plan*>(opaque);
  auto* address=plan.host_buffers[index];
  if(!address)return -2;
  return cudaMemcpy(host,address,bytes,cudaMemcpyDeviceToHost)==cudaSuccess?0:-3;
}

extern "C" void tm_plan_destroy(void* opaque) {
  delete static_cast<tilemega::codegen::forward_f3::Plan*>(opaque);
}

#if TILEMEGA_TRACE_STAGE
extern "C" unsigned tm_forward_trace_schema() {return 1;}
extern "C" unsigned tm_forward_trace_stages() {return TILEMEGA_FORWARD_TRACE_STAGES;}
extern "C" unsigned tm_forward_trace_grid() {return TILEMEGA_FORWARD_TRACE_GRID;}
extern "C" unsigned tm_forward_trace_segments() {
  return tilemega::backend::kSm90F3TraceSegments;
}
extern "C" unsigned tm_forward_trace_record_bytes() {
  return sizeof(tilemega::backend::Sm90F3TraceRecord);
}
extern "C" unsigned long long tm_forward_trace_bytes() {
  return sizeof(tilemega::backend::sm90_f3_trace_records);
}
extern "C" char const* tm_forward_trace_segment_name(unsigned index) {
  using namespace tilemega::backend;
  return index<kSm90F3TraceSegments?kSm90F3TraceNames[index]:nullptr;
}
extern "C" int tm_forward_trace_reset() {
  using namespace tilemega::backend;
  auto status=cudaDeviceSynchronize();
  if(status!=cudaSuccess)return int(status);
  void* pointer=nullptr;
  status=cudaGetSymbolAddress(&pointer,sm90_f3_trace_records);
  if(status==cudaSuccess)status=cudaMemset(pointer,0,sizeof(sm90_f3_trace_records));
  if(status==cudaSuccess)status=cudaDeviceSynchronize();
  return int(status);
}
extern "C" int tm_forward_trace_copy(void* output,unsigned long long bytes) {
  using namespace tilemega::backend;
  if(!output || bytes!=sizeof(sm90_f3_trace_records))return -1;
  auto status=cudaDeviceSynchronize();
  if(status==cudaSuccess)
    status=cudaMemcpyFromSymbol(output,sm90_f3_trace_records,sizeof(sm90_f3_trace_records));
  return int(status);
}
#endif
