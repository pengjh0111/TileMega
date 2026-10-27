// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/Sm90ForwardProgram.cuh>
#include <tilemega/Codegen/executor/TensorMap.h>
#include <tilemega/Codegen/executor/TensorMapTiled.h>
#include <tilemega/Codegen/executor/TensorIm2col.h>
#include <tilemega/Target/TargetSpec.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

// Host runtime of a native DNN forward plan (prompt §10 N-1 item 1). The
// generated translation unit defines kNativeForwardBuffers, the tensor-map
// table and the NativeForwardProgram type, then includes this header, which
// supplies the forward ABI: one plan, one launch, no step ring and no KV
// state, so `step` and the iteration counter are unused.
namespace tilemega::codegen::native_forward {

inline void Check(cudaError_t status) {
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
template<class T>
inline T* Allocate(std::size_t count) {
  T* result=nullptr;
  Check(cudaMalloc(&result,count*sizeof(T)));
  auto status=cudaMemset(result,0,count*sizeof(T));
  if(status!=cudaSuccess) {cudaFree(result);Check(status);}
  return result;
}

struct Plan {
  typename NativeForwardProgram::Binding bindings{};
  ForwardCounter* events=nullptr;
  std::vector<void*> owned;
  int grid=0;
  ~Plan() {
    for(void* pointer:owned)cudaFree(pointer);
    cudaFree(events);
  }
};

inline int Grid() {
  auto target=TargetSpec::Probe();
  return target.res.num_sms*NativeForwardProgram::kResidency;
}

}  // namespace tilemega::codegen::native_forward

// The coordinates the program was generated from, so an artifact can be
// identified without its manifest.
extern "C" char const* tm_plan_identity() {
  return tilemega::codegen::native_forward::kNativeForwardIdentity;
}
extern "C" int tm_plan_static_dense_abi() {return 0;}
extern "C" unsigned tm_plan_pdl_enabled(void*) {return 0;}
extern "C" unsigned tm_plan_cluster_dim() {return 1;}
extern "C" unsigned tm_plan_loop_modes(void*) {return 0;}
extern "C" unsigned long long tm_plan_shared_bytes() {
  return tilemega::codegen::native_forward::NativeForwardProgram::kSharedBytes;
}
extern "C" unsigned long long tm_plan_execution_smem(void*,unsigned mode) {
  return mode==TM_SERVING_L1 || mode==TM_SERVING_L2?tm_plan_shared_bytes():0;
}
extern "C" int tm_plan_watchdog(void*,std::uint64_t[12]) {return 0;}

extern "C" int tm_plan_query(tm_plan_info* output) {
  using namespace tilemega::codegen::native_forward;
  if(!output)return -1;
  try {
    *output={TM_SERVING_ABI_VERSION,TM_SERVING_FORWARD,
        NativeForwardProgram::kBatch,NativeForwardProgram::kBatch,1,0,0,0,
        Grid(),NativeForwardProgram::kResidency,TM_SERVING_L1|TM_SERVING_L2,
        NativeForwardProgram::kBuffers};
    return 0;
  }catch(std::exception const& error) {
    std::fprintf(stderr,"tm_plan_query: %s\n",error.what());return -2;
  }
}

extern "C" int tm_plan_buffer(unsigned index,tm_buffer_info* output) {
  using namespace tilemega::codegen::native_forward;
  if(!output || index>=kNativeForwardBuffers.size())return -1;
  *output=kNativeForwardBuffers[index];return 0;
}

extern "C" int tm_plan_set_steps(void*,const std::int32_t*,unsigned count) {
  // A forward pass has exactly one step; the harness may still declare it.
  return count<=1?0:-1;
}

extern "C" void* tm_plan_create(int batch,void* const* external,int device) {
  using namespace tilemega::codegen::native_forward;
  if(batch!=NativeForwardProgram::kBatch || !external)return nullptr;
  try {
    Check(cudaSetDevice(device));
    auto plan=std::make_unique<Plan>();
    plan->grid=Grid();
    auto kernel=sm90_forward_native_kernel<NativeForwardProgram>;
    Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,
        NativeForwardProgram::kSharedBytes));
    int resident=0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,
        NativeForwardProgram::kThreads,NativeForwardProgram::kSharedBytes));
    if(resident<NativeForwardProgram::kResidency)
      throw std::invalid_argument("native forward plan exceeds its measured residency");
    for(unsigned i=0;i<kNativeForwardBuffers.size();++i) {
      auto const& desc=kNativeForwardBuffers[i];
      if(desc.role==TM_BUFFER_EXTERNAL) {
        plan->bindings.buffers[i]=static_cast<Element*>(external[i]);
        continue;
      }
      std::size_t bytes=desc.elements_constant*
          (desc.dtype==TM_DTYPE_F32?4u:desc.dtype==TM_DTYPE_I32?4u:2u);
      void* allocation=nullptr;
      Check(cudaMalloc(&allocation,bytes?bytes:16));
      Check(cudaMemset(allocation,0,bytes?bytes:16));
      plan->owned.push_back(allocation);
      plan->bindings.buffers[i]=static_cast<Element*>(allocation);
    }
    for(unsigned i=0;i<kNativeForwardMaps.size();++i) {
      auto const& m=kNativeForwardMaps[i];
      namespace ex=tilemega::codegen::executor;
      auto* address=plan->bindings.buffers[m.buffer]+m.offset;
      if(m.kind==1) {
        auto shape=ex::TensorIm2colShape::Build(m.conv,m.layout);
        auto status=ex::EncodeTensorIm2col(
            *reinterpret_cast<ex::TensorMap*>(&plan->bindings.maps[i]),address,shape);
        if(status!=CUDA_SUCCESS)
          throw std::invalid_argument("the driver rejected a native forward im2col map");
      }else if(m.rank==2)ex::EncodeTiledMap<2>(plan->bindings.maps[i],address,
          {{m.dims[0],m.dims[1]},{m.strides[0]},{m.box[0],m.box[1]}});
      else if(m.rank==3)ex::EncodeTiledMap<3>(plan->bindings.maps[i],address,
          {{m.dims[0],m.dims[1],m.dims[2]},{m.strides[0],m.strides[1]},
           {m.box[0],m.box[1],m.box[2]}});
      else if(m.rank==4)ex::EncodeTiledMap<4>(plan->bindings.maps[i],address,
          {{m.dims[0],m.dims[1],m.dims[2],m.dims[3]},
           {m.strides[0],m.strides[1],m.strides[2]},
           {m.box[0],m.box[1],m.box[2],m.box[3]}});
      else throw std::invalid_argument("native forward tensor map rank must be 2, 3 or 4");
    }
    plan->events=Allocate<ForwardCounter>(NativeForwardProgram::kStages);
    static_assert(alignof(decltype(NativeForwardProgram::Binding::maps))>=64,
        "an im2col descriptor is written through a 64-byte aligned TensorMap view");
    return plan.release();
  }catch(std::exception const& error) {
    std::fprintf(stderr,"tm_plan_create: %s\n",error.what());return nullptr;
  }
}

extern "C" int tm_plan_launch(void* opaque,unsigned,unsigned mode,
                              std::uint64_t,void* stream) {
  using namespace tilemega::codegen::native_forward;
  if(!opaque || (mode!=TM_SERVING_L1 && mode!=TM_SERVING_L2))return -1;
  auto& plan=*static_cast<Plan*>(opaque);
  // Every boundary counter is consumed by its own stage, so a pass starts
  // from zero; the forward program has no cross-launch phase to carry.
  if(cudaMemsetAsync(plan.events,0,NativeForwardProgram::kStages*sizeof(ForwardCounter),
                     static_cast<cudaStream_t>(stream))!=cudaSuccess)return -2;
  sm90_forward_native_kernel<NativeForwardProgram><<<plan.grid,NativeForwardProgram::kThreads,
      NativeForwardProgram::kSharedBytes,static_cast<cudaStream_t>(stream)>>>(
      plan.bindings,plan.events);
  return int(cudaGetLastError());
}

extern "C" void tm_plan_destroy(void* opaque) {
  delete static_cast<tilemega::codegen::native_forward::Plan*>(opaque);
}
