// V-I: instantiate every TaskBody and the cross-block event path from one
// architecture-neutral megakernel source (skeleton §5.3, §8).
#include <cuda_runtime.h>

#include <tilemega/Codegen/tasks/AttentionChunkTaskBody.h>
#include <tilemega/Codegen/tasks/AttentionCombineTaskBody.h>
#include <tilemega/Codegen/tasks/ElementwiseTaskBody.h>
#include <tilemega/Codegen/tasks/GemmSplitKTaskBody.h>
#include <tilemega/Codegen/tasks/GemmStageTaskBody.h>
#include <tilemega/Codegen/tasks/GemmTaskBody.h>
#include <tilemega/Codegen/tasks/KVAppendTaskBody.h>
#include <tilemega/Codegen/tasks/MoERouterTaskBody.h>
#include <tilemega/Codegen/tasks/RMSNormTaskBody.h>
#include <tilemega/Codegen/tasks/RoPETaskBody.h>
#include <tilemega/Codegen/tasks/SchedulerTaskBody.h>
#include <tilemega/Target/ArchDispatch.h>

#ifndef TILEMEGA_STAGES
#error TILEMEGA_STAGES must be computed from the target configuration
#endif

namespace {
using Arch = tilemega::arch::CurrentArch;
using Gemm = tilemega::codegen::GemmTaskBody<Arch, TILEMEGA_STAGES>;
using SplitK = tilemega::codegen::GemmSplitKTaskBody<Arch, TILEMEGA_STAGES>;
union TaskStorage;
using RuntimeGemm = tilemega::codegen::GemmStageTaskBody<Arch, TaskStorage, 256>;
using Attn = tilemega::codegen::AttentionTaskBody<Arch, TaskStorage, 256>;
using Combine = tilemega::codegen::AttentionCombineTaskBody<Arch>;
using Norm = tilemega::codegen::RMSNormTaskBody<Arch, TaskStorage, 256>;
using Rope = tilemega::codegen::RoPETaskBody<Arch, TaskStorage, 256>;
using Elementwise = tilemega::codegen::ElementwiseTaskBody<Arch, TaskStorage, 256>;
using Kv = tilemega::codegen::KVAppendTaskBody<Arch, TaskStorage, 256>;
using Moe = tilemega::codegen::MoERouterTaskBody<Arch>;
using Scheduler = tilemega::codegen::SchedulerTaskBody<Arch>;

union TaskStorage {
  Gemm::SharedStorage legacy_gemm;
  SplitK::SharedStorage splitk;
  RuntimeGemm::SharedStorage gemm;
  Attn::SharedStorage attn;
  float attention[256];
  Combine::SharedStorage combine;
  Norm::SharedStorage norm;
  float rms[256];
  Rope::SharedStorage rope;
  Elementwise::SharedStorage elementwise;
  Kv::SharedStorage kv;
  Moe::SharedStorage moe;
  Scheduler::SharedStorage scheduler;
};

struct alignas(16) Event { unsigned long long value; unsigned long long epoch; };

__device__ void Wait(Event* event, unsigned long long need) {
  if (threadIdx.x == 0) {
    while (atomicAdd(&event->value, 0ull) < need) __nanosleep(64);
  }
  __syncthreads();
}

__device__ void Signal(Event* event, unsigned long long value) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicExch(&event->value, value);
  }
}
}  // namespace

extern "C" __global__ __launch_bounds__(256)
void tilemega_megakernel(int* output, Event* events, int extent,
                         tilemega::codegen::Params const* params,
                         tilemega::codegen::StageDesc const* stage) {
  extern __shared__ unsigned char dynamic_storage[];
  auto& storage = *reinterpret_cast<TaskStorage*>(dynamic_storage);
  // Keep the caller-owned pipeline allocation observable in generated code.
  // The byte count remains a launch property, so ptxas reports static SHM=0;
  // the matrix records sizeof(TaskStorage) as required dynamic SHM.
  unsigned touch = (threadIdx.x * 257u + blockIdx.x) % sizeof(TaskStorage);
  dynamic_storage[touch] = static_cast<unsigned char>(threadIdx.x);
  tilemega::codegen::TaskContext context;
  context.output = output;
  context.logical_tile = blockIdx.x;
  context.iteration = 1;
  context.extent = extent;

  if (blockIdx.x != 0) Wait(&events[blockIdx.x - 1], 1);
  switch (blockIdx.x % 11) {
    case 0: Gemm::Run(context, storage.legacy_gemm); break;
    case 1: SplitK::Run(context, storage.splitk); break;
    case 2: RuntimeGemm{}(*params, *stage, storage); break;
    case 3: Attn{}(*params, *stage, storage); break;
    case 4: Combine::Run(context, storage.combine); break;
    case 5: Norm{}(*params, *stage, storage); break;
    case 6: Rope{}(*params, *stage, storage); break;
    case 7: Elementwise{}(*params, *stage, storage); break;
    case 8: Kv{}(*params, *stage, storage); break;
    case 9: Moe::Run(context, storage.moe); break;
    default: Scheduler::Run(context, storage.scheduler); break;
  }
  Signal(&events[blockIdx.x], 1);
}

static_assert(sizeof(TaskStorage) == Gemm::Traits::kSharedStorageBytes,
              "TaskBody shared storage must be a max-sized union");
static_assert(!tilemega::arch::Caps<tilemega::arch::Sm120>::kTcgen05,
              "sm_120 must never select the tcgen05 path");
