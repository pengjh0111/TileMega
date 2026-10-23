// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Codegen/tasks/ServingAbi.h>
#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Included after codegen defines NativeProgram, kNativeBuffers, kNativeStages
// and kNativeEpsilon in native_dense. These are host recipes and constexpr
// types; no device-side descriptor table or kind/variant dispatch is emitted.
namespace tilemega::codegen::native_dense {

static_assert(kNativeBuffers.size()==NativeProgram::kBuffers);
static_assert(kNativeStages.size()==NativeProgram::kStages);
static_assert(kNativeEpsilon>0);

inline void Check(cudaError_t status) {
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
inline int Grid() {
  int device=0;cudaDeviceProp properties{};
  Check(cudaGetDevice(&device));Check(cudaGetDeviceProperties(&properties,device));
  if(properties.major!=9 || properties.minor!=0)
    throw std::invalid_argument("native dense plans require Hopper");
  return properties.multiProcessorCount*NativeProgram::kResidency;
}
struct Plan {
  NativeProgram::Binding bindings{};
  std::vector<void*> owned;
  int *pasts=nullptr,grid=0;
  unsigned steps=0;
  unsigned long long iteration=0;
  unsigned long long* stamps=nullptr;
  GridCounter* events=nullptr;
  TraceState trace{};
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
  boundary_diagnostic::Config boundary_config{};
  std::vector<int> boundary_pasts;
  std::vector<unsigned long long> boundary_iterations;
  std::vector<int> boundary_slot_pasts;
#endif
  ~Plan() {
    for(auto* p:owned)cudaFree(p);
    cudaFree(pasts);cudaFree(stamps);cudaFree(events);
    cudaFree(trace.stages);cudaFree(trace.steps);
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
    cudaFree(trace.boundary_releases);
#endif
#if TILEMEGA_TRACE_NATIVE_PHASES
    cudaFree(trace.diagnostics);
#endif
  }
};
template<class T>
inline T* Allocate(std::size_t count) {
  T* result=nullptr;
  Check(cudaMalloc(&result,count*sizeof(T)));
  auto status=cudaMemset(result,0,count*sizeof(T));
  if(status!=cudaSuccess) {cudaFree(result);Check(status);}
  return result;
}
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
inline void ConfigureBoundary(Plan& p,boundary_diagnostic::Config config) {
  Check(cudaDeviceSynchronize());
  auto count=std::size_t(p.trace.slots)*NativeProgram::kStages*p.grid;
  Check(cudaMemset(p.trace.stages,0,count*sizeof(StageTrace)));
  Check(cudaMemset(p.trace.boundary_releases,0,count*sizeof(unsigned long long)));
  p.boundary_iterations.assign(p.trace.slots,~0ull);
  p.boundary_slot_pasts.assign(p.trace.slots,-1);
  p.boundary_config=config;
  p.trace.boundary_observe=config.observe;p.trace.boundary_probe_mask=config.probe_mask;
}
#endif
inline int Launch(Plan& p,unsigned first,unsigned count,unsigned mode,
                  unsigned long long iteration,void* stream) {
  if(mode!=TM_SERVING_L1 || !count || first>=p.steps || count>p.steps-first ||
     iteration!=p.iteration)return -1;
  sm90_dense_native_kernel<NativeProgram><<<p.grid,NativeProgram::kThreads,
      NativeProgram::kSharedBytes,static_cast<cudaStream_t>(stream)>>>(p.bindings,p.pasts,
      first,count,kNativeEpsilon,p.events,iteration,p.stamps,p.trace);
  auto status=cudaGetLastError();
  if(status==cudaSuccess) {
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
    for(unsigned i=0;i<count;++i) {
      auto slot=(iteration+i)%p.trace.slots;
      p.boundary_iterations[slot]=iteration+i;p.boundary_slot_pasts[slot]=p.boundary_pasts[first+i];
    }
#endif
    p.iteration+=count;
  }
  return int(status);
}

}  // namespace tilemega::codegen::native_dense

extern "C" int tm_plan_static_dense_abi() {return 1;}
extern "C" unsigned tm_plan_pdl_enabled(void*) {return 0;}
extern "C" unsigned tm_plan_cluster_dim() {return 1;}
extern "C" unsigned tm_plan_requested_residency() {
  return tilemega::codegen::native_dense::NativeProgram::kResidency;
}
extern "C" unsigned tm_plan_loop_modes(void* p) {return p?TM_SERVING_L1:0;}
extern "C" unsigned long long tm_plan_shared_bytes() {
  return tilemega::codegen::native_dense::NativeProgram::kSharedBytes;
}
extern "C" unsigned long long tm_plan_execution_smem(void*,unsigned mode) {
  return mode==TM_SERVING_L1?tm_plan_shared_bytes():0;
}
extern "C" int tm_plan_watchdog(void*,std::uint64_t[12]) {return 0;}
extern "C" int tm_plan_query(tm_plan_info* output) {
  using namespace tilemega::codegen::native_dense;
  if(!output)return -1;
  try {
    int grid=Grid();
    *output={TM_SERVING_ABI_VERSION,TM_SERVING_DECODE,NativeProgram::kBatch,
        NativeProgram::kBatch,1,64,NativeProgram::kCapacity-2,NativeProgram::kCapacity,
        grid,NativeProgram::kResidency,TM_SERVING_L1,NativeProgram::kBuffers};
    return 0;
  }catch(std::exception const& e) {std::fprintf(stderr,"tm_plan_query: %s\n",e.what());return -2;}
}
extern "C" int tm_plan_buffer(unsigned index,tm_buffer_info* output) {
  using namespace tilemega::codegen::native_dense;
  if(!output || index>=kNativeBuffers.size())return -1;
  *output=kNativeBuffers[index];return 0;
}
extern "C" void* tm_plan_create(int batch,void* const* external,int device) {
  using namespace tilemega::codegen::native_dense;
  if(batch!=NativeProgram::kBatch || !external)return nullptr;
  try {
    Check(cudaSetDevice(device));auto plan=std::make_unique<Plan>();plan->grid=Grid();
    auto kernel=sm90_dense_native_kernel<NativeProgram>;
    Check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,NativeProgram::kSharedBytes));
    int resident=0;
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,
        NativeProgram::kThreads,NativeProgram::kSharedBytes));
    if(resident<NativeProgram::kResidency)
      throw std::invalid_argument("native dense plan exceeds measured residency");
    for(unsigned i=0;i<kNativeBuffers.size();++i) {
      auto const& desc=kNativeBuffers[i];
      auto elements=desc.elements_constant+desc.elements_per_batch*batch;
      void* pointer=nullptr;
      if(desc.role==TM_BUFFER_EXTERNAL) {
        pointer=external[i];
        if(elements && !pointer)throw std::invalid_argument("unbound native external buffer");
      }else if(elements) {
        auto bytes=elements*(desc.dtype==TM_DTYPE_BF16?2:4);
        Check(cudaMalloc(&pointer,bytes));plan->owned.push_back(pointer);
        Check(cudaMemset(pointer,0,bytes));
      }
      if(pointer && reinterpret_cast<std::uintptr_t>(pointer)%16)
        throw std::invalid_argument("native dense buffers require 16-byte alignment");
      plan->bindings.buffers[i]=static_cast<Element*>(pointer);
    }
    plan->events=Allocate<GridCounter>(NativeProgram::kStages);
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY
    auto env=std::getenv("TILEMEGA_TRACE_LAUNCHES");
    unsigned long long slots=env?std::strtoull(env,nullptr,10):16;
    if(!slots || slots>4096)throw std::invalid_argument("invalid native trace slot count");
    plan->trace.slots=unsigned(slots);
#if TILEMEGA_TRACE_STAGE
    plan->trace.stages=Allocate<StageTrace>(slots*NativeProgram::kStages*plan->grid);
#endif
#if TILEMEGA_TRACE_STEP
    plan->trace.steps=Allocate<StepTrace>(slots*plan->grid);
#endif
#if TILEMEGA_TRACE_NATIVE_PHASES
    plan->trace.diagnostics=Allocate<tilemega::backend::Sm90DenseDiagnosticRecord>(slots*NativeProgram::kTotalTasks);
#endif
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
    for(unsigned i=0;i<NativeProgram::kStages;++i)
      if(!boundary_diagnostic::Matches(boundary_diagnostic::ClassOf(i,NativeProgram::kStages),
          kNativeStages[i].kind,kNativeStages[i].name))
        throw std::invalid_argument("native boundary diagnostic stage class mismatch");
    plan->trace.stages=Allocate<StageTrace>(slots*NativeProgram::kStages*plan->grid);
    plan->trace.boundary_releases=Allocate<unsigned long long>(slots*NativeProgram::kStages*plan->grid);
    auto mode=std::getenv("TILEMEGA_NATIVE_BOUNDARY_MODE");
    auto selected=std::getenv("TILEMEGA_NATIVE_BOUNDARY_CLASS");
    ConfigureBoundary(*plan,boundary_diagnostic::Parse(mode?mode:"off",selected?selected:""));
#endif
#endif
    return plan.release();
  }catch(std::exception const& e) {std::fprintf(stderr,"tm_plan_create: %s\n",e.what());return nullptr;}
}
extern "C" int tm_plan_set_steps(void* opaque,int const* past,unsigned count) {
  using namespace tilemega::codegen::native_dense;
  auto* p=static_cast<Plan*>(opaque);
  if(!p || !past || !count)return -1;
  for(unsigned i=0;i<count;++i)
    if(past[i]<64 || past[i]>NativeProgram::kCapacity-2)return -1;
  try {
    int* next=Allocate<int>(count);
    auto status=cudaMemcpy(next,past,std::size_t(count)*sizeof(int),cudaMemcpyHostToDevice);
    if(status!=cudaSuccess) {cudaFree(next);Check(status);}
    unsigned long long* stamps=nullptr;
    try {stamps=Allocate<unsigned long long>(std::size_t(count)+1);}
    catch(...) {cudaFree(next);throw;}
    cudaFree(p->pasts);cudaFree(p->stamps);p->pasts=next;p->stamps=stamps;p->steps=count;
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
    p->boundary_pasts.assign(past,past+count);
#endif
    return 0;
  }catch(std::exception const& e) {std::fprintf(stderr,"tm_plan_set_steps: %s\n",e.what());return -2;}
}
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
extern "C" int tm_plan_set_boundary_diagnostic(void* opaque,char const* mode,char const* selected) {
  using namespace tilemega::codegen::native_dense;
  auto* p=static_cast<Plan*>(opaque);
  if(!p || !mode)return -1;
  try {ConfigureBoundary(*p,boundary_diagnostic::Parse(mode,selected?selected:""));return 0;}
  catch(std::exception const& e) {
    std::fprintf(stderr,"tm_plan_set_boundary_diagnostic: %s\n",e.what());return -2;
  }
}
#endif
extern "C" int tm_plan_launch(void* opaque,unsigned step,unsigned mode,
                               std::uint64_t iteration,void* stream) {
  using namespace tilemega::codegen::native_dense;
  return opaque?Launch(*static_cast<Plan*>(opaque),step,1,mode,iteration,stream):-1;
}
extern "C" int tm_plan_launch_steps(void* opaque,unsigned first,unsigned count,
    unsigned mode,std::uint64_t iteration,void* stream) {
  using namespace tilemega::codegen::native_dense;
  return opaque?Launch(*static_cast<Plan*>(opaque),first,count,mode,iteration,stream):-1;
}
extern "C" int tm_plan_read_step_ns(void* opaque,unsigned first,unsigned count,
                                    std::uint64_t* host) {
  using namespace tilemega::codegen::native_dense;
  auto* p=static_cast<Plan*>(opaque);
  if(!p || !p->stamps || !host || first>p->steps || count>p->steps-first+1)return -1;
  return int(cudaMemcpy(host,p->stamps+first,std::size_t(count)*8,cudaMemcpyDeviceToHost));
}
extern "C" int tm_plan_read_buffer(void* opaque,unsigned index,void* host,std::uint64_t bytes) {
  using namespace tilemega::codegen::native_dense;
  auto* p=static_cast<Plan*>(opaque);
  if(!p || !host || index>=kNativeBuffers.size())return -1;
  auto const& desc=kNativeBuffers[index];
  auto maximum=(desc.elements_constant+desc.elements_per_batch*NativeProgram::kBatch)*
      (desc.dtype==TM_DTYPE_BF16?2:4);
  if(bytes>maximum)return -1;
  return int(cudaMemcpy(host,p->bindings.buffers[index],bytes,cudaMemcpyDeviceToHost));
}
extern "C" void tm_plan_destroy(void* opaque) {
  delete static_cast<tilemega::codegen::native_dense::Plan*>(opaque);
}

#if TILEMEGA_TRACE_NATIVE_PHASES
__global__ void sm90_dense_diagnostic_clock(unsigned long long* output) {
  auto begin=tilemega::backend::Sm90DenseDiagnostic::Global();auto cycles=clock64();
  while(tilemega::backend::Sm90DenseDiagnostic::Global()-begin<1000000ull) {}
  output[0]=clock64()-cycles;output[1]=tilemega::backend::Sm90DenseDiagnostic::Global()-begin;
}
#endif
#if TILEMEGA_TRACE_STAGE || TILEMEGA_TRACE_STEP || TILEMEGA_TRACE_NATIVE_PHASES || TILEMEGA_TRACE_NATIVE_BOUNDARY
extern "C" int tm_plan_dump_serving_trace(void* opaque,char const* directory) {
  using namespace tilemega::codegen::native_dense;
  auto* p=static_cast<Plan*>(opaque);
  if(!p || !directory || cudaDeviceSynchronize()!=cudaSuccess)return -1;
  auto dump=[&](char const* file,auto* data,std::size_t count,char const* header) {
    using Row=std::remove_pointer_t<decltype(data)>;
    std::vector<Row> host(count);
    if(cudaMemcpy(host.data(),data,count*sizeof(Row),cudaMemcpyDeviceToHost)!=cudaSuccess)return false;
    auto* stream=std::fopen((std::string(directory)+"/"+file).c_str(),"w");
    if(!stream)return false;
    std::fprintf(stream,"%s\n",header);
    for(std::size_t i=0;i<count;++i) {
      auto const* words=reinterpret_cast<unsigned long long const*>(&host[i]);
      if(!words[2])continue;
      std::fprintf(stream,"%zu",i%p->grid);
      for(unsigned j=0;j<sizeof(Row)/8;++j)std::fprintf(stream,"\t%llu",words[j]);
      std::fputc('\n',stream);
    }
    return std::fclose(stream)==0;
  };
#if TILEMEGA_TRACE_STAGE
  if(!dump("stage_trace.tsv",p->trace.stages,std::size_t(p->trace.slots)*NativeProgram::kStages*p->grid,
      "worker\titeration\tpast\tt_begin\tt_tasks_end\tt_release\ttasks\tsmid\tstage"))return -2;
#endif
#if TILEMEGA_TRACE_STEP
  if(!dump("step_trace.tsv",p->trace.steps,std::size_t(p->trace.slots)*p->grid,
      "worker\titeration\tpast\tkernel_begin\tkernel_end\tfirst_task\tlast_task\ttoken_lag\tkv_lag\tlast_barrier_wait\tstage_has_tasks"))return -2;
#endif
#if TILEMEGA_TRACE_NATIVE_BOUNDARY
  {
    auto count=std::size_t(p->trace.slots)*NativeProgram::kStages*p->grid;
    std::vector<unsigned long long> releases(count);
    if(cudaMemcpy(releases.data(),p->trace.boundary_releases,count*8,cudaMemcpyDeviceToHost)!=cudaSuccess)return -4;
    auto* stream=std::fopen((std::string(directory)+"/boundary_release.tsv").c_str(),"w");
    if(!stream)return -4;
    std::fprintf(stream,"worker\titeration\tpast\tstage\tclass\tt_release\n");
    for(unsigned slot=0;slot<p->trace.slots;++slot) {
      if(p->boundary_iterations[slot]==~0ull)continue;
      for(unsigned stage=0;stage<NativeProgram::kStages;++stage)
        for(int worker=0;worker<p->grid;++worker) {
          auto value=releases[(std::size_t(slot)*NativeProgram::kStages+stage)*p->grid+worker];
          if(value)std::fprintf(stream,"%d\t%llu\t%d\t%u\t%s\t%llu\n",worker,
              p->boundary_iterations[slot],p->boundary_slot_pasts[slot],stage,
              boundary_diagnostic::kNames[static_cast<unsigned>(boundary_diagnostic::ClassOf(stage,NativeProgram::kStages))],value);
        }
    }
    if(std::fclose(stream))return -4;
    if(!dump("boundary_probes.tsv",p->trace.stages,count,
        "worker\titeration\tpast\tt_begin\tt_tasks_end\tt_release\ttasks\tsmid\tstage"))return -4;
    auto* manifest=std::fopen((std::string(directory)+"/boundary_diagnostic.json").c_str(),"w");
    if(!manifest)return -4;
    std::fprintf(manifest,"{\n  \"schema\": 1, \"performance_eligible\": false,\n"
        "  \"mode\": \"%s\", \"selected_class\": \"%s\", \"probe_mask\": %u,\n"
        "  \"observation\": \"%s\", \"all_cta_release\": %s,\n"
        "  \"grid\": %d, \"stages\": %u, \"slots\": %u,\n"
        "  \"attribution\": \"conditional marginal; no additive correction\",\n"
        "  \"class_map\": [\n",p->boundary_config.mode,p->boundary_config.selected_class,
        p->trace.boundary_probe_mask,p->trace.boundary_observe?"release_after_cta_barrier":"none",
        p->trace.boundary_observe?"true":"false",p->grid,NativeProgram::kStages,p->trace.slots);
    for(unsigned i=0;i<NativeProgram::kStages;++i) {
      auto c=boundary_diagnostic::ClassOf(i,NativeProgram::kStages);
      std::fprintf(manifest,"    {\"stage\": %u, \"class\": \"%s\", \"extra_probe\": %s}%s\n",i,
          boundary_diagnostic::kNames[static_cast<unsigned>(c)],
          (p->trace.boundary_probe_mask&boundary_diagnostic::Bit(c))?"true":"false",
          i+1==NativeProgram::kStages?"":",");
    }
    std::fprintf(manifest,"  ]\n}\n");if(std::fclose(manifest))return -4;
  }
#endif
#if TILEMEGA_TRACE_NATIVE_PHASES
  {
    using Row=tilemega::backend::Sm90DenseDiagnosticRecord;
    static_assert(sizeof(Row)==29*8);
    std::vector<Row> rows(std::size_t(p->trace.slots)*NativeProgram::kTotalTasks);
    Check(cudaMemcpy(rows.data(),p->trace.diagnostics,rows.size()*sizeof(Row),cudaMemcpyDeviceToHost));
    auto* phases=std::fopen((std::string(directory)+"/native_phases.tsv").c_str(),"w");
    if(!phases)return -4;
    std::fprintf(phases,"iteration\tpast\tstage\ttask\tworker\tsmid\tglobal_begin\tglobal_end\tprevious_release_global\tclock_begin\tclock_end\tprevious_release_0\tprevious_release_1\tprevious_release_2\tprevious_release_3\tfirst_mma_0\tfirst_mma_1\tfirst_mma_2\tfirst_mma_3\tquery_0\tquery_1\tquery_2\tquery_3\tnew_row_cycles\twait_cycles\tcompute_cycles\tpartial_cycles\tpublish_cycles\tmerge_cycles\n");
    for(auto const& row:rows)if(row.global_begin) {
      auto const* words=reinterpret_cast<unsigned long long const*>(&row);
      for(unsigned i=0;i<29;++i)std::fprintf(phases,i?"\t%llu":"%llu",words[i]);
      std::fputc('\n',phases);
    }
    if(std::fclose(phases))return -4;
    auto* clock=std::fopen((std::string(directory)+"/native_clock.tsv").c_str(),"w");
    if(!clock)return -4;
    std::fprintf(clock,"cycles\tglobal_ns\n");
    auto* values=Allocate<unsigned long long>(2);
    for(int sample=0;sample<5;++sample) {
      sm90_dense_diagnostic_clock<<<1,1>>>(values);unsigned long long host[2];
      Check(cudaMemcpy(host,values,16,cudaMemcpyDeviceToHost));
      std::fprintf(clock,"%llu\t%llu\n",host[0],host[1]);
    }
    cudaFree(values);if(std::fclose(clock))return -4;
  }
#endif
  auto* stream=std::fopen((std::string(directory)+"/runtime_stages.tsv").c_str(),"w");
  if(!stream)return -3;
  std::fprintf(stream,"stage\tkind\tkind_name\tname\tweight_bytes\textent\tgroup\twidth\tkv_block\telided\treducer\tnative_chunks\n");
  for(unsigned i=0;i<kNativeStages.size();++i) {
    auto const& stage=kNativeStages[i];
    std::fprintf(stream,"%u\t0\t%s\t%s\t%llu\t%u\t%u\t%u\t%u\t0\t4294967295\t%u\n",
        i,stage.kind,stage.name,stage.weight_bytes,stage.heads,stage.group,stage.width,
        NativeProgram::kCapacity,stage.chunks);
  }
  return std::fclose(stream)==0?0:-3;
}
#endif
