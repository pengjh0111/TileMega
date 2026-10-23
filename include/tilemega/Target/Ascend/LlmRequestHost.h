// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmPlanABI.h>
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tilemega::ascend::llm_host {
namespace fs=std::filesystem;
#ifdef TILEMEGA_LLM_VARIANT
inline constexpr unsigned kCompiledVariant=TILEMEGA_LLM_VARIANT;
#else
inline constexpr unsigned kCompiledVariant=0;
#endif
inline void Check(aclError code,char const* action) {
  if(code)throw std::runtime_error(std::string(action)+": "+std::to_string(code));
}
template<class T> std::vector<T> Read(fs::path const& path) {
  auto bytes=fs::file_size(path);
  if(bytes%sizeof(T))throw std::invalid_argument("Malformed request table size");
  std::vector<T> data(bytes/sizeof(T));std::ifstream f(path,std::ios::binary);
  f.read(reinterpret_cast<char*>(data.data()),bytes);
  if(!f)throw std::runtime_error("Cannot read request table: "+path.string());return data;
}
inline void Save(fs::path const& path,void const* data,size_t bytes) {
  std::ofstream f(path,std::ios::binary);f.write(static_cast<char const*>(data),bytes);
  if(!f)throw std::runtime_error("Cannot save request evidence");
}
inline void ValidateCompiledBodies(std::vector<llm::Stage> const& stages) {
  for(auto const& s:stages)if(s.kind==6 || (s.kind==0 && s.extent>1)) {
#if !defined(TILEMEGA_LLM_SPLIT_K) || !TILEMEGA_LLM_SPLIT_K
    throw std::invalid_argument("This binary does not register CTG split-K/combine");
#else
    const bool family=(s.extent==3 && s.columns==2048 && (s.k==6144 || s.k==8192)) ||
                      (s.extent==2 && s.columns==3072 && s.k==8192) ||
                      (s.extent==7 && s.columns==1024 && s.k==4096);
    if(!family || s.variant!=1 || s.width!=(s.k+s.extent-1)/s.extent || s.group!=1 ||
        (s.rows!=1 && s.rows!=64 && !(s.rows==16 && s.extent==3 && s.columns==2048 && s.k==8192)) ||
        (s.kind==0?s.epilogue!=0:s.epilogue!=(s.extent==7?0u:1u)))
      throw std::invalid_argument("Unregistered split-K/combine shape or rounding contract");
#endif
  }
  for(auto const& s:stages)if(s.kind==25 &&
      (s.extent!=8 || s.width!=768 || s.group!=1 || s.query_rows!=4 || s.kv_block!=128 || s.variant))
    throw std::invalid_argument("Unregistered QKV pack geometry");
  for(auto const& s:stages)if(s.kind==13 && s.variant>6)
    throw std::invalid_argument("Unregistered native attention implementation");
  for(auto const& s:stages)if(s.kind==1 && s.variant!=0 &&
      ((s.variant!=2 && s.variant!=3) ||
       (s.width!=1024 && s.width!=2048 && !((s.width==3072 || s.width==4096) && s.variant==3))))
    throw std::invalid_argument("Unregistered full-row normalization implementation");
#ifndef TILEMEGA_AS2_CANN_NORM_PRIMITIVES
  for(auto const& s:stages)if((s.kind==13 && (s.variant==5 || s.variant==6)) || (s.kind==1 && (s.variant==2 || s.variant==3)))
    throw std::invalid_argument("Newton QK norm requires explicitly configured pinned CANN device primitives");
#endif
}
struct Plan {
  llm::Metadata m;
  std::vector<llm::Stage> stages;
  std::vector<llm::Task> tasks;
  std::vector<llm::Wait> waits;
  std::vector<llm::Allocation> allocations;
  std::vector<uint32_t> offsets,queue;
  uint32_t minimum_past=0,maximum_past=1086;
  explicit Plan(fs::path const& root) {
    auto metadata=Read<llm::Metadata>(root/"metadata.bin");
    if(metadata.size()!=1)throw std::invalid_argument("Malformed request metadata");m=metadata[0];
    stages=Read<llm::Stage>(root/"stages.bin");tasks=Read<llm::Task>(root/"tasks.bin");
    ValidateCompiledBodies(stages);
    waits=Read<llm::Wait>(root/"waits.bin");allocations=Read<llm::Allocation>(root/"allocations.bin");
    offsets=Read<uint32_t>(root/"offsets.bin");queue=Read<uint32_t>(root/"queue.bin");
    if(m.version!=llm::kVersion || (m.groups!=2 && m.groups!=4 && m.groups!=8) || m.workers!=3*m.groups ||
        m.stages!=stages.size() || !m.tasks || m.tasks!=tasks.size() || m.waits!=waits.size() ||
        m.queue_refs!=queue.size() || offsets.size()!=m.workers+1 || offsets.front()!=0 || offsets.back()!=queue.size() ||
        (m.seq!=1 && m.seq!=64) || (m.batch!=1 && m.batch!=4 && m.batch!=16) || m.capacity!=1088 ||
        !m.epoch_stride || !m.arena_bytes || m.arena_bytes>(64ULL<<30) || m.worker_scratch_bytes!=131072 ||
        m.wait_ticks!=2000000000ULL || m.diagnostic>1)throw std::invalid_argument("Unsupported request plan");
    if(!m.diagnostic) {
      auto bounds=Read<uint32_t>(root/"pricing_bounds.bin");
      if(bounds.size()!=4 || bounds[0]!=1 || bounds[1]>bounds[2] || bounds[2]>1086 || bounds[3]!=m.groups)
        throw std::invalid_argument("Missing/invalid empirical price bounds");
      minimum_past=bounds[1];maximum_past=bounds[2];
    }
    uint64_t previous_end=0;
    for(auto const& a:allocations) {
      if(!a.bytes || a.offset<previous_end || a.offset>m.arena_bytes || a.bytes>m.arena_bytes-a.offset ||
          a.allocated<a.bytes+512 || a.allocated>m.arena_bytes-a.offset || a.required>1 || a.reserved)
        throw std::invalid_argument("Invalid shared allocation range");
      previous_end=a.offset+a.allocated;
    }
    auto valid=[&](uint64_t offset) {
      return offset==llm::kAbsent || std::any_of(allocations.begin(),allocations.end(),
        [&](auto const& a){return a.offset==offset;});
    };
    for(auto const& s:stages) {
      if(s.kind==0 && s.variant!=kCompiledVariant)
        throw std::invalid_argument("Request GEMM geometry differs from compiled Body");
      for(auto offset:s.operands)if(!valid(offset))throw std::invalid_argument("Unbound request operand");
      if(!valid(s.workspace))throw std::invalid_argument("Unbound request workspace");
      if(s.kind==13 && (s.variant==4 || s.variant==5 || s.variant==6)) {
        auto found=std::find_if(allocations.begin(),allocations.end(),[&](auto const& a){return a.offset==s.workspace;});
        if(found==allocations.end() || found->bytes<uint64_t(m.groups)*llm::attention::kGroupBytes)
          throw std::invalid_argument("Cooperative attention lacks its per-group workspace");
      }
    }
    for(unsigned w=0;w<m.workers;++w) {
      if(offsets[w]>offsets[w+1])throw std::invalid_argument("Invalid request FIFO");
      for(unsigned i=offsets[w];i<offsets[w+1];++i) {
        auto id=queue[i];if(id>=tasks.size())throw std::invalid_argument("Invalid request task");
        auto const& t=tasks[id];
        if(t.stage>=stages.size() || t.owner>=m.workers || !t.ordinal || t.ordinal>=m.epoch_stride ||
            t.wait_begin>waits.size() || t.wait_count>waits.size()-t.wait_begin || t.mix_aiv_lanes>1 ||
            (t.owner!=w && !(t.mix_aiv_lanes && t.owner<m.groups && w==m.groups+2*t.owner)))
          throw std::invalid_argument("Invalid request task owner/participant");
      }
    }
    for(auto const& wait:waits)if(wait.worker>=m.workers || !wait.prefix || wait.prefix>=m.epoch_stride)
      throw std::invalid_argument("Invalid request wait");
  }
};
struct DevicePlan {uint8_t *metadata,*stages,*tasks,*waits,*offsets,*queue,*control,*trace;};
struct Device {
  aclrtStream stream=nullptr;bool initialized=false;
  std::vector<void*> owned;
  Device() {
    Check(aclInit(nullptr),"request aclInit");initialized=true;uint32_t count=0;
    Check(aclrtGetDeviceCount(&count),"request device count");
    if(count!=1)throw std::runtime_error("Use the single-device AS-2 wrapper");
    Check(aclrtSetDevice(0),"request set device");auto sku=aclrtGetSocName();
    if(!sku || std::string(sku).find("Ascend950PR")!=0)throw std::runtime_error("Unexpected request SKU");
    Check(aclrtCreateStream(&stream),"request stream");
  }
  ~Device() {
    if(stream)aclrtSynchronizeStream(stream);
    for(auto p:owned)aclrtFree(p);
    if(stream)aclrtDestroyStream(stream);
    if(initialized)aclFinalize();
  }
  uint8_t* Allocate(size_t bytes,void const* source=nullptr) {
    void* p=nullptr;auto size=std::max<size_t>(bytes,512);
    Check(aclrtMalloc(&p,size,ACL_MEM_MALLOC_HUGE_FIRST),"request allocate");owned.push_back(p);
    if(source && bytes)Check(aclrtMemcpy(p,bytes,source,bytes,ACL_MEMCPY_HOST_TO_DEVICE),"request upload");
    else Check(aclrtMemset(p,size,0,size),"request initialize");return static_cast<uint8_t*>(p);
  }
  template<class T> uint8_t* Upload(std::vector<T> const& data) {
    return Allocate(data.size()*sizeof(T),data.data());
  }
  DevicePlan Upload(Plan const& p,unsigned diagnostic) {
    return {Allocate(sizeof(p.m),&p.m),Upload(p.stages),Upload(p.tasks),Upload(p.waits),
      Upload(p.offsets),Upload(p.queue),Allocate((p.m.workers+2*p.m.groups+1)*512),
      Allocate(diagnostic==2?p.m.queue_refs*512ULL:diagnostic?p.m.tasks*64ULL:512)};
  }
};
inline void Complete(Plan const& p,DevicePlan const& d,unsigned epoch,fs::path const& out) {
  auto n=(p.m.workers+2*p.m.groups+1)*512;std::vector<uint32_t> values(n/4);
  Check(aclrtMemcpy(values.data(),n,d.control,n,ACL_MEMCPY_DEVICE_TO_HOST),"request progress download");
  Save(out/"control.bin",values.data(),n);
  if(values[(p.m.workers+2*p.m.groups)*128])throw std::runtime_error("Request device abort; control.bin retained");
  for(unsigned w=0;w<p.m.workers;++w)
    if(values[w*128]!=uint64_t(epoch)*p.m.epoch_stride+p.offsets[w+1]-p.offsets[w])
      throw std::runtime_error("Incomplete request worker prefix");
}
inline void Dump(Plan const& p,DevicePlan const& d,uint8_t* arena,unsigned past,unsigned diagnostic,fs::path const& out) {
  for(unsigned id=0;id<p.allocations.size();++id) {
    auto const& a=p.allocations[id];bool logits=false;
    for(auto const& s:p.stages)if(s.kind==0 && s.epilogue==3 && s.workspace==a.offset)logits=true;
    auto limit=diagnostic==1 && p.m.batch==1?4ULL*1048576:1048576ULL;
    if(!a.required && (a.bytes<=limit || logits)) {
      std::vector<uint8_t> data(a.bytes);
      Check(aclrtMemcpy(data.data(),data.size(),arena+a.offset,data.size(),ACL_MEMCPY_DEVICE_TO_HOST),"request diagnostic output");
      Save(out/(std::to_string(id)+".bin"),data.data(),data.size());
    }
  }
  // Cache snapshots are also available in fast mode, outside request timing.
  // Removing task tracing must not make C-2 cache comparisons impossible.
  for(unsigned i=0;i<p.stages.size();++i) {
    auto const& s=p.stages[i];if(s.kind!=13)continue;
    std::vector<uint8_t> row(uint64_t(p.m.batch)*s.extent*s.width*2);
    for(unsigned side=0;side<2;++side) {
      for(unsigned b=0;b<p.m.batch;++b)for(unsigned g=0;g<s.extent;++g) {
        auto index=uint64_t(b)*s.extent+g;auto n=s.width*2;
        auto source=arena+s.operands[1+side]+(index*p.m.capacity+past+p.m.seq-1)*n;
        Check(aclrtMemcpy(row.data()+index*n,n,source,n,ACL_MEMCPY_DEVICE_TO_HOST),"request KV row");
      }
      Save(out/(std::string(side?"kv_v_s":"kv_k_s")+std::to_string(i)+".bin"),row.data(),row.size());
      unsigned history_limit=diagnostic==1 && p.m.batch==1?p.m.capacity:128;
      if(past+p.m.seq<=history_limit) {
        unsigned valid=past+p.m.seq;std::vector<uint8_t> history(uint64_t(p.m.batch)*s.extent*valid*s.width*2);
        for(unsigned b=0;b<p.m.batch;++b)for(unsigned g=0;g<s.extent;++g) {
          auto index=uint64_t(b)*s.extent+g;auto n=uint64_t(valid)*s.width*2;
          auto source=arena+s.operands[1+side]+index*p.m.capacity*s.width*2;
          Check(aclrtMemcpy(history.data()+index*n,n,source,n,ACL_MEMCPY_DEVICE_TO_HOST),"request short KV history");
        }
        Save(out/(std::string(side?"kv_v_history_s":"kv_k_history_s")+std::to_string(i)+".bin"),history.data(),history.size());
      }
    }
  }
  if(diagnostic) {
    std::vector<uint8_t> data(diagnostic==2?p.m.queue_refs*512ULL:p.m.tasks*64ULL);
    Check(aclrtMemcpy(data.data(),data.size(),d.trace,data.size(),ACL_MEMCPY_DEVICE_TO_HOST),"request trace");
    Save(out/(diagnostic==2?"profile.bin":"trace.bin"),data.data(),data.size());
  }
}
template<class Launch> int RunRequest(fs::path const& binding,fs::path const& out,unsigned outputs,
    unsigned warmup,unsigned repetitions,unsigned diagnostic,Launch launch,bool teacher_forced=false) {
  Plan pre(binding/"prefill"),dec(binding/"decode");
  if(outputs<2 || outputs>1024 || (teacher_forced && (outputs>513 || warmup || repetitions!=1)) ||
      warmup>1 || !repetitions || repetitions>3 || diagnostic>2 || fs::exists(out) ||
      pre.m.seq!=64 || dec.m.seq!=1 || pre.m.batch!=dec.m.batch || pre.m.groups!=dec.m.groups ||
      pre.minimum_past || dec.minimum_past>64 || outputs+62>dec.maximum_past ||
      pre.m.arena_bytes!=dec.m.arena_bytes || pre.allocations.size()!=dec.allocations.size() ||
      std::memcmp(pre.allocations.data(),dec.allocations.data(),pre.allocations.size()*sizeof(llm::Allocation)) ||
      uint64_t(outputs)*dec.m.epoch_stride+dec.m.epoch_stride-1>UINT32_MAX/4)
    throw std::invalid_argument("Incompatible shared plans or bounded request arguments");
  auto reset=Read<uint32_t>(binding/"reset.bin");
  if(reset.size()<2 || reset.size()!=2ULL+reset[1] || reset[0]>=pre.allocations.size())
    throw std::invalid_argument("Invalid request reset metadata");
  auto const& tokens=pre.allocations[reset[0]];
  if(!tokens.required || tokens.bytes!=uint64_t(pre.m.batch)*1088*4)
    throw std::invalid_argument("Invalid request token allocation");
  for(unsigned i=2;i<reset.size();++i)if(reset[i]>=pre.allocations.size() || pre.allocations[reset[i]].required)
    throw std::invalid_argument("Invalid request KV reset allocation");
  auto token_initial=Read<uint32_t>(binding/"inputs"/(std::to_string(reset[0])+".bin"));
  if(token_initial.size()*4!=tokens.bytes)throw std::invalid_argument("Wrong prompt token size");
  auto setup=std::chrono::steady_clock::now();Device device;
  auto arena=device.Allocate(pre.m.arena_bytes),scratch=device.Allocate(pre.m.workers*pre.m.worker_scratch_bytes);
  auto pd=device.Upload(pre,diagnostic),dd=device.Upload(dec,diagnostic);
  std::vector<uint8_t> chunk(1<<20),guard(512,0xa5);
  for(unsigned id=0;id<pre.allocations.size();++id) {
    auto const& a=pre.allocations[id];
    Check(aclrtMemcpy(arena+a.offset+a.bytes,512,guard.data(),512,ACL_MEMCPY_HOST_TO_DEVICE),"request guard upload");
    if(!a.required)continue;
    auto file=binding/"inputs"/(std::to_string(id)+".bin");
    if(fs::file_size(file)!=a.bytes)throw std::invalid_argument("Wrong shared input byte count");
    std::ifstream f(file,std::ios::binary);
    for(uint64_t first=0;first<a.bytes;first+=chunk.size()) {
      auto n=std::min<uint64_t>(chunk.size(),a.bytes-first);f.read(reinterpret_cast<char*>(chunk.data()),n);
      if(!f)throw std::runtime_error("Cannot stream shared weights");
      Check(aclrtMemcpy(arena+a.offset+first,n,chunk.data(),n,ACL_MEMCPY_HOST_TO_DEVICE),"request weight upload");
    }
  }
  fs::create_directory(out);std::ofstream samples(out/"samples.jsonl"),receipts(out/"launches.jsonl");
  samples<<std::setprecision(17);unsigned launches=0;
  auto setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count();
  for(int repetition=-int(warmup);repetition<int(repetitions);++repetition) {
    Check(aclrtSynchronizeStream(device.stream),"request reset boundary");
    auto control_bytes=(pre.m.workers+2*pre.m.groups+1)*512;
    Check(aclrtMemset(pd.control,control_bytes,0,control_bytes),"reset prefill epoch");
    Check(aclrtMemset(dd.control,control_bytes,0,control_bytes),"reset decode epoch");
    Check(aclrtMemcpy(arena+tokens.offset,tokens.bytes,token_initial.data(),tokens.bytes,ACL_MEMCPY_HOST_TO_DEVICE),"reset prompt");
    for(unsigned i=2;i<reset.size();++i) {
      auto const& a=pre.allocations[reset[i]];
      Check(aclrtMemset(arena+a.offset,a.bytes,0,a.bytes),"reset request KV");
    }
    Check(aclrtSynchronizeStream(device.stream),"request t0 boundary");
    auto t0=std::chrono::steady_clock::now();
    launch(pd,arena,scratch,device.stream,pre.m.groups,0,0,diagnostic);++launches;
    Check(aclrtSynchronizeStream(device.stream),"request first token");
    uint32_t prefill_abort=0;
    Check(aclrtMemcpy(&prefill_abort,sizeof(prefill_abort),pd.control+(pre.m.workers+2*pre.m.groups)*512,
      sizeof(prefill_abort),ACL_MEMCPY_DEVICE_TO_HOST),"request prefill abort check");
    if(prefill_abort) {
      // Decode has a separate progress namespace; its zero abort cell must
      // never allow execution after an unsuccessful prefill produced no KV.
      auto failure=out/"failed_prefill";fs::create_directory(failure);
      receipts<<"{\"kernel\":\"tilemega_llm\",\"phase\":\"prefill\",\"repetition\":"<<repetition
        <<",\"forward_index\":0,\"past\":0,\"epoch\":0,\"aborted\":true}\n";receipts.flush();
      Complete(pre,pd,0,failure);
    }
    auto first=std::chrono::steady_clock::now();
    for(unsigned step=1;step<outputs;++step) {
      launch(dd,arena,scratch,device.stream,dec.m.groups,step-1,63+step,diagnostic);++launches;
    }
    Check(aclrtSynchronizeStream(device.stream),"request final token");
    auto last=std::chrono::steady_clock::now();
    auto result=out/("request_"+std::to_string(repetition));fs::create_directory(result);
    auto pre_out=result/"prefill",dec_out=result/"decode";fs::create_directory(pre_out);fs::create_directory(dec_out);
    Complete(pre,pd,0,pre_out);Complete(dec,dd,outputs-2,dec_out);
    if(diagnostic) {
      std::vector<uint8_t> trace(diagnostic==2?pre.m.queue_refs*512ULL:pre.m.tasks*64ULL);
      Check(aclrtMemcpy(trace.data(),trace.size(),pd.trace,trace.size(),ACL_MEMCPY_DEVICE_TO_HOST),"prefill task receipt");
      Save(pre_out/(diagnostic==2?"profile.bin":"trace.bin"),trace.data(),trace.size());
    }
    for(auto const& a:pre.allocations) {
      Check(aclrtMemcpy(guard.data(),512,arena+a.offset+a.bytes,512,ACL_MEMCPY_DEVICE_TO_HOST),"request check guard");
      if(!std::all_of(guard.begin(),guard.end(),[](auto x){return x==0xa5;}))throw std::runtime_error("Shared request guard overwritten");
    }
    std::vector<uint32_t> all(token_initial.size()),generated(uint64_t(pre.m.batch)*outputs);
    Check(aclrtMemcpy(all.data(),tokens.bytes,arena+tokens.offset,tokens.bytes,ACL_MEMCPY_DEVICE_TO_HOST),"request token output");
    for(unsigned b=0;b<pre.m.batch;++b)std::copy_n(all.data()+uint64_t(b)*1088+64,outputs,generated.data()+uint64_t(b)*outputs);
    Save(result/(teacher_forced?"teacher_feedback.bin":"tokens.bin"),generated.data(),generated.size()*4);
    Save(dec_out/"tokens.bin",all.data(),tokens.bytes);
    Dump(dec,dd,arena,outputs+62,diagnostic,dec_out);
    auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(a-b).count();};
    if(teacher_forced)samples<<"{\"request_mode\":\"teacher_forced\",\"timing_admitted\":false,\"decoder_steps\":"<<outputs-1<<"}\n";
    else samples<<"{\"repetition\":"<<repetition<<",\"warmup\":"<<(repetition<0?"true":"false")
      <<",\"output_count\":"<<outputs<<",\"decode_forwards\":"<<outputs-1
      <<",\"ttft_ms\":"<<ms(first,t0)<<",\"e2e_ms\":"<<ms(last,t0)<<",\"tpot_ms\":"<<ms(last,first)/(outputs-1)
      <<",\"output_tokens_per_second\":"<<pre.m.batch*outputs*1000/ms(last,t0)<<"}\n";samples.flush();
    for(unsigned step=0;step<outputs;++step)receipts<<"{\"repetition\":"<<repetition<<",\"kernel\":\"tilemega_llm\",\"forward_index\":"<<step
      <<",\"phase\":\""<<(step?"decode":"prefill")<<"\",\"past\":"<<(step?63+step:0)<<",\"epoch\":"<<(step?step-1:0)
      <<",\"seq\":"<<(step?1:64)<<",\"groups\":"<<pre.m.groups<<",\"logical_device\":0}\n";
    receipts.flush();std::cout<<"request "<<repetition<<" completed; outputs="<<outputs<<std::endl;
  }
  if(!samples || !receipts)throw std::runtime_error("Cannot save request receipts");
  std::ofstream summary(out/"result.json");summary<<std::setprecision(17)
    <<"{\"status\":\"executed_requires_numeric_compare\",\"binding_directory\":"<<std::quoted(binding.string())
    <<",\"isl\":64,\"osl\":"<<outputs<<",\"batch\":"<<pre.m.batch<<",\"capacity\":1088,\"warmup\":"<<warmup
    <<",\"repetitions\":"<<repetitions<<",\"diagnostic\":"<<diagnostic<<",\"host_kernel_launches\":"<<launches
    <<",\"request_mode\":\""<<(teacher_forced?"teacher_forced":"natural_greedy")<<"\""
    <<",\"groups\":"<<pre.m.groups<<",\"workers\":"<<pre.m.workers
    <<",\"arena_bytes\":"<<pre.m.arena_bytes<<",\"setup_seconds\":"<<setup_seconds
    <<",\"priced_plans_executed\":"<<(!pre.m.diagnostic && !dec.m.diagnostic?"true":"false")
    <<",\"complete_generation_length_executed\":"<<(outputs==1024?"true":"false")
    <<",\"optimized_pipeline_admitted\":false,\"correctness_gate\":\"not_compared\",\"performance_gate\":\"not_admitted\"}\n";
  if(!summary)throw std::runtime_error("Cannot save request summary");return 0;
}
} // namespace tilemega::ascend::llm_host
