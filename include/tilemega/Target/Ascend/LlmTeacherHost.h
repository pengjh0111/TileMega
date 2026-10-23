// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Target/Ascend/LlmRequestHost.h>

namespace tilemega::ascend::llm_host {
inline std::vector<unsigned> TeacherCheckpoints(unsigned steps) {
  std::vector<unsigned> result{0,steps};
  for(unsigned n:{1u,2u,64u,65u,66u,192u,193u,448u,449u})if(n<=steps)result.push_back(n);
  std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());
  return result;
}

template<class Launch> int RunTeacher(fs::path const& binding,fs::path const& trace,fs::path const& out,
    unsigned steps,Launch launch) {
  Plan pre(binding/"prefill"),dec(binding/"decode");
  auto header=Read<uint32_t>(trace/"header.bin"),inputs=Read<uint32_t>(trace/"tokens.bin");
  auto reset=Read<uint32_t>(binding/"reset.bin");
  if(header.size()!=5 || header[0]!=1 || header[1]!=16 || header[2]!=512 || header[3]!=1088 ||
      !steps || steps>512 || inputs.size()!=16*512 || (pre.m.batch!=1 && pre.m.batch!=16) ||
      reset.empty() || reset[0]>=pre.allocations.size())throw std::invalid_argument("Invalid bounded teacher trace");
  auto tokens=pre.allocations[reset[0]];
  auto head=[](Plan const& p) {
    auto found=std::find_if(p.stages.begin(),p.stages.end(),[](auto const& s){return s.kind==0 && s.epilogue==3;});
    if(found==p.stages.end() || std::count_if(p.stages.begin(),p.stages.end(),[](auto const& s){return s.kind==0 && s.epilogue==3;})!=1)
      throw std::invalid_argument("Teacher requires the complete vocabulary projection");
    return *found;
  };
  auto ph=head(pre),dh=head(dec);
  if(ph.columns!=header[4] || dh.columns!=header[4] || !header[4] ||
      std::any_of(inputs.begin(),inputs.end(),[&](auto n){return n>=header[4];}))
    throw std::invalid_argument("Teacher token/vocabulary mismatch");
  auto checkpoints=TeacherCheckpoints(steps);
  std::vector<uint32_t> predictions(uint64_t(pre.m.batch)*(steps+1));
  std::vector<uint8_t> logits(uint64_t(pre.m.batch)*header[4]*2);
  unsigned observed=0;
  auto callback=[&](DevicePlan const& p,uint8_t* arena,uint8_t* scratch,aclrtStream stream,
      unsigned groups,unsigned epoch,unsigned past,unsigned diagnostic) {
    unsigned step=past?past-63:0;
    if(step!=observed++ || step>steps)throw std::runtime_error("Unexpected teacher forward order");
    if(past)for(unsigned b=0;b<pre.m.batch;++b) {
      auto target=arena+tokens.offset+(uint64_t(b)*1088+past)*4;
      Check(aclrtMemcpy(target,4,&inputs[b*512+step-1],4,ACL_MEMCPY_HOST_TO_DEVICE),"teacher input token");
    }
    launch(p,arena,scratch,stream,groups,epoch,past,diagnostic);
    // This diagnostic mode observes each forward and is never a timing path.
    Check(aclrtSynchronizeStream(stream),"teacher forward boundary");
    uint32_t abort=0;
    Check(aclrtMemcpy(&abort,4,p.control+(pre.m.workers+2*groups)*512,4,ACL_MEMCPY_DEVICE_TO_HOST),"teacher abort");
    if(abort) {
      auto failed=out/("teacher_failed_"+std::to_string(step));fs::create_directory(failed);
      Complete(past?dec:pre,p,epoch,failed);
    }
    unsigned output_position=past?past+1:64;
    for(unsigned b=0;b<pre.m.batch;++b)
      Check(aclrtMemcpy(&predictions[b*(steps+1)+step],4,arena+tokens.offset+(uint64_t(b)*1088+output_position)*4,
        4,ACL_MEMCPY_DEVICE_TO_HOST),"teacher predicted token");
    if(std::binary_search(checkpoints.begin(),checkpoints.end(),step)) {
      auto directory=out/"teacher_checkpoints";fs::create_directories(directory);
      Check(aclrtMemcpy(logits.data(),logits.size(),arena+(past?dh.workspace:ph.workspace),logits.size(),
        ACL_MEMCPY_DEVICE_TO_HOST),"teacher complete logits");
      Save(directory/("logits_s"+std::to_string(step)+".bin"),logits.data(),logits.size());
    }
  };
  auto result=RunRequest(binding,out,steps+1,0,1,1,callback,true);
  if(observed!=steps+1)throw std::runtime_error("Incomplete teacher sequence");
  Save(out/"teacher_predictions.bin",predictions.data(),predictions.size()*4);
  std::ofstream record(out/"teacher.json");
  record<<"{\"request_mode\":\"teacher_forced\",\"timing_admitted\":false,\"trace_directory\":"<<std::quoted(trace.string())
    <<",\"decoder_steps\":"<<steps<<",\"model_forwards\":"<<steps+1<<",\"batch\":"<<pre.m.batch
    <<",\"vocab\":"<<header[4]<<",\"checkpoints\":[";
  for(unsigned i=0;i<checkpoints.size();++i){if(i)record<<',';record<<checkpoints[i];}
  record<<"]}\n";if(!record)throw std::runtime_error("Cannot save teacher receipt");return result;
}
} // namespace tilemega::ascend::llm_host
