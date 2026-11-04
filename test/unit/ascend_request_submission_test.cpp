#include <tilemega/Target/Ascend/RequestSubmission.h>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  using namespace tilemega::ascend::llm_host;
  auto require=[](bool b){if(!b)throw std::runtime_error("Submission contract violated");};
  require(DiagnosticSubmissionWindow(nullptr,true,false)==0);
  for(auto value:{"1","8","32","64"})require(DiagnosticSubmissionWindow(value,false,true)>0);
  for(auto value:{"","0","01","65","128","99999999999999999999999","-1","+1","1x"," 1"}) {
    bool rejected=false;try {DiagnosticSubmissionWindow(value,false,true);}catch(std::invalid_argument const&){rejected=true;}
    require(rejected);
  }
  for(bool timed:{false,true}) {
    bool rejected=false;try {DiagnosticSubmissionWindow("8",timed,false);}catch(std::invalid_argument const&){rejected=true;}
    require(rejected);
  }
  bool rejected=false;try {DiagnosticSubmissionWindow("8",true,true);}catch(std::invalid_argument const&){rejected=true;}
  require(rejected);
  unsigned trials=0;
  for(unsigned outputs:{2u,16u,64u,65u,66u,129u,1024u})for(unsigned window:{0u,1u,8u,32u,64u})
    for(unsigned request=0;request<2;++request) {
      std::vector<unsigned> pending,executed;unsigned peak=0,boundaries=0;
      auto drain=[&](unsigned last) {
        require(!pending.empty() && pending.back()==last);
        executed.insert(executed.end(),pending.begin(),pending.end());pending.clear();++boundaries;
      };
      SubmitDecodeForwards(outputs,window,[&](unsigned step) {
        pending.push_back(step);peak=std::max<unsigned>(peak,pending.size());
      },drain);
      if(!window) {require(boundaries==0);drain(outputs-1);}
      require(pending.empty() && executed.size()==outputs-1 && peak<=(window?window:outputs-1));
      require(boundaries==(window?(outputs-2)/window+1:1));
      for(unsigned i=0;i<executed.size();++i)require(executed[i]==i+1);
      ++trials;
    }
  unsigned submitted=0;
  try {
    SubmitDecodeForwards(1024,8,[&](unsigned){++submitted;},[](unsigned){throw std::runtime_error("simulated device abort");});
    return 1;
  }catch(std::runtime_error const&) {require(submitted==8);}
  for(auto args:{std::pair{1u,8u},std::pair{1025u,8u},std::pair{16u,65u}}) {
    submitted=0;bool rejected=false;
    try {SubmitDecodeForwards(args.first,args.second,[&](unsigned){++submitted;},[](unsigned){});}
    catch(std::invalid_argument const&) {rejected=true;}
    require(rejected && !submitted);
  }
  std::cout<<"{\"status\":\"pass\",\"scope\":\"CPU Host submission contract only\",\"reset_trials\":"<<trials
    <<",\"full_request_decode_forwards\":1023,\"acl_initialized\":false}\n";
}
