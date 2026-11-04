// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <stdexcept>

namespace tilemega::ascend::llm_host {
inline unsigned DiagnosticSubmissionWindow(char const* value,bool performance_only,bool host_progress) {
  if(!value)return 0;
  if(performance_only || !host_progress)
    throw std::invalid_argument("Submission window requires diagnostic Host progress and cannot be timed");
  unsigned window=0;
  if(*value=='0' || !*value)throw std::invalid_argument("Diagnostic submission window must be1..64");
  for(auto p=value;*p;++p) {
    if(*p<'0' || *p>'9' || window>6)
      throw std::invalid_argument("Diagnostic submission window must be1..64");
    window=window*10+unsigned(*p-'0');
  }
  if(window>64)throw std::invalid_argument("Diagnostic submission window must be1..64");
  return window;
}

// A window bounds Host submissions only. It changes no device Plan, token,
// epoch or forward count, and never substitutes Host work for a model task.
template<class Submit,class Drain> inline void SubmitDecodeForwards(unsigned outputs,
    unsigned window,Submit submit,Drain drain) {
  if(outputs<2 || outputs>1024 || window>64)throw std::invalid_argument("Unbounded request submission");
  if(!window) {
    for(unsigned step=1;step<outputs;++step)submit(step);
    return;
  }
  for(unsigned step=1;step<outputs;++step) {
    submit(step);
    if(step%window==0 || step+1==outputs)drain(step);
  }
}
} // namespace tilemega::ascend::llm_host
