// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <chrono>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace tilemega::solver {
struct SolverTiming {
  struct Entry { std::uint64_t count=0; double total_ms=0; };
  std::map<std::string,Entry> phases;
  struct Event { std::string candidate,phase; };
  std::vector<Event> events;
  std::string candidate;
  void Add(std::string const& phase,double ms=0,std::uint64_t count=1) {
    auto& e=phases[phase];e.count+=count;e.total_ms+=ms;
    events.push_back({candidate,phase});
  }
  void Write(std::ostream& out,std::string const& solver,std::string const& model,int seq) const {
    out<<"solver\tmodel\tseq\tphase\tcount\ttotal_ms\n";
    for(auto const& [name,e]:phases)
      out<<solver<<'\t'<<model<<'\t'<<seq<<'\t'<<name<<'\t'<<e.count<<'\t'<<e.total_ms<<'\n';
  }
  void WriteEvents(std::ostream& out) const {
    out<<"index\tcandidate\tphase\n";
    for(std::size_t i=0;i<events.size();++i)
      out<<i<<'\t'<<events[i].candidate<<'\t'<<events[i].phase<<'\n';
  }
};
class SolverPhase {
 public:
  SolverPhase(SolverTiming* timing,std::string phase):timing_(timing),phase_(std::move(phase)),start_(Clock::now()) {}
  ~SolverPhase() { if(timing_)timing_->Add(phase_,std::chrono::duration<double,std::milli>(Clock::now()-start_).count()); }
 private:
  using Clock=std::chrono::steady_clock;
  SolverTiming* timing_;std::string phase_;Clock::time_point start_;
};
}
